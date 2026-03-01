import copy
import random

import torch
from torch.nn import functional as F
from .utils import parent_module, brackets_to_periods, EarlyStopMeter, EditingMeanAct
from ...models.quantization.quantizer import W8A16Model
import transformers
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from .merge import slerp, GTA, linear
import torch.nn as nn
import gc
import time


class WISEZeroOrderTrainer:
    def __init__(self, config, model, zo_eps=1e-5, device='cuda:0'):
        self.config = config
        self.model = model  # WISE实例
        self.device = device
        self.zo_eps = zo_eps
        self.current_iteration = 0
        self.max_iterations = config.n_iter

        self.initial_lr = config.edit_lr

    def _get_learning_rate(self):
        """
        动态学习率：
        - Start with warmup: small -> peak.
        - Then cosine decay to small.
        """
        warmup_ratio = 1  # 比如前10%步是warmup
        max_lr = self.initial_lr   # 配置最大学习率
        min_lr = self.initial_lr * 0.01  # 最小学习率是最大值的1%

        warmup_steps = int(self.max_iterations * warmup_ratio)
        step = self.current_iteration

        if step < warmup_steps:
            # Linear Warmup: 从 0 -> max_lr
            lr = max_lr * (step / warmup_steps)
        else:
            # Cosine Annealing
            progress = (step - warmup_steps) / (self.max_iterations - warmup_steps)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

        return lr
    
    def generate_activation_mask(self, mask_ratio=0.1):
        adapter_layer = self.model.get_adapter_layer()
        p_grad = adapter_layer.new_weight.reshape(-1)
        p_mask = np.random.choice(
            [1, 0], size=p_grad.size(0), p=[mask_ratio, 1-mask_ratio]
        )
        p_mask = torch.from_numpy(p_mask).float().to(p_grad.device)
        self.weight_mask = p_mask

    def calculate_jvp(self, func, param, v, h):
        with torch.no_grad():
            f_plus = func(param + h * v).detach()
            f_minus = func(param - h * v).detach()

            avg_loss = (f_plus + f_minus) / 2
            jvp = (f_plus - f_minus) / (2 * h)

        gc.collect()
        torch.cuda.empty_cache()
        print(f"loss:{avg_loss}, jvp:{jvp}")
        return avg_loss, jvp

    def zero_order_loss(self, tokens, new_weight, act_mask=None, deact_mask=None):
        adapter_layer = self.model.get_adapter_layer()

        # 暂时替换new_weight
        old_weight = adapter_layer.new_weight.data.clone()
        adapter_layer.new_weight.data.copy_(new_weight)

        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        ft_loss = self.model._WISE__cal_ft_loss(tokens, last_prompt_token_loc)
        act_loss = self.model._WISE__cal_activation_loss(
            adapter_layer.original_layer_output,
            adapter_layer.new_weight_layer_output,
            config=self.config,
            act_mask=act_mask,
            deact_mask=deact_mask
        )
        print(f"ft_loss:{ft_loss}, act_loss:{act_loss}")

        total_loss = ft_loss

        adapter_layer.new_weight.data.copy_(old_weight)  # 恢复
        return total_loss

    def forward_grad_step(self, tokens, act_mask=None, deact_mask=None):
        """
        执行完整 Zero-Order 训练 n_iter步。
        """
        adapter_layer = self.model.get_adapter_layer()
        param = adapter_layer.new_weight
        self.generate_activation_mask(mask_ratio=0.05)

        loss_meter = EarlyStopMeter()
        # old_weight = param.data.clone()
        # previous_loss = None

        v_nums = 10  # v向量数量
        losses = []

        for step in range(self.max_iterations):
            self.current_iteration = step
            v_buffer = []
            jvp_buffer = []

            torch.manual_seed(42)  # 避免随机偏差

            for _ in range(v_nums):
                vs = torch.normal(mean=0, std=1, size=param.size(), device=param.device)
                if hasattr(self, 'weight_mask'):
                    vs = vs * self.weight_mask.view_as(vs)  # mask掉无关扰动
                v_buffer.append(vs)

            f = lambda p: self.zero_order_loss(tokens, p, act_mask=act_mask, deact_mask=deact_mask)

            for vs in v_buffer:
                loss_avg, jvp = self.calculate_jvp(f, param, vs, self.zo_eps)
                jvp_buffer.append(jvp)

            grad_estimate = torch.zeros_like(param)
            with torch.no_grad():
                for i, vs in enumerate(v_buffer):
                    grad_estimate += vs * jvp_buffer[i]
                # grad_estimate /= v_nums

            if hasattr(self, 'weight_mask'):
                grad_estimate *= self.weight_mask.view_as(grad_estimate)

                # 更新参数
            adapter_layer.new_weight.data -= self._get_learning_rate() * grad_estimate
            # cur_loss = self.zero_order_loss(tokens, adapter_layer.new_weight, act_mask, deact_mask).item()
            #     # gradient mask处理
            #     # adapter_layer.mask_new_weight_gradient()
            # # 回溯判定
            # if previous_loss is not None and cur_loss > previous_loss:
            #     # 恢复参数（Rollback）
            #     adapter_layer.new_weight.data.copy_(old_weight)
            #     print(f"Step [{step+1}/{self.max_iterations}]: Loss↑ ({cur_loss:.6f} > {previous_loss:.6f}) ROLLBACK!")

            # else:
            #     # 接受更新，保存这步loss和参数
            #     old_weight = adapter_layer.new_weight.data.clone()
            #     previous_loss = cur_loss

            # 保存loss
            losses.append(loss_avg.item())
            loss_meter.update(loss_avg.item())

            print(f"[{step+1}/{self.max_iterations}] Loss={loss_avg.item():.4f}")

            # early stopping
            if loss_meter.stop():
                print("Early stopping triggered.")
                adapter_layer.save_editing_activation()
                break

            if step == self.max_iterations - 1:
                adapter_layer.save_editing_activation()

            # Norm约束
            if isinstance(self.config.norm_constraint, float):
                self.__norm_constraint(self.config.norm_constraint)

            torch.cuda.empty_cache()

        return losses

    def __norm_constraint(self, norm_constraint):
        """对new_weight做 norm范围限制"""
        adapter_layer = self.model.get_adapter_layer()
        new_weight = adapter_layer.new_weight
        original_weight = adapter_layer.weight

        with torch.no_grad():
            new_weight[...] = torch.clamp(
                new_weight,
                min=original_weight - norm_constraint,
                max=original_weight + norm_constraint
            )