import torch
import numpy as np
from functools import partial
from torch.autograd.functional import jvp
from ...util import nethook
import csv

# para
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DataParallel
import math
from time import sleep

class ZOTrainer:
    def __init__(self, args, model, tokenizer, lr_scheduler=None, zo_eps=1e-2, para=False):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.lr_scheduler = lr_scheduler
        self.zo_random_seed = None
        self.projected_grad = None
        self.zo_eps = zo_eps
        self.initial_lr = args.v_lr
        self.max_iterations = args.v_num_grad_steps
        self.current_iteration = 0  
        print("self.zo_eps:", zo_eps)
        # self.delta = torch.normal(mean=0, std=1, size=(model.config.hidden_size,), requires_grad=False, device=f"cuda:{args.device}")
        self.delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{args.device}")
        # z = torch.normal(mean=0, std=1, size=self.delta.size(), device=self.delta.device)
        self.target_init = None
        # self.kl_distr_init = None
        self.grad = None
        self.v_buffer = []
        self.Hessian_matrix = torch.ones((model.config.hidden_size,), requires_grad=False, device=f"cuda:{args.device}")
        self.Hessian_smooth = 1e-8
        self.hist_perturbation = torch.zeros_like(self.delta)
        self.prefix_cache_flag = args.use_random_prefix
        self.loss_history=[]
        self.no_prefix_cooldown = 0
        if para == True:
            print(f"model device:{next(self.model.parameters()).device}")
            self.model_cache_per_gpu = {}
            ## 使用torch的广播机制拷贝模型
            # from torch.nn.parallel.replicate import replicate

            # # 获取所有目标设备，如 ["cuda:0", "cuda:1", ...]（注意，此处编号已根据 CUDA_VISIBLE_DEVICES 重映射）
            # devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

            # # replicate 会在各个 GPU 上创建模型副本
            # model_copies = replicate(self.model, devices)

            # # 存储到你的缓存字典中
            # self.model_cache_per_gpu = {i: model for i, model in enumerate(model_copies)}

            # for i, model in self.model_cache_per_gpu.items():
            #     print(f"Model on {devices[i]}: {next(model.parameters()).device}")

            # 拷贝模型
            import copy
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs")
                for gpu_id in range(torch.cuda.device_count()):
                    # print(f"Creating model copy for GPU {gpu_id}")
                    if gpu_id == 0:
                        print(f"Using original model on GPU {gpu_id}")
                        self.model_cache_per_gpu[gpu_id] = self.model
                        # sleep(15)
                    else:
                        print(f"Creating model copy for GPU {gpu_id}")
                        model_copy = copy.deepcopy(self.model).to(f"cuda:{gpu_id}")
                    # model_copy.eval()
                        self.model_cache_per_gpu[gpu_id] = model_copy
                        # sleep(15)

    def _get_learning_rate(self):
        cosine_decay = 0.5 * (1 + np.cos(np.pi * self.current_iteration / self.max_iterations))
        # cosine_decay = 1.0
        current_lr = self.initial_lr * cosine_decay
        print(f"current_lr:{current_lr}")
        self.current_iteration += 1
        return current_lr
        # if self.lr_scheduler:
        #     for param_group in self.lr_scheduler.optimizer.param_groups:
        #         return param_group["lr"]
        # return self.args.v_lr

    def zo_forward(self, model, inputs, delta):
        """Performs a forward pass without gradients to compute the loss."""
        model.eval()
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            loss = self.zero_order_loss(inputs, delta)
        return loss.detach()
    
    def zero_order_loss_para(self, inputs, delta, device_id=None):
        """
        计算零阶优化的损失函数。
        增加 device_id 参数来指定使用哪个 GPU 进行计算。
        如果 device_id 为 None，则使用默认设备。
        """
        # 确定要使用的设备
        print(f"device_id: {delta.device.index}")
        model = self.model_cache_per_gpu[delta.device.index]
        device = f"cuda:{device_id}" if device_id is not None else delta.device
        print(f"Using device: {device}")

        # 将输入和模型移动到指定设备
        input_tok = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs["input_tok"].items()}
        rewriting_targets = inputs["rewriting_targets"].to(device)
        kl_distr_init = inputs.get("kl_distr_init", None)
        if kl_distr_init is not None:
            kl_distr_init = kl_distr_init.to(device)

        lookup_idxs = inputs["lookup_idxs"]
        hparams = inputs["hparams"]
        layer = inputs["layer"]
        loss_layer = inputs["loss_layer"]

        # 将 delta 移动到指定设备
        delta = delta.to(device)

        # 创建用于当前设备的模型副本
        # 注意：这可能需要根据模型的具体特性调整
        # if device_id is not None and hasattr(self, f'model_device_{device_id}'):
        #     # 使用已缓存的设备模型
        #     model = getattr(self, f'model_device_{device_id}')
        # else:
        #     # 为设备创建新的模型副本
        #     model = self.model
        # 创建模型的深层复制并移动到目标设备
        # print(f"Using model copy for GPU {device_id}")
        # model = self.model_cache_per_gpu[device_id]

        def edit_output_fn(cur_out, cur_layer):
            # 使用当前设备上的目标初始值
            if cur_layer == hparams.mlp_module_tmp.format(layer):
                # 初始值的处理
                if self.target_init is None:
                    print("Recording initial value of v*")
                    # 记录在当前设备上的初始值
                    self.target_init = cur_out[0, lookup_idxs[0]].detach().clone()

                # 确保 target_init 在当前设备上
                target_init_device = self.target_init.to(device)

                # 应用 delta 修改
                for i, idx in enumerate(lookup_idxs):
                    if len(lookup_idxs) != len(cur_out):
                        cur_out[idx, i, :] += delta
                    else:
                        cur_out[i, idx, :] += delta

            return cur_out

        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # 处理 KL 散度计算
            kl_prompts_indices = list(range(len(inputs["lookup_idxs"]) - len(inputs["kl_prompts"]), len(inputs["lookup_idxs"])))
            kl_logits = torch.stack(
                [
                    logits[i - len(inputs["kl_prompts"]), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(inputs["kl_prompts"]):])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)

            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()
                inputs["kl_distr_init"] = kl_distr_init

            # 计算损失
            log_probs = torch.log_softmax(logits, dim=2)
            loss_raw = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, torch.tensor(0, device=device)).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            nll_loss_each = -(loss_raw * mask).sum(1) / inputs["target_ids"].size(0)
            nll_loss = nll_loss_each.mean()

            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )

            weight_decay_ = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(self.target_init.to(device)) ** 2)

            total_loss = nll_loss + kl_loss + weight_decay_

            # 确保返回的损失在正确的设备上
            return total_loss

    def zero_order_loss(self, inputs, delta):
        """Calculates the loss for ZO optimization."""
        input_tok = inputs["input_tok"]
        rewriting_targets = inputs["rewriting_targets"]
        kl_distr_init = inputs.get("kl_distr_init", None)
        # print("kl_distr_init:", inputs.get("kl_distr_init", None))
        lookup_idxs = inputs["lookup_idxs"]
        hparams = inputs["hparams"]
        layer = inputs["layer"]
        loss_layer = inputs["loss_layer"]

        # 将编辑向量delta插入到mlp.downproj后
        def edit_output_fn(cur_out, cur_layer):
            # nonlocal target_init
            if cur_layer == hparams.mlp_module_tmp.format(layer):
                # Store initial value of the vector of interest
                if self.target_init is None:
                    print("Recording initial value of v*")
                    # Initial value is recorded for the clean sentence
                    self.target_init = cur_out[0, lookup_idxs[0]].detach().clone()
                    
                for i, idx in enumerate(lookup_idxs):
                    # print(f"Delta norm: {torch.norm(delta).item()}")
                    if len(lookup_idxs)!=len(cur_out):
                        cur_out[idx, i, :] += delta
                    else:
                        cur_out[i, idx, :] += delta

            return cur_out

        if len(self.loss_history) == 2:
            loss_gap = self.loss_history[0] - self.loss_history[1]
            print("loss gap:", loss_gap)

        with nethook.TraceDict(
            module=self.model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            # print("self.prefix_cache_flag:", self.prefix_cache_flag)
            # print("self.current_iteration:", self.current_iteration)
            if self.prefix_cache_flag == False or self.current_iteration >= 400:
                print("===no prefix===")
                logits = self.model(**input_tok).logits
            elif self.prefix_cache_flag == True and len(self.loss_history) == 2 and loss_gap <= 0.001:
                print("===update prefix===")
                logits = self.model(**input_tok, update_prefix=True).logits
            elif self.prefix_cache_flag == True:
                logits = self.model(**input_tok, prefix=10).logits
            # print("logits:", logits)

            kl_logits = torch.stack(
                [
                    logits[i - len(inputs["kl_prompts"]), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(inputs["kl_prompts"]):])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)

            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()
                inputs["kl_distr_init"] = kl_distr_init

            log_probs = torch.log_softmax(logits, dim=2)
            loss_raw = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, torch.tensor(0, device=logits.device)).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            nll_loss_each = -(loss_raw * mask).sum(1) / inputs["target_ids"].size(0)
            nll_loss = nll_loss_each.mean()

            # print("KL distr init:", kl_distr_init)
            # print("KL log probs:", kl_log_probs)
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )

            weight_decay_ = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(self.target_init) ** 2)

            total_loss = nll_loss + kl_loss + weight_decay_
            # print(f"nll_loss: {nll_loss.item()}, kl_loss: {kl_loss.item()}, weight_decay_: {weight_decay_.item()}, total_loss: {total_loss.item()}")

            return total_loss

    def zo_step(self, model, inputs):
        """Estimates the gradient using ZO method and updates delta only."""
        self.zo_random_seed = np.random.randint(1000000000)
        torch.manual_seed(self.zo_random_seed)

        z = torch.normal(mean=0, std=1, size=self.delta.size(), device=self.delta.device)

        # Compute f(delta + eps * z)
        delta = self.delta + self.zo_eps * z
        loss1 = self.zo_forward(model, inputs, delta)
        print("loss1:", loss1)

        # Compute f(delta - eps * z)
        delta = self.delta - self.zo_eps * z
        loss2 = self.zo_forward(model, inputs, delta)
        print("loss2:", loss2)

        # Restore delta to original state
        # self.delta += self.zo_eps * z
        print(f"loss: {(loss1 + loss2) / 2}")
        print(f"self.zo_eps:{self.zo_eps}")
        # Estimate gradient
        self.projected_grad = (loss1 - loss2) / (2 * self.zo_eps)
        print(f"Projected gradient: {self.projected_grad}")

        # Update delta
        # self.update_delta()
        grad_est = self.projected_grad * z
        # print(f"projected_grad:{self.projected_grad}")
        # print(f"z:{z}")
        # print(f"grad_est:{grad_est}")
        self.delta -= self._get_learning_rate() * grad_est

        # Clamp delta to remain within L2 constraint
        max_norm = self.args.clamp_norm_factor * torch.norm(self.target_init)
        if torch.norm(self.delta) > max_norm:
            self.delta = self.delta * (max_norm / torch.norm(self.delta))
        # print(f"Delta norm after update: {torch.norm(self.delta).item()}")

        return (loss1 + loss2) / 2
    
    def update_delta_hissan(self):
        """Updates delta explicitly during optimization."""
        z = torch.normal(mean=0, std=1, size=self.delta.size(), device=self.delta.device)
        grad_est = self.projected_grad * z
        self.delta -= self._get_learning_rate() * grad_est

        # Clamp delta to remain within L2 constraint
        max_norm = self.args.clamp_norm_factor * torch.norm(self.target_init)
        if torch.norm(self.delta) > max_norm:
            self.delta = self.delta * (max_norm / torch.norm(self.delta))
        # print(f"Delta norm after update: {torch.norm(self.delta).item()}")
        # print(f"Delta:", self.delta)

    def update_delta(self):
        """Updates delta explicitly during optimization."""
        z = torch.normal(mean=0, std=1, size=self.delta.size(), device=self.delta.device)
        grad_est = self.projected_grad * z
        self.delta -= self._get_learning_rate() * grad_est

        # Clamp delta to remain within L2 constraint
        max_norm = self.args.clamp_norm_factor * torch.norm(self.target_init)
        if torch.norm(self.delta) > max_norm:
            self.delta = self.delta * (max_norm / torch.norm(self.delta))
        # print(f"Delta norm after update: {torch.norm(self.delta).item()}")
        # print(f"Delta:", self.delta)
        
    def _prepare_inputs(self, inputs):
        """Prepares inputs for the forward pass."""
        inputs["delta"] = self.delta
        # if self.target_init is None:
            # self.target_init = torch.zeros_like(self.delta)
        return inputs
    
    def cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
    
    def loss_backward(self, func, inputs, params):
        params.requires_grad_(True)
        loss = func(params)  # loss at param
        print(loss.requires_grad)
        loss.backward()
        real_grad = params.grad.clone()
        return real_grad
    
    def calculate_jvp_hessian(self, func, inputs, params, v, Hessian_matrix, h=1e-5):
        """
        Calculate the Jacobian-vector product (JVP) using numerical differentiation.

        Args:
            func: The function for which we are calculating the JVP.
            params: The model parameters.
            v: The vector with which we compute the product.
            h: The small perturbation used for numerical differentiation.

        Returns:
            avg_loss: The average loss from perturbations.
            jvp: The computed Jacobian-vector product.
        """
        # print("inputs:", inputs)
        # print("params:", params)
        # params = params - h * v
        # print(f"before GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
        # print(f"before GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")    

        with torch.no_grad():
            # 使用no_grad()可以减少内存使用
            params_minus = params - 1.0 / torch.sqrt(Hessian_matrix) * h * v
            print(f"params_minus:{params_minus}")
            params_plus = params + 1.0 / torch.sqrt(Hessian_matrix) * h * v
            print(f"params_plus:{params_plus}")
            
            ori_loss = func(params).detach()  # loss at params
            loss = func(params_minus).detach()  # perturb params with -hv
            turbulence_loss = func(params_plus).detach()  # perturb params with +hv
            
            # 立即删除不需要的变量
            del params_minus, params_plus
            
            # 计算结果
            avg_loss = (turbulence_loss + loss) / 2
            jvp = (turbulence_loss - loss) / (2 * h)
            
            # 确保结果是分离的，不连接到计算图
            avg_loss = avg_loss.clone()
            jvp = jvp.clone()
            
            # 删除中间变量
            # del loss, turbulence_loss
        
        # 强制进行垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
        # print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")   
        return ori_loss, loss, turbulence_loss 

    def calculate_jvp_zoadamu(self, func, inputs, params, v, h=1e-5):
        """
        Calculate the Jacobian-vector
        """
        std_current_step = self._ema_weight()
        v_history = 0.9 * torch.normal(mean=self.hist_perturbation, std=math.sqrt(1. - std_current_step))
        v_current = 0.1 * torch.normal(mean=0, std=math.sqrt(std_current_step), size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        v = v_history + v_current
    
    def calculate_jvp_para(self, func, inputs, params, v, h=1e-5, device_id=None):
        """
        Calculate the Jacobian-vector product (JVP) using numerical differentiation.
        Added device_id parameter to specify which GPU to use.
        """
        with torch.no_grad():
            # 将参数和向量移动到指定设备
            device = f"cuda:{device_id}" if device_id is not None else params.device
            params_device = params.to(device)
            v_device = v.to(device)

            # 计算扰动后的参数
            params_minus = params_device - h * v_device
            params_plus = params_device + h * v_device

            # 使用指定设备计算损失
            ori_loss = self.zero_order_loss_para(inputs, params_device, device_id=device_id).detach()
            loss = self.zero_order_loss_para(inputs, params_minus, device_id=device_id).detach()
            turbulence_loss = self.zero_order_loss_para(inputs, params_plus, device_id=device_id).detach()

            # 立即删除不需要的变量
            del params_minus, params_plus

            # 计算结果
            avg_loss = (turbulence_loss + loss) / 2
            jvp = (turbulence_loss - loss) / (2 * h)

            # 将结果移回原始设备
            ori_device = params.device
            avg_loss = avg_loss.to(ori_device)
            jvp = jvp.to(ori_device)

            # 删除中间变量
            del loss, turbulence_loss
    
        # 强制进行垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return ori_loss.to(ori_device), jvp

    def calculate_jvp(self, func, inputs, params, v, h=1e-5):
        """
        Calculate the Jacobian-vector product (JVP) using numerical differentiation.

        Args:
            func: The function for which we are calculating the JVP.
            params: The model parameters.
            v: The vector with which we compute the product.
            h: The small perturbation used for numerical differentiation.

        Returns:
            avg_loss: The average loss from perturbations.
            jvp: The computed Jacobian-vector product.
        """

        with torch.no_grad():
            # 扰动delta
            params_minus = params - h * v
            params_plus = params + h * v
            
            # 计算未扰动loss （可去掉这一步）
            ori_loss = func(params).detach()  # loss at params
            
            # 计算两次扰动的loss，func为zero_order_loss
            loss = func(params_minus).detach()  # perturb params with -hv
            turbulence_loss = func(params_plus).detach()  # perturb params with +hv
            
            # 立即删除不需要的变量
            del params_minus, params_plus
            
            # 计算结果
            avg_loss = (turbulence_loss + loss) / 2
            jvp = (turbulence_loss - loss) / (2 * h)
            
            # 确保结果是分离的，不连接到计算图
            avg_loss = avg_loss.clone()
            jvp = jvp.clone()
            
            # 删除中间变量
            del loss, turbulence_loss
        
        # 强制进行垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
        # print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")    


        return ori_loss, jvp
    
    def forward_grad_step_hessian(self, inputs):
        """
        Forward Gradient Method

        Paper: Gradients without Backpropagation
        https://arxiv.org/pdf/2202.08587.pdf
        """
        args = self.args
        delta = self.delta

        # Sample the random seed for sampling vs
        self.zo_random_seed = np.random.randint(42)
        torch.manual_seed(self.zo_random_seed)

        loss = 0
        v_nums = 3
        v_buffer = []
        loss_buffer = []
        jvp_buffer = []
        cos_sims = []
        Hessian_estimators = []
        # if len(self.v_buffer) == 0:
        for _ in range(v_nums):
            vs = torch.normal(mean=0, std=1, size=self.delta.size(), device=self.delta.device)
            v_buffer.append(vs)
        # vs2 = torch.normal(mean=0, std=2, size=self.delta.size(), device=self.delta.device)
        # print(f"vs norm: {torch.norm(vs).item()}")
        # Partial function for loss computation
        # f = partial(
        #     self.zero_order_loss,
        #     inputs=inputs,
        #     delta=delta
        # )
        f = lambda delta: self.zero_order_loss(inputs, delta)

        real_grad = self.loss_backward(f, inputs, delta)
        real_grad = real_grad.cpu().numpy()
        # print(f"real_grad:{real_grad}")
        print(f"test cos sim: {self.cosine_similarity(real_grad, real_grad)}")
        torch.cuda.empty_cache()

        

        for vs in v_buffer:
        # Perform JVP computation
        # loss_, jvp_ = jvp(f, (delta,), (vs,))
            Hessian_temp = self.Hessian_matrix * vs * vs
            loss_original, loss1, loss2 = self.calculate_jvp_hessian(f, inputs, delta, vs, self.Hessian_matrix)
            print(f"loss_original:{loss_original}, loss1:{loss1}, loss2:{loss2}")
            Hessian_estimator = (torch.abs(loss1+loss2-2 * loss_original)* Hessian_temp * self.Hessian_smooth /(2 * self.zo_eps*self.zo_eps))
            print(f"Hessian_estimator:{Hessian_estimator}")
            Hessian_estimators.append(Hessian_estimator)
            jvp_ = (loss2 - loss1) / (2 * self.zo_eps)
            # print(f"loss_:{loss_}, jvp_:{jvp_}")
            # loss_buffer.append(loss_)
            jvp_buffer.append(jvp_)
            
        # loss_2, jvp_2 = self.calculate_jvp(f, inputs, delta, vs2)
        # print(f"loss_2:{loss_2}, jvp_2:{jvp_2}")

        # perturbed_delta = delta + torch.normal(mean=0, std=1000, size=self.delta.size(), device=self.delta.device)
        # loss_perturbed = self.zero_order_loss(inputs, perturbed_delta)
        # print(f"Loss with perturbed delta: {loss_perturbed}")
        Hessian_estimators_sum = torch.zeros_like(self.delta)
        for i, Hessian_estimator in enumerate(Hessian_estimators):
            # print(f"Hessian_estimator_{i}: {Hessian_estimator}")
            Hessian_estimators_sum += Hessian_estimators[i]

        # Update delta
        grad_est = torch.zeros_like(self.delta)
        with torch.no_grad():
            for i, vs in enumerate(v_buffer):
                grad_est += vs * jvp_buffer[i]
                # tmp_grad =  vs * jvp_buffer[i]
                # Hessian_temp = self.Hessian_matrix * vs * vs
                # print(f"Hessian_temp:{Hessian_temp}")
                # Hessian_estimator = (torch.abs(loss1+loss2-2 * loss_original)* Hessian_temp * self.Hessian_smooth /(2 * self.zo_eps*self.zo_eps))
                # print(f"Hessian_estimator:{Hessian_estimator}")
            self.Hessian_matrix = ((1-self.Hessian_smooth) * self.Hessian_matrix +  Hessian_estimators_sum)
            print(f"Hessian_matrix:{self.Hessian_matrix}")
            grad_est = grad_est / torch.sqrt(self.Hessian_matrix)
            print(f"grad_est:{grad_est}")
                # tmp_grad = tmp_grad.cpu().numpy()
                # print(f"tmp_grad:{tmp_grad}")
                # cos = self.cosine_similarity(real_grad, tmp_grad)
                # cos_sims.append(cos)
            # print(f"Projected gradient: {jvp_}")
            # print(f"vs:{vs}")
            # print(f"grad_est:{grad_est}")
            # grad_est = grad_est / v_nums
            # self.lr = self.lr * 0.9
            # self.delta -= self.lr * grad_est

            self.delta -= self._get_learning_rate() * grad_est
            print(f"cos similarity of real grad and jvp:{cos_sims}")
            grad_est = grad_est.cpu().numpy()
            final_cos_sim = self.cosine_similarity(real_grad, grad_est)
            print(f"cos similarity of real grad and est grad:{final_cos_sim}")

            # with open("cos_sims_fg1_new.csv", mode="a", newline="") as file:
            #     writer = csv.writer(file)
            #     writer.writerow(cos_sims)  # 逐行追加

            # **将最终的 cos 相似度追加写入 CSV 文件**
            with open("final_cosine_similarity_fg3_hessian.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([final_cos_sim])  # 逐行追加

            # Clamp delta to remain within L2 constraint
            max_norm = args.clamp_norm_factor * torch.norm(self.target_init)
            if torch.norm(self.delta) > max_norm:
                self.delta = self.delta * (max_norm / torch.norm(self.delta))
            print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
            print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")    

            # print(f"Delta norm after update: {torch.norm(self.delta).item()}")
        # print(f"Delta norm: {torch.norm(self.delta).item()}, Projected Grad: {grad_est}")

        loss += loss_original.item()
        return torch.tensor(loss)
    
    def forward_grad_step_zoadamu(self, inputs):
        """
        Forward Gradient Method

        Paper: Gradients without Backpropagation
        https://arxiv.org/pdf/2202.08587.pdf
        """
        args = self.args
        delta = self.delta

        # Sample the random seed for sampling vs
        self.zo_random_seed = np.random.randint(42)
        torch.manual_seed(self.zo_random_seed)

        loss = 0
        v_nums = 1
        v_buffer = []
        loss_buffer = []
        jvp_buffer = []
        cos_sims = []
        # if len(self.v_buffer) == 0:
        for _ in range(v_nums):
            vs = torch.normal(mean=0, std=1, size=self.delta.size(), device=self.delta.device)
            v_buffer.append(vs)
        # vs2 = torch.normal(mean=0, std=2, size=self.delta.size(), device=self.delta.device)
        # print(f"vs norm: {torch.norm(vs).item()}")
        # Partial function for loss computation
        # f = partial(
        #     self.zero_order_loss,
        #     inputs=inputs,
        #     delta=delta
        # )
        f = lambda delta: self.zero_order_loss(inputs, delta)

        real_grad = self.loss_backward(f, inputs, delta)
        real_grad = real_grad.cpu().numpy()
        # print(f"real_grad:{real_grad}")
        print(f"test cos sim: {self.cosine_similarity(real_grad, real_grad)}")
        torch.cuda.empty_cache()

        for vs in v_buffer:
        # Perform JVP computation
        # loss_, jvp_ = jvp(f, (delta,), (vs,))
            loss_, jvp_ = self.calculate_jvp(f, inputs, delta, vs)
            print(f"loss_:{loss_}, jvp_:{jvp_}")
            # loss_buffer.append(loss_)
            jvp_buffer.append(jvp_)
            
        # loss_2, jvp_2 = self.calculate_jvp(f, inputs, delta, vs2)
        # print(f"loss_2:{loss_2}, jvp_2:{jvp_2}")

        # perturbed_delta = delta + torch.normal(mean=0, std=1000, size=self.delta.size(), device=self.delta.device)
        # loss_perturbed = self.zero_order_loss(inputs, perturbed_delta)
        # print(f"Loss with perturbed delta: {loss_perturbed}")

        # Update delta
        grad_est = torch.zeros_like(self.delta)
        with torch.no_grad():
            for i, vs in enumerate(v_buffer):
                grad_est += vs * jvp_buffer[i]
                tmp_grad =  vs * jvp_buffer[i]
                tmp_grad = tmp_grad.cpu().numpy()
                # print(f"tmp_grad:{tmp_grad}")
                cos = self.cosine_similarity(real_grad, tmp_grad)
                cos_sims.append(cos)
            # print(f"Projected gradient: {jvp_}")
            # print(f"vs:{vs}")
            # print(f"grad_est:{grad_est}")
            # grad_est = grad_est / v_nums
            # self.lr = self.lr * 0.9
            # self.delta -= self.lr * grad_est
            # if self.current_iteration == 0:
            #     self.delta -= self._get_learning_rate() * real_grad
            # else:
            self.delta -= self._get_learning_rate() * grad_est
            print(f"cos similarity of real grad and jvp:{cos_sims}")
            grad_est = grad_est.cpu().numpy()
            final_cos_sim = self.cosine_similarity(real_grad, grad_est)
            print(f"cos similarity of real grad and est grad:{final_cos_sim}")

            with open("cos_sims_fg1_40fact.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(cos_sims)  # 逐行追加

            # **将最终的 cos 相似度追加写入 CSV 文件**
            with open("final_cosine_similarity_fg1_40fact.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([final_cos_sim])  # 逐行追加

            # Clamp delta to remain within L2 constraint
            max_norm = args.clamp_norm_factor * torch.norm(self.target_init)
            if torch.norm(self.delta) > max_norm:
                self.delta = self.delta * (max_norm / torch.norm(self.delta))
            print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
            print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")    

            # print(f"Delta norm after update: {torch.norm(self.delta).item()}")
        # print(f"Delta norm: {torch.norm(self.delta).item()}, Projected Grad: {grad_est}")

        loss += loss_.item()
        return torch.tensor(loss)

    def forward_grad_step(self, inputs):
        """
        Forward Gradient Method

        Paper: Gradients without Backpropagation
        https://arxiv.org/pdf/2202.08587.pdf
        """
        args = self.args
        delta = self.delta

        # Sample the random seed for sampling vs
        self.zo_random_seed = np.random.randint(42)
        torch.manual_seed(self.zo_random_seed)

        loss = 0
        v_nums = 5 # 扰动数量
        v_buffer = []
        loss_buffer = []
        jvp_buffer = []
        cos_sims = []

        #初始化扰动向量
        for _ in range(v_nums):
            vs = torch.normal(mean=0, std=1, size=self.delta.size(), device=self.delta.device)
            v_buffer.append(vs)

        f = lambda delta: self.zero_order_loss(inputs, delta)

        #计算每个扰动向量对应的loss和梯度标量（JVP）
        jvpzeros = 0
        for vs in v_buffer:
        # Perform JVP computation
            loss_, jvp_ = self.calculate_jvp(f, inputs, delta, vs, h=self.zo_eps)
            print(f"loss_:{loss_}, jvp_:{jvp_}")
            jvp_buffer.append(jvp_)
            if jvp_ == 0:
                jvpzeros += 1
        if self.prefix_cache_flag == True and jvpzeros == v_nums:
            print("All jvp are zeros, prefix_cache_flag = False")
            self.prefix_cache_flag = False

        # 计算梯度估计
        grad_est = torch.zeros_like(self.delta)
        with torch.no_grad():
            for i, vs in enumerate(v_buffer):
                grad_est += vs * jvp_buffer[i]
                tmp_grad =  vs * jvp_buffer[i]
                tmp_grad = tmp_grad.cpu().numpy()

            # 更新编辑向量delta
            if self.prefix_cache_flag == True and self.current_iteration >= 400:
                self.delta -= 1.5 * self._get_learning_rate() * grad_est
            else:
                self.delta -= self._get_learning_rate() * grad_est
            

            # Clamp delta to remain within L2 constraint
            max_norm = args.clamp_norm_factor * torch.norm(self.target_init)
            if torch.norm(self.delta) > max_norm:
                self.delta = self.delta * (max_norm / torch.norm(self.delta))
            print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
            print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")    

            print(f"delta:{self.delta}")

        # record 5 step loss
        loss += loss_.item()
        self.loss_history.append(loss)
        if len(self.loss_history) > 2:
            self.loss_history.pop(0)
        print(f"loss history: {self.loss_history}")
        return torch.tensor(loss)
    
    def compute_jvp_worker(self, vs_list, gpu_id, delta, inputs, device):
        results = []

        with torch.cuda.device(gpu_id):
            delta_device = delta.to(f'cuda:{gpu_id}')
            for vs in vs_list:
                vs_device = vs.to(f'cuda:{gpu_id}')
                loss_, jvp_ = self.calculate_jvp_para(
                    lambda d: self.zero_order_loss_para(inputs, d.to(device)).to(f'cuda:{gpu_id}'), 
                    inputs, 
                    delta_device, 
                    vs_device
                )
                results.append((loss_.cpu(), jvp_.cpu()))

        # 返回结果，不能用 extend + 共享 list，避免多进程通信复杂
        return results

    
    def forward_grad_step_para(self, inputs):
        """
        Forward Gradient Method with multi-GPU parallelization
        Paper: Gradients without Backpropagation
        https://arxiv.org/pdf/2202.08587.pdf
        """
        args = self.args
        delta = self.delta
        if self.target_init is None:
            self.zero_order_loss_para(inputs, delta, 0) #为了初始化target_init（必须在主进程里做，否则多进程子进程中会出错）

        # 获取可用的GPU数量
        n_gpus = torch.cuda.device_count()
        print(f"Using {n_gpus} GPUs for parallel computation")

        # 如果只有1个GPU，就使用原来的方法
        if n_gpus <= 1:
            return self._forward_grad_step(inputs)

        # 设置随机种子
        self.zo_random_seed = np.random.randint(42)
        torch.manual_seed(self.zo_random_seed)

        # 设置向量样本数量
        v_nums = 4

        # 决定使用多少GPU
        gpus_to_use = min(n_gpus, v_nums)
        print(f"有 {n_gpus} 个可用GPU，使用 {gpus_to_use} 个GPU进行计算")

        # 生成所有随机向量
        v_buffer = [torch.normal(mean=0, std=1, size=self.delta.size(), device=self.delta.device) 
                    for _ in range(v_nums)]

        # 计算每个GPU上的向量数量
        if gpus_to_use > 0:
            base_vecs_per_gpu = v_nums // gpus_to_use
            extra_vecs = v_nums % gpus_to_use
        else:
            base_vecs_per_gpu = 0
            extra_vecs = 0

        # 按GPU分组，确保每个被使用的GPU至少有一个向量
        v_groups = []
        start_idx = 0
        for i in range(gpus_to_use):
            # 前extra_vecs个GPU多分配一个向量
            vecs_for_this_gpu = base_vecs_per_gpu + (1 if i < extra_vecs else 0)
            end_idx = start_idx + vecs_for_this_gpu

            if start_idx < end_idx:  # 确保这个区间有效
                v_groups.append(v_buffer[start_idx:end_idx])

            start_idx = end_idx

        # 打印分配情况
        for i, group in enumerate(v_groups):
            print(f"GPU {i}: 分配了 {len(group)} 个向量")

        # 创建损失函数
        f = lambda delta: self.zero_order_loss_para(inputs, delta)

        # 并行计算
        results = []
        processes = []

        # 使用Python多进程来并行处理
        from torch.multiprocessing import Pool
        import functools
        # set_start_method('spawn')
        with Pool(processes=gpus_to_use) as pool:
            # 组织每个进程的参数
            param_list = []
            for gpu_id in range(gpus_to_use):
                param_list.append((
                    v_groups[gpu_id], 
                    gpu_id, 
                    self.delta, 
                    inputs, 
                    self.delta.device, 
                ))

            # 使用 starmap 并行调用
            multi_results = pool.starmap(self.compute_jvp_worker, param_list)

        # 把所有结果平铺展开
        results = [item for sublist in multi_results for item in sublist]

        # # 等待所有进程完成
        # for p in processes:
        #     p.join()

        # 提取结果
        loss_buffer = [r[0] for r in results]
        jvp_buffer = [r[1] for r in results]

        # 选择一个损失值作为返回值
        loss = loss_buffer[0].item() if loss_buffer else 0

        # 更新delta
        grad_est = torch.zeros_like(self.delta)
        with torch.no_grad():
            for i, vs in enumerate(v_buffer):
                grad_est += vs * jvp_buffer[i]

            self.delta -= self._get_learning_rate() * grad_est

            # Clamp delta to remain within L2 constraint
            max_norm = args.clamp_norm_factor * torch.norm(self.target_init)
            if torch.norm(self.delta) > max_norm:
                self.delta = self.delta * (max_norm / torch.norm(self.delta))

            print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
            print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")    

        return torch.tensor(loss)
    
    def forward_grad_choosev_step(self, inputs):
        """
        Forward Gradient Method

        Paper: Gradients without Backpropagation
        https://arxiv.org/pdf/2202.08587.pdf
        """
        args = self.args
        delta = self.delta

        # Sample the random seed for sampling vs
        self.zo_random_seed = np.random.randint(1000000000)
        torch.manual_seed(self.zo_random_seed)

        loss = 0
        v_nums = 300
        v_buffer = []
        loss_buffer = []
        jvp_buffer = []

        if self.grad is not None:
            selected_cos = (torch.rand(v_nums,1)*0.5+0.5).to(self.delta.device)
            target_grad = self.grad
            target_grad = torch.flatten(target_grad)
            target_grad = target_grad/target_grad.norm()
            # 随机向量
            r = torch.stack([torch.randn_like(target_grad) for _ in range(v_nums)],dim=0)
            # 根据随机向量算出的垂直grad的向量
            grad_perp = r - (r@(target_grad.unsqueeze(-1)))*target_grad
            # Make it a unit vector:
            grad_perp = grad_perp / (grad_perp.norm(dim=-1)[:,None])
            # 算出v
            candidate_v = selected_cos*target_grad + torch.sqrt(torch.tensor(1,device=self.delta.device) - selected_cos**2)*grad_perp
            candidate_v = candidate_v*torch.tensor((target_grad.shape[0]**0.5),device=self.delta.device)

            for i in range(v_nums):
                # 取出第 i 行
                single_v = candidate_v[i]  # (D,)
                # 再 reshape
                single_v = single_v.reshape(self.delta.size())
                v_buffer.append(single_v)

        else:
            for _ in range(v_nums):
                vs = torch.normal(mean=0, std=1, size=self.delta.size(), device=self.delta.device)
                v_buffer.append(vs)
        # vs2 = torch.normal(mean=0, std=2, size=self.delta.size(), device=self.delta.device)
        # print(f"vs norm: {torch.norm(vs).item()}")
        # Partial function for loss computation
        # f = partial(
        #     self.zero_order_loss,
        #     inputs=inputs,
        #     delta=delta
        # )
        f = lambda delta: self.zero_order_loss(inputs, delta)

        for vs in v_buffer:
        # Perform JVP computation
        # loss_, jvp_ = jvp(f, (delta,), (vs,))
            loss_, jvp_ = self.calculate_jvp(f, inputs, delta, vs)
            print(f"loss_:{loss_}, jvp_:{jvp_}")
            loss_buffer.append(loss_)
            jvp_buffer.append(jvp_)
        # loss_2, jvp_2 = self.calculate_jvp(f, inputs, delta, vs2)
        # print(f"loss_2:{loss_2}, jvp_2:{jvp_2}")

        # perturbed_delta = delta + torch.normal(mean=0, std=1000, size=self.delta.size(), device=self.delta.device)
        # loss_perturbed = self.zero_order_loss(inputs, perturbed_delta)
        # print(f"Loss with perturbed delta: {loss_perturbed}")

        # Update delta
        grad_est = torch.zeros_like(self.delta)
        with torch.no_grad():
            for i, vs in enumerate(v_buffer):
                grad_est += vs * jvp_buffer[i]
            # print(f"Projected gradient: {jvp_}")
            # print(f"vs:{vs}")
            # print(f"grad_est:{grad_est}")
            self.grad = grad_est
            self.delta -= self._get_learning_rate() * self.grad

            # Clamp delta to remain within L2 constraint
            max_norm = args.clamp_norm_factor * torch.norm(self.target_init)
            if torch.norm(self.delta) > max_norm:
                self.delta = self.delta * (max_norm / torch.norm(self.delta))
            # print(f"Delta norm after update: {torch.norm(self.delta).item()}")
        # print(f"Delta norm: {torch.norm(self.delta).item()}, Projected Grad: {grad_est}")

        loss += loss_.item()
        return torch.tensor(loss)
    

