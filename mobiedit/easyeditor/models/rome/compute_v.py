from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook
from .zo import ZOTrainer
from ...evaluate import compute_edit_quality

from .rome_hparams import ROMEHyperParams
from ...models.quantization.quantizer import W8A16Model

import torch
from datetime import datetime
import time
import sys

# torch.cuda.set_device(4)

def get_tensor_info(tensor, name=""):
    """获取tensor的详细信息"""
    if tensor is None:
        return f"{name} is None"
        
    dtype = tensor.dtype
    shape = tensor.shape
    device = tensor.device
    mem = tensor.element_size() * tensor.nelement() / (1024**2)  # MB
    return f"{name}:\n - 类型: {dtype}\n - 形状: {shape}\n - 设备: {device}\n - 内存: {mem:.2f}MB"

def print_model_memory_usage(model):
    """打印模型内存使用情况"""
    total_params_memory = 0
    total_grads_memory = 0
    
    print("\n模型内存使用情况:")
    for name, param in model.named_parameters():
        param_mem = param.element_size() * param.nelement() / (1024**2)  # MB
        grad_mem = param.grad.element_size() * param.grad.nelement() / (1024**2) if param.grad is not None else 0
        
        total_params_memory += param_mem
        total_grads_memory += grad_mem
        
        # print(f"\n{name}:")
        # print(get_tensor_info(param, "参数"))
        # if param.grad is not None:
            # print(get_tensor_info(param.grad, "梯度"))
    
    # print(f"\n总计:")
    print(f" - 参数总内存: {total_params_memory:.2f}MB")
    # print(f" - 梯度总内存: {total_grads_memory:.2f}MB")

def format_time(seconds):
    """将秒数转换为可读性更强的格式"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    if minutes > 0:
        return f"{minutes}min {seconds:.2f}s"
    return f"{seconds:.2f}s"

def compute_v_randomprefix_zo(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # 记录开始时间
    start_time = time.time()
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")

    print("Computing right vector (v)")
    print("len of context_templates:", len(context_templates))
    print("context_templates:", context_templates)

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]


    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # 存储目标向量的初始值
            if target_init is None:
                print("Recording initial value of v*")
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
                
            # 在指定位置插入 delta 向量
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs)!=len(cur_out):
                    cur_out[idx, i, :] += delta
                else:
                    cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    trainer = ZOTrainer(hparams, model, tok, zo_eps=1e-3)  # 初始化 ZOTrainer，无需传入 lr_scheduler

    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        
        # 随机选择 18 个 context_templates
        import random
        import string
        # selected_context_templates = random.sample(context_templates, 18)
        # print("selected_context_templates:", selected_context_templates)

        def random_string(length):
            letters = string.ascii_letters + string.digits
            return ''.join(random.choice(letters) for i in range(length))

        selected_context_templates = [random_string(random.randint(10, 10)) + " {}" for _ in range(18)]


        # Compile list of rewriting and KL x/y pairs
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context in selected_context_templates
        ], ["{} is a"]
        all_prompts = rewriting_prompts + kl_prompts
        print("len of all_prompts:", len(all_prompts))
        print("all_prompts:", all_prompts)

        input_tok = tok(
            [prompt.format(request["subject"]) for prompt in all_prompts],
            return_tensors="pt",
            padding=True,
        ).to(f"cuda:{hparams.device}")

        # Compute rewriting targets
        rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
            len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
        )
        for i in range(len(rewriting_prompts)):
            ex_len = input_tok["attention_mask"][i].sum()
            rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

        # Compute indices of the tokens where the fact is looked up
        vanilla_input_prompts = [
            context.format(request["prompt"]).format(request['subject'])
            for context in selected_context_templates
        ] + [f"{request['subject']} is a"]
        lookup_idxs = [
            find_fact_lookup_idx(
                prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
            )
            for i, prompt in enumerate(all_prompts)
        ]

        # 将所有输入数据打包
        inputs = {
            "input_tok": input_tok,
            "rewriting_targets": rewriting_targets,
            "lookup_idxs": lookup_idxs,
            "kl_prompts": kl_prompts,
            "hparams": hparams,
            "request": request,
            "target_ids": target_ids,
            "layer": layer,
            "loss_layer": loss_layer,
        }

        # 调用 trainer 的 zo_step 方法进行一次梯度估计
        # 前向传播开始
        forward_start = time.time()
        loss = trainer.forward_grad_step(inputs)
        # 前向传播结束
        forward_time = time.time() - forward_start
        print(f"前向传播耗时: {format_time(forward_time)}") 

        # 输出
        print(f"Iter {it}/{hparams.v_num_grad_steps - 1}, ZO loss={loss:.4f}")

        # 若损失小于阈值，提前退出
        if loss < 5e-2:
            break


    # target = target_init + delta (存储在 trainer 内部的 delta)
    target = trainer.target_init + trainer.delta.to(trainer.target_init.dtype)


    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {trainer.target_init.norm().item()} to {target.norm().item()} => {(target.norm() - trainer.target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    # 记录总耗时
    total_time = time.time() - start_time
    print(f"总耗时: {format_time(total_time)}")

    return right_vector

def compute_v_randomprefix(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # 记录开始时间
    start_time = time.time()
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")

    print("Computing right vector (v)")
    print("len of context_templates:", len(context_templates))
    print("context_templates:", context_templates)

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]


    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # 存储目标向量的初始值
            if target_init is None:
                print("Recording initial value of v*")
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
                
            # 在指定位置插入 delta 向量
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs)!=len(cur_out):
                    cur_out[idx, i, :] += delta
                else:
                    cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        # 前向传播开始
        forward_start = time.time()
        opt.zero_grad()

        
        # 随机选择 18 个 context_templates
        import random
        import string
        # selected_context_templates = random.sample(context_templates, 18)
        # print("selected_context_templates:", selected_context_templates)

        def random_string(length):
            letters = string.ascii_letters + string.digits
            return ''.join(random.choice(letters) for i in range(length))

        selected_context_templates = [random_string(random.randint(20, 20)) + " {}" for _ in range(18)]


        # Compile list of rewriting and KL x/y pairs
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context in selected_context_templates
        ], ["{} is a"]
        all_prompts = rewriting_prompts + kl_prompts
        print("len of all_prompts:", len(all_prompts))
        print("all_prompts:", all_prompts)

        input_tok = tok(
            [prompt.format(request["subject"]) for prompt in all_prompts],
            return_tensors="pt",
            padding=True,
        ).to(f"cuda:{hparams.device}")

        # Compute rewriting targets
        rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
            len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
        )
        for i in range(len(rewriting_prompts)):
            ex_len = input_tok["attention_mask"][i].sum()
            rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

        # Compute indices of the tokens where the fact is looked up
        vanilla_input_prompts = [
            context.format(request["prompt"]).format(request['subject'])
            for context in selected_context_templates
        ] + [f"{request['subject']} is a"]
        lookup_idxs = [
            find_fact_lookup_idx(
                prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
            )
            for i, prompt in enumerate(all_prompts)
        ]

        
        # Forward propagation
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
            logits = model(**input_tok, prefix=20).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        loss = nll_loss + kl_loss + weight_decay
        print(f"nll_loss: {nll_loss.item()}, kl_loss: {kl_loss.item()}, weight_decay: {weight_decay.item()}, total_loss: {loss.item()}")

        
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # 前向传播结束
        forward_time = time.time() - forward_start
        print(f"前向传播耗时: {format_time(forward_time)}")

        print(f"Iter {it}/{hparams.v_num_grad_steps - 1}, ZO loss={loss:.4f}")
        #Delta mlp backward
        backward_start = time.time()
        loss.backward(retain_graph=True)
        print_model_memory_usage(model)
        backward_time = time.time() - backward_start
        print(f"反向传播耗时: {format_time(backward_time)}")
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
        # 在计算完成后添加最终显存统计
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")  
        print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    target = target_init + delta.to(target_init.dtype)

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")
    # 记录总耗时
    total_time = time.time() - start_time
    print(f"总耗时: {format_time(total_time)}")

    return right_vector

def compute_v_zo(
    weights: torch.Tensor,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    torch.cuda.set_device(hparams.device)
    # 记录开始时间
    start_time = time.time()
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    # 创建request的副本
    request_copy = request.copy()  # 浅拷贝足够了

    # 更新副本中的prompt，将{}替换为subject值
    request_copy['prompt'] = request['prompt'].format(request['subject'])
    # 删除target_new中的第一个空格
    if 'target_new' in request_copy and isinstance(request_copy['target_new'], str) and request_copy['target_new'].startswith(' '):
        request_copy['target_new'] = request_copy['target_new'].lstrip(' ')
        print(f"已删除target_new中的首部空格，现在为: '{request_copy['target_new']}'")


    print("-----Computing right vector (v) ZO-----")

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]

    # if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
    #     target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts
    # all_prompts = [[{"role":"user", "content":m}] for m in all_prompts]
    # all_prompts= tok.apply_chat_template(all_prompts,
    #                                 add_generation_prompt=True,
    #                                 tokenize=False)
    # print("len of all_prompts:", len(all_prompts))
    print("all_prompts:", all_prompts)
    # print("prompt.format:", [prompt.format(request["subject"]) for prompt in all_prompts])

    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context in context_templates
    ] + [f"{request['subject']} is a"]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
        )
        for i, prompt in enumerate(all_prompts)
    ]
    print("lookup_idx:", lookup_idxs)

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
            
            # 创建新的tensor而不是修改原始tensor
            new_out = cur_out.clone()
            
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(cur_out):
                    print(get_tensor_info(new_out[idx, i, :], f"MLP output 激活值"))
                    # 使用加法创建新tensor而不是in-place操作
                    new_out[idx, i, :] = cur_out[idx, i, :] + delta(cur_out[idx, i, :])
                else:
                    new_out[i, idx, :] = cur_out[i, idx, :] + delta(cur_out[i, idx, :])
            
            return new_out
        return cur_out
    
    if hparams.quantize == True:
        # 应用W8A16量化
        skip_modules = ["model.layers.5.mlp.down_proj"]  # 可以指定不需要量化的模块
        quan_model = W8A16Model(
            model=model,
            device=f"cuda:{hparams.device}",
            skip_modules=skip_modules)

        # 修改模型的get_parameter方法
        def get_parameter_wrapper(name):
            return quan_model.get_parameter(name)
        quan_model.get_parameter = get_parameter_wrapper

        # calibrate
        quan_model.calibrate(input_tok)

        # #打印量化误差
        # after_quan_logits = model(**input_tok).logits
        # print(f"after quan model(input_tok).logits:{after_quan_logits}")

        # print(f"before_quan_logits.shape:{before_quan_logits.shape}")
        # print(f"after_quan_logits.shape:{after_quan_logits.shape}")
        # before_quan_logits = before_quan_logits.view(len(all_prompts), -1)
        # after_quan_logits = after_quan_logits.view(len(all_prompts), -1)
        # cos_sim = torch.nn.functional.cosine_similarity(before_quan_logits, after_quan_logits, dim=1)
        # print(f"cos_sim:{cos_sim}")

        model = quan_model

    # Optimizer
    # opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    
    # 初始化 ZOTrainer
    trainer = ZOTrainer(hparams, model, tok, zo_eps=1e-3, para=False)

    # 将所有输入数据打包
    inputs = {
        "input_tok": input_tok,
        "rewriting_targets": rewriting_targets,
        "lookup_idxs": lookup_idxs,
        "kl_prompts": kl_prompts,
        "hparams": hparams,
        "request": request,
        "target_ids": target_ids,
        "layer": layer,
        "loss_layer": loss_layer,
    }

    # 3. ========== ZO 优化过程 ==========

    for it in range(hparams.v_num_grad_steps):
        # 调用 trainer 的 forward_grad_step 方法进行一次梯度估计
        # 前向开始
        forward_start = time.time()
        loss = trainer.forward_grad_step(inputs)
        # 前向结束
        forward_time = time.time() - forward_start
        print(f"前向传播耗时: {format_time(forward_time)}")

        # 输出
        print(f"Iter {it}/{hparams.v_num_grad_steps - 1}, ZO loss={loss:.4f}")

        # 每隔20stepevaluate一次     
        if it != 0 and it % 20 == 0:
            target = trainer.target_init + trainer.delta.to(trainer.target_init.dtype)

            # Retrieve cur_input, the current input to the 2nd MLP layer, and
            # cur_output, the original output of the 2nd MLP layer.
            cur_input, cur_output = get_module_input_output_at_word(
                model,
                tok,
                layer,
                context_template=request["prompt"],
                word=request["subject"],
                module_template=hparams.rewrite_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )

            # Solving the linear system to compute the right vector
            right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
            deltas = {}
            with torch.no_grad():
                # Determine correct transposition of delta matrix
                weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"

                deltas[weight_name] = (
                    left_vector.detach(),
                    right_vector.detach(),
                )

            print(f"Deltas successfully computed for {list(weights.keys())}")
            weight_copy = {}
            with torch.no_grad():
                for w_name, (delta_u, delta_v) in deltas.items():
                    upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                    w = nethook.get_parameter(model, w_name)
                    weight_copy[w_name] = w.detach().clone()
                    from .rome_main import upd_matrix_match_shape
                    upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                    w[...] += upd_matrix
                print(f"New weights successfully inserted into {list(deltas.keys())}")
            eval_metric = "exact match"
            edit_quality = compute_edit_quality(model, "qwen2", hparams, tok, request_copy, hparams.device, eval_metric=eval_metric, test_generation=True)
            print("compute v Edit succ:",edit_quality)
            # print("compute v Edit succ:",compute_edit_quality(model, "qwen2", hparams, tok, request_copy, hparams.device, eval_metric=eval_metric, test_generation=True))
            # print(f"edited_model: {model}, self.model_name: qwen2, self.hparams: {hparams}, self.tok: {tok}, request: {request_copy}, self.hparams.device: {hparams.device}, eval_metric: exact match, test_generation: True")
            
            # # Early stopping
            # if edit_quality['rewrite_acc'][0] == 1.0:
            #     return right_vector

            # Restore state of original model
            for w_name in deltas.keys():
                    w = nethook.get_parameter(model, w_name)
                    # print(f"w[{w_name}]:{w})")
                    w[...] = weight_copy[w_name]
                    # print(f"w[{w_name}]:{w})")
            print("Restore succ:",compute_edit_quality(model, "qwen2", hparams, tok, request_copy, hparams.device))


        # 若损失小于阈值，提前退出
        if loss < 1e-5:
            break

    # 4. ========== 获取最终的目标向量 ==========
    target = trainer.target_init + trainer.delta.to(trainer.target_init.dtype)


    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {trainer.target_init.norm().item()} to {target.norm().item()} => {(target.norm() - trainer.target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")
    # 记录总耗时
    total_time = time.time() - start_time
    print(f"总耗时: {format_time(total_time)}")

    return right_vector

def compute_v(
    weights: torch.Tensor,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("-----Computing right vector (v)------")
    # 记录开始时间
    start_time = time.time()
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")

    target_ids = tok.encode(request["subject"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['subject']:", request["subject"])
    print("subject target_ids.shape:", target_ids.shape)
    
    
    target_ids = tok.encode(request["prompt"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['prompt']:", request["prompt"])
    print("prompt target_ids.shape:", target_ids.shape)


    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['target_new']:", request["target_new"])
    print("target target_ids.shape:", target_ids.shape)

    # if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
    #     target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts
        # all_prompts = rewriting_prompts + kl_prompts
    all_prompts = [[{"role":"user", "content":m}] for m in all_prompts]
    all_prompts= tok.apply_chat_template(all_prompts,
                                    add_generation_prompt=True,
                                    tokenize=False)
    print("request['prompt']:", request["prompt"])
    print("len of all_prompts:", len(all_prompts))
    print("all_prompts:", all_prompts)
    print("prompt.format:", [prompt.format(request["subject"]) for prompt in all_prompts])

    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")
    print("request['subject']:", request["subject"])
    token_length = input_tok.input_ids.size(1)  # 获取序列长度
    print(f"token_length: {token_length}")
    token_length = input_tok.attention_mask.sum(1)
    print(f"token_length_unpadding: {token_length}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    print("input token shape:", input_tok["input_ids"].shape)
    
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context in context_templates
    ] + [f"{request['subject']} is a"]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
        )
        for i, prompt in enumerate(all_prompts)
    ]
    print("lookup_idx:", lookup_idxs)

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.randn((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        delta = torch.randn((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
        # delta = DeltaMLP(model.config.hidden_size).to(f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        print("cur_layer:", cur_layer)
        if cur_layer == "model.layers.5.mlp":
        # if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
                print(f"target_init:{target_init}")
            for i, idx in enumerate(lookup_idxs):
                # print(f"cur_out.dtype:{cur_out.dtype}")
                # print("======================insert delta=======================")
                # print(f"cur_out:{cur_out}")
                if len(lookup_idxs)!=len(cur_out):
                    print(get_tensor_info(cur_out[idx, i, :], f"MLP output 激活值"))
                    # print(f"idx:{idx}, i:{i}")
                    # print(f"before delta cur_out[idx, i, :]:{cur_out[idx, i, :]}")
                    # print(f"before delta cur_out[idx, i, :].norm:{cur_out[idx, i, :].norm()}")

                    cur_out[idx, i, :] += delta
                    
                    # print(f"after cur_out[idx, i, :]:{cur_out[idx, i, :]}")
                    # print(f"after cur_out[idx, i, :].norm:{cur_out[idx, i, :].norm()}")
                    # cur_out[idx, i, :] += delta_coef1 * delta1 + delta_coef2 * delta2
                    # cur_out[idx, i, :] += delta(cur_out[idx, i, :])
                    # cur_out[idx, i, :] = (cur_out[idx, i, :] + delta).to(torch.float16)
                else:
                    # print(f"idx:{idx}, i:{i}")
                    # print(f"before delta cur_out[i, idx, :]:{cur_out[i, idx, :]}")
                    # print(f"before delta cur_out[i, idx, :].norm:{cur_out[i, idx, :].norm()}")
                    # print(f"cur_out[i, idx, :]:{cur_out[i, idx, :]}")
                    cur_out[i, idx, :] += delta
                    # print(f"after delta cur_out[i, idx, :]:{cur_out[i, idx, :]}")
                    # print(f"after delta cur_out[i, idx, :].norm:{cur_out[i, idx, :].norm()}")
                # print("========================================================")
                    # cur_out[i, idx, :] += delta_coef1 * delta1 + delta_coef2 * delta2
                    # cur_out[i, idx, :] += delta(cur_out[i, idx, :])
                    # cur_out[i, idx, :] = (cur_out[i, idx, :] + delta).to(torch.float16)
        # return cur_out
        if isinstance(cur_out, torch.Tensor):
            # print(f"edit_output_fn {cur_layer}: input id={id(cur_out)} grad_fn={cur_out.grad_fn}")
            out_new = cur_out.clone()
            # print(f"edit_output_fn {cur_layer}: output id={id(out_new)} grad_fn={out_new.grad_fn}")
            return out_new
        elif isinstance(cur_out, tuple):
            outputs = []
            for idx, o in enumerate(cur_out):
                if isinstance(o, torch.Tensor):
                    # print(f"edit_output_fn {cur_layer}[{idx}]: input id={id(o)} grad_fn={o.grad_fn}")
                    o_new = o.clone()
                    # print(f"edit_output_fn {cur_layer}[{idx}]: output id={id(o_new)} grad_fn={o_new.grad_fn}")
                    outputs.append(o_new)
                else:
                    outputs.append(o)
            return tuple(outputs)
        else:
            # print(f"edit_output_fn {cur_layer}: not Tensor/tuple")
            return cur_out
        # if isinstance(cur_out, torch.Tensor):
        #     return cur_out.clone()
        # elif isinstance(cur_out, tuple):
        # # 对每个 Tensor 部分clone，其它类型（如None）不变
        #     return tuple(o.clone() if isinstance(o, torch.Tensor) else o for o in cur_out)
        # else:
        # 其它情况直接返回
            # return cur_out
    nethook.set_requires_grad(False, model)
    # nethook.set_requires_grad(True, model.model.layers[5].mlp.down_proj)
    print(f"model:{model}")
    module_path = f"model.model.layers[5].mlp.down_proj"
    mod = eval(module_path)
    # Optimizer
    # opt = torch.optim.Adam([delta] + list(mod.parameters()), lr=hparams.v_lr)
    # opt = torch.optim.AdamW([delta, delta_coef], lr=2e-1, betas=(0.9, 0.95), eps=1e-4)
    # lr = hparams.v_lr
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    # from lion_pytorch import Lion
    # opt = Lion([delta1, delta2, delta_coef1, delta_coef2], lr=2e-1, betas=(0.9, 0.95))

    if hparams.quantize == True:
    
        before_quan_logits = model(**input_tok).logits

        print(f"before quan model(input_tok).logits:{before_quan_logits}")

        # 应用W8A16量化
        # skip_modules = [hparams.mlp_module_tmp.format(layer)+".down_proj", "model.layers.6", "model.layers.7", "model.layers.8", "model.layers.9","model.layers.10"]  # 可以指定不需要量化的模块
        # skip_modules = [hparams.mlp_module_tmp.format(layer)+".down_proj", "model.layers"]
        skip_modules = [hparams.mlp_module_tmp.format(layer)+".down_proj"]
        quan_model = W8A16Model(
            model=model,
            device=f"cuda:{hparams.device}",
            skip_modules=skip_modules)

        # quan_model.enable_debug()

        # 修改模型的get_parameter方法
        def get_parameter_wrapper(name):
            return quan_model.get_parameter(name)
        quan_model.get_parameter = get_parameter_wrapper

        # calibrate
        quan_model.calibrate(input_tok, insert_module_name=hparams.mlp_module_tmp.format(layer)+".down_proj")

        after_quan_logits = model(**input_tok).logits
        print(f"after quan model(input_tok).logits:{after_quan_logits}")

        print(f"before_quan_logits.shape:{before_quan_logits.shape}")
        print(f"after_quan_logits.shape:{after_quan_logits.shape}")
        before_quan_logits = before_quan_logits.view(len(all_prompts), -1)
        after_quan_logits = after_quan_logits.view(len(all_prompts), -1)
        cos_sim = torch.nn.functional.cosine_similarity(before_quan_logits, after_quan_logits, dim=1)
        print(f"cos_sim:{cos_sim}")

        model = quan_model

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    # layer = model.model.model.layers[5].mlp.down_proj
    # for param in layer.parameters():
    #     param.requires_grad = True
        
    # layer6 = model.model.model.layers[6].mlp.down_proj

    # act5_out = None
    # # act6_in = None

    # def hook_layer5_out(module, inputs, output):
    #     print("hook_layer5_out output type:", type(output))
    #     if isinstance(output, torch.Tensor):
    #         print("output.requires_grad:", output.requires_grad)
    #     else:
    #         print("output不是Tensor类型")
    #     global act5_out
    #     act5_out = output
    #     act5_out.retain_grad()
    #     def grad_hook(grad):
    #         print('第5层输出激活值的梯度:', grad)
    #     act5_out.register_hook(grad_hook)

    # # def hook_layer6_in(module, inputs, output):
    # #     global act6_in
    # #     act6_in = inputs[0] # inputs是元组, 第一项是本层输入
    # #     act6_in.retain_grad()
    # #     def grad_hook(grad):
    # #         print('第6层输入激活值的梯度:', grad)
    # #     act6_in.register_hook(grad_hook)

    # model.model.model.layers[5].mlp.down_proj.register_forward_hook(hook_layer5_out)
    # model.model.model.layers[6].mlp.down_proj.register_forward_hook(hook_layer6_in)


    last_logits = model(**input_tok).logits
    old_delta = delta.data.clone()
    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        # 前向传播开始
        forward_start = time.time()
        opt.zero_grad()

        # # 随机选择 18 个提示
        # indices = torch.randperm(len(all_prompts))[:18]
        # selected_input_tok = {key: val[indices] for key, val in input_tok.items()}

        # Forward propagation
        print("TraceDict will hook:", hparams.layer_module_tmp.format(loss_layer))
        print("TraceDict will hook:", hparams.mlp_module_tmp.format(layer))
        # for n, _ in model.named_modules():
        #     print(n)
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                # hparams.layer_module_tmp.format(6),
                # hparams.mlp_module_tmp.format(layer),
                "model.layers.5.mlp",
                # "model.layers.5.mlp.down_proj",
                # "model.layers.6.mlp.down_proj",
                # # "model.layers.6",
                # "model.layers.6.mlp.down_proj",
                # "model.layers.33.mlp.down_proj",
                # "model.layers.34.mlp.down_proj",
                # "model.layers.35.mlp.up_proj",
                # "model.layers.35.mlp.down_proj",
            ],
            retain_input=True,
            retain_output=True,
            # retain_grad=True,
            # clone=True,
            edit_output=edit_output_fn,
        ) as tr:
            if hparams.use_random_prefix == False:
                logits = model(**input_tok).logits
            else:
                logits = model(**input_tok, prefix = 10).logits
            # print("logits:", logits)
            # print(f"cos_sim:{torch.nn.functional.cosine_similarity(last_logits, logits, dim=1)}")
            print(f"cos_sim.mean:{torch.nn.functional.cosine_similarity(last_logits, logits, dim=1).mean()}")
            last_logits = logits
            # logits = model(**input_tok).logits

            # model.calibrate(input_tok, edit_output=edit_output_fn, insert_module_name=hparams.mlp_module_tmp.format(layer)+".down_proj")

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(f"nll_loss: {nll_loss.item()}, kl_loss: {kl_loss.item()}, weight_decay_: {weight_decay.item()}, total_loss: {loss.item()}")

        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # 前向传播结束
        forward_time = time.time() - forward_start
        print(f"前向传播耗时: {format_time(forward_time)}")

        print(f"Iter {it}/{hparams.v_num_grad_steps - 1}, ZO loss={loss:.4f}")
        #Delta mlp backward
        backward_start = time.time()

        # tr["model.layers.5.mlp.down_proj"].output.retain_grad()
        # # tr["model.layers.6"].input[0].retain_grad()
        # tr["model.layers.6.mlp.down_proj"].output.retain_grad()
        # tr["model.layers.33.mlp.down_proj"].output.retain_grad()
        # tr["model.layers.34.mlp.down_proj"].output.retain_grad()
        # tr["model.layers.35.mlp.down_proj"].output.retain_grad()
        # tr["model.layers.35.mlp.up_proj"].output.retain_grad()
        
        # outputs = [
        #     tr["model.layers.5.mlp.down_proj"].output,
        #     tr["model.layers.6.mlp.down_proj"].output,
        #     tr["model.layers.33.mlp.down_proj"].output,
        #     tr["model.layers.34.mlp.down_proj"].output,
        #     tr["model.layers.35.mlp.down_proj"].output,
        # ]
        # loss = 0
        # for i, out in enumerate(outputs):
        #     k = 1.0 + 0.1 * i  # 不同分支不同权重
        #     loss = loss + k * out.sum()
        # print(f"loss:{loss}")


        loss.backward()
        # from torchviz import make_dot
        # dot = make_dot(loss, params=dict(model.named_parameters()))
        # dot.render('quan_graph_new', format="pdf", view=False)
        # dot = make_dot(tr["model.layers.5.mlp.down_proj"].output, params=dict(model.named_parameters()))
        # dot.render('quan_graph_5', format="png", view=False)
        # # dot = make_dot(tr["model.layers.6"].input[0], params=dict(model.named_parameters()))
        # # dot.render('quan_graph_6', format="png", view=False)
        # dot = make_dot(tr["model.layers.6.mlp.down_proj"].output, params=dict(model.named_parameters()))
        # dot.render('quan_graph_6_mlp', format="png", view=False)
        # dot = make_dot(tr["model.layers.33.mlp.down_proj"].output, params=dict(model.named_parameters()))
        # dot.render('quan_graph_33', format="png", view=False)
        # dot = make_dot(tr["model.layers.34.mlp.down_proj"].output, params=dict(model.named_parameters()))
        # dot.render('quan_graph_34', format="png", view=False)
        # dot = make_dot(tr["model.layers.35.mlp.down_proj"].output, params=dict(model.named_parameters()))
        # dot.render('quan_graph_35', format="png", view=False)
        

        # for name in ["model.layers.5.mlp.down_proj", 
        #              "model.layers.6.mlp.down_proj", "model.layers.33.mlp.down_proj", 
        #              "model.layers.34.mlp.down_proj", "model.layers.35.mlp.down_proj",
        #              "model.layers.35.mlp.up_proj"]:
        #     obj = tr[name].output
        #     print(f"{name} obj:{obj}")
        #     print(f"{name} output id:", id(obj))
        #     print(f"{name} grad_fn:", obj.grad_fn)
        #     print(f"{name} grad_fn.next_functions:", obj.grad_fn.next_functions)
        
        # for idx in [5,6,33,34,35]:
        #     out = tr[f'model.layers.{idx}.mlp.down_proj'].output
        #     print(f"Layer {idx} output.data_ptr: {out.data_ptr()}")
        #     print(f"Layer {idx} grad.data_ptr: {out.grad.data_ptr() if out.grad is not None else None}")
        #     print(f"Layer {idx}, id: {id(out)}, grad_fn: {out.grad_fn}")
        #     g = out.grad_fn
        #     nstep = 0
        #     while g is not None:
        #         print(" ", g)
        #         try:
        #             g = g.next_functions[0][0]
        #         except:
        #             break
        #         nstep += 1

        # for name in ["model.layers.5.mlp.down_proj", 
        #              "model.layers.6.mlp.down_proj", "model.layers.33.mlp.down_proj", 
        #              "model.layers.34.mlp.down_proj", "model.layers.35.mlp.down_proj",
        #              "model.layers.35.mlp.up_proj"]:
        #     grad = tr[name].output.grad
        #     print(f"{name} grad id:", id(grad))
            
        # grad_5_out = tr["model.layers.5.mlp.down_proj"].output.grad
        # print(f"grad_5_out.norm:{grad_5_out.norm()}")

        # grad_6_out = tr["model.layers.6.mlp.down_proj"].output.grad
        # print(f"grad_6_out.norm:{grad_6_out.norm()}")

        # grad_33_out = tr["model.layers.33.mlp.down_proj"].output.grad
        # print(f"grad_33_out.norm:{grad_33_out.norm()}")

        # grad_34_out = tr["model.layers.34.mlp.down_proj"].output.grad
        # print(f"grad_34_out.norm:{grad_34_out.norm()}")

        # grad_35_up_out = tr["model.layers.35.mlp.up_proj"].output.grad
        # print(f"grad_35_up_out.norm:{grad_35_up_out.norm()}")

        # grad_35_out = tr["model.layers.35.mlp.down_proj"].output.grad
        # print(f"grad_35_out.norm:{grad_35_out.norm()}")

        # print("grad_33-34 diff max:", (grad_33_out - grad_34_out).abs().max())
        # print("grad_34-35 diff max:", (grad_34_out - grad_35_out).abs().max())

        # quan_model.print_gradient_stats()
        print_model_memory_usage(model)
        backward_time = time.time() - backward_start
        print(f"反向传播耗时: {format_time(backward_time)}")

        # 获取 delta 的梯度范数（所有参数合成一个向量）
        real_grad = delta.grad.clone()
        # coef_grad_norm = delta_coef.grad.norm()
        print(f"Real gradient norm: {torch.norm(real_grad)}")
        print(f"Real gradient (mean ± std): {real_grad.mean():.4e} ± {real_grad.std():.4e}")
        # print(f"coef_grad_norm: {coef_grad_norm}")
        
        # for param_group in opt.param_groups:  
        #     param_group['lr'] = param_group['lr'] * 10
        #     print(f"param_group['lr']:{param_group['lr']}")
        # 更新 delta
        opt.step()        
        step_size = (delta.data - old_delta.data).abs().max()
        print(f"step_size:{step_size}")
        print(f"cos_sim:{torch.nn.functional.cosine_similarity(delta.data, old_delta.data, dim=0)}")
        print(f"cos_sim.mean:{torch.nn.functional.cosine_similarity(delta.data, old_delta.data, dim=0).mean()}")
        old_delta = delta.data.clone()


        # print(f"model.model.layers[5].mlp.down_proj.weight.grad:{model.model.model.layers[5].mlp.down_proj.weight.grad}")
        # print(f"model.model.layers[5].mlp.down_proj.weight:{model.model.model.layers[5].mlp.down_proj.weight}")
        # print(f"model.model.layers[5].mlp.down_proj.weight.norm:{model.model.model.layers[5].mlp.down_proj.weight.norm()}")
        # 裁剪 delta 参数的 L2 范数
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
        print(f"Delta norm after update: {torch.norm(delta).item()}")

        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
        print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # sys.exit(1)



        # Backpropagate
        # backward_start = time.time()
        # loss.backward()
        # print_model_memory_usage(model)
        # backward_time = time.time() - backward_start
        # print(f"反向传播耗时: {format_time(backward_time)}")
        # real_grad = delta.grad.clone()
        # print(f"Real gradient norm: {torch.norm(real_grad)}")
        # print(f"Real gradient: {real_grad}")
        # opt.step()

        # # Project within L2 ball
        # max_norm = hparams.clamp_norm_factor * target_init.norm()
        # if delta.norm() > max_norm:
        #     with torch.no_grad():
        #         delta[...] = delta * max_norm / delta.norm()
        # print(f"Delta norm after update: {torch.norm(delta).item()}")
        # print(f"Delta:", delta)
        
        # print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
        # print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")    

    target = target_init + delta.to(target_init.dtype)

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")
    # 记录总耗时
    total_time = time.time() - start_time
    print(f"总耗时: {format_time(total_time)}")
    # sys.exit(1)

    return right_vector

def compute_v_rank2(
    weights: torch.Tensor,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("-----Computing right vector (v)------")
    # 记录开始时间
    start_time = time.time()
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")

    target_ids = tok.encode(request["subject"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['subject']:", request["subject"])
    print("subject target_ids.shape:", target_ids.shape)
    
    
    target_ids = tok.encode(request["prompt"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['prompt']:", request["prompt"])
    print("prompt target_ids.shape:", target_ids.shape)


    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['target_new']:", request["target_new"])
    print("target target_ids.shape:", target_ids.shape)

    # if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
    #     target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts
    print("request['prompt']:", request["prompt"])
    print("len of all_prompts:", len(all_prompts))
    print("all_prompts:", all_prompts)
    print("prompt.format:", [prompt.format(request["subject"]) for prompt in all_prompts])

    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")
    print("request['subject']:", request["subject"])
    token_length = input_tok.input_ids.size(1)  # 获取序列长度
    print(f"token_length: {token_length}")
    token_length = input_tok.attention_mask.sum(1)
    print(f"token_length_unpadding: {token_length}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    print("input token shape:", input_tok["input_ids"].shape)
    
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context in context_templates
    ] + [f"{request['subject']} is a"]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
        )
        for i, prompt in enumerate(all_prompts)
    ]
    print("lookup_idx:", lookup_idxs)

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta_num = 5
    if hasattr(model.config, 'n_embd'):
        delta = torch.randn((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        delta_list = []
        delta_coef_list = []
        for i in range(delta_num):
            delta_list.append(torch.randn((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}"))
            delta_coef_list.append(torch.nn.Parameter(torch.tensor(1.0, device=f"cuda:{hparams.device}"), requires_grad=True))

        # delta_coef1 = torch.nn.Parameter(torch.tensor(1.0, device=f"cuda:{hparams.device}"), requires_grad=True)
        # delta2 = torch.randn((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
        # delta_coef2 = torch.nn.Parameter(torch.tensor(1.0, device=f"cuda:{hparams.device}"), requires_grad=True)
        # delta = DeltaMLP(model.config.hidden_size).to(f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
                print(f"target_init:{target_init}")
            for seq_idx, cur_lookup_list in enumerate(lookup_idxs):
                for loc in range(cur_lookup_list, token_length[seq_idx]):
                # loc: 这是该seq内第loc号被编辑token, 用delta_list[loc]
                    if loc < len(delta_list):
                        print(f"seq_idx:{seq_idx}, loc:{loc}")  # 防止越界
                        cur_out[seq_idx, loc, :] += delta_list[loc]
            # for i, idx in enumerate(lookup_idxs):
            #     # print(f"cur_out.dtype:{cur_out.dtype}")
            #     if len(lookup_idxs)!=len(cur_out):
            #         print(get_tensor_info(cur_out[idx, i, :], f"MLP output 激活值"))
            #         cur_out[idx, i, :] += delta_list[i] + delta_coef_list[i]
            #         # for j in range(delta_num):
            #         #     cur_out[idx, i, :] += delta_coef_list[j] * delta_list[j]
            #         # cur_out[idx, i, :] += delta(cur_out[idx, i, :])
            #         # cur_out[idx, i, :] = (cur_out[idx, i, :] + delta).to(torch.float16)
            #     else:
            #         cur_out[i, idx, :] += delta_list[i] + delta_coef_list[i]
            #         # for j in range(delta_num):
            #         #     cur_out[i, idx, :] += delta_coef_list[j] * delta_list[j]
            #         # cur_out[i, idx, :] += delta(cur_out[i, idx, :])
            #         # cur_out[i, idx, :] = (cur_out[i, idx, :] + delta).to(torch.float16)

        return cur_out

    # Optimizer
    # opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    # opt = torch.optim.AdamW([delta, delta_coef], lr=2e-1, betas=(0.9, 0.95), eps=1e-4)
    # lr = hparams.v_lr
    # opt = torch.optim.Adam(delta.parameters(), lr=hparams.v_lr)
    from lion_pytorch import Lion
    opt = Lion(delta_list + delta_coef_list, lr=2e-1, betas=(0.9, 0.95))


    nethook.set_requires_grad(False, model)
    before_quan_logits = model(**input_tok).logits

    print(f"before quan model(input_tok).logits:{before_quan_logits}")

    # 应用W8A16量化
    skip_modules = ["hparams.mlp_module_tmp.format(layer)"+".down_proj"]  # 可以指定不需要量化的模块
    quan_model = W8A16Model(
        model=model,
        device=f"cuda:{hparams.device}",
        skip_modules=skip_modules)
    
    # 修改模型的get_parameter方法
    def get_parameter_wrapper(name):
        return quan_model.get_parameter(name)
    quan_model.get_parameter = get_parameter_wrapper

    # calibrate
    quan_model.calibrate(input_tok, insert_module_name=skip_modules[0])

    after_quan_logits = model(**input_tok).logits
    print(f"after quan model(input_tok).logits:{after_quan_logits}")

    print(f"before_quan_logits.shape:{before_quan_logits.shape}")
    print(f"after_quan_logits.shape:{after_quan_logits.shape}")
    before_quan_logits = before_quan_logits.view(len(all_prompts), -1)
    after_quan_logits = after_quan_logits.view(len(all_prompts), -1)
    cos_sim = torch.nn.functional.cosine_similarity(before_quan_logits, after_quan_logits, dim=1)
    print(f"cos_sim:{cos_sim}")

    model = quan_model
    last_logits = model(**input_tok).logits
    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        # 前向传播开始
        forward_start = time.time()
        opt.zero_grad()

        # # 随机选择 18 个提示
        # indices = torch.randperm(len(all_prompts))[:18]
        # selected_input_tok = {key: val[indices] for key, val in input_tok.items()}

        # Forward propagation
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
            # logits = model(**input_tok).logits
            print("logits:", logits)
            print(f"cos_sim:{torch.nn.functional.cosine_similarity(last_logits, logits, dim=1)}")
            print(f"cos_sim.mean:{torch.nn.functional.cosine_similarity(last_logits, logits, dim=1).mean()}")
            last_logits = logits
            # print("logits:", logits)
            # logits = model(**input_tok).logits

            # model.calibrate(input_tok, edit_output=edit_output_fn)

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        delta_norm = 0
        for i in range(delta_num):
            delta_norm += torch.norm(delta_list[i])
        delta_norm = delta_norm / delta_num
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta_norm) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(f"nll_loss: {nll_loss.item()}, kl_loss: {kl_loss.item()}, weight_decay_: {weight_decay.item()}, total_loss: {loss.item()}")

        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # 前向传播结束
        forward_time = time.time() - forward_start
        print(f"前向传播耗时: {format_time(forward_time)}")

        print(f"Iter {it}/{hparams.v_num_grad_steps - 1}, ZO loss={loss:.4f}")
        #Delta mlp backward
        backward_start = time.time()
        loss.backward()
        print_model_memory_usage(model)
        backward_time = time.time() - backward_start
        print(f"反向传播耗时: {format_time(backward_time)}")

        # 获取 delta 的梯度范数（所有参数合成一个向量）
        # real_grad = delta.grad.clone()
        # coef_grad_norm = delta_coef.grad.norm()
        # print(f"Real gradient norm: {torch.norm(real_grad)}")
        # print(f"Real gradient (mean ± std): {real_grad.mean():.4e} ± {real_grad.std():.4e}")
        # print(f"coef_grad_norm: {coef_grad_norm}")

        grad_norm_list = []
        for i in range(delta_num):
            grad_norm_list.append(delta_list[i].grad.norm())
        grad_norm_coef_list = []
        # for i in range(delta_num):
        #     grad_norm_coef_list.append(delta_coef_list[i].grad.norm())

        # print(f"Grad Norm ➔ delta1:{grad_norm_list[0]:.4e}, delta2:{grad_norm_list[1]:.4e}, coef1:{grad_norm_coef_list[0]:.4e}, coef2:{grad_norm_coef_list[1]:.4e}")
        
        # for param_group in opt.param_groups:
        #     param_group['lr'] = param_group['lr'] * 10
        #     print(f"param_group['lr']:{param_group['lr']}")
        # 更新 delta
        opt.step()

        # 裁剪 delta 参数的 L2 范数
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        # if delta.norm() > max_norm:
        #     with torch.no_grad():
        #         delta[...] = delta * max_norm / delta.norm()
        # print(f"Delta norm after update: {torch.norm(delta).item()}")
        with torch.no_grad():
            final_norm = 0
            for i in range(delta_num):
                final_norm += torch.norm(delta_list[i])
            final_norm = final_norm / delta_num
            if final_norm > max_norm:
                scale = max_norm / final_norm
                for i in range(delta_num):
                    delta_list[i].mul_(scale)

            for i in range(delta_num):
                delta_coef_list[i].clamp_(-5.0, 5.0)

        print(f"Delta Norm after update ➔ delta1:{torch.norm(delta_list[0]):.4e}, delta2:{torch.norm(delta_list[1]):.4e}")
        # print(f"Coef after update ➔ coef1:{delta_coef_list[0].item():.4f}, coef2:{delta_coef_list[1].item():.4f}")

        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
        print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")



        # Backpropagate
        # backward_start = time.time()
        # loss.backward()
        # print_model_memory_usage(model)
        # backward_time = time.time() - backward_start
        # print(f"反向传播耗时: {format_time(backward_time)}")
        # real_grad = delta.grad.clone()
        # print(f"Real gradient norm: {torch.norm(real_grad)}")
        # print(f"Real gradient: {real_grad}")
        # opt.step()

        # # Project within L2 ball
        # max_norm = hparams.clamp_norm_factor * target_init.norm()
        # if delta.norm() > max_norm:
        #     with torch.no_grad():
        #         delta[...] = delta * max_norm / delta.norm()
        # print(f"Delta norm after update: {torch.norm(delta).item()}")
        # print(f"Delta:", delta)
        
        # print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
        # print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")    

    target = target_init + delta.to(target_init.dtype)

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")
    # 记录总耗时
    total_time = time.time() - start_time
    print(f"总耗时: {format_time(total_time)}")
    # sys.exit(1)

    return right_vector, model


def compute_v_delta_mlp(
    weights: torch.Tensor,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    torch.autograd.set_detect_anomaly(True)

    print("-----Computing right vector (v)------")
    # 记录开始时间
    start_time = time.time()
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")

    target_ids = tok.encode(request["subject"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['subject']:", request["subject"])
    print("subject target_ids.shape:", target_ids.shape)
    
    
    target_ids = tok.encode(request["prompt"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['prompt']:", request["prompt"])
    print("prompt target_ids.shape:", target_ids.shape)


    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['target_new']:", request["target_new"])
    print("target target_ids.shape:", target_ids.shape)

    # if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
    #     target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts
    print("request['prompt']:", request["prompt"])
    print("len of all_prompts:", len(all_prompts))
    print("all_prompts:", all_prompts)
    print("prompt.format:", [prompt.format(request["subject"]) for prompt in all_prompts])

    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")
    print("request['subject']:", request["subject"])
    token_length = input_tok.input_ids.size(1)  # 获取序列长度
    print(f"token_length: {token_length}")
    token_length = input_tok.attention_mask.sum(1)
    print(f"token_length_unpadding: {token_length}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    print("input token shape:", input_tok["input_ids"].shape)
    
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context in context_templates
    ] + [f"{request['subject']} is a"]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
        )
        for i, prompt in enumerate(all_prompts)
    ]
    print("lookup_idx:", lookup_idxs)

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.ones((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        # delta = torch.ones((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
        delta = DeltaMLP(model.config.hidden_size).to(f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
            
            # 创建新的tensor而不是修改原始tensor
            new_out = cur_out.clone()
            
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(cur_out):
                    print(get_tensor_info(new_out[idx, i, :], f"MLP output 激活值"))
                    # 使用加法创建新tensor而不是in-place操作
                    new_out[idx, i, :] = cur_out[idx, i, :] + delta(cur_out[idx, i, :])
                else:
                    new_out[i, idx, :] = cur_out[i, idx, :] + delta(cur_out[i, idx, :])
            
            return new_out
        return cur_out

    # Optimizer
    # opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    opt = torch.optim.Adam(delta.parameters(), lr=hparams.v_lr)

    nethook.set_requires_grad(False, model)
    before_quan_logits = model(**input_tok).logits

    print(f"before quan model(input_tok).logits:{before_quan_logits}")

    # 应用W8A16量化
    skip_modules = []  # 可以指定不需要量化的模块
    quan_model = W8A16Model(
        model=model,
        device=f"cuda:{hparams.device}",
        skip_modules=skip_modules)
    
    # 修改模型的get_parameter方法
    def get_parameter_wrapper(name):
        return quan_model.get_parameter(name)
    quan_model.get_parameter = get_parameter_wrapper

    # calibrate
    quan_model.calibrate(input_tok)

    after_quan_logits = model(**input_tok).logits
    print(f"after quan model(input_tok).logits:{after_quan_logits}")

    print(f"before_quan_logits.shape:{before_quan_logits.shape}")
    print(f"after_quan_logits.shape:{after_quan_logits.shape}")
    before_quan_logits = before_quan_logits.view(len(all_prompts), -1)
    after_quan_logits = after_quan_logits.view(len(all_prompts), -1)
    cos_sim = torch.nn.functional.cosine_similarity(before_quan_logits, after_quan_logits, dim=1)
    print(f"cos_sim:{cos_sim}")

    model = quan_model
    last_logits = model(**input_tok).logits
    # print("last_logits:", last_logits)
    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        # 前向传播开始
        forward_start = time.time()
        opt.zero_grad()

        # # 随机选择 18 个提示
        # indices = torch.randperm(len(all_prompts))[:18]
        # selected_input_tok = {key: val[indices] for key, val in input_tok.items()}

        # Forward propagation
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
            print("logits:", logits)
            print(f"cos_sim:{torch.nn.functional.cosine_similarity(last_logits, logits, dim=1)}")
            print(f"cos_sim.mean:{torch.nn.functional.cosine_similarity(last_logits, logits, dim=1).mean()}")
            last_logits = logits
            # logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = 0
        # weight_decay = hparams.v_weight_decay * (
        #     torch.norm(delta) / torch.norm(target_init) ** 2
        # )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(f"nll_loss: {nll_loss.item()}, kl_loss: {kl_loss.item()}, weight_decay_: {weight_decay}, total_loss: {loss.item()}")

        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay, 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # 前向传播结束
        forward_time = time.time() - forward_start
        print(f"前向传播耗时: {format_time(forward_time)}")

        print(f"Iter {it}/{hparams.v_num_grad_steps - 1}, ZO loss={loss:.4f}")
        #Delta mlp backward
        backward_start = time.time()
        loss.backward()
        print_model_memory_usage(model)
        backward_time = time.time() - backward_start
        print(f"反向传播耗时: {format_time(backward_time)}")

        # 获取 delta 的梯度范数（所有参数合成一个向量）
        real_grad = torch.cat([
            p.grad.view(-1) for p in delta.parameters() if p.grad is not None
        ])
        print(f"Real gradient norm: {torch.norm(real_grad)}")
        print(f"Real gradient (mean ± std): {real_grad.mean():.4e} ± {real_grad.std():.4e}")

        # 更新 delta
        opt.step()

        # 裁剪 delta 参数的 L2 范数
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        total_norm = torch.norm(torch.cat([p.data.view(-1) for p in delta.parameters()]))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            with torch.no_grad():
                for p in delta.parameters():
                    p.data.mul_(scale)

        print(f"Delta total norm after update: {total_norm.item():.4f}")
        for name, p in delta.named_parameters():
            print(f"{name} norm after update: {p.data.norm().item():.4f}")

        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
        print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


        # Backpropagate
        # backward_start = time.time()
        # loss.backward()
        # print_model_memory_usage(model)
        # backward_time = time.time() - backward_start
        # print(f"反向传播耗时: {format_time(backward_time)}")
        # real_grad = delta.grad.clone()
        # print(f"Real gradient norm: {torch.norm(real_grad)}")
        # print(f"Real gradient: {real_grad}")
        # opt.step()

        # # Project within L2 ball
        # max_norm = hparams.clamp_norm_factor * target_init.norm()
        # if delta.norm() > max_norm:
        #     with torch.no_grad():
        #         delta[...] = delta * max_norm / delta.norm()
        # print(f"Delta norm after update: {torch.norm(delta).item()}")
        # print(f"Delta:", delta)
        
        # print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
        # print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")    

    target = target_init + delta.to(target_init.dtype)

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")
    # 记录总耗时
    total_time = time.time() - start_time
    print(f"总耗时: {format_time(total_time)}")
    sys.exit(1)

    return right_vector

def compute_v_eval(
    weights: torch.Tensor,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("-----Computing right vector (v)------")
    # 记录开始时间
    start_time = time.time()
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    # 创建request的副本
    request_copy = request.copy()  # 浅拷贝足够了

    # 更新副本中的prompt，将{}替换为subject值
    request_copy['prompt'] = request['prompt'].format(request['subject'])
    # 删除target_new中的第一个空格
    if 'target_new' in request_copy and isinstance(request_copy['target_new'], str) and request_copy['target_new'].startswith(' '):
        request_copy['target_new'] = request_copy['target_new'].lstrip(' ')
        print(f"已删除target_new中的首部空格，现在为: '{request_copy['target_new']}'")

    # 打印结果查看
    # print("原始request['prompt']:", request['prompt'])  # Which family does {} belong to?
    # print("新的request_copy['prompt']:", request_copy['prompt'])  # Which family does Epaspidoceras belong to?

    target_ids = tok.encode(request["subject"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['subject']:", request["subject"])
    print("subject target_ids.shape:", target_ids.shape)
    
    
    target_ids = tok.encode(request["prompt"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['prompt']:", request["prompt"])
    print("prompt target_ids.shape:", target_ids.shape)


    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
    print("request['target_new']:", request["target_new"])
    print("target target_ids.shape:", target_ids.shape)

    # if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
    #     target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts
    print("request['prompt']:", request["prompt"])
    print("len of all_prompts:", len(all_prompts))
    print("all_prompts:", all_prompts)
    print("prompt.format:", [prompt.format(request["subject"]) for prompt in all_prompts])

    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")
    print("request['subject']:", request["subject"])
    token_length = input_tok.input_ids.size(1)  # 获取序列长度
    print(f"token_length: {token_length}")
    token_length = input_tok.attention_mask.sum(1)
    print(f"token_length_unpadding: {token_length}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context in context_templates
    ] + [f"{request['subject']} is a"]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
        )
        for i, prompt in enumerate(all_prompts)
    ]
    print("lookup_idx:", lookup_idxs)

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
            
            # 创建新的tensor而不是修改原始tensor
            # new_out = cur_out.clone()
            
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(cur_out):
                    # print(get_tensor_info(new_out[idx, i, :], f"MLP output 激活值"))
                    # 使用加法创建新tensor而不是in-place操作
                    cur_out[idx, i, :] += delta
                    # new_out[idx, i, :] = cur_out[idx, i, :] + delta(cur_out[idx, i, :])
                else:
                    new_out[i, idx, :] = cur_out[i, idx, :] + delta(cur_out[i, idx, :])
            
            return new_out
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        # 前向传播开始
        forward_start = time.time()
        opt.zero_grad()

        # # 随机选择 18 个提示
        # indices = torch.randperm(len(all_prompts))[:18]
        # selected_input_tok = {key: val[indices] for key, val in input_tok.items()}

        # Forward propagation
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
            # print("logits:", logits)
            # logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(f"nll_loss: {nll_loss.item()}, kl_loss: {kl_loss.item()}, weight_decay_: {weight_decay.item()}, total_loss: {loss.item()}")

        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # 前向传播结束
        forward_time = time.time() - forward_start
        print(f"前向传播耗时: {format_time(forward_time)}")

        print(f"Iter {it}/{hparams.v_num_grad_steps - 1}, ZO loss={loss:.4f}")
        # Backpropagate
        backward_start = time.time()
        loss.backward()
        print_model_memory_usage(model)
        backward_time = time.time() - backward_start
        print(f"反向传播耗时: {format_time(backward_time)}")
        real_grad = delta.grad.clone()
        print(f"Real gradient norm: {torch.norm(real_grad)}")
        print(f"Real gradient: {real_grad}")
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
        print(f"Delta norm after update: {torch.norm(delta).item()}")
        # print(f"Delta:", delta)
        
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
        print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")    

        target = target_init + delta.to(target_init.dtype)
    
        # Retrieve cur_input, the current input to the 2nd MLP layer, and
        # cur_output, the original output of the 2nd MLP layer.
        cur_input, cur_output = get_module_input_output_at_word(
            model,
            tok,
            layer,
            context_template=request["prompt"],
            word=request["subject"],
            module_template=hparams.rewrite_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )
    
        # Solving the linear system to compute the right vector
        right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
        # print(f"right_vector:{right_vector}")
        deltas = {}
        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            # upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            # from .rome_main import upd_matrix_match_shape
            # upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # # # Update model weights and record desired changes in `delta` variable
            # weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )
            # print(f"Delta norm: {deltas[weight_name][0].norm()}")
            # print(f"Delta norm: {deltas[weight_name][1].norm()}")
        # print(f"deltas:{deltas}")

        # # # Restore state of original model
        # # with torch.no_grad():
        # #     for k, v in weights.items():
        # #         v[...] = weights_copy[k]
        print(f"Deltas successfully computed for {list(weights.keys())}")
        weight_copy = {}
        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                # print(f"upd_matrix:{upd_matrix}")
                w = nethook.get_parameter(model, w_name)
                weight_copy[w_name] = w.detach().clone()
                from .rome_main import upd_matrix_match_shape
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                # print(f"upd_matrix.norm(): {upd_matrix.norm()}")
                # print("Weight norm before:", w.norm().item())
                # print("Weight norm after:", (w + upd_matrix).norm().item())
                # print(f"upd_matrix:{upd_matrix}")

                # if return_orig_weights and w_name not in weights_copy:
                #     weights_copy[w_name] = w.detach().clone()
                # print(f"w[{w_name}]:{w})")
                w[...] += upd_matrix
                # print(f"w[{w_name}]:{w})")
            print(f"New weights successfully inserted into {list(deltas.keys())}")
        eval_metric = "exact match"
        print("compute v Edit succ:",compute_edit_quality(model, "qwen2", hparams, tok, request_copy, hparams.device, eval_metric=eval_metric, test_generation=True))
        # print(f"edited_model: {model}, self.model_name: qwen2, self.hparams: {hparams}, self.tok: {tok}, request: {request_copy}, self.hparams.device: {hparams.device}, eval_metric: exact match, test_generation: True")
                
        # Restore state of original model
        for w_name in deltas.keys():
                w = nethook.get_parameter(model, w_name)
                # print(f"w[{w_name}]:{w})")
                w[...] = weight_copy[w_name]
                # print(f"w[{w_name}]:{w})")
        print("Restore succ:",compute_edit_quality(model, "qwen2", hparams, tok, request_copy, hparams.device))

    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")
        # 记录总耗时
    total_time = time.time() - start_time
    print(f"总耗时: {format_time(total_time)}")

    return right_vector


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
    input_prompt=None
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = len(tok.encode(input_prompt)) - 1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret

# 定义 delta 为一个小型 MLP 模块（两层 MLP）
import torch.nn as nn
class DeltaMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
