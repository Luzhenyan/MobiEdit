from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook
from ...util.generate import generate_fast
from ...evaluate import compute_edit_quality

from .compute_u import compute_u
from .compute_v import compute_v, compute_v_randomprefix, compute_v_zo, compute_v_randomprefix_zo, compute_v_eval, compute_v_delta_mlp, compute_v_rank2
from .rome_hparams import ROMEHyperParams
from transformers import BitsAndBytesConfig




CONTEXT_TEMPLATES_CACHE = None


def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    request = request[0]
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    # from optimum.gptq import GPTQQuantizedModelForCausalLM
    # # # 配置量化为8-bit
    # # quantization_config = BitsAndBytesConfig(
    # #     load_in_8bit=True,  # 启用8-bit量化
    # #     llm_int8_enable_fp32_cpu_offload=False,  # 禁用CPU上的FP32 offload
    # #     llm_int8_has_fp16_weight=False,  # 不使用FP16权重
    # # )

    # # 加载并量化Qwen2.5-3B-Instruct模型
    # model_name = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8"  # 根据您的模型路径/名称进行修改
    # # 加载分词器
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # # 加载 GPTQ 量化模型
    # model = GPTQQuantizedModelForCausalLM.from_pretrained(model_name, device_map="auto")


    # 使用INT8进行编辑
    # with torch.cuda.amp.autocast(enabled=True, dtype=torch.int8):
        # deltas = execute_rome(model, tok, request, hparams)
    # print(f"before model:{model}")
    deltas = execute_rome(model, tok, request, hparams)
    # print(f"after model:{model}")
    print("---return rome main-----")
    # is_bnb_quantized = hasattr(model, "is_loaded_in_8bit") or hasattr(model, "is_loaded_in_4bit")
    # print(f"模型是否使用BitsAndBytes量化: {is_bnb_quantized}")
    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            print(f"\nProcessing weight: {w_name}")
            print(f"delta_u type: {delta_u.dtype}, shape: {delta_u.shape}")
            print(f"delta_u: {delta_u}")
            print(f"delta_v type: {delta_v.dtype}, shape: {delta_v.shape}")
            print(f"delta_v: {delta_v}")
            
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            print(f"upd_matrix type: {upd_matrix.dtype}, shape: {upd_matrix.shape}")
            print(f"upd_matrix: {upd_matrix}")
            w = nethook.get_parameter(model, w_name)
            print(f"w type: {w.dtype}, shape: {w.shape}")
            
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
            print(f"upd_matrix after match shape type: {upd_matrix.dtype}, shape: {upd_matrix.shape}")
            
            # 确保w是浮点类型
            if w.dtype != torch.float32:
                print(f"Converting w from {w.dtype} to float32")
                w = w.to(torch.float32)
            
            print("Weight norm before:", w.norm().item())
            print("Weight norm after:", (w + upd_matrix).norm().item())

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            print(f"before w: {w}")
            w[...] += upd_matrix
            print(f"after w: {w}")
            # w = w.quantize() if is_bnb_quantized else w

        print(f"New weights successfully inserted into {list(deltas.keys())}")
        eval_metric = "exact match"
        print("rome main Edit succ:",compute_edit_quality(model, "qwen", hparams, tok, request, hparams.device, eval_metric=eval_metric, test_generation=True))
        

    return model, weights_copy


def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # # 打印模型结构以进行调试
    # print("Model structure:")
    # for name, module in model.named_modules():
    #     print(f"Module name: {name}")
    #     if hasattr(module, 'weight'):
    #         print(f"  Has weight: {module.weight.shape if hasattr(module.weight, 'shape') else 'No shape'}")
    
    # Update target and print info
    request = deepcopy(request)
    if request["target_new"] != " ":
        # Space required for correct tokenization
        request["target_new"] = " " + request["target_new"]
        # request["target_new"] = request["target_new"]

    if '{}' not in request['prompt']:
        assert request['subject'] in request['prompt'] or \
               print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

        request['prompt'] = request['prompt'].replace(request['subject'], '{}')

    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates = get_context_templates(model, tok, hparams.context_template_length_params)
    
    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            context_templates,
        )
        print("Left vector shape:", left_vector.shape)

        if hparams.use_zo == False:
            right_vector: torch.Tensor = compute_v(
                weights,
                model,
                tok,
                request,
                hparams,
                layer,
                left_vector,
                context_templates,
            )
        else:
            right_vector = compute_v_zo(
                weights,
                model,
                tok,
                request,
                hparams,
                layer,
                left_vector,
                context_templates,
            )
        print("Right vector shape:", right_vector.shape)
        # right_vector: torch.Tensor = compute_v_randomprefix_zo(
        #     model,
        #     tok,
        #     request,
        #     hparams,
        #     layer,
        #     left_vector,
        #     get_context_templates(model, tok, hparams.context_template_length_params),
        # )
        # print("Right vector shape:", right_vector.shape)
        # left_vector: torch.Tensor = compute_u(
        #     model,
        #     tok,
        #     request,
        #     hparams,
        #     layer,
        #     context_templates,
        # )


        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            # upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            # upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            # weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )
            print(f"Delta norm: {deltas[weight_name][0].norm()}")
            print(f"Delta norm: {deltas[weight_name][1].norm()}")

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        matrix = matrix.reshape(shape)
        return matrix
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
