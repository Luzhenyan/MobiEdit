import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..quantization.quantizer import W8A16Model
from ..rome import repr_tools
from ...util.globals import *

from .layer_stats import layer_stats
from .rome_hparams import ROMEHyperParams

# Cache variables
inv_mom2_cache = {}


def get_inv_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    hparams=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams
        )
        inv_mom2_cache[key] = torch.inverse(
            stat.mom2.moment().to(f"cuda:{hparams.device}")
        ).float()  # Cast back to float32

    return inv_mom2_cache[key]


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")
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

    if hparams.quantize == True:
        print("=====compute u quantization=====")
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
    
        before_quan_logits = model(**input_tok).logits
    
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

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request["subject"]
        print(f"Selected u projection object {word}")
        
        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.format(request["prompt"]) for templ in context_templates
            ],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_") :],
            **word_repr_args,
        ).mean(0)

    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply inverse second moment adjustment
    u = cur_repr
    if hparams.mom2_adjustment:
        u = get_inv_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            hparams=hparams,
        ) @ u.unsqueeze(1)
        u = u.squeeze()

    return u / u.norm()
