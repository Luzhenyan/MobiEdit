import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *
from ...util.nethook import Trace, set_requires_grad
from ...util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    LengthCollator,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        # Load_From_File
        from datasets import Dataset, load_from_disk
        # raw_ds = load_dataset(
        #     ds_name,
        #     dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name]
        # )
        print(f"Loading dataset from disk...")
        raw_ds = load_from_disk("/home/pcllzy/wikipedia")
        print(f"raw_ds[train] size: {len(raw_ds['train'])}")
        print(f"raw_ds[train] first item: {raw_ds['train'][0]}")
        # wikipedia_path = '/home/pcllzy/wikipedia/train/data-00000-of-00037.arrow'
        # raw_ds = Dataset.from_file(wikipedia_path).select(range(10))
        # raw_ds = {'train': raw_ds}
        # print(f"raw_ds[train] size: {len(raw_ds['train'])}")
        # print(f"raw_ds[train] first item: {raw_ds['train'][0]}")
        # from datasets import load_from_disk, concatenate_datasets
        # # 正确加载数据集
        # wikipedia_path = '/home/pcllzy/wikipedia'
        # dataset_dict = load_from_disk(wikipedia_path)
        
        # # 检查数据集结构
        # print(f"数据集类型: {type(dataset_dict)}")
        # print(f"数据集键: {list(dataset_dict.keys()) if hasattr(dataset_dict, 'keys') else 'No keys (单一数据集)'}")
        
        # # 根据数据集结构选择正确的加载方式
        # if hasattr(dataset_dict, 'keys'):
        #     # 如果是DatasetDict类型，包含多个分割
        #     if 'train' in dataset_dict:
        #         raw_ds = dataset_dict['train']
        #         print(f"使用train分割，大小: {len(raw_ds)}")
        #     else:
        #         # 如果没有train分割，使用第一个可用的分割
        #         first_key = list(dataset_dict.keys())[0]
        #         raw_ds = dataset_dict[first_key]
        #         print(f"使用{first_key}分割，大小: {len(raw_ds)}")
        # else:
        #     # 如果直接是Dataset类型
        #     raw_ds = dataset_dict
        #     print(f"直接使用数据集，大小: {len(raw_ds)}")
        
        # # 如果raw_ds仍然为空或很小，尝试直接加载train目录
        # if len(raw_ds) < 10:
        #     print("尝试直接加载train目录...")
        #     train_path = f"{wikipedia_path}/train"
        #     raw_ds = load_from_disk(train_path)
        #     print(f"直接从train加载，大小: {len(raw_ds)}")
        
        # # 确认数据集大小，并检查其内容
        # print(f"最终数据集大小: {len(raw_ds)}")
        # if len(raw_ds) > 0:
        #     print(f"第一条数据示例: {next(iter(raw_ds))}")
        
        # # 设置希望使用的样本数量
        # n_samples = min(1000, len(raw_ds))
        # print(f"将使用 {n_samples} 个样本")
        
        # # 安全采样
        # if n_samples == len(raw_ds):
        #     return raw_ds
        # else:
        #     import random
        #     random_indices = random.sample(range(len(raw_ds)), n_samples)
        #     return raw_ds.select(random_indices)
        # raw_ds = load_dataset(
        #     ds_name,
        #     dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name]
        # )
        if hasattr(model.config, 'n_positions'):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config,'seq_length'):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError
                
        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
        if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config,'seq_length'):
        npos = model.config.seq_length
    else:
        raise NotImplementedError
        
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        # model_name = model.config._name_or_path.replace("/", "_")
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    print(f"Computing Cov locally....")

    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=LengthCollator(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, f"cuda:{hparams.device}")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat


if __name__ == "__main__":
    main()
