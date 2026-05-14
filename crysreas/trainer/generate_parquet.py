from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
import random

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
import tyro
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _normalize_prompt_to_chat(prompt_obj: Any) -> list[dict[str, str]]:
    if isinstance(prompt_obj, list):
        if all(isinstance(x, dict) and "role" in x and "content" in x for x in prompt_obj):
            return [{"role": str(x["role"]), "content": str(x["content"])} for x in prompt_obj]
        return [{"role": "user", "content": "\n".join(str(x) for x in prompt_obj)}]
    if isinstance(prompt_obj, dict):
        if "role" in prompt_obj and "content" in prompt_obj:
            return [{"role": str(prompt_obj["role"]), "content": str(prompt_obj["content"])}]
        return [{"role": "user", "content": str(prompt_obj.get("content", ""))}]
    return [{"role": "user", "content": str(prompt_obj)}]


@dataclass
class Args:
    input_path: str
    output_path: str
    model_path: str
    prompt_key: str = "prompt"
    response_key: str = "responses"
    batch_size: int = 64
    num_samples: int = 1
    seed: int = 42
    prompt_length: int = 2048
    trust_remote_code: bool = False
    nnodes: int = 1
    n_gpus_per_node: int = 1
    device: str = "cuda"
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.7
    response_length: int = 3584
    rollout_dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = True
    free_cache_engine: bool = True
    load_format: str = "auto"
    tensor_model_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    max_model_len: int | None = None
    max_num_seqs: int = 1024
    log_prob_micro_batch_size: int | None = None
    log_prob_micro_batch_size_per_gpu: int = 8
    do_sample: bool = True
    disable_log_stats: bool = True
    enable_chunked_prefill: bool = True
    rollout_n: int = 1
    calculate_log_probs: bool = False
    num_cpus: int | None = None
    preformatted_prompt: bool = False
    squeeze_single_sample: bool = True


def _ensure_ray_initialized(*, num_cpus: int | None = None) -> None:
    if not ray.is_initialized():
        ray_kwargs: dict[str, Any] = {
            "runtime_env": {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        }
        if num_cpus is not None:
            ray_kwargs["num_cpus"] = num_cpus
        ray.init(**ray_kwargs)


def generate_from_dataframe(
    input_df: pd.DataFrame,
    *,
    model_path: str,
    prompt_key: str = "prompt",
    response_key: str = "responses",
    batch_size: int = 64,
    num_samples: int = 1,
    seed: int = 42,
    n_samples: int | None = None,
    prompt_length: int = 8192,
    trust_remote_code: bool = False,
    nnodes: int = 1,
    n_gpus_per_node: int = 8,
    device: str = "cuda",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.7,
    response_length: int = 3584,
    rollout_dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.6,
    ignore_eos: bool = False,
    enforce_eager: bool = True,
    free_cache_engine: bool = True,
    load_format: str = "auto",
    tensor_model_parallel_size: int = 1,
    data_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    max_num_batched_tokens: int = 8192,
    max_model_len: int | None = None,
    max_num_seqs: int = 1024,
    log_prob_micro_batch_size: int | None = None,
    log_prob_micro_batch_size_per_gpu: int = 8,
    do_sample: bool = True,
    disable_log_stats: bool = True,
    enable_chunked_prefill: bool = True,
    rollout_n: int = 1,
    calculate_log_probs: bool = False,
    num_cpus: int | None = None,
    preformatted_prompt: bool = False,
    squeeze_single_sample: bool = True,
) -> pd.DataFrame:
    # Backward compatibility: prefer num_samples, but accept legacy n_samples.
    if n_samples is not None:
        num_samples = n_samples
    random.seed(seed)
    np.random.seed(seed)
    _ensure_ray_initialized(num_cpus=num_cpus)
    print(f"[generate_parquet] num_samples={num_samples}, batch_size={batch_size}")

    local_path = copy_to_local(model_path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert num_samples >= 1, "num_samples should always >= 1"
    if prompt_key not in input_df.columns:
        raise ValueError(f"Input parquet missing prompt column: {prompt_key!r}")
    prompt_values = input_df[prompt_key].tolist()
    chat_lst = [_normalize_prompt_to_chat(v) for v in prompt_values]
    text_prompts = [str(v) for v in prompt_values]

    worker_config = {
        "model": {"path": model_path, "trust_remote_code": trust_remote_code},
        "actor": {"strategy": "fsdp", "fsdp_config": {"fsdp_size": -1}},
        "rollout": {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "vllm",
            "mode": "sync",
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "dtype": rollout_dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "ignore_eos": ignore_eos,
            "enforce_eager": enforce_eager,
            "free_cache_engine": free_cache_engine,
            "load_format": load_format,
            "tensor_model_parallel_size": tensor_model_parallel_size,
            "data_parallel_size": data_parallel_size,
            "pipeline_model_parallel_size": pipeline_model_parallel_size,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "log_prob_micro_batch_size": log_prob_micro_batch_size,
            "log_prob_micro_batch_size_per_gpu": log_prob_micro_batch_size_per_gpu,
            "do_sample": do_sample,
            "disable_log_stats": disable_log_stats,
            "enable_chunked_prefill": enable_chunked_prefill,
            "n": rollout_n,
            "calculate_log_probs": calculate_log_probs,
        },
        "trainer": {"nnodes": nnodes, "n_gpus_per_node": n_gpus_per_node, "device": device},
    }
    worker_config = OmegaConf.create(worker_config)

    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=worker_config,
        role="rollout",
    )
    resource_pool = RayResourcePool(process_on_nodes=[n_gpus_per_node] * nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=device,
    )
    wg.init_model()

    total_samples = len(input_df)
    num_batch = -(-total_samples // batch_size)
    output_lst = [[] for _ in range(num_samples)]

    for batch_idx in tqdm(range(num_batch)):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        left = batch_idx * batch_size
        right = (batch_idx + 1) * batch_size
        if preformatted_prompt:
            batch_prompts = text_prompts[left:right]
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
        else:
            batch_chat_lst = chat_lst[left:right]
            inputs = tokenizer.apply_chat_template(
                batch_chat_lst,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(num_samples):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            output_texts: list[str] = []
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                output_texts.append(response_str)

            output_lst[n_sample].extend(output_texts)

    output_arr = np.array(output_lst, dtype=object)
    responses = np.transpose(output_arr, axes=(1, 0)).tolist()

    output_df = input_df.copy()
    if num_samples == 1 and squeeze_single_sample:
        output_df[response_key] = [row[0] for row in responses]
    else:
        output_df[response_key] = responses
    return output_df


def generate_parquet_file(args: Args) -> None:
    input_df = pd.read_parquet(args.input_path)
    output_df = generate_from_dataframe(
        input_df,
        model_path=args.model_path,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
        prompt_length=args.prompt_length,
        trust_remote_code=args.trust_remote_code,
        nnodes=args.nnodes,
        n_gpus_per_node=args.n_gpus_per_node,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        response_length=args.response_length,
        rollout_dtype=args.rollout_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        ignore_eos=args.ignore_eos,
        enforce_eager=args.enforce_eager,
        free_cache_engine=args.free_cache_engine,
        load_format=args.load_format,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        data_parallel_size=args.data_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        log_prob_micro_batch_size=args.log_prob_micro_batch_size,
        log_prob_micro_batch_size_per_gpu=args.log_prob_micro_batch_size_per_gpu,
        do_sample=args.do_sample,
        disable_log_stats=args.disable_log_stats,
        enable_chunked_prefill=args.enable_chunked_prefill,
        rollout_n=args.rollout_n,
        calculate_log_probs=args.calculate_log_probs,
        num_cpus=args.num_cpus,
        preformatted_prompt=args.preformatted_prompt,
        squeeze_single_sample=args.squeeze_single_sample,
    )
    output_dir = os.path.dirname(args.output_path)
    makedirs(output_dir, exist_ok=True)
    output_df.to_parquet(args.output_path)
    print(f"[generate_parquet] Wrote parquet to {args.output_path}")


if __name__ == "__main__":
    tyro.cli(generate_parquet_file)
