# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import os
from typing import Any

import hydra
import numpy as np
import ray
import torch

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf
import shelve, json
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from crysreas.data.prompt_generator import get_info
from crysreas import Config

_CIF_OPEN = "<CIF>"
_CIF_CLOSE = "</CIF>"


def _find_subsequence_first(haystack: list[int], needle: list[int]) -> int:
    """First start index of ``needle`` in ``haystack``, or -1 if not found."""
    if not needle:
        return 0
    n, m = len(haystack), len(needle)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return -1


def _cif_inner_information_from_rollout(
    tokenizer: Any,
    response_ids: torch.Tensor,
    rollout_log_probs: torch.Tensor | None,
) -> tuple[dict[str, Any], str]:
    """
    Sum / average of (-log p) over response tokens strictly between ``<CIF>`` and ``</CIF>``
    (boundary tag tokens are excluded). ``token_num`` is the number of those inner tokens.
    """
    empty = {"sum_of_information": None, "average_of_information": None, "token_num": None}
    if rollout_log_probs is None:
        return empty, "missing_log_probs"

    ids = response_ids.detach().cpu().long().tolist()
    open_ids = tokenizer.encode(_CIF_OPEN, add_special_tokens=False)
    close_ids = tokenizer.encode(_CIF_CLOSE, add_special_tokens=False)
    o = _find_subsequence_first(ids, open_ids)
    if o < 0:
        return empty, "missing_open_cif"

    after_open = o + len(open_ids)
    c_rel = _find_subsequence_first(ids[after_open:], close_ids)
    if c_rel < 0:
        content_start, content_end_excl = after_open, len(ids)
    else:
        content_start, content_end_excl = after_open, after_open + c_rel

    token_num = int(max(0, content_end_excl - content_start))
    if token_num == 0:
        return {"sum_of_information": 0.0, "average_of_information": None, "token_num": 0}, "empty_content"

    lp = rollout_log_probs.detach().cpu().float()
    if lp.numel() < content_end_excl:
        return empty, "logprob_too_short"

    info_sum = 0.0
    for j in range(content_start, content_end_excl):
        v = float(lp[j].item())
        # Log-probabilities are normally <= 0. Only treat non-finite values as invalid.
        if not np.isfinite(v):
            return empty, "invalid_non_finite_logprob"
        info_sum += -v

    avg = info_sum / float(token_num)
    return (
        {
            "sum_of_information": float(info_sum),
            "average_of_information": float(avg),
            "token_num": token_num,
        },
        "ok",
    )


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # When ``RUNPY_GENERATION_DEBUG=1`` (e.g. ``scripts/run.py generate_* --level=debug``), run the driver
    # loop in this process so ``tqdm`` and batch prints go to the job terminal instead of a Ray worker.
    if os.environ.get("RUNPY_GENERATION_DEBUG") == "1":
        _main_task_impl(config)
    else:
        ray.get(main_task.remote(config))


def _main_task_impl(config) -> None:
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    print(
        f"[main_generation] data.n_samples={config.data.n_samples} -> that many generations per prompt "
        "(parquet responses column list length). "
        "vLLM logs often show kwargs with n=1 (SamplingParams); that is not data.n_samples."
    )

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    db = shelve.open(config.data.custom_data.db_path)
    with open(config.data.custom_data.split_path, "r") as f:
        split = json.load(f)[config.data.custom_data.split_type]
    chat_lst = [[{"role": "user", "content": get_info(db[key], config.data.custom_data.prompt_type, seed=key[3:])["question"]}] for key in split]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    total_samples = len(split)
    config_batch_size = config.data.batch_size
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]
    extra_args_lst = [[] for _ in range(config.data.n_samples)]
    warned_no_logprobs = False
    extra_args_reason_counts: dict[str, int] = {}
    # Store one example key per reason to make debugging straightforward.
    extra_args_reason_examples: dict[str, str] = {}

    for batch_idx in tqdm(range(num_batch)):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(config.data.n_samples):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            output_texts = []
            batch_extra_args: list[dict[str, Any]] = []
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_length = int(valid_response_length.item())
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                rollout_lps = (
                    data_item.batch["rollout_log_probs"][:valid_response_length]
                    if "rollout_log_probs" in data_item.batch.keys()
                    else None
                )
                if rollout_lps is None and not warned_no_logprobs:
                    print(
                        "[main_generation] rollout_log_probs missing; "
                        "extra_args information fields will be null. "
                        "Set rollout.calculate_log_probs=true in Hydra config.",
                        flush=True,
                    )
                    warned_no_logprobs = True
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                output_texts.append(response_str)
                info_dict, info_reason = _cif_inner_information_from_rollout(
                    tokenizer, valid_response_ids, rollout_lps
                )
                batch_extra_args.append(info_dict)

                extra_args_reason_counts[info_reason] = extra_args_reason_counts.get(info_reason, 0) + 1
                if info_reason not in extra_args_reason_examples:
                    global_i = batch_idx * config_batch_size + i
                    if global_i < len(split):
                        extra_args_reason_examples[info_reason] = str(split[global_i])
                    else:
                        extra_args_reason_examples[info_reason] = f"out_of_range_idx_{global_i}"

            output_lst[n_sample].extend(output_texts)
            extra_args_lst[n_sample].extend(batch_extra_args)
    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    extra_args_arr = np.array(extra_args_lst, dtype=object)
    extra_args_out = np.transpose(extra_args_arr, axes=(1, 0)).tolist()

    data_dict = {}
    data_dict["mp_id"] = split
    data_dict["prompt"] = chat_lst
    data_dict["responses"] = output_lst
    data_dict["extra_args"] = extra_args_out
    dataframe = pd.DataFrame(data_dict)
    print("[main_generation] extra_args reason counts:")
    for reason, cnt in sorted(extra_args_reason_counts.items(), key=lambda x: x[0]):
        ex = extra_args_reason_examples.get(reason, "n/a")
        print(f"  - {reason}: {cnt} (example mp_id: {ex})")

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataframe.to_parquet(config.data.output_path)


@ray.remote(num_cpus=8)
def main_task(config):
    _main_task_impl(config)


if __name__ == "__main__":
    main()
