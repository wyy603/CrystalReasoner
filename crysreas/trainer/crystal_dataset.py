import json
import random
import shelve
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.model import compute_position_id_with_mask

from crysreas.data.prompt_generator import (
    PRIOR_WORK_TRAIN_TO_GENERATION,
    get_info,
    get_info_infill,
    simple_cif_positions,
)


class CrystalDataset(Dataset):
    def __init__(self, parquet_files: str, tokenizer, config, max_samples: int = -1):
        print(config)
        with open(config.custom_data.split_path, "r") as f:
            self.split = json.load(f)

        self.split_name = parquet_files
        if self.split_name == "test":
            random.seed(42)
            self.split[self.split_name] = random.sample(self.split[self.split_name], 256)

        self.db = shelve.open(config.custom_data.db_path)
        self.prompt_type = config.custom_data.prompt_type

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<CIF>", "</CIF>"]})
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_samples = max_samples

        self.max_length = config.get("max_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.use_shm = config.get("use_shm", False)
        self.seed = config.get("seed")
        random.seed(self.seed)
        self.shuffle = config.get("shuffle", False)
        assert self.truncation in ["error", "left", "right"]
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

    def __len__(self):
        print("split_name", self.split_name)
        return len(self.split[self.split_name])

    def _sample_seed(self, item: int) -> int:
        base = 0 if self.seed is None else int(self.seed)
        split_hash = sum(ord(ch) for ch in self.split_name)
        return base * 1000003 + split_hash * 1009 + int(item)

    def _build_info(self, elem: Any, item: int) -> dict[str, Any]:
        prompt_family = self.prompt_type.split("+", 1)[0]
        if prompt_family in PRIOR_WORK_TRAIN_TO_GENERATION:
            seed = self._sample_seed(item)
            rng = random.Random(seed)
            if rng.random() < 0.66:
                generation_prompt_type = f"{PRIOR_WORK_TRAIN_TO_GENERATION[prompt_family]}+no_thinking"
                return get_info(
                    elem,
                    generation_prompt_type,
                    seed=seed,
                )
            return get_info_infill(
                elem,
                self.prompt_type,
                seed=seed,
            )
        return get_info(elem, self.prompt_type)

    def _task_span(self, info: dict[str, Any], response: str) -> tuple[int, int]:
        if "task_span" in info:
            return tuple(info["task_span"])
        positions = simple_cif_positions(response)
        if not positions:
            return (0, len(response))
        return positions[0]

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        elem = self.db[self.split[self.split_name][item]]
        info = self._build_info(elem, item)
        prompt = info["question"]
        response = info["answer"]

        prompt_chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )

        task_l, task_r = self._task_span(info, response)
        parts = [prompt_chat, response[:task_l], response[task_l:task_r], response[task_r:], tokenizer.eos_token]

        lengths = []
        input_ids = None
        attention_mask = None
        for part in parts:
            tokenizer_output = tokenizer(part, return_tensors="pt", add_special_tokens=False)
            # Some tokenizers return float32 empty tensors for "", which later
            # breaks default_collate when mixed with normal int64 samples.
            current_ids = tokenizer_output["input_ids"][0].to(dtype=torch.long)
            current_attn = tokenizer_output["attention_mask"][0].to(dtype=torch.long)
            input_ids = torch.cat((input_ids, current_ids), dim=-1) if input_ids is not None else current_ids
            lengths.append(input_ids.shape[-1])
            attention_mask = (
                torch.cat((attention_mask, current_attn), dim=-1) if attention_mask is not None else current_attn
            )

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if lengths[0] > 1:
            loss_mask[: min(lengths[0], loss_mask.size(0)) - 1] = 0

        task_mask = attention_mask.clone()
        task_mask[: lengths[1]] = 0
        task_mask[lengths[2] :] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "task_mask": task_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
