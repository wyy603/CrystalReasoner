from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from verl.utils import hf_tokenizer
from verl.utils.model import compute_position_id_with_mask


class OfflineDPODataset(Dataset):
    """Parquet dataset for offline DPO preference pairs."""

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer,
        config,
        max_samples: int = -1,
    ):
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = [os.path.expanduser(path) for path in parquet_files]
        self.max_samples = max_samples
        self.shuffle = config.get("shuffle", False)
        self.seed: Optional[int] = config.get("seed")

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.prompt_key = config.get("prompt_key", "prompt")
        self.chosen_key = config.get("chosen_key", "chosen")
        self.rejected_key = config.get("rejected_key", "rejected")
        self.max_length = config.get("max_length", 4096)
        self.truncation = config.get("truncation", "right")
        self.add_eos = config.get("add_eos", True)
        self.apply_chat_template = config.get("apply_chat_template", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.prompt_type = config.get("custom_data", {}).get("prompt_type")
        if self.prompt_type != "plaid_wyckoff_train+no_thinking":
            raise ValueError(
                "OfflineDPODataset is currently scoped to "
                "plaid_wyckoff_train+no_thinking preference pairs."
            )

        self._read_files()

    def _read_files(self):
        frames = [pd.read_parquet(path) for path in self.parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        missing = {self.prompt_key, self.chosen_key, self.rejected_key} - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"DPO parquet is missing required columns: {sorted(missing)}")

        total = len(self.dataframe)
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def _prompt_text(self, prompt: str) -> str:
        if not self.apply_chat_template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )

    def _tokenize_pair_side(self, prompt: str, response: str) -> dict[str, torch.Tensor]:
        prompt_text = self._prompt_text(prompt)
        response_text = response + (self.tokenizer.eos_token if self.add_eos else "")

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(
            dtype=torch.long
        )
        response_ids = self.tokenizer(response_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(
            dtype=torch.long
        )
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[: prompt_ids.shape[0]] = -100

        if input_ids.shape[0] < self.max_length:
            pad_len = self.max_length - input_ids.shape[0]
            pad_ids = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros((pad_len,), dtype=attention_mask.dtype)
            pad_labels = torch.full((pad_len,), -100, dtype=labels.dtype)
            input_ids = torch.cat((input_ids, pad_ids))
            attention_mask = torch.cat((attention_mask, pad_mask))
            labels = torch.cat((labels, pad_labels))
        elif input_ids.shape[0] > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                labels = labels[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                labels = labels[-self.max_length :]
            elif self.truncation == "error":
                raise NotImplementedError(f"sequence length {input_ids.shape[0]} exceeds max_length={self.max_length}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
        }

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]
        prompt = str(row[self.prompt_key])
        chosen = str(row[self.chosen_key])
        rejected = str(row[self.rejected_key])
        chosen_tensors = self._tokenize_pair_side(prompt, chosen)
        rejected_tensors = self._tokenize_pair_side(prompt, rejected)
        return {
            "chosen_input_ids": chosen_tensors["input_ids"],
            "chosen_attention_mask": chosen_tensors["attention_mask"],
            "chosen_position_ids": chosen_tensors["position_ids"],
            "chosen_labels": chosen_tensors["labels"],
            "rejected_input_ids": rejected_tensors["input_ids"],
            "rejected_attention_mask": rejected_tensors["attention_mask"],
            "rejected_position_ids": rejected_tensors["position_ids"],
            "rejected_labels": rejected_tensors["labels"],
        }
