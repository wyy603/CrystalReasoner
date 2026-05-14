from types import SimpleNamespace

import pandas as pd
import torch

from crysreas.trainer.dpo_dataset import OfflineDPODataset
from crysreas.trainer.dpo_loss import compute_dpo_loss
from crysreas.trainer.main_dpo import build_plaid_dpo_pairs


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = "<pad>"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        assert not tokenize
        text = "".join(message["content"] for message in messages)
        if add_generation_prompt:
            return f"user:{text}\nassistant:"
        return f"user:{text}"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        assert return_tensors == "pt"
        ids = [(ord(ch) % 97) + 3 for ch in text]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }


def test_compute_dpo_loss_prefers_larger_policy_margin():
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    good_loss, _ = compute_dpo_loss(
        policy_chosen_logps=torch.tensor([2.0]),
        policy_rejected_logps=torch.tensor([0.0]),
        reference_chosen_logps=ref_chosen,
        reference_rejected_logps=ref_rejected,
        beta=1.0,
    )
    bad_loss, _ = compute_dpo_loss(
        policy_chosen_logps=torch.tensor([0.0]),
        policy_rejected_logps=torch.tensor([2.0]),
        reference_chosen_logps=ref_chosen,
        reference_rejected_logps=ref_rejected,
        beta=1.0,
    )
    assert good_loss < bad_loss


def test_offline_dpo_dataset_masks_prompt_and_returns_pair_tensors(tmp_path):
    path = tmp_path / "pairs.parquet"
    pd.DataFrame(
        [
            {
                "prompt": "make a Wyckoff crystal",
                "chosen": "stable response",
                "rejected": "unstable response",
            }
        ]
    ).to_parquet(path)
    cfg = SimpleNamespace(
        prompt_key="prompt",
        chosen_key="chosen",
        rejected_key="rejected",
        max_length=96,
        truncation="right",
        add_eos=True,
        apply_chat_template=True,
        apply_chat_template_kwargs={},
        custom_data={"prompt_type": "plaid_wyckoff_train+no_thinking"},
        get=lambda key, default=None: getattr(cfg, key, default),
    )

    tokenizer = TinyTokenizer()
    ds = OfflineDPODataset(str(path), tokenizer, cfg)
    item = ds[0]

    expected_keys = {
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_position_ids",
        "chosen_labels",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_position_ids",
        "rejected_labels",
    }
    assert set(item) == expected_keys
    assert item["chosen_input_ids"].shape == (96,)
    assert item["rejected_input_ids"].shape == (96,)

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "make a Wyckoff crystal"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_len = tokenizer(prompt_text)["input_ids"].shape[-1]
    assert torch.all(item["chosen_labels"][:prompt_len] == -100)
    assert torch.any(item["chosen_labels"][prompt_len:] != -100)


def test_build_plaid_dpo_pairs_groups_by_prompt(tmp_path):
    scored_path = tmp_path / "scored.parquet"
    out_path = tmp_path / "pairs.parquet"
    prompt_a = [{"role": "user", "content": "prompt A"}]
    prompt_b = [{"role": "user", "content": "prompt B"}]
    pd.DataFrame(
        [
            {
                "mp_id": "mp-a",
                "prompt": prompt_a,
                "responses": "stable A",
                "energy_above_hull": -0.01,
                "is_novel": True,
            },
            {
                "mp_id": "mp-a",
                "prompt": prompt_a,
                "responses": "metastable A",
                "energy_above_hull": 0.04,
                "is_novel": True,
            },
            {
                "mp_id": "mp-a",
                "prompt": prompt_a,
                "responses": "unstable A",
                "energy_above_hull": 0.2,
                "is_novel": False,
            },
            {
                "mp_id": "mp-b",
                "prompt": prompt_b,
                "responses": "stable B",
                "energy_above_hull": -0.02,
                "is_novel": True,
            },
        ]
    ).to_parquet(scored_path)

    n_pairs = build_plaid_dpo_pairs(scored_path, out_path, seed=1)
    pairs = pd.read_parquet(out_path)

    assert n_pairs == len(pairs)
    assert n_pairs >= 3
    assert set(pairs["prompt"]) == {"prompt A"}
    assert set(pairs["chosen"]) <= {"stable A", "metastable A"}
    assert "stable B" not in set(pairs["chosen"])
    assert "stable B" not in set(pairs["rejected"])
