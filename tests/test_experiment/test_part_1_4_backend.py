"""Tests for experiment thinking part 1.4 backend."""

from __future__ import annotations

import math

import torch

from crysreas.experiment.thinking.part_1_4_backend import _locate_cif_span, _score_cif_information


class _FakeTokenizer:
    """Character-level tokenizer for deterministic unit tests."""

    def __call__(
        self,
        texts,
        *,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
        return_offsets_mapping=True,
    ):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        assert padding is True
        assert return_offsets_mapping is True

        max_len = max(len(t) for t in texts)
        input_ids = []
        attention_mask = []
        offset_mapping = []
        for text in texts:
            ids = [ord(ch) % 13 for ch in text]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [0] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            offsets = [[i, i + 1] for i in range(len(text))] + [[0, 0]] * pad_len
            offset_mapping.append(offsets)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "offset_mapping": torch.tensor(offset_mapping, dtype=torch.long),
        }


class _FakeModelOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeModel:
    """Uniform logits model so each token NLL is ln(vocab_size)."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _FakeModelOutput:
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), dtype=torch.float32)
        return _FakeModelOutput(logits=logits)


def test_score_cif_information_single_sample_average() -> None:
    text = "prefix ## CIF File\nabc\n</CIF> suffix"
    span = _locate_cif_span(text)
    assert span is not None

    tokenizer = _FakeTokenizer()
    model = _FakeModel(vocab_size=13)
    info_sum, n_tok = _score_cif_information(
        model=model,
        tokenizer=tokenizer,
        texts=[text],
        spans=[span],
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert int(n_tok[0]) > 0
    avg_info = float(info_sum[0]) / float(n_tok[0])
    assert math.isclose(avg_info, math.log(13.0), rel_tol=1e-6, abs_tol=1e-6)
