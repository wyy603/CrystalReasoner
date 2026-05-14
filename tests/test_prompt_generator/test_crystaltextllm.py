from __future__ import annotations

import json
import re
import shelve
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pymatgen.core import Lattice, Structure

from crysreas.data.prompt_generator import get_info, get_info_infill
from crysreas.metric_process.helpers import parse_crystaltext_structure_from_response
from crysreas.trainer.crystal_dataset import CrystalDataset
from crysreas.utils.crystaltext import parse_crystaltext


class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]


def _elem() -> dict:
    structure = Structure(Lattice.cubic(5.43), ["Si", "O"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    return {
        "material_id": "mp-1",
        "structure": structure,
        "condensed_structure": structure.composition,
        "formation_energy_per_atom": -1.234,
        "band_gap": 1.5,
        "energy_above_hull": 0.01,
        "description": "A test crystalline material.",
    }


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def add_special_tokens(self, _kwargs):
        return 0

    def apply_chat_template(self, chat, add_generation_prompt=True, tokenize=False, **_kwargs):
        text = "".join(msg["content"] for msg in chat)
        return f"<user>{text}</user><assistant>"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False, **_kwargs):
        import torch

        ids = torch.tensor([[ord(ch) % 251 + 1 for ch in text]], dtype=torch.long)
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn}


class EmptyFloatTokenizer(FakeTokenizer):
    def __call__(self, text, return_tensors="pt", add_special_tokens=False, **_kwargs):
        import torch

        if text == "":
            return {
                "input_ids": torch.empty((1, 0), dtype=torch.float32),
                "attention_mask": torch.empty((1, 0), dtype=torch.float32),
            }
        return super().__call__(text, return_tensors=return_tensors, add_special_tokens=add_special_tokens, **_kwargs)


def test_get_info_crystaltextllm_generation():
    info = get_info(_elem(), "crystaltextllm_generation+no_thinking", debug=False, seed=7)
    assert "Below is a description of a bulk material." in info["question"]
    assert "The chemical formula is" in info["question"]
    assert "<CIF>" in info["answer"] and "</CIF>" in info["answer"]
    inner = info["answer"].split("<CIF>", 1)[1].split("</CIF>", 1)[0]
    assert "P1" not in inner
    lines = [line.strip() for line in inner.splitlines() if line.strip()]
    assert re.fullmatch(r"\d+\.\d \d+\.\d \d+\.\d", lines[0])
    assert re.fullmatch(r"\d+ \d+ \d+", lines[1])
    assert re.fullmatch(r"-?\d+\.\d{2} -?\d+\.\d{2} -?\d+\.\d{2}", lines[3])


def test_get_info_crystaltextllm_8_generation():
    info = get_info(_elem(), "crystaltextllm_8_generation+no_thinking", debug=False, seed=7)
    assert "<CIF>" in info["answer"] and "</CIF>" in info["answer"]
    inner = info["answer"].split("<CIF>", 1)[1].split("</CIF>", 1)[0]
    lines = [line.strip() for line in inner.splitlines() if line.strip()]
    assert re.fullmatch(r"\d+\.\d{8} \d+\.\d{8} \d+\.\d{8}", lines[0])
    assert re.fullmatch(r"\d+ \d+ \d+", lines[1])
    assert re.fullmatch(r"-?\d+\.\d{8} -?\d+\.\d{8} -?\d+\.\d{8}", lines[3])


def test_get_info_infill_crystaltextllm():
    info = get_info_infill(_elem(), "crystaltextllm_train+no_thinking", debug=False, seed=9)
    assert "[MASK]" in info["question"]
    assert len(info["answer"]) > 0
    assert info["task_type"] == "infill"
    assert info["task_span"] == (0, len(info["answer"]))


def test_get_info_infill_crystaltextllm_8():
    info = get_info_infill(_elem(), "crystaltextllm_8_train+no_thinking", debug=False, seed=9)
    assert "[MASK]" in info["question"]
    assert re.search(r"\d+\.\d{8} \d+\.\d{8} \d+\.\d{8}", info["question"])
    assert info["task_type"] == "infill"


def test_parse_crystaltext_structure_from_response():
    info = get_info(_elem(), "crystaltextllm_generation+no_thinking", debug=False, seed=11)
    structure = parse_crystaltext_structure_from_response(info["answer"])
    assert structure is not None
    assert structure.num_sites == 2


def test_parse_crystaltext_malformed_returns_none():
    assert parse_crystaltext("5.4 5.4\n90 90 90\nSi") is None
    assert parse_crystaltext("not crystaltext") is None
    assert parse_crystaltext(None) is None


def test_crystal_dataset_train_mixes_generation_and_infill(tmp_path: Path):
    split_path = tmp_path / "split.json"
    db_path = tmp_path / "db"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"train": ["mp-1", "mp-2"], "test": [], "val": []}, f)
    with shelve.open(str(db_path)) as db:
        db["mp-1"] = _elem()
        db["mp-2"] = _elem()

    cfg = AttrDict(
        {
            "max_length": 512,
            "truncation": "right",
            "use_shm": False,
            "seed": 1,
            "shuffle": False,
            "apply_chat_template_kwargs": {},
            "custom_data": SimpleNamespace(
                split_path=str(split_path),
                db_path=str(db_path),
                prompt_type="crystaltextllm_train+no_thinking",
            ),
        }
    )

    with patch("random.Random.random", return_value=0.1):
        ds = CrystalDataset("train", FakeTokenizer(), cfg)
        out = ds[0]
        assert out["task_mask"].sum().item() > 10
        ds.db.close()

    with patch("random.Random.random", return_value=0.9):
        ds = CrystalDataset("train", FakeTokenizer(), cfg)
        out = ds[0]
        assert 0 < out["task_mask"].sum().item() < out["loss_mask"].sum().item()
        ds.db.close()


def test_crystal_dataset_infill_empty_segments_stay_integral(tmp_path: Path):
    split_path = tmp_path / "split.json"
    db_path = tmp_path / "db"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"train": ["mp-1"], "test": [], "val": []}, f)
    with shelve.open(str(db_path)) as db:
        db["mp-1"] = _elem()

    cfg = AttrDict(
        {
            "max_length": 128,
            "truncation": "right",
            "use_shm": False,
            "seed": 1,
            "shuffle": False,
            "apply_chat_template_kwargs": {},
            "custom_data": SimpleNamespace(
                split_path=str(split_path),
                db_path=str(db_path),
                prompt_type="crystaltextllm_train+no_thinking",
            ),
        }
    )

    with patch("random.Random.random", return_value=0.9):
        ds = CrystalDataset("train", EmptyFloatTokenizer(), cfg)
        out = ds[0]
        assert all(t.dtype.is_floating_point is False for t in out.values())
        ds.db.close()
