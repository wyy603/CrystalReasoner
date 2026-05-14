from __future__ import annotations

import json
import os
import re
import shelve
from pathlib import Path

import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from crysreas.data.prompt_generator import get_info, get_info_infill
from crysreas.metric_process.helpers import parse_plaid_wyckoff_structure_from_response
from crysreas.utils.plaid_wyckoff import parse_plaid_wyckoff, plaid_wyckoff_string
from tqdm import tqdm

def _elem() -> dict:
    structure = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    return {
        "material_id": "mp-1",
        "structure": structure,
        "condensed_structure": structure.composition,
        "formation_energy_per_atom": -1.234,
        "band_gap": 1.5,
        "energy_above_hull": 0.01,
        "description": "A test crystalline material.",
    }


def test_get_info_plaid_wyckoff_generation():
    info = get_info(_elem(), "plaid_wyckoff_generation+no_thinking", debug=False, seed=7)
    assert "Below is a description of a bulk material." in info["question"]
    assert "<CIF>" in info["answer"] and "</CIF>" in info["answer"]
    inner = info["answer"].split("<CIF>", 1)[1].split("</CIF>", 1)[0]
    assert "Spacegroup:" in inner
    assert "Sites (" in inner
    assert re.search(r"^abc:\s+\d+\.\d{2}\s+\d+\.\d{2}\s+\d+\.\d{2}$", inner, re.MULTILINE)
    assert re.search(r"^angles:\s+\d+\.\d{2}\s+\d+\.\d{2}\s+\d+\.\d{2}$", inner, re.MULTILINE)
    assert re.search(
        r"^[A-Z][a-z]?\s+-?\d+\.\d{3}\s+-?\d+\.\d{3}\s+-?\d+\.\d{3}\s+\d+[A-Za-z]+$",
        inner,
        re.MULTILINE,
    )


def test_get_info_plaid_wyckoff_8_generation():
    info = get_info(_elem(), "plaid_wyckoff_8_generation+no_thinking", debug=False, seed=7)
    assert "<CIF>" in info["answer"] and "</CIF>" in info["answer"]
    inner = info["answer"].split("<CIF>", 1)[1].split("</CIF>", 1)[0]
    assert re.search(r"^abc:\s+\d+\.\d{8}\s+\d+\.\d{8}\s+\d+\.\d{8}$", inner, re.MULTILINE)
    assert re.search(r"^angles:\s+\d+\.\d{2}\s+\d+\.\d{2}\s+\d+\.\d{2}$", inner, re.MULTILINE)
    assert re.search(
        r"^[A-Z][a-z]?\s+-?\d+\.\d{8}\s+-?\d+\.\d{8}\s+-?\d+\.\d{8}\s+\d+[A-Za-z]+$",
        inner,
        re.MULTILINE,
    )


def test_get_info_infill_plaid_wyckoff():
    info = get_info_infill(_elem(), "plaid_wyckoff_train+no_thinking", debug=False, seed=9)
    assert "[MASK]" in info["question"]
    assert len(info["answer"]) > 0
    assert info["task_type"] == "infill"
    assert info["task_span"] == (0, len(info["answer"]))


def test_get_info_infill_plaid_wyckoff_8():
    info = get_info_infill(_elem(), "plaid_wyckoff_8_train+no_thinking", debug=False, seed=9)
    assert "[MASK]" in info["question"]
    assert re.search(r"^abc:\s+\d+\.\d{8}\s+\d+\.\d{8}\s+\d+\.\d{8}$", info["question"], re.MULTILINE)
    assert info["task_type"] == "infill"


def test_parse_plaid_wyckoff_structure_from_response():
    info = get_info(_elem(), "plaid_wyckoff_generation+no_thinking", debug=False, seed=11)
    structure = parse_plaid_wyckoff_structure_from_response(info["answer"])
    assert structure is not None
    assert StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5).fit(_elem()["structure"], structure)


def test_parse_plaid_wyckoff_malformed_returns_none():
    assert parse_plaid_wyckoff("not plaid wyckoff") is None
    assert parse_plaid_wyckoff(None) is None


def test_split_cdvae_test_roundtrip_small_subset():
    split_path = Path(os.environ.get("AI4SCI_PLAID_WYCKOFF_SPLIT", "assets/MP/split_cdvae.json"))
    db_path = Path(os.environ.get("AI4SCI_PLAID_WYCKOFF_DB", "assets/MP/MP_shelve"))
    limit = int(os.environ.get("AI4SCI_PLAID_WYCKOFF_LIMIT", "8"))
    if not split_path.exists() or not (db_path.with_suffix(".dat").exists() or db_path.exists()):
        pytest.skip("split_cdvae.json or MP_shelve is not available")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    keys = split["test"] if limit <= 0 else split["test"][:limit]
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    failures: list[str] = []

    with shelve.open(str(db_path), flag="r") as db:
        for key in tqdm(keys):
            structure = db[key]["structure"]
            serialized = plaid_wyckoff_string(structure)
            parsed = parse_plaid_wyckoff(serialized)
            if parsed is None or not matcher.fit(structure, parsed):
                failures.append(str(key))

    assert not failures, f"PLaID++ Wyckoff roundtrip failed for {failures[:20]} out of {len(keys)}"

test_split_cdvae_test_roundtrip_small_subset()
