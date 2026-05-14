"""Tests for crysreas.metric_process basic metrics (no MLIP)."""

from __future__ import annotations

import json
import shelve
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from crysreas import Config

from crysreas.metric_process.helpers import df_deserialize
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from crysreas.metric_process import MetricProcess, merge_metric_process_config, run_metrics
from crysreas.trainer.crystal_dataset import CrystalDataset
from crysreas.utils.crystal import SimpleCrystal
from crysreas.metric_process.basic import (
    ensure_cte_reward,
    ensure_fit_format,
    ensure_simple_structure,
    ensure_smact_validity,
    ensure_stable_unique_novel,
    ensure_structure_validity,
)
from crysreas.utils.crystaltext import crystaltext_string

_CFG_DEFAULT = merge_metric_process_config(None, None)


class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]


def _minimal_si_simple_response() -> str:
    """LLM block uses SimpleCrystal simple text, not standard CIF (see crystal.py)."""
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si"], [[0, 0, 0]])
    simple = SimpleCrystal(structure).to_simple_no_sym()
    return f"prefix<CIF>{simple}</CIF>suffix"


def test_ensure_simple_structure_parses_cif():
    df = pd.DataFrame({"responses": [_minimal_si_simple_response()]})
    ensure_simple_structure(df, _CFG_DEFAULT)
    assert "simple_structure" in df.columns
    assert df["simple_structure"].iloc[0] is not None
    assert len(df["simple_structure"].iloc[0]) == 1


def test_ensure_simple_structure_invalid_cif_yields_none():
    df = pd.DataFrame({"responses": ["<CIF>not a cif</CIF>"]})
    ensure_simple_structure(df, _CFG_DEFAULT)
    assert df["simple_structure"].iloc[0] is None


def test_ensure_simple_structure_dispatches_to_crystaltext_parser():
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "O"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    response = f"CIF File: <CIF>{crystaltext_string(structure, seed=0, translate=False)}</CIF>"
    cfg = merge_metric_process_config(None, {"prompt_type": ["crystaltextllm_generation", "no_thinking"]})
    df = pd.DataFrame({"responses": [response]})
    ensure_simple_structure(df, cfg)
    parsed = df["simple_structure"].iloc[0]
    assert parsed is not None
    assert parsed.num_sites == 2


def test_ensure_simple_structure_dispatches_to_crystaltext_8_parser():
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "O"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    response = f"CIF File: <CIF>{crystaltext_string(structure, seed=0, translate=False, precision_8=True)}</CIF>"
    cfg = merge_metric_process_config(None, {"prompt_type": ["crystaltextllm_8_generation", "no_thinking"]})
    df = pd.DataFrame({"responses": [response]})
    ensure_simple_structure(df, cfg)
    parsed = df["simple_structure"].iloc[0]
    assert parsed is not None
    assert parsed.num_sites == 2


def test_crystaltextllm_dataset_item_generation_response_runs_metric_process(tmp_path: Path):
    structure = Structure(Lattice.cubic(5.43), ["Si", "O"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    elem = {
        "material_id": "mp-1",
        "structure": structure,
        "condensed_structure": structure.composition,
        "formation_energy_per_atom": -1.234,
        "band_gap": 1.5,
        "energy_above_hull": 0.01,
        "description": "A test crystalline material.",
    }

    split_path = tmp_path / "split.json"
    db_path = tmp_path / "db"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"train": ["mp-1"], "test": [], "val": []}, f)
    with shelve.open(str(db_path)) as db:
        db["mp-1"] = elem

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
        ds = CrystalDataset("train", "checkpoints_merged/no_thinking", cfg)
        sample = ds[0]
        assert set(sample) == {
            "input_ids",
            "attention_mask",
            "task_mask",
            "position_ids",
            "loss_mask",
        }

        valid_input_ids = sample["input_ids"][sample["attention_mask"].bool()]
        prompt_plus_target = ds.tokenizer.decode(valid_input_ids, skip_special_tokens=False)
        assert "Generate a description of the lengths and angles of the lattice vectors" in prompt_plus_target

        generated_response = (
            f"CIF File: <CIF>{crystaltext_string(structure, seed=0, translate=False)}</CIF>"
            "<|im_end|><|endoftext|>"
        )
        generation_like_df = pd.DataFrame(
            {
                "mp_id": ["mp-1"],
                "prompt": [prompt_plus_target],
                "responses": [generated_response],
            }
        )

        mp_cfg = merge_metric_process_config(None, {"prompt_type": ["crystaltextllm_generation", "no_thinking"]})
        with MetricProcess(mp_cfg) as proc:
            out = proc.process(generation_like_df, ["simple_structure", "structure_validity"])

        parsed = out["simple_structure"].iloc[0]
        assert parsed is not None
        assert parsed.num_sites == 2
        assert bool(out["structure_validity"].iloc[0])
        ds.db.close()


def test_fit_format_accepts_crystaltext_outer_wrapper():
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si"], [[0, 0, 0]])
    response = (
        "prefix"
        + f"CIF File: <CIF>{crystaltext_string(structure, seed=0, translate=False)}</CIF>"
        + "<|im_end|><|endoftext|>"
    )
    df = pd.DataFrame({"responses": [response]})
    cfg = merge_metric_process_config(None, {"prompt_type": ["crystaltextllm_generation", "no_thinking"]})
    ensure_fit_format(df, cfg)
    assert df["fit_format"].iloc[0] == 1


def test_ensure_simple_structure_skips_if_column_exists():
    sentinel = object()
    df = pd.DataFrame(
        {
            "responses": [_minimal_si_simple_response()],
            "simple_structure": [sentinel],
        }
    )
    ensure_simple_structure(df, _CFG_DEFAULT)
    assert df["simple_structure"].iloc[0] is sentinel


def test_structure_and_smact_after_simple():
    df = pd.DataFrame({"responses": [_minimal_si_simple_response()]})
    ensure_structure_validity(df, _CFG_DEFAULT)
    assert df["structure_validity"].iloc[0]
    ensure_smact_validity(df, _CFG_DEFAULT)
    assert df["smact_validity"].iloc[0]


def test_run_metrics_simple_structure_registered():
    df = pd.DataFrame({"responses": [_minimal_si_simple_response()]})
    out = run_metrics(df, ["simple_structure"], config=_CFG_DEFAULT)
    assert out["simple_structure"].iloc[0] is not None


def test_fit_format():
    # Two token ranges: position 100 lands in second segment; CIF end inside last 4 tokens of "completion"
    response_ok = "x" * 100 + "y" * 20 + "<CIF>a</CIF><|im_end|><|endoftext|>"
    df = pd.DataFrame({"responses": [response_ok]})
    ensure_fit_format(df, _CFG_DEFAULT)
    assert df["fit_format"].iloc[0] == 1

    response_bad = "no cif markers at all"  # start_idx -1 -> 0
    df2 = pd.DataFrame({"responses": [response_bad]})
    ensure_fit_format(df2, _CFG_DEFAULT)
    assert df2["fit_format"].iloc[0] == 0

    response_bad_2 = "x" * 100 + "y" * 20 + "<CIF>a</CIF>bcdefghu<|endoftext|>"
    df3 = pd.DataFrame({"responses": [response_bad_2]})
    ensure_fit_format(df3, _CFG_DEFAULT)
    assert df3["fit_format"].iloc[0] == 0


def test_gt_mock_shelve():
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si"], [[0, 0, 0]])
    elem = {
        "material_id": "mp-999",
        "structure": structure,
        "bulk_modulus": 100.0,
        "shear_modulus": 50.0,
    }

    mock_db = MagicMock()
    mock_db.__getitem__ = lambda self, k: elem

    class MockShelve:
        def __enter__(self):
            return mock_db

        def __exit__(self, *args):
            return False

    with patch("crysreas.metric_process.basic.shelve.open", return_value=MockShelve()):
        cfg = merge_metric_process_config(None, {"prompt_type": ["elastic"]})
        df = pd.DataFrame({"mp_id": ["mp-999"]})
        out = run_metrics(df, ["gt"], config=cfg)
        gt0 = out["gt"].iloc[0]
        assert gt0["mp_id"] == "mp-999"
        assert "comp" in gt0
        assert "spg" in gt0


def test_composition_and_spacegroup_with_mock_gt():
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si"], [[0, 0, 0]])
    simple = SimpleCrystal(structure).to_simple_no_sym()
    st = SimpleCrystal.from_simple_no_sym(simple).structure
    spg = SpacegroupAnalyzer(st, symprec=0.1, angle_tolerance=5).get_space_group_number()
    gt = {
        "mp_id": "mp-999",
        "comp": str(st.composition.reduced_composition),
        "spg": spg,
    }
    response = f"<CIF>{simple}</CIF>"

    df = pd.DataFrame(
        {
            "responses": [response],
            "gt": [gt],
        }
    )
    out = run_metrics(
        df, ["composition_consistency", "spacegroup_consistency"], config=_CFG_DEFAULT
    )
    assert out["composition_consistency"].iloc[0]
    assert out["spacegroup_consistency"].iloc[0]


def test_ensure_cte_reward_recompute_qha_only_top16_lowest_energy():
    n = 20
    structures = [f"s{i}" for i in range(n)]
    df = pd.DataFrame(
        {
            # Provide a bin interval so ensure_cte_reward can use _range_reward_quadratic (smooth reward).
            "gt": [{"thermal_expansion_300k": (0.0, 2.0)}] * n,
            "simple_structure": structures,
            "structure_validity": [True] * n,
            "smact_validity": [True] * n,
            "small_atoms": [True] * n,
            "energies": [float(i) for i in range(n)],
        }
    )

    class _FakeQHA:
        def __init__(self, val: float):
            self._val = val

        def model_dump(self):
            return {
                "temperatures": [200, 300, 400],
                "thermal_expansion": [0.0, self._val, 0.0],
            }

    captured = {"items": None, "remote_method": None}

    def _fake_batched(remote_method, items, metric_name, batch_size=16):
        captured["items"] = list(items)
        captured["remote_method"] = remote_method
        return [_FakeQHA(1.0) for _ in items]

    fake_calculate_qha = object()
    with patch("crysreas.metric_process.basic._ensure_registered_dependencies", return_value=None), patch(
        "crysreas.mlip.cte.calculate_qha", fake_calculate_qha
    ), patch(
        "crysreas.metric_process.basic._batched_remote_results_with_debug_tqdm",
        side_effect=_fake_batched,
    ):
        ensure_cte_reward(df, _CFG_DEFAULT)

    assert "cte_reward" in df.columns
    assert captured["remote_method"] is fake_calculate_qha
    assert captured["items"] == structures[:16]
    assert int(df["cte_reward"].sum()) == 16
    assert (df.loc[:15, "cte_reward"] == 1).all()
    assert (df.loc[16:, "cte_reward"] == 0).all()


def test_ensure_stable_unique_novel_coerces_dirty_is_stable_column():
    df = pd.DataFrame(
        {
            "simple_structure": list("abcdefgh"),
            "relaxed_structures": list("ABCDEFGH"),
            "energies": [0.0] * 8,
            "energy_above_hull": [0.0] * 8,
            "is_stable": [True, np.nan, None, "False", "true", "yes", "bad", 1.0],
        }
    )
    captured = {}

    def _fake_sun_batch(structures, relaxed_structures, energies, is_stable):
        captured["is_stable"] = is_stable
        is_novel = np.ones(len(structures), dtype=np.bool_)
        is_unique = np.ones(len(structures), dtype=np.bool_)
        return is_novel, is_unique, is_stable

    with patch(
        "crysreas.mlip.sun.compute_stable_unique_novel_batch",
        side_effect=_fake_sun_batch,
    ):
        ensure_stable_unique_novel(df, _CFG_DEFAULT)

    assert captured["is_stable"].dtype == np.bool_
    np.testing.assert_array_equal(
        captured["is_stable"],
        np.array([True, False, False, False, True, True, False, True]),
    )
    np.testing.assert_array_equal(df["stable_unique_novel"].to_numpy(), captured["is_stable"])


def _nullable_bool_equal(a, b) -> bool:
    na = a is None or (isinstance(a, float) and pd.isna(a))
    nb = b is None or (isinstance(b, float) and pd.isna(b))
    if na and nb:
        return True
    if na or nb:
        return False
    return bool(a) == bool(b)


def _simple_structure_canonical(s) -> str | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    if isinstance(s, str):
        return s.strip()
    return SimpleCrystal(s).to_simple_no_sym().strip()


def _assert_gt_row_equal(a, b) -> None:
    if a is None and b is None:
        return
    assert a is not None and b is not None
    assert dict(a) == dict(b)


def test_parquet_first64_multiprocess_matches_ground_truth():
    """Slice parquet (mp_id, prompt, responses only), run metrics with workers=4; match golden columns."""
    parquet_path = Path(__file__).resolve().parent / "ground_truth" / "conditional+thinking.parquet"
    metric_names = [
        "gt",
        "simple_structure",
        "structure_validity",
        "smact_validity",
        "spacegroup_consistency",
        "composition_consistency",
    ]

    mp_cfg = merge_metric_process_config(
        None, {"prompt_type": ["conditional", "prompt"]}
    )

    inputs = pd.read_parquet(
       parquet_path,
        columns=["mp_id", "prompt", "responses"],
    ).iloc[:64].copy()
    df_deserialize(inputs)

    expected = pd.read_parquet(
        parquet_path,
        columns=metric_names,
    ).iloc[:64].copy()
    df_deserialize(expected)

    with MetricProcess(mp_cfg) as proc:
        actual = proc.process(inputs.copy(), metric_names)
    actual = actual.sort_index()
    expected = expected.sort_index()

    assert len(actual) == len(expected) == 64
    for col in metric_names:
        assert col in actual.columns
    for i in range(64):
        _assert_gt_row_equal(expected["gt"].iloc[i], actual["gt"].iloc[i])
        assert _simple_structure_canonical(expected["simple_structure"].iloc[i]) == _simple_structure_canonical(
            actual["simple_structure"].iloc[i]
        )
        for col in (
            "structure_validity",
            "smact_validity",
            "spacegroup_consistency",
            "composition_consistency",
        ):
            assert _nullable_bool_equal(
                expected[col].iloc[i],
                actual[col].iloc[i],
            ), f"row {i} col {col}"
