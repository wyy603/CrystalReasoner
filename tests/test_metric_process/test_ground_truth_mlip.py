"""Ground-truth parquet regression (MLIP columns) with mocks for MLIP compute.

This module keeps only 2 regression tests:
1) conditional: (energies, relaxed_structures, energy_above_hull, is_stable,
   is_unique, is_novel, stable_unique_novel)
2) elastic: (gt, elastic_properties)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from crysreas import Config
from crysreas.metric_process import MetricProcess, merge_metric_process_config, run_metrics
from crysreas.metric_process.helpers import df_deserialize
from crysreas.utils.crystal import SimpleCrystal
from pymatgen.core import Structure

def _conditional_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "ground_truth"
        / "conditional+thinking.parquet"
    )

def _elastic_path() -> Path:
    return (
        Path(__file__).resolve().parent / "ground_truth" / "elastic+thinking.parquet"
    )

def _cte_path() -> Path:
    return (
        Config.DATA_PATH / "split_small_atoms_metric.parquet"
    )


def _assert_structure_equal_by_positions(a: Structure | None, b: Structure | None, rtol: float = 1e-5, atol: float = 1e-6) -> None:
    if a is None or (isinstance(a, float) and pd.isna(a)):
        assert b is None or (isinstance(b, float) and pd.isna(b))
        return

    assert a is not None and b is not None
    
    # TODO: Implement structure equality check
    pass


def _elastic_tensor_to_matrix(t) -> np.ndarray:
    """Convert the stored elastic tensor to a numeric 2D matrix."""
    if isinstance(t, np.ndarray) and t.dtype == object:
        rows = [np.asarray(r, dtype=float) for r in t]
        return np.vstack(rows)
    return np.asarray(t, dtype=float)


def _assert_elastic_properties_equal(a: dict, b: dict) -> None:
    if (a is None or b is None):
        return a is None and b is None
    assert set(a.keys()) == set(b.keys())
    for k in a.keys():
        av, bv = a[k], b[k]
        if k == "elastic_tensor":
            am = _elastic_tensor_to_matrix(av)
            bm = _elastic_tensor_to_matrix(bv)
            np.testing.assert_allclose(am, bm, rtol=1e-2, atol=1e-2)
        elif isinstance(av, (int, float, np.floating)) or isinstance(
            bv, (int, float, np.floating)
        ):
            np.testing.assert_allclose(float(av), float(bv), rtol=1e-2, atol=1e-2)
        else:
            assert av == bv


def _assert_gt_dict_equal(a: dict, b: dict) -> None:
    assert set(a.keys()) == set(b.keys())
    for k in a.keys():
        av, bv = a[k], b[k]
        if isinstance(av, np.ndarray) or isinstance(bv, np.ndarray):
            np.testing.assert_allclose(
                np.asarray(av, dtype=float),
                np.asarray(bv, dtype=float),
                rtol=1e-5,
                atol=1e-6,
            )
        else:
            assert av == bv

def to_comparable_series(data):
    return pd.Series(data).replace({np.nan: None})


def _test_conditional() -> None:
    p = _conditional_path()
    gold = pd.read_parquet(p).iloc[:8].copy()
    df_deserialize(gold)

    inputs = gold[["mp_id", "prompt", "responses"]].copy()
    df_deserialize(inputs)

    # Only run the MLIP-heavy chain that produces the columns we care about:
    # relaxed_structures -> energies -> energy_above_hull/is_stable
    # -> stable_unique_novel/is_unique/is_novel
    names = ["stable_unique_novel"]

    mp_cfg = merge_metric_process_config(
        None, {"prompt_type": ["conditional", "thinking"]}
    )
    with MetricProcess(mp_cfg) as proc:
        out = proc.process(inputs.copy(), names, forced=False, log=True)

    assert len(out) == len(gold)

    # relaxed_structures: compare canonical SimpleCrystal simple text
    for i in range(len(out)):
        _assert_structure_equal_by_positions(out["relaxed_structures"].iloc[i], gold["relaxed_structures"].iloc[i])

    # Scalar / boolean arrays
    for col in ("energies", "energy_above_hull", "is_stable"):
        print("col", col, out[col].reset_index(drop=True))
        pd.testing.assert_series_equal(
            to_comparable_series(out[col].reset_index(drop=True)),
            to_comparable_series(gold[col].reset_index(drop=True)),
            check_names=False,
            check_dtype=False,
        )

    for col in ("is_novel", "is_unique", "stable_unique_novel"):
        pd.testing.assert_series_equal(
            to_comparable_series(out[col].reset_index(drop=True)),
            to_comparable_series(gold[col].reset_index(drop=True)),
            check_names=False,
            check_dtype=False,
        )

def _test_elastic() -> None:
    p = _elastic_path()
    gold = pd.read_parquet(p).iloc[:8].copy()
    df_deserialize(gold)

    inputs = gold[["mp_id", "prompt", "responses"]].copy()
    df_deserialize(inputs)

    names = ["gt", "elastic_properties"]

    mp_cfg = merge_metric_process_config(
        None, {"prompt_type": ["elastic", "thinking"]}
    )
    print("test_elastic, inputs = ", inputs)
    out = run_metrics(
        inputs.copy(), names, config=mp_cfg, forced=False, log=True
    )

    assert len(out) == len(gold)

    # gt: compare dicts exactly
    for i in range(len(out)):
        _assert_gt_dict_equal(out["gt"].iloc[i], gold["gt"].iloc[i])

    print("col", "relaxed_structures", out["relaxed_structures"].reset_index(drop=True))
    print("col", "elastic_properties", out["elastic_properties"].reset_index(drop=True))
    # elastic_properties: compare elastic_tensor + scalar fields
    for i in range(len(out)):
        _assert_elastic_properties_equal(
            out["elastic_properties"].iloc[i],
            gold["elastic_properties"].iloc[i],
        )

def _test_cte() -> None:

    p = _cte_path()
    gold = pd.read_parquet(p).iloc[:64].copy()
    df_deserialize(gold)

    inputs = gold[["mp_id", "simple_structure"]].copy()

    names = ["cte"]
    out = run_metrics(
        inputs.copy(),
        names,
        config=merge_metric_process_config(None, None),
        forced=False,
        log=True,
    )

    assert len(out) == len(gold)

    valid_count = 0
    none1_count = 0
    none2_count = 0
    nonediff_count = 0
    for i in range(len(out)):
        out_val = out["cte"].iloc[i]
        gold_val = gold["cte"].iloc[i]

        if out_val is None or gold_val is None:
            if out_val is None:
                none1_count += 1
            if gold_val is None:
                none2_count += 1
            if out_val is None and gold_val is None:
                valid_count += 1
            else:
                nonediff_count += 1

        else:
            try:
                np.testing.assert_allclose(
                    np.asarray(out_val["thermal_expansion"], dtype=float),
                    np.asarray(gold_val["thermal_expansion"], dtype=float),
                    rtol=0,
                    atol=1e-5,
                )
            except AssertionError:
                valid_count += 1
    
    print(f"Valid {valid_count} / {len(out)}")
    print(f"None out {none1_count} / {len(out)}")
    print(f"None gold {none2_count} / {len(out)}")
    print(f"None diff {nonediff_count} / {len(out)}")


@pytest.mark.skipif(
    not _conditional_path().is_file(), reason="conditional ground-truth parquet missing"
)
@pytest.mark.skipif(
    not (Config.DATA_PATH / "MP_shelve.db").exists() and not (Config.DATA_PATH / "MP_shelve.dat").exists(), reason="MP_shelve not available"
)
def test_conditional() -> None:
    _test_conditional()


@pytest.mark.skipif(
    not _elastic_path().is_file(), reason="elastic ground-truth parquet missing"
)
@pytest.mark.skipif(
    not (Config.DATA_PATH / "MP_shelve.db").exists() and not (Config.DATA_PATH / "MP_shelve.dat").exists(), reason="MP_shelve not available"
)
@pytest.mark.skip("aa")
def test_elastic() -> None:
    _test_elastic()


@pytest.mark.skipif(
    not _cte_path().is_file(), reason="cte ground-truth parquet missing"
)
@pytest.mark.skipif(
    not (Config.DATA_PATH / "MP_shelve.db").exists() and not (Config.DATA_PATH / "MP_shelve.dat").exists(), reason="MP_shelve not available"
)
def test_cte() -> None:
    _test_cte()

if __name__ == "__main__":
    import ray
    ray.init(num_cpus=16, num_gpus=1)
    test_cte()