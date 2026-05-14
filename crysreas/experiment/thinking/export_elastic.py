"""
Export balanced elastic samples from merged parquet files.

Rules:
- Read from both:
  - checkpoints_merged/rl_thinking_mix/elastic+thinking.parquet
  - checkpoints_merged/rl_elastic_thinking_new/elastic+thinking.parquet
- Randomly choose 6 (bulk_range, shear_range) target pairs while covering all
  observed bulk ranges and all observed shear ranges.
- For each chosen target pair, sample 12 rows:
  - 3 for fit_in_range == (0,0)
  - 3 for fit_in_range == (0,1)
  - 3 for fit_in_range == (1,0)
  - 3 for fit_in_range == (1,1)
- Exclude rows where elastic_properties is missing/invalid, or predicted
  bulk_modulus < 0, or predicted shear_modulus < 0.
- Export only requested columns + fit_in_range + origin.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import THINKING_DATA_DIR, ensure_thinking_dirs
from crysreas.utils.crystal import SimpleCrystal

INPUT_PARQUETS = (
    Path("checkpoints_merged/rl_thinking_mix/elastic+thinking.parquet"),
    Path("checkpoints_merged/rl_elastic_thinking_new/elastic+thinking.parquet"),
)
OUTPUT_PARQUET = THINKING_DATA_DIR / "elastic_samples.parquet"
OUTPUT_CSV = THINKING_DATA_DIR / "elastic_samples.csv"

PROPERTIES = ("bulk_modulus", "shear_modulus")
OUTPUT_COLUMNS = [
    "mp_id",
    "prompt",
    "responses",
    "gt",
    "energies",
    "relaxed_structures",
    "elastic_properties",
]
FIT_COLUMN = "fit_in_range"
ORIGIN_COLUMN = "origin"
TOTAL_RANGE_PAIRS = 6
SAMPLES_PER_COMBO = 3
FIT_COMBOS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _valid_elastic_properties(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    bulk = _to_float(obj.get("bulk_modulus"))
    shear = _to_float(obj.get("shear_modulus"))
    if bulk is None or shear is None:
        return False
    if bulk < 0 or shear < 0:
        return False
    return True


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(x):
        return None
    return x


def _parse_range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        seq = value.tolist()
    elif isinstance(value, (list, tuple)):
        seq = list(value)
    else:
        return None
    if len(seq) != 2:
        return None
    left = _to_float(seq[0])
    right = _to_float(seq[1])
    if left is None or right is None:
        return None
    return (left, right)


def _in_range(pred_value: Any, gt_range: tuple[float, float]) -> bool:
    pred = _to_float(pred_value)
    if pred is None:
        return False
    left, right = gt_range
    return (pred >= left) and (pred <= right)


def _simple_to_cif(simple_structure: Any) -> str | None:
    if not isinstance(simple_structure, str) or not simple_structure.strip():
        return None
    try:
        crystal = SimpleCrystal.from_simple_no_sym(simple_structure)
        return crystal.to_cif()
    except Exception:
        return None


def _choose_range_pairs(
    df: pd.DataFrame,
    random_state: int,
    n_pairs: int = TOTAL_RANGE_PAIRS,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    observed_pairs = sorted(
        df[["__bulk_gt_range", "__shear_gt_range"]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if len(observed_pairs) < n_pairs:
        raise ValueError(f"Observed range pairs ({len(observed_pairs)}) < required {n_pairs}.")

    all_bulk = set(df["__bulk_gt_range"].dropna().unique().tolist())
    all_shear = set(df["__shear_gt_range"].dropna().unique().tolist())

    rng = random.Random(random_state)
    max_tries = 50_000
    for _ in range(max_tries):
        picked = rng.sample(observed_pairs, n_pairs)
        bulk_cov = {x[0] for x in picked}
        shear_cov = {x[1] for x in picked}
        if bulk_cov == all_bulk and shear_cov == all_shear:
            return list(picked)
    raise ValueError(
        "Failed to find random range-pair subset with full bulk/shear coverage."
    )


def _sample_for_target_pair(
    df: pd.DataFrame,
    target_pair: tuple[tuple[float, float], tuple[float, float]],
    random_state: int,
    used_indices: set[int],
) -> pd.DataFrame:
    bulk_range, shear_range = target_pair
    bulk_in = df["elastic_properties"].map(
        lambda x: _in_range(x.get("bulk_modulus") if isinstance(x, dict) else None, bulk_range)
    )
    shear_in = df["elastic_properties"].map(
        lambda x: _in_range(x.get("shear_modulus") if isinstance(x, dict) else None, shear_range)
    )
    fit = pd.Series(
        list(zip(bulk_in.astype(int).tolist(), shear_in.astype(int).tolist())),
        index=df.index,
    )

    chosen_parts: list[pd.DataFrame] = []
    pair_local = df[
        (df["__bulk_gt_range"] == bulk_range) & (df["__shear_gt_range"] == shear_range)
    ]

    for k, combo in enumerate(FIT_COMBOS):
        rs = random_state + (k * 97)
        combo_mask = fit == combo

        local_pool = pair_local[combo_mask.loc[pair_local.index] & (~pair_local.index.isin(used_indices))]
        local_take_n = min(SAMPLES_PER_COMBO, len(local_pool))
        if local_take_n > 0:
            take_local = local_pool.sample(n=local_take_n, random_state=rs)
            chosen_parts.append(take_local)
            used_indices.update(take_local.index.tolist())
        else:
            local_take_n = 0

        remain = SAMPLES_PER_COMBO - local_take_n
        if remain > 0:
            global_pool = df[combo_mask & (~df.index.isin(used_indices))]
            if len(global_pool) < remain:
                raise ValueError(
                    f"Target pair {target_pair} combo {combo} needs {remain} more rows, "
                    f"but only {len(global_pool)} available globally."
                )
            take_global = global_pool.sample(n=remain, random_state=rs + 1)
            chosen_parts.append(take_global)
            used_indices.update(take_global.index.tolist())

    out = pd.concat(chosen_parts, ignore_index=False).copy()
    out["__target_bulk_range"] = [bulk_range] * len(out)
    out["__target_shear_range"] = [shear_range] * len(out)
    out[FIT_COLUMN] = fit.loc[out.index].map(lambda x: f"({int(x[0])},{int(x[1])})").tolist()
    return out


def run(
    input_paths: tuple[Path, Path] = INPUT_PARQUETS,
    output_path: Path = OUTPUT_PARQUET,
    random_state: int = 42,
) -> None:
    np.random.seed(random_state)
    random.seed(random_state)
    ensure_thinking_dirs(data_dir=output_path.parent)
    frames: list[pd.DataFrame] = []
    for input_path in input_paths:
        frame = pd.read_parquet(input_path, columns=OUTPUT_COLUMNS).copy()
        frame[ORIGIN_COLUMN] = str(input_path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["elastic_properties"].map(_valid_elastic_properties)].copy()

    df["__bulk_gt_range"] = df["gt"].map(
        lambda x: _parse_range(x.get("bulk_modulus")) if isinstance(x, dict) else None
    )
    df["__shear_gt_range"] = df["gt"].map(
        lambda x: _parse_range(x.get("shear_modulus")) if isinstance(x, dict) else None
    )
    df = df[df["__bulk_gt_range"].notna() & df["__shear_gt_range"].notna()].copy()
    df = df[
        df["__bulk_gt_range"].map(lambda x: x[0] >= 0.0)
        & df["__shear_gt_range"].map(lambda x: x[0] >= 0.0)
    ].copy()

    selected_pairs = _choose_range_pairs(df, random_state=random_state, n_pairs=TOTAL_RANGE_PAIRS)
    sampled_blocks: list[pd.DataFrame] = []
    used_indices: set[int] = set()
    for i, pair in enumerate(selected_pairs):
        block = _sample_for_target_pair(
            df=df,
            target_pair=pair,
            random_state=random_state + i * 1000,
            used_indices=used_indices,
        )
        sampled_blocks.append(block)

    sampled_df = pd.concat(sampled_blocks, ignore_index=True)
    sampled_df["__bulk_gt_sort"] = sampled_df["gt"].map(
        lambda x: _parse_range(x.get("bulk_modulus")) if isinstance(x, dict) else None
    )
    sampled_df["__shear_gt_sort"] = sampled_df["gt"].map(
        lambda x: _parse_range(x.get("shear_modulus")) if isinstance(x, dict) else None
    )
    sampled_df = sampled_df.sort_values(
        by=["__bulk_gt_sort", "__shear_gt_sort", "__target_bulk_range", "__target_shear_range", FIT_COLUMN, "mp_id"],
        kind="stable",
    ).reset_index(drop=True)
    sampled_df["relaxed_structures"] = sampled_df["relaxed_structures"].map(_simple_to_cif)
    sampled_df = sampled_df[OUTPUT_COLUMNS + [FIT_COLUMN, ORIGIN_COLUMN]]

    if output_path.suffix == ".parquet":
        sampled_df.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        sampled_df.to_csv(output_path, index=False)

    n_bulk = int(df["__bulk_gt_range"].dropna().nunique())
    n_shear = int(df["__shear_gt_range"].dropna().nunique())
    expected_rows = TOTAL_RANGE_PAIRS * len(FIT_COMBOS) * SAMPLES_PER_COMBO
    combo_counts = sampled_df[FIT_COLUMN].value_counts().to_dict()
    print(f"Saved: {output_path}")
    print(f"Rows: {len(sampled_df)} (expected: {expected_rows})")
    print(f"Observed gt range counts: bulk_modulus={n_bulk}, shear_modulus={n_shear}")
    print(f"Selected target range pairs ({len(selected_pairs)}): {selected_pairs}")
    print(f"fit_in_range counts: {combo_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export balanced elastic samples by gt ranges.")
    parser.add_argument(
        "--input-rl-thinking",
        type=str,
        default=str(INPUT_PARQUETS[0]),
        help="Input parquet path for rl_thinking_mix.",
    )
    parser.add_argument(
        "--input-rl-elastic",
        type=str,
        default=str(INPUT_PARQUETS[1]),
        help="Input parquet path for rl_elastic_thinking_new.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_CSV),
        help="Output parquet path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()
    run(
        input_paths=(Path(args.input_rl_thinking), Path(args.input_rl_elastic)),
        output_path=Path(args.output),
        random_state=int(args.seed),
    )


if __name__ == "__main__":
    main()
