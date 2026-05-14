"""
Backend for EXPERIMENT §1.6: test whether thinking traces reduce CIF-token information.

This experiment compares the same CIF suffix under the same thinking model:
- with the original thinking trace kept before the CIF block
- with the thinking trace removed and only the CIF suffix retained
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crysreas.experiment.assets.plot_helpers import mean_ci_95

from .common import ComparePaths, count_atoms, ensure_thinking_dirs, split_thinking_and_tail
from .part_1_4_backend import _locate_cif_span, _load_model_and_tokenizer, _score_cif_information

DATA_1_6_PER_ROW = "thinking_exp_1_6_average_of_information_per_row.parquet"
DATA_1_6_CURVE = "thinking_exp_1_6_average_of_information_vs_atoms_curve.parquet"


def run(paths: ComparePaths, *, batch_size: int = 2, max_rows: int | None = None) -> None:
    ensure_thinking_dirs(data_dir=paths.data_dir)
    cols = ["mp_id", "responses", "simple_structure"]
    df = pd.read_parquet(paths.thinking_parquet, columns=cols)
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    keep_idx: list[int] = []
    text_with_trace: list[str] = []
    span_with_trace: list[tuple[int, int]] = []
    text_without_trace: list[str] = []
    span_without_trace: list[tuple[int, int]] = []

    for idx, response in enumerate(df["responses"].tolist()):
        thinking_text, cif_tail = split_thinking_and_tail(response)
        if not isinstance(response, str) or not isinstance(cif_tail, str):
            continue
        if not thinking_text.strip() or not cif_tail.strip():
            continue

        full_span = _locate_cif_span(response)
        tail_span = _locate_cif_span(cif_tail)
        if full_span is None or tail_span is None:
            continue

        keep_idx.append(idx)
        text_with_trace.append(str(response))
        span_with_trace.append(full_span)
        text_without_trace.append(cif_tail)
        span_without_trace.append(tail_span)

    if not keep_idx:
        raise ValueError("No rows with both thinking trace and CIF block were found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_model_and_tokenizer(paths.thinking_parquet.parent, device)

    info_sum_with, token_count_with = _score_cif_information(
        model,
        tokenizer,
        text_with_trace,
        span_with_trace,
        device,
        batch_size=batch_size,
    )
    info_sum_without, token_count_without = _score_cif_information(
        model,
        tokenizer,
        text_without_trace,
        span_without_trace,
        device,
        batch_size=batch_size,
    )

    info_avg_with = np.divide(
        info_sum_with,
        token_count_with,
        out=np.full_like(info_sum_with, np.nan, dtype=float),
        where=token_count_with > 0,
    )
    info_avg_without = np.divide(
        info_sum_without,
        token_count_without,
        out=np.full_like(info_sum_without, np.nan, dtype=float),
        where=token_count_without > 0,
    )

    base = pd.DataFrame(
        {
            "mp_id": df.iloc[keep_idx]["mp_id"].astype(str).to_numpy(),
            "n_atoms": [count_atoms(x) for x in df.iloc[keep_idx]["simple_structure"].tolist()],
            "average_of_information_with_trace": info_avg_with.astype(float),
            "average_of_information_without_trace": info_avg_without.astype(float),
            "token_num_with_trace": token_count_with.astype(int),
            "token_num_without_trace": token_count_without.astype(int),
        }
    )
    per_row = base[base["n_atoms"].notna()].copy()
    per_row["n_atoms"] = per_row["n_atoms"].astype(int)

    curve = pd.concat(
        [
            per_row.loc[:, ["mp_id", "n_atoms", "average_of_information_with_trace"]]
            .rename(columns={"average_of_information_with_trace": "average_of_information"})
            .assign(condition="with_trace"),
            per_row.loc[:, ["mp_id", "n_atoms", "average_of_information_without_trace"]]
            .rename(columns={"average_of_information_without_trace": "average_of_information"})
            .assign(condition="without_trace"),
        ],
        ignore_index=True,
    )
    curve = curve[curve["average_of_information"].notna()].copy()
    curve_rows: list[dict[str, object]] = []
    for (condition, n_atoms), group in curve.groupby(["condition", "n_atoms"], observed=True):
        stats = mean_ci_95(group["average_of_information"])
        curve_rows.append(
            {
                "condition": condition,
                "n_atoms": int(n_atoms),
                "n_samples": stats["n_samples"],
                "average_of_information": stats["mean"],
                "average_of_information_sem": stats["sem"],
                "average_of_information_ci_low": stats["ci_low"],
                "average_of_information_ci_high": stats["ci_high"],
            }
        )
    curve = pd.DataFrame(curve_rows).sort_values(["condition", "n_atoms"]).reset_index(drop=True)

    p_row = paths.data_dir / DATA_1_6_PER_ROW
    p_curve = paths.data_dir / DATA_1_6_CURVE
    per_row.to_parquet(p_row, index=False)
    curve.to_parquet(p_curve, index=False)
    print(f"1.6 backend wrote {p_row} and {p_curve}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.6 backend (data only).")
    parser.add_argument("--thinking", type=str, default="checkpoints_merged/thinking/conditional+thinking.parquet")
    parser.add_argument("--data-dir", type=str, default=str(ComparePaths().data_dir))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    run(
        ComparePaths(
            thinking_parquet=Path(args.thinking),
            data_dir=Path(args.data_dir),
        ),
        batch_size=int(args.batch_size),
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
