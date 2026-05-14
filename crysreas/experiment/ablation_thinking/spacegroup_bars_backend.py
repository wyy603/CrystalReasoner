"""
Backend for ablation thinking: aggregate spacegroup consistency ratio by variant,
then export a compact table for grouped bar charts (remove vs replace × part1–3)
plus an original (no remove / no replace) baseline from conditional+thinking parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crysreas.experiment.assets.plot_helpers import wilson_ci_95

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_BAR = "ablation_thinking_spacegroup_part_bars.parquet"
OUT_RATIO_CSV = "ablation_spacegroup_ratio.csv"

DEFAULT_INPUT = Path(__file__).resolve().parent / "ablation_results.parquet"
DEFAULT_BASELINE_PARQUET = Path("checkpoints_merged/thinking/conditional+thinking.parquet")


def _to_bool(series: pd.Series) -> pd.Series:
    def convert(x: object) -> bool | None:
        if pd.isna(x):
            return None
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        if s in {"true", "1"}:
            return True
        if s in {"false", "0"}:
            return False
        return None

    return series.apply(convert)


def _aggregate_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    if "variant" not in df.columns or "spacegroup_consistency" not in df.columns:
        raise ValueError("Raw input must contain 'variant' and 'spacegroup_consistency' columns")
    work = df.copy()
    work["_sg_ok"] = _to_bool(work["spacegroup_consistency"])
    valid = work[work["_sg_ok"].notna()].copy()
    stats = (
        valid.groupby("variant", as_index=False)["_sg_ok"]
        .agg(n_samples="count", n_true="sum")
        .sort_values("variant")
    )
    stats["ratio"] = stats["n_true"] / stats["n_samples"]
    return stats.rename(columns={"n_true": "true_count"})


def _aggregate_baseline_from_conditional_thinking(
    *,
    baseline_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
) -> dict[str, object]:
    if "mp_id" not in ablation_df.columns:
        raise ValueError("Ablation input must contain 'mp_id' to build baseline")
    if "mp_id" not in baseline_df.columns or "spacegroup_consistency" not in baseline_df.columns:
        raise ValueError("Baseline parquet must contain 'mp_id' and 'spacegroup_consistency' columns")

    target_mp_ids = set(ablation_df["mp_id"].astype(str).unique().tolist())
    work = baseline_df.copy()
    work["mp_id"] = work["mp_id"].astype(str)
    work = work[work["mp_id"].isin(target_mp_ids)].copy()
    work["_sg_ok"] = _to_bool(work["spacegroup_consistency"])
    valid = work[work["_sg_ok"].notna()].copy()

    n_samples = int(len(valid))
    true_count = int(valid["_sg_ok"].sum())
    ratio = float(true_count / n_samples) if n_samples > 0 else float("nan")
    return {
        "variant": "original_no_ablation",
        "n_samples": n_samples,
        "true_count": true_count,
        "ratio": ratio,
    }


def _load_summary(path: Path) -> pd.DataFrame:
    """Load per-variant table from analyze_ablation.py output (variant, total, true_count, ratio)."""
    df = pd.read_csv(path)
    need = {"variant", "ratio"}
    if not need.issubset(df.columns):
        raise ValueError(f"Summary CSV at {path} must contain columns {sorted(need)}")
    out = df[["variant", "ratio"]].copy()
    if "total" in df.columns:
        out["n_samples"] = df["total"].astype(int)
    else:
        out["n_samples"] = pd.NA
    if "true_count" in df.columns:
        out["true_count"] = df["true_count"].astype(int)
    return out


def _parse_variant(variant: str) -> tuple[str, str]:
    s = str(variant).strip()
    op, _, rest = s.partition("_")
    if not op or not rest:
        raise ValueError(f"Unexpected variant label: {variant!r}")
    return op, rest


def _build_bar_rows(stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, r in stats.iterrows():
        operation, target_section = _parse_variant(str(r["variant"]))
        n_s = r["n_samples"] if "n_samples" in r.index else pd.NA
        n_t = r["true_count"] if "true_count" in r.index else pd.NA
        rows.append(
            {
                "operation": operation,
                "target_section": target_section,
                "ratio": float(r["ratio"]),
                "n_samples": int(n_s) if pd.notna(n_s) else None,
                "true_count": int(n_t) if pd.notna(n_t) else None,
                "ci_low": (
                    wilson_ci_95(int(n_t), int(n_s))["ci_low"]
                    if pd.notna(n_s) and pd.notna(n_t)
                    else float("nan")
                ),
                "ci_high": (
                    wilson_ci_95(int(n_t), int(n_s))["ci_high"]
                    if pd.notna(n_s) and pd.notna(n_t)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def run(
    *,
    input_path: Path,
    data_dir: Path,
    use_summary_csv: bool = False,
    baseline_parquet: Path = DEFAULT_BASELINE_PARQUET,
    ratio_csv_path: Path | None = None,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    if use_summary_csv:
        stats = _load_summary(input_path)
    else:
        if input_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
        stats = _aggregate_from_raw(df)
        baseline_row = _aggregate_baseline_from_conditional_thinking(
            baseline_df=pd.read_parquet(baseline_parquet),
            ablation_df=df,
        )
        stats = pd.concat([stats, pd.DataFrame([baseline_row])], ignore_index=True)

    if "variant" not in stats.columns:
        raise ValueError("Expected 'variant' column after aggregation")

    bar_stats = stats[stats["variant"] != "original_no_ablation"].copy()
    bar_df = _build_bar_rows(bar_stats)
    baseline_row = stats[stats["variant"] == "original_no_ablation"].head(1)
    if len(baseline_row) > 0:
        baseline_n = int(baseline_row["n_samples"].iloc[0])
        baseline_true = int(baseline_row["true_count"].iloc[0])
        baseline_stats = wilson_ci_95(baseline_true, baseline_n)
        bar_df["baseline_ratio"] = baseline_stats["ratio"]
        bar_df["baseline_n_samples"] = baseline_stats["n_samples"]
        bar_df["baseline_true_count"] = baseline_stats["true_count"]
        bar_df["baseline_ci_low"] = baseline_stats["ci_low"]
        bar_df["baseline_ci_high"] = baseline_stats["ci_high"]

    out_path = data_dir / OUT_BAR
    bar_df.to_parquet(out_path, index=False)
    print(f"ablation spacegroup bars backend wrote {out_path}")

    ratio_out = ratio_csv_path if ratio_csv_path is not None else (Path(__file__).resolve().parent / OUT_RATIO_CSV)
    ratio_df = stats.copy()
    ratio_df = ratio_df.rename(columns={"n_samples": "total"})
    ratio_df = ratio_df[["variant", "total", "true_count", "ratio"]]
    ratio_df.to_csv(ratio_out, index=False)
    print(f"ablation spacegroup ratio wrote {ratio_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ablation thinking spacegroup bar chart data.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Raw ablation_results parquet/csv, or summary CSV from analyze_ablation.py",
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument(
        "--baseline-parquet",
        type=str,
        default=str(DEFAULT_BASELINE_PARQUET),
        help="Source conditional+thinking parquet for no-ablation baseline ratio.",
    )
    parser.add_argument(
        "--ratio-csv-path",
        type=str,
        default=str(Path(__file__).resolve().parent / OUT_RATIO_CSV),
        help="CSV output path for variant + baseline ratios.",
    )
    parser.add_argument(
        "--summary-csv",
        action="store_true",
        help="Treat --input as analyze_ablation.py output (variant,total,true_count,ratio).",
    )
    args = parser.parse_args()
    run(
        input_path=Path(args.input),
        data_dir=Path(args.data_dir),
        use_summary_csv=bool(args.summary_csv),
        baseline_parquet=Path(args.baseline_parquet),
        ratio_csv_path=Path(args.ratio_csv_path),
    )


if __name__ == "__main__":
    main()
