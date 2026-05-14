from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro


@dataclass
class Args:
    csv_path: Path = Path("crysreas/experiment/ablation_thinking/ablation_results.csv")
    output_path: Path = Path("crysreas/experiment/ablation_thinking/ablation_spacegroup_ratio.csv")


def to_bool(series: pd.Series) -> pd.Series:
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


def main(args: Args) -> None:
    df = pd.read_csv(args.csv_path)
    if "variant" not in df.columns or "spacegroup_consistency" not in df.columns:
        raise ValueError("CSV must contain 'variant' and 'spacegroup_consistency' columns")

    df = df.copy()
    df["spacegroup_consistency_bool"] = to_bool(df["spacegroup_consistency"])
    valid_df = df[df["spacegroup_consistency_bool"].notna()].copy()

    stats = (
        valid_df.groupby("variant", as_index=False)["spacegroup_consistency_bool"]
        .agg(total="count", true_count="sum")
        .sort_values("variant")
    )
    stats["ratio"] = stats["true_count"] / stats["total"]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.output_path, index=False)

    print("Spacegroup consistency ratio by variant:")
    print(stats.to_string(index=False))
    print(f"\nWrote analysis to {args.output_path}")


if __name__ == "__main__":
    tyro.cli(main)
