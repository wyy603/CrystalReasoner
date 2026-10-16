"""Aggregate the rerun parquets with the repo's own statistics and diff them
against the numbers the paper reports.

Uses crysreas.experiment.make_graph.compute verbatim, so the aggregation and the
error bars are identical to the ones that produced the committed result CSVs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

sys.path.insert(0, str(ROOT))
from crysreas.experiment.make_graph import compute
from crysreas.metric_process.io import load_metrics_dataframe

OUT = DATA_DIR / "out"

# Paper table values, keyed by rerun tag. Source: the paper draft; the repo CSVs
# under crysreas/experiment/ are listed alongside where they differ.
PAPER = {
    # Synced 2026-09-24 to papers/CrystalReasoner_paper/iclr2027_conference.tex
    # (Tables 1-3 + Table 5). The previous dict held the PRE-CORRECTION draft
    # (uniqueness 82.49/87.23%, S.U.N. 1.23/1.70%) and is no longer the paper.
    "main.base":          {"structure_validity": "83.84%", "smact_validity": "90.45%",
                           "composition_consistency": "90.64%", "spacegroup_consistency": "40.77%",
                           "is_unique": "35.63%", "Energy": "0.59", "stable_unique_novel": "0.49%"},
    "main.thinking":      {"structure_validity": "91.30%", "smact_validity": "91.78%",
                           "composition_consistency": "97.25%", "spacegroup_consistency": "47.76%",
                           "is_unique": "38.91%", "Energy": "0.52", "stable_unique_novel": "0.57%"},
    "main.rl":            {"structure_validity": "89.85%", "smact_validity": "91.10%",
                           "composition_consistency": "92.75%", "spacegroup_consistency": "41.44%",
                           "is_unique": "35.66%", "Energy": "0.53", "stable_unique_novel": "0.54%"},
    "main.full":          {"structure_validity": "94.92%", "smact_validity": "91.78%",
                           "composition_consistency": "97.32%", "spacegroup_consistency": "50.21%",
                           "is_unique": "36.34%", "Energy": "0.45", "stable_unique_novel": "0.56%",
                           "is_stable": "40.95%"},
    "main.validity_only": {"structure_validity": "95.81%", "is_stable": "38.30%",
                           "is_unique": "41.64%", "stable_unique_novel": "0.54%"},
    "main.energy_only":   {"structure_validity": "95.31%", "is_stable": "40.66%",
                           "is_unique": "36.29%", "stable_unique_novel": "0.43%"},
    "main.crystaltextllm":  {"structure_validity": "90.01%", "smact_validity": "91.59%",
                             "composition_consistency": "95.39%", "spacegroup_consistency": "22.27%",
                             "is_unique": "47.40%", "Energy": "0.61", "stable_unique_novel": "0.38%",
                             "is_stable": "24.57%"},
    "main.plaid_wyckoff":   {"structure_validity": "89.06%", "smact_validity": "91.59%",
                             "composition_consistency": "96.95%", "spacegroup_consistency": "41.56%",
                             "is_unique": "40.70%", "Energy": "0.57", "stable_unique_novel": "0.50%",
                             "is_stable": "32.03%"},
    "elastic.full":        {"structure_validity": "95.29%", "smact_validity": "99.85%",
                            "elastic_reward_all": "56.85%"},
    "elastic.specialist":  {"structure_validity": "94.20%", "smact_validity": "99.82%",
                            "elastic_reward_all": "61.22%"},
}

BOOL_METRICS = ["structure_validity", "smact_validity", "composition_consistency",
                "spacegroup_consistency", "is_stable", "is_novel", "is_unique",
                "stable_unique_novel"]


def stat(df: pd.DataFrame, metric: str) -> str:
    if metric == "Energy":
        if "energy_above_hull" not in df.columns:
            return "N/A"
        d = df[(~df["energy_above_hull"].isna()) & (df["energy_above_hull"] >= -1)
               & (df["energy_above_hull"] <= 10)]
        return compute(d, "energy_above_hull", mode="float")
    if metric == "elastic_reward_all":
        return compute(df, metric, mode="2")
    if metric == "cte_reward_all":
        return compute(df, metric, mode="1")
    return compute(df, metric)


def main() -> None:
    rows = []
    for path in sorted(OUT.glob("*.parquet")):
        tag = path.stem
        if tag == "smoke_metrics":
            continue
        try:
            df = load_metrics_dataframe(path)
        except Exception as e:
            print(f"!! {tag}: cannot load ({e})")
            continue
        paper = PAPER.get(tag, {})
        metrics = list(paper) or (BOOL_METRICS + ["Energy"])
        for m in metrics:
            got = stat(df, m)
            want = paper.get(m, "")
            delta = ""
            if want and got != "N/A":
                try:
                    g = float(got.split("±")[0].strip().rstrip("%"))
                    w = float(want.rstrip("%"))
                    delta = f"{g - w:+.2f}"
                except ValueError:
                    pass
            rows.append({"run": tag, "n_samples": len(df), "metric": m,
                         "rerun": got, "paper": want, "delta": delta})
    if not rows:
        print("no metric parquets found under data/out/")
        return
    out = pd.DataFrame(rows)
    out_path = DATA_DIR / "comparison_to_paper.parquet"
    out.to_parquet(out_path)
    print(out.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
