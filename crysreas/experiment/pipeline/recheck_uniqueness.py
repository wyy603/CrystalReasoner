"""Recompute uniqueness from the *stored* relaxed structures of the paper's own runs.

Groups by reduced formula and applies MatterGen's greedy first-wins matcher directly
(the same logic OrderedDatasetUniquenessComputer uses internally), so the comparison is
same-structures / same-matcher and isolates the metric computation from generation.
"""
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "submodules" / "mattergen"))
from mattergen.evaluation.utils.dataset_matcher import get_unique
from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher
from crysreas.metric_process.io import load_metrics_dataframe

MATCHER = DefaultDisorderedStructureMatcher()
RUNS = ["no_thinking", "thinking", "rl_thinking_mix",
        "thinking_only_validity", "thinking_only_energy", "rl_no_thinking"]

rows = []
for name in RUNS:
    p = ROOT / "checkpoints_merged" / name / "conditional+thinking.parquet"
    if not p.exists():
        print(f".. {name}: no parquet", flush=True); continue
    df = load_metrics_dataframe(p)
    if "relaxed_structures" not in df.columns or "is_unique" not in df.columns:
        print(f".. {name}: missing columns", flush=True); continue
    rel = df["relaxed_structures"].notna()
    structs = df.loc[rel, "relaxed_structures"].tolist()

    groups = defaultdict(list)
    for s in structs:
        groups[s.composition.reduced_formula].append(s)
    n_unique = sum(len(get_unique(MATCHER, v)) for v in groups.values())

    stored = int(df["is_unique"].fillna(False).astype(bool).sum())
    rows.append({
        "run": name, "n": len(df), "relaxed": int(rel.sum()), "formula_groups": len(groups),
        "stored_unique": stored, "stored_%": f"{100*stored/len(df):.2f}%",
        "recomputed_unique": n_unique, "recomputed_%": f"{100*n_unique/len(df):.2f}%",
    })
    print(rows[-1], flush=True)

if rows:
    out = pd.DataFrame(rows)
    out.to_parquet(DATA_DIR / "uniqueness_recheck.parquet")
    pd.set_option("display.width", 220)
    print(); print(out.to_string(index=False))
