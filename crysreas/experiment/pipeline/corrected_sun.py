"""Corrected S.U.N. for the paper's own runs: recompute is_unique, keep stored
is_novel / is_stable, and re-form S.U.N. = unique & novel & stable."""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
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
RUNS = ["no_thinking", "thinking", "rl_no_thinking", "rl_thinking_mix",
        "thinking_only_validity", "thinking_only_energy"]
rows = []
for name in RUNS:
    p = ROOT / "checkpoints_merged" / name / "conditional+thinking.parquet"
    if not p.exists():
        print(f".. {name}: missing", flush=True); continue
    df = load_metrics_dataframe(p)
    if "is_unique" not in df.columns:
        print(f".. {name}: no metrics", flush=True); continue
    n = len(df)
    rel_idx = np.flatnonzero(df["relaxed_structures"].notna().to_numpy())
    structs = df["relaxed_structures"].to_numpy()[rel_idx]

    groups = defaultdict(list)
    for local, s in enumerate(structs):
        groups[s.composition.reduced_formula].append(local)
    uniq = np.zeros(n, dtype=bool)
    for locals_ in groups.values():
        keep = get_unique(MATCHER, [structs[i] for i in locals_])
        uniq[rel_idx[[locals_[k] for k in keep]]] = True

    novel = df["is_novel"].fillna(False).astype(bool).to_numpy()
    stable = df["is_stable"].fillna(False).astype(bool).to_numpy()
    stored_u = df["is_unique"].fillna(False).astype(bool).to_numpy()
    stored_sun = df["stable_unique_novel"].fillna(False).astype(bool).to_numpy()
    rows.append({
        "run": name, "n": n,
        "uniq_stored": f"{100*stored_u.sum()/n:.2f}%",
        "uniq_fixed":  f"{100*uniq.sum()/n:.2f}%",
        "SUN_stored":  f"{100*stored_sun.sum()/n:.2f}%",
        "SUN_fixed":   f"{100*(uniq & novel & stable).sum()/n:.2f}%",
    })
    print(rows[-1], flush=True)

if rows:
    out = pd.DataFrame(rows)
    out.to_parquet(DATA_DIR / "corrected_sun.parquet")
    pd.set_option("display.width", 200)
    print(); print(out.to_string(index=False))
