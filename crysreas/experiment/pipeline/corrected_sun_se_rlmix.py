"""Recompute is_unique per-row for the paper's own 16,384-structure evaluations
and report mean +- clustered s.e. using the exact formula of
crysreas/experiment/make_graph.py::compute (group by mp_id)."""
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

def compute(df, key, group_key="mp_id"):
    t = df[[group_key, key]].copy()
    t[key] = (t[key] == True).astype(float)
    m = t.groupby(group_key)[key].mean().mean()
    gs = t.groupby(group_key)[key].agg(['var', 'count'])
    gs['var'] = gs['var'].fillna(0)
    M = len(gs)
    sigma = np.sqrt((gs['var'] / gs['count']).sum() / M**2)
    return f"{m*100:.2f}% ± {sigma*100:.2f}%"

MATCHER = DefaultDisorderedStructureMatcher()
RUNS = ["rl_no_thinking_mix"]
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
    out = df[["mp_id"]].copy()
    out["is_unique_stored"] = df["is_unique"].fillna(False).astype(bool).to_numpy()
    out["sun_stored"] = df["stable_unique_novel"].fillna(False).astype(bool).to_numpy()
    out["is_unique_fixed"] = uniq
    out["sun_fixed"] = uniq & novel & stable
    out["is_stable"] = stable
    out["is_novel"] = novel
    out.to_parquet(DATA_DIR / f"fixed_rows_{name}.parquet")
    rows.append({
        "run": name, "n": n,
        "uniq_stored": compute(out, "is_unique_stored"),
        "uniq_fixed":  compute(out, "is_unique_fixed"),
        "SUN_stored":  compute(out, "sun_stored"),
        "SUN_fixed":   compute(out, "sun_fixed"),
        "stable":      compute(out, "is_stable"),
        "novel_fixed": compute(out, "is_novel"),
    })
    print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_parquet(DATA_DIR / "corrected_sun_se_rlmix.parquet")
pd.set_option("display.width", 250)
print(); print(out.to_string(index=False))
