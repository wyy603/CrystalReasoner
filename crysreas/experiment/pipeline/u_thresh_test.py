"""Diagnostic: S.U.N. / S.S.U.N. under the strict stability threshold E_hull <= 0
(PLAID++'s "stable"), next to the paper's E_hull < 0.1 (metastable).

Same per-seed uniqueness as pipeline/u_per_seed.py.  Also caches the per-row
per-seed uniqueness mask so further threshold variants cost nothing.
"""
from __future__ import annotations
import sys, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
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
TAGS = ["full", "base", "thinking", "rl", "validity_only", "energy_only",
        "crystaltextllm", "plaid_wyckoff"]
CACHE = DATA_DIR / "u_seed_masks.parquet"


def unique_mask(structs, idx, n):
    groups = defaultdict(list)
    for local, s in enumerate(structs):
        groups[s.composition.reduced_formula].append(local)
    out = np.zeros(n, dtype=bool)
    for locals_ in groups.values():
        keep = get_unique(MATCHER, [structs[i] for i in locals_])
        out[idx[[locals_[k] for k in keep]]] = True
    return out


cache = pd.read_parquet(CACHE) if CACHE.exists() else None
masks, rows = {}, []
for tag in TAGS:
    df = load_metrics_dataframe(DATA_DIR / f"out/main.{tag}.parquet")
    sun = pd.read_parquet(DATA_DIR / f"sun/out.main.{tag}.parquet")
    t0 = pd.read_parquet(DATA_DIR / f"t0/out.main.{tag}.parquet")
    n = len(df)
    b = lambda s: s.astype("boolean").fillna(False).to_numpy(dtype=bool)
    novel = b(sun.is_novel_train)
    valid = b(t0.structure_validity) & b(t0.smact_validity)
    sgc = (t0.sgc_supergroup == True).to_numpy() & valid
    e = t0.energy_above_hull.astype(float).to_numpy()
    seed = df.groupby("mp_id").cumcount().to_numpy(); ns = seed.max() + 1

    if cache is not None and tag in cache.columns:
        u = cache[tag].to_numpy(dtype=bool)
    else:
        rel = df["relaxed_structures"].to_numpy(); has = df["relaxed_structures"].notna().to_numpy()
        u = np.zeros(n, dtype=bool)
        for s in range(ns):
            r = np.flatnonzero(seed == s); idx = r[has[r]]
            u |= unique_mask(rel[idx], idx, n)
        print(f".. {tag}: uniqueness recomputed", flush=True)
    masks[tag] = u

    r = {"tag": tag}
    for name, stable in [("metastable (<0.1)", np.nan_to_num(e, nan=1e9) < 0.1),
                         ("stable (<=0)",      np.nan_to_num(e, nan=1e9) <= 0.0)]:
        for mname, m in [("Stability", stable), ("S.U.N.", stable & u & novel),
                         ("S.S.U.N.", stable & u & novel & sgc)]:
            per = np.array([m[seed == s].mean() for s in range(ns)])
            r[f"{name}|{mname}"] = (100 * per.mean(), 100 * per.std(ddof=1) / np.sqrt(ns),
                                    per.mean() * 1024)
    rows.append(r)

if cache is None:
    pd.DataFrame(masks).to_parquet(CACHE)

hdr = ["Stability", "S.U.N.", "S.S.U.N."]
print(f"\n{'':16s}" + "".join(f"{'  E_hull < 0.1':>26s}" for _ in hdr))
print(f"{'model':16s}" + "".join(f"{h:>26s}" for h in hdr))
for r in rows:
    print(f"{r['tag']:16s}" + "".join(
        f"{r['metastable (<0.1)|'+h][0]:10.2f}±{r['metastable (<0.1)|'+h][1]:.2f}"
        f" ({r['metastable (<0.1)|'+h][2]:6.1f}/seed)" for h in hdr))
print(f"\n{'model':16s}" + "".join(f"{h+'  (E_hull <= 0)':>26s}" for h in hdr))
for r in rows:
    print(f"{r['tag']:16s}" + "".join(
        f"{r['stable (<=0)|'+h][0]:10.2f}±{r['stable (<=0)|'+h][1]:.2f}"
        f" ({r['stable (<=0)|'+h][2]:6.1f}/seed)" for h in hdr))
