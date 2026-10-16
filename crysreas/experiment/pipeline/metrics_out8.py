"""Table 1 columns for the 8-decimal baselines, on the CrysReas protocol.

Same 1,024 prompts as data/out/; the stored runs have 16 samples per prompt,
so the first 10 are used, one per seed, to match the 10 seeds of every other
model.  Thresholds follow PLAID++: Stability and S.U.N. at E_hull <= 0,
S.S.U.N. at E_hull <= 0.1.
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
M = DefaultDisorderedStructureMatcher()
NSEED = 10


def unique_mask(structs, idx, n):
    g = defaultdict(list)
    for i, s in enumerate(structs):
        g[s.composition.reduced_formula].append(i)
    out = np.zeros(n, dtype=bool)
    for loc in g.values():
        for k in get_unique(M, [structs[i] for i in loc]):
            out[idx[loc[k]]] = True
    return out


def run(tag):
    df = load_metrics_dataframe(DATA_DIR / f"out8/main.{tag}.parquet")
    sun = pd.read_parquet(DATA_DIR / f"sun/out8.main.{tag}.parquet")
    t0 = pd.read_parquet(DATA_DIR / f"t0/out8.main.{tag}.parquet")
    assert (df.mp_id.values == sun.mp_id.values).all() and (df.mp_id.values == t0.mp_id.values).all()
    seed = df.groupby("mp_id").cumcount().to_numpy()
    keep = seed < NSEED                      # first 10 samples of each prompt
    n = len(df)
    b = lambda s: s.astype("boolean").fillna(False).to_numpy(dtype=bool)
    nov = b(sun.is_novel_train)
    valid = b(t0.structure_validity) & b(t0.smact_validity)
    sgc = (t0.sgc_supergroup == True).to_numpy() & valid
    e = np.nan_to_num(t0.energy_above_hull.astype(float).to_numpy(), nan=1e9)

    u = np.zeros(n, dtype=bool)
    rel = df["relaxed_structures"].to_numpy(); has = df["relaxed_structures"].notna().to_numpy()
    for s in range(NSEED):
        r = np.flatnonzero(seed == s); idx = r[has[r]]
        u |= unique_mask(rel[idx], idx, n)

    cols = {"StructVal": b(t0.structure_validity), "ChemVal": b(t0.smact_validity),
            "CompCons": b(t0.composition_consistency), "SGC": sgc,
            "Stability": e <= 0.0, "Uniqueness": u,
            "SUN": (e <= 0.0) & u & nov, "SSUN": (e <= 0.1) & u & nov & sgc}
    out = {}
    for k, v in cols.items():
        per = np.array([v[keep & (seed == s)].mean() for s in range(NSEED)])
        out[k] = (100 * per.mean(), 100 * per.std(ddof=1) / np.sqrt(NSEED))
    ev = t0.energy_above_hull.astype(float).to_numpy()
    per = np.array([np.nanmean(ev[keep & (seed == s)]) for s in range(NSEED)])
    out["Energy"] = (per.mean(), per.std(ddof=1) / np.sqrt(NSEED))
    return out


if __name__ == "__main__":
    ORDER = ["StructVal", "ChemVal", "CompCons", "SGC", "Stability", "Uniqueness",
             "Energy", "SUN", "SSUN"]
    res = {t: run(t) for t in ["crystaltextllm_8", "plaid_wyckoff_8"]}
    OLD = {"crystaltextllm": dict(StructVal=89.61, ChemVal=89.98, CompCons=95.30, SGC=31.62,
                                  Stability=1.91, Uniqueness=80.71, Energy=0.572, SUN=1.73, SSUN=16.00),
           "plaid_wyckoff": dict(StructVal=88.33, ChemVal=89.38, CompCons=97.10, SGC=50.78,
                                 Stability=3.13, Uniqueness=79.61, Energy=0.530, SUN=2.86, SSUN=23.94)}
    print(f"{'metric':12s}{'CTLLM 2dp':>12s}{'CTLLM 8dp':>18s}{'PLAID 3dp':>12s}{'PLAID 8dp':>18s}")
    for m in ORDER:
        f = (lambda v: f"{v:.3f}") if m == "Energy" else (lambda v: f"{v:.2f}")
        a, b_ = res["crystaltextllm_8"][m], res["plaid_wyckoff_8"][m]
        print(f"{m:12s}{f(OLD['crystaltextllm'][m]):>12s}{f(a[0])+'±'+f'{a[1]:.2f}':>18s}"
              f"{f(OLD['plaid_wyckoff'][m]):>12s}{f(b_[0])+'±'+f'{b_[1]:.2f}':>18s}")
