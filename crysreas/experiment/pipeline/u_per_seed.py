"""Uniqueness under the per-seed definition, and the metrics that contain it.

Protocol: data/out/  -- the 1,024 prompts of assets/MP/split_generation.json,
10 samples per prompt.  Sample j of every prompt is read as seed j, so each
seed is a complete 1,024-structure evaluation.

Uniqueness is computed inside one seed: a duplicate can then only arise between
two *different* requested compositions, i.e. the model returned the same
structure whatever it was asked for.  Under the pooled definition every
comparison instead falls between the samples of a single composition, because
the matcher only compares structures sharing a reduced formula.

Everything else keeps the definitions fixed in report.md:
  stable   E_hull < 0.1 eV/atom
  novel    absent from the 24,231-entry SFT training split
  sgc      generated structure carries at least the requested symmetry
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


def unique_mask(structs: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    """MatterGen uniqueness over the given rows: greedy first-wins per reduced formula."""
    groups = defaultdict(list)
    for local, s in enumerate(structs):
        groups[s.composition.reduced_formula].append(local)
    out = np.zeros(n, dtype=bool)
    for locals_ in groups.values():
        keep = get_unique(MATCHER, [structs[i] for i in locals_])
        out[idx[[locals_[k] for k in keep]]] = True
    return out


def run(tag: str) -> dict:
    df = load_metrics_dataframe(DATA_DIR / f"out/main.{tag}.parquet")
    sun = pd.read_parquet(DATA_DIR / f"sun/out.main.{tag}.parquet")
    t0 = pd.read_parquet(DATA_DIR / f"t0/out.main.{tag}.parquet")
    assert (df.mp_id.values == sun.mp_id.values).all(), tag
    assert (t0.mp_id.values == sun.mp_id.values).all(), tag
    n = len(df)

    b = lambda s: s.astype("boolean").fillna(False).to_numpy(dtype=bool)
    stable = b(sun.is_stable); novel_t = b(sun.is_novel_train); novel_mp = b(sun.is_novel_mp)
    valid = b(t0.structure_validity) & b(t0.smact_validity)
    sgc = (t0.sgc_supergroup == True).to_numpy() & valid   # same gate as the repo metric

    rel = df["relaxed_structures"].to_numpy()
    has_rel = df["relaxed_structures"].notna().to_numpy()
    seed = df.groupby("mp_id").cumcount().to_numpy()
    n_seeds = int(seed.max()) + 1

    # ---- pooled definition, for reference: already stored by the original pipeline ----
    u_pool = b(sun.is_unique)

    # ---- per-seed definition ----
    u_seed = np.zeros(n, dtype=bool)
    per = []
    for s in range(n_seeds):
        rows = np.flatnonzero(seed == s)
        idx = rows[has_rel[rows]]
        m = unique_mask(rel[idx], idx, n)
        u_seed |= m
        k = len(rows)
        per.append(dict(
            seed=s, n=k,
            uniq=m[rows].mean(),
            sun=(stable & m & novel_t)[rows].mean(),
            ssun=(stable & m & novel_t & sgc)[rows].mean(),
            sun_mp=(stable & m & novel_mp)[rows].mean(),
            uniq_pool=u_pool[rows].mean(),
            sun_pool=(stable & u_pool & novel_t)[rows].mean(),
            ssun_pool=(stable & u_pool & novel_t & sgc)[rows].mean(),
        ))
    p = pd.DataFrame(per)
    out = {"tag": tag, "n_seeds": n_seeds, "n_per_seed": int(p.n.iloc[0])}
    for c in ("uniq", "sun", "ssun", "sun_mp", "uniq_pool", "sun_pool", "ssun_pool"):
        out[c] = 100 * p[c].mean()
        out[c + "_se"] = 100 * p[c].std(ddof=1) / np.sqrt(len(p))
    return out


if __name__ == "__main__":
    rows = []
    for tag in TAGS:
        rows.append(run(tag))
        r = rows[-1]
        print(f"{tag:16s} uniq {r['uniq']:6.2f}±{r['uniq_se']:.2f}  "
              f"S.U.N. {r['sun']:6.2f}±{r['sun_se']:.2f}  "
              f"S.S.U.N. {r['ssun']:6.2f}±{r['ssun_se']:.2f}  "
              f"| pooled: uniq {r['uniq_pool']:6.2f}  S.U.N. {r['sun_pool']:5.2f}  "
              f"S.S.U.N. {r['ssun_pool']:5.2f}  | S.U.N.(MP+Alex) {r['sun_mp']:.2f}±{r['sun_mp_se']:.2f}",
              flush=True)
    pd.DataFrame(rows).to_parquet(DATA_DIR / "u_per_seed.parquet")
    print("\nwrote data/u_per_seed.parquet")
