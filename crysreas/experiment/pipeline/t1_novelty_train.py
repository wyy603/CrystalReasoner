"""Tier-1: recompute S.U.N. with novelty = "not in the SFT training split".

Why change the criterion. In the draft, `novel` means "absent from MP+Alexandria",
while every prompt fixes a *known* MP formula. A model that gets the structure
right therefore reproduces a known phase and scores novel=False; a model that
gets it wrong scores novel=True but is almost never stable. Measured on the
1024x10 rerun: P(stable | unique & novel) = 1.0-2.5% versus
P(stable | not novel) = 49-66%. The metric penalises accuracy.

Replacing the reference with the training split asks a different, answerable
question: did the model produce something it was not trained on? Caveat, to be
stated wherever the number is used: train and test ids are disjoint and only
21.5% of test materials even share a reduced formula with the training split,
so novelty-vs-train can only fail for that minority. `sun_train` is therefore
close to "stable AND unique" in practice.

Stability is NOT recomputed: `is_stable` keeps MatterGen's MP2020 convex hull.
Uniqueness is NOT recomputed: it is a property of the generated set. Only the
novelty term changes, so the three factors stay comparable to the paper.
"""
from __future__ import annotations

import json
import pickle
import shelve
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(ROOT))

CACHE = DATA_DIR / "train_formula_index.pkl"
_INDEX: dict[str, list] | None = None      # formula -> [Structure], inherited by fork
_MATCHER = None


def build_index() -> dict[str, list]:
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    ids = json.load(open(ROOT / "assets/MP/split_cdvae.json"))["train"]
    db = shelve.open(str(ROOT / "assets/MP/MP_shelve"), flag="r")
    idx: dict[str, list] = {}
    for m in ids:
        try:
            s = db[m]["structure"]
            idx.setdefault(s.composition.reduced_formula, []).append(s)
        except Exception:
            continue
    db.close()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(idx, f)
    print(f"[novelty] train index: {sum(len(v) for v in idx.values())} structures, "
          f"{len(idx)} formulas -> {CACHE}", flush=True)
    return idx


def _novel(rel_str) -> bool | None:
    """True when no training structure of the same reduced formula matches."""
    global _MATCHER
    from crysreas.utils.crystal import SimpleCrystal
    if rel_str is None or (isinstance(rel_str, float) and np.isnan(rel_str)):
        return None
    try:
        s = SimpleCrystal.from_simple_no_sym(str(rel_str)).structure
    except Exception:
        return None
    cands = _INDEX.get(s.composition.reduced_formula, ())
    if not cands:
        return True                       # formula absent from training split
    if _MATCHER is None:
        from mattergen.evaluation.utils.structure_matcher import (
            DefaultDisorderedStructureMatcher,
        )
        _MATCHER = DefaultDisorderedStructureMatcher()
    for c in cands:
        try:
            if _MATCHER.fit(s, c):
                return False
        except Exception:
            continue
    return True


def process(path: Path, out_path: Path, workers: int = 8) -> None:
    df = pd.read_parquet(path)
    if "relaxed_structures" not in df.columns:
        print(f"[skip] {path.name}: no relaxed_structures"); return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        novel = list(ex.map(_novel, df["relaxed_structures"].tolist(), chunksize=32))

    b = lambda c: df[c].astype("boolean").fillna(False) if c in df.columns else False  # noqa: E731
    res = pd.DataFrame({
        "mp_id": df["mp_id"],
        "is_stable": b("is_stable"),
        "is_unique": b("is_unique"),
        "is_novel_mp": b("is_novel"),
        "sun_mp": b("stable_unique_novel"),
        "is_novel_train": pd.Series(novel, index=df.index).astype("boolean").fillna(False),
    })
    res["sun_train"] = res.is_stable & res.is_unique & res.is_novel_train
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_path, index=False)
    print(f"{path.name:32s} n={len(res):6d}  novel_mp={100*res.is_novel_mp.mean():5.2f}%  "
          f"novel_train={100*res.is_novel_train.mean():5.2f}%  "
          f"SUN_mp={100*res.sun_mp.mean():5.3f}%  SUN_train={100*res.sun_train.mean():5.3f}%",
          flush=True)


def main() -> None:
    global _INDEX
    _INDEX = build_index()
    src_name = sys.argv[1] if len(sys.argv) > 1 else "denovo"
    src_dir = DATA_DIR / src_name
    dst = DATA_DIR / "sun"
    for p in sorted(src_dir.glob("main.*.parquet")):
        out = dst / f"{src_name}.{p.stem}.parquet"
        if out.exists():
            print(f"[skip] {out.name}"); continue
        process(p, out)


if __name__ == "__main__":
    main()
