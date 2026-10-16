"""Recompute the S.U.N. decomposition with stability = E_hull <= 0 (on/below hull).

The repo's `is_stable` is `E_hull < 0.1` (MatterGen's DEFAULT_STABILITY_THRESHOLD,
`crysreas/mlip/energy.py:53,69`). This script replaces it with the strictest
possible reading -- at or below the convex hull -- and re-derives every S.U.N.
quantity from it. Uniqueness and both novelty flags are reused unchanged, so the
only thing that moves is the stability factor.
"""
from __future__ import annotations

import glob
import os
import sys
import warnings
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
from crysreas.experiment.make_graph import compute  # noqa: E402

LAB = {"base": "Base", "thinking": "Thinking", "rl": "RL", "full": "CrysReas",
       "validity_only": "Validity Only", "energy_only": "Energy Only",
       "crystaltextllm": "CrystalTextLLM", "plaid_wyckoff": "PLAID++ Wyckoff"}
ORDER = list(LAB)


def load(proto: str, tag: str):
    raw = pd.read_parquet(DATA_DIR / f"{proto}/main.{tag}.parquet")
    sun = pd.read_parquet(DATA_DIR / f"sun/{proto}.main.{tag}.parquet")
    e = pd.to_numeric(raw["energy_above_hull"], errors="coerce")
    df = pd.DataFrame({
        "mp_id": sun["mp_id"],
        "relaxed": raw["relaxed_structures"].notna().to_numpy(),
        "e_hull": e.to_numpy(),
        "is_unique": sun["is_unique"].astype(bool).to_numpy(),
        "novel_mp": sun["is_novel_mp"].astype(bool).to_numpy(),
        "novel_train": sun["is_novel_train"].astype(bool).to_numpy(),
        "stable_old": sun["is_stable"].astype(bool).to_numpy(),
    })
    df["stable_new"] = (df["e_hull"] <= 0).fillna(False)
    return df


def sun_table(proto: str) -> pd.DataFrame:
    rows = []
    for tag in ORDER:
        try:
            d = load(proto, tag)
        except FileNotFoundError:
            continue
        r = {"protocol": proto, "model": LAB[tag], "tag": tag, "n": len(d)}
        for sname, scol in (("old", "stable_old"), ("new", "stable_new")):
            d["_st"] = d[scol]
            r[f"stable_{sname}"] = compute(d, "_st")
            for nname, ncol in (("mp", "novel_mp"), ("train", "novel_train")):
                d["_sun"] = d[scol] & d["is_unique"] & d[ncol]
                r[f"sun_{nname}_{sname}"] = compute(d, "_sun")
                r[f"n_sun_{nname}_{sname}"] = int(d["_sun"].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def bucket_table(proto: str) -> pd.DataFrame:
    """Three mutually exclusive buckets, restricted to rows that relaxed."""
    rows = []
    for tag in ORDER:
        try:
            d = load(proto, tag)
        except FileNotFoundError:
            continue
        d = d[d["relaxed"]]
        A = d["novel_mp"].to_numpy()
        B = (d["novel_train"] & ~d["novel_mp"]).to_numpy()
        C = (~d["novel_train"]).to_numpy()
        r = {"model": LAB[tag], "n_relax": len(d),
             "A_pct": 100 * A.mean(), "B_pct": 100 * B.mean(), "C_pct": 100 * C.mean()}
        for sname, scol in (("old", "stable_old"), ("new", "stable_new")):
            s = d[scol].to_numpy()
            for bn, bm in (("A", A), ("B", B), ("C", C)):
                r[f"{bn}_stable_{sname}"] = 100 * (s & bm).sum() / max(bm.sum(), 1)
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    out = DATA_DIR / "tables"
    for proto in ("out", "denovo"):
        if not glob.glob(str(DATA_DIR / f"sun/{proto}.main.*.parquet")):
            continue
        t = sun_table(proto)
        t.to_parquet(out / f"t1_sun_strict_{proto}.parquet", index=False)
        print(f"\n=== S.U.N. -- {proto} -- 旧 (E_hull<0.1) vs 新 (E_hull<=0) ===")
        print(t[["model", "stable_old", "stable_new",
                 "sun_mp_old", "sun_mp_new", "sun_train_old", "sun_train_new"]]
              .to_string(index=False))
        print(t[["model", "n_sun_mp_old", "n_sun_mp_new",
                 "n_sun_train_old", "n_sun_train_new"]].to_string(index=False))
    b = bucket_table("denovo")
    b.to_parquet(out / "t1_buckets_strict.parquet", index=False)
    print("\n=== 三桶分解 (de-novo, 仅成功弛豫的行) ===")
    print(b[["model", "n_relax", "A_pct", "B_pct", "C_pct",
             "A_stable_old", "A_stable_new", "B_stable_old", "B_stable_new",
             "C_stable_old", "C_stable_new"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
