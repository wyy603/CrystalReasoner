"""Subgroup-aware space-group consistency for the trace-shuffle ablation rows.

The stored `spacegroup_consistency` column is integer equality against the
requested group. This adds the criterion used throughout report.md: a structure
counts when it carries at least the requested symmetry, i.e. the requested group
equals or is a proper subgroup of the detected group. Gating is copied from
`crysreas/metric_process/basic.py::ensure_spacegroup_consistency` so the two
columns are comparable.
"""
from __future__ import annotations

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

SRC = DATA_DIR / "shuffle" / "shuffle_ablation.parquet"
DST = DATA_DIR / "shuffle" / "shuffle_sgc.parquet"


def _detect(s):
    from crysreas.utils.crystal import SimpleCrystal
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        st = SimpleCrystal.from_simple_no_sym(str(s)).structure
        return int(SpacegroupAnalyzer(st, symprec=0.1, angle_tolerance=5)
                   .get_space_group_number())
    except Exception:
        return None


def _sub(req, det) -> bool | None:
    from pymatgen.symmetry.groups import SpaceGroup
    if req is None or det is None:
        return None
    if int(req) == int(det):
        return True
    try:
        return bool(SpaceGroup.from_int_number(int(req))
                    .is_subgroup(SpaceGroup.from_int_number(int(det))))
    except Exception:
        return False


def main() -> None:
    df = pd.read_parquet(SRC)
    with ProcessPoolExecutor(max_workers=4) as ex:
        det = list(ex.map(_detect, df["simple_structure"].tolist(), chunksize=32))
    req = [(g or {}).get("spg") if isinstance(g, dict) else None for g in df["gt"]]

    ok = (df["structure_validity"].astype("boolean").fillna(False)
          & df["smact_validity"].astype("boolean").fillna(False)).to_numpy()
    pairs = {(a, b) for a, b in zip(req, det)}
    cache = {p: _sub(*p) for p in pairs}
    sub = [cache[(a, b)] for a, b in zip(req, det)]

    out = df[["mp_id", "variant", "structure_validity", "smact_validity",
              "composition_consistency", "spacegroup_consistency"]].copy()
    out["req_spg"] = req
    out["det_spg"] = det
    out["sgc_sub"] = [s if o else None for s, o in zip(sub, ok)]
    out.to_parquet(DST, index=False)
    print(f"n={len(out)} -> {DST}")
    print(out.groupby("variant")[["spacegroup_consistency", "sgc_sub"]]
          .apply(lambda d: pd.Series({"strict": 100*(d.spacegroup_consistency == True).mean(),
                                      "sub": 100*(d.sgc_sub == True).mean()})))


if __name__ == "__main__":
    main()
