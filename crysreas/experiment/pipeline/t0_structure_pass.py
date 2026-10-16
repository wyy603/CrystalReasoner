"""T0-a + T0-b per-row structure pass over existing generation parquets.

Writes one parquet per input run with these extra columns:

T0-a (space-group consistency, group/subgroup aware)
  req_spg          requested space-group number (from `gt`)
  spg_p1           detected number at symprec=0.1 (what the repo metric uses)
  sgc_strict       spg_p1 == req_spg                 -- reproduces the repo metric
  sgc_supergroup   spg_p1 == req_spg OR req is a proper subgroup of detected,
                   i.e. the structure carries *at least* the requested symmetry.
                   One-sided: it can only ever raise the score, so it is an
                   alternative reading, not a correction.

T0-b (numerical precision)
  spg_1e2/1e3/1e4  detected number at tightening symprec -- a structure whose
                   symmetry survives a tighter tolerance carries more precise
                   coordinates.
  sym_survives_*   detected number at that symprec still equals spg_p1
  relax_rmsd       RMSD (A) between generated and MatterSim-relaxed positions,
                   min-image fractional delta mapped through the generated cell
  relax_maxdisp    max single-atom displacement (A)
  dvol_frac        |V_relaxed - V_gen| / V_gen
  dabc_frac        mean relative change of the three lattice lengths
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

SYMPRECS = [("spg_p1", 0.1), ("spg_1e2", 0.01), ("spg_1e3", 0.001), ("spg_1e4", 0.0001)]


def _parse(s):
    from crysreas.utils.crystal import SimpleCrystal
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        return SimpleCrystal.from_simple_no_sym(str(s)).structure
    except Exception:
        return None


def _row(args):
    gen_s, rel_s, req = args
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    out = {k: None for k, _ in SYMPRECS}
    out.update(relax_rmsd=None, relax_maxdisp=None, dvol_frac=None, dabc_frac=None,
               req_spg=req, n_atoms=None)

    g = _parse(gen_s)
    if g is None:
        return out
    out["n_atoms"] = len(g)

    for key, sp in SYMPRECS:
        try:
            out[key] = int(SpacegroupAnalyzer(g, symprec=sp, angle_tolerance=5).get_space_group_number())
        except Exception:
            out[key] = None

    r = _parse(rel_s)
    if r is not None and len(r) == len(g):
        try:
            d = r.frac_coords - g.frac_coords
            d -= np.round(d)                       # minimum image
            cart = d @ g.lattice.matrix            # in the generated cell's frame
            dist = np.linalg.norm(cart, axis=1)
            out["relax_rmsd"] = float(np.sqrt((dist ** 2).mean()))
            out["relax_maxdisp"] = float(dist.max())
            v0, v1 = g.volume, r.volume
            out["dvol_frac"] = float(abs(v1 - v0) / v0) if v0 > 0 else None
            a0 = np.array(g.lattice.abc); a1 = np.array(r.lattice.abc)
            out["dabc_frac"] = float(np.mean(np.abs(a1 - a0) / a0))
        except Exception:
            pass
    return out


def _as_int(x):
    """None for missing; pandas turns our None space groups into NaN."""
    if x is None:
        return None
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
    except TypeError:
        return None
    return int(x)


def _supergroup_ok(req, det) -> bool | None:
    from pymatgen.symmetry.groups import SpaceGroup
    req, det = _as_int(req), _as_int(det)
    if req is None or det is None:
        return None
    if int(req) == int(det):
        return True
    try:
        return bool(SpaceGroup.from_int_number(int(req)).is_subgroup(
            SpaceGroup.from_int_number(int(det))))
    except Exception:
        return False


def process(path: Path, out_path: Path, workers: int = 8) -> None:
    df = pd.read_parquet(path)
    req = [(g or {}).get("spg") if isinstance(g, dict) else None for g in df["gt"]]
    # the space-group-conditioned runs only computed the non-MLIP metrics, so
    # they carry no relaxed structures; the displacement columns stay empty there
    rel = (df["relaxed_structures"].tolist() if "relaxed_structures" in df.columns
           else [None] * len(df))
    args = list(zip(df["simple_structure"].tolist(), rel, req))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_row, args, chunksize=64))
    add = pd.DataFrame(rows, index=df.index)

    # cache the supergroup test on the small set of (req, det) pairs actually seen
    norm = [(_as_int(a), _as_int(b)) for a, b in zip(add["req_spg"], add["spg_p1"])]
    cache = {p: _supergroup_ok(*p) for p in set(norm)}   # NaN != NaN, so normalise first
    add["sgc_strict"] = [None if (_as_int(a) is None or _as_int(b) is None)
                         else (_as_int(a) == _as_int(b))
                         for a, b in zip(add["req_spg"], add["spg_p1"])]
    add["sgc_supergroup"] = [cache[p] for p in norm]
    for key, _ in SYMPRECS[1:]:
        add["sym_survives_" + key.split("_")[1]] = [
            None if (_as_int(a) is None or _as_int(b) is None)
            else (_as_int(a) == _as_int(b))
            for a, b in zip(add["spg_p1"], add[key])]

    keep = [c for c in ("mp_id", "structure_validity", "smact_validity",
                        "composition_consistency", "spacegroup_consistency",
                        "is_stable", "is_unique", "is_novel", "stable_unique_novel",
                        "energy_above_hull") if c in df.columns]
    res = pd.concat([df[keep].reset_index(drop=True), add.reset_index(drop=True)], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_path, index=False)
    print(f"{path.name:34s} n={len(res):6d} -> {out_path}", flush=True)


def main() -> None:
    src_dir = DATA_DIR / (sys.argv[1] if len(sys.argv) > 1 else "out")
    dst_dir = DATA_DIR / "t0"
    pats = sys.argv[2] if len(sys.argv) > 2 else "main.*.parquet"
    for p in sorted(src_dir.glob(pats)):
        if p.stem == "smoke_metrics":
            continue
        out = dst_dir / f"{src_dir.name}.{p.stem}.parquet"
        if out.exists():
            print(f"[skip] {out.name}"); continue
        process(p, out)


if __name__ == "__main__":
    main()
