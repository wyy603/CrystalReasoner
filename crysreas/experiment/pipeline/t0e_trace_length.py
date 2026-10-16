"""T0-e: is the trace-length/atom-count scaling *adaptive*, or just imitation?

The draft's Figure 3(a) shows generated trace length rising with atom count and
calls it an adaptive reasoning budget. But the appendix already reports the same
correlation inside the SFT corpus (Pearson r = 0.57) and concedes it is "partly
by construction, since larger cells have more orbits to enumerate". Reproducing
a correlation the template already has shows imitation, not adaptivity.

The discriminating test is the SLOPE. Fit tokens ~ n_atoms for
  (a) the ground-truth traces of the same prompts (what the template does), and
  (b) the model's generated traces.
Slope_model > slope_corpus  => the model allocates budget beyond what it copied.
Slope_model == slope_corpus => the scaling is inherited, and the claim should go.

Both are fitted against the SAME x (the ground-truth atom count of the prompt's
material), so the comparison is paired and cannot be moved by the model getting
the cell size wrong. A second fit against each side's own structure reproduces
the paper's plotting convention.
"""
from __future__ import annotations

import json
import shelve
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

from crysreas.data.prompt_generator.get_info import get_info  # noqa: E402
from crysreas.utils.crystal import SimpleCrystal  # noqa: E402

OUT = DATA_DIR / "t0"
CIF_MARK = "## CIF File"


def thinking_part(resp: str) -> str | None:
    i = str(resp).find(CIF_MARK)
    return None if i == -1 else str(resp)[:i]


def n_atoms_of(simple_str) -> int | None:
    try:
        return len(SimpleCrystal.from_simple_no_sym(str(simple_str)).structure)
    except Exception:
        return None


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def boot_ci(x: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    s = [ols_slope(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints_merged/thinking"))
    ntok = lambda t: len(tok.encode(t, add_special_tokens=False))  # noqa: E731

    ids = json.load(open(ROOT / "assets/MP/split_generation.json"))["test"]

    # (a) ground-truth traces for exactly these prompts
    db = shelve.open(str(ROOT / "assets/MP/MP_shelve"), flag="r")
    gt_rows = []
    for mp in ids:
        try:
            item = get_info(db[mp], "conditional+thinking", seed=mp[3:])
            th = thinking_part(item["answer"])
            if th is None:
                continue
            gt_rows.append(dict(mp_id=mp, gt_tokens=ntok(th),
                                gt_atoms=len(db[mp]["structure"])))
        except Exception:
            continue
    db.close()
    gt = pd.DataFrame(gt_rows)
    print(f"[t0e] ground-truth traces: {len(gt)}/{len(ids)} prompts", flush=True)

    # (b) generated traces
    recs = []
    for tag in ("thinking", "full"):
        src = DATA_DIR / f"out/main.{tag}.parquet"
        if not src.exists():
            continue
        d = pd.read_parquet(src)
        for mp, resp, simple in zip(d["mp_id"], d["responses"], d["simple_structure"]):
            th = thinking_part(resp)
            if th is None:
                continue
            recs.append(dict(model=tag, mp_id=str(mp), tokens=ntok(th),
                             gen_atoms=n_atoms_of(simple)))
    gen = pd.DataFrame(recs).merge(gt, on="mp_id", how="inner")
    OUT.mkdir(parents=True, exist_ok=True)
    gen.to_parquet(OUT / "t0e_trace_length_per_row.parquet", index=False)
    gt.to_parquet(OUT / "t0e_trace_length_gt.parquet", index=False)

    rows = []
    lo, hi = 1, 21                       # the paper's plotted range
    m = gt.gt_atoms.between(lo, hi)
    x, y = gt.loc[m, "gt_atoms"].to_numpy(float), gt.loc[m, "gt_tokens"].to_numpy(float)
    c = boot_ci(x, y)
    rows.append(dict(series="SFT corpus (ground-truth traces)", x="gt_atoms", n=len(x),
                     slope=ols_slope(x, y), ci_lo=c[0], ci_hi=c[1],
                     pearson_r=float(np.corrcoef(x, y)[0, 1])))
    for tag, g in gen.groupby("model"):
        for xcol, label in (("gt_atoms", "gt_atoms"), ("gen_atoms", "generated_atoms")):
            gg = g[g[xcol].between(lo, hi) & g["tokens"].notna()]
            if len(gg) < 50:
                continue
            x, y = gg[xcol].to_numpy(float), gg["tokens"].to_numpy(float)
            c = boot_ci(x, y)
            rows.append(dict(series=f"{tag} (generated traces)", x=label, n=len(x),
                             slope=ols_slope(x, y), ci_lo=c[0], ci_hi=c[1],
                             pearson_r=float(np.corrcoef(x, y)[0, 1])))
    res = pd.DataFrame(rows)
    res.to_parquet(OUT / "t0e_slopes.parquet", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
