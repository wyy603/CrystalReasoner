"""Verify generated samples against the prompt constraints using the repo's own checkers."""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "scripts")
from crysreas.metric_process.helpers import (
    parse_simple_structure_from_response,
    structure_validity,
    spacegroup_number_safe,
    return_gt_dict,
)
from crysreas.utils.basic_eval import smact_validity
from pymatgen.core.periodic_table import Element

ap = argparse.ArgumentParser()
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ap.add_argument("--path", default=str(DATA_DIR / "task4_samples.parquet"))
ap.add_argument("--records", default=str(DATA_DIR / "test_records.pkl"))
args = ap.parse_args()

recs = pickle.load(open(args.records, "rb"))
df = pd.read_parquet(args.path)
prompts = json.load(open(args.path + ".prompts.json"))

def smact_ok(s):
    try:
        comp = s.composition.reduced_composition.as_dict()
        elems = [Element(k).Z for k in comp]
        counts = [int(round(v)) for v in comp.values()]
        import numpy as np
        g = np.gcd.reduce(counts)
        return bool(smact_validity(tuple(elems), tuple(c // g for c in counts)))
    except Exception:
        return None

for _, row in df.iterrows():
    mp_id = row["mp_id"]
    gt = return_gt_dict(recs[mp_id], "conditional")
    print("=" * 100)
    print(f"{mp_id}  target comp={gt['comp']}  target spg={gt['spg']}")
    print("PROMPT:", prompts[mp_id].strip())
    for i, resp in enumerate(row["responses"]):
        s = parse_simple_structure_from_response(resp)
        if s is None:
            print(f"  [{i:2d}] PARSE FAIL")
            continue
        comp_ok = str(s.composition.reduced_composition) == gt["comp"]
        spg = spacegroup_number_safe(s)
        sv = structure_validity(s)
        sm = smact_ok(s)
        flag = "ALL-OK" if (comp_ok and spg == gt["spg"] and sv and sm) else ""
        print(f"  [{i:2d}] natoms={len(s):3d} comp={comp_ok!s:5s} spg={spg} (want {gt['spg']}) "
              f"struct={sv!s:5s} smact={sm!s:5s} vol={s.volume:8.2f} {flag}")
