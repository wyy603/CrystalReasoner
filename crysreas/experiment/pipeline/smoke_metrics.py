"""Smoke-test the metric pipeline on the small task-4 parquet and time it."""
import subprocess, sys, time
from pathlib import Path
import pandas as pd
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SRC = Path(str(DATA_DIR / "task4_samples.parquet"))
DST = Path(str(DATA_DIR / "out/smoke_metrics.parquet"))
DST.parent.mkdir(parents=True, exist_ok=True)
pd.read_parquet(SRC).to_parquet(DST)

cmd = [".venv/bin/python", "-m", "crysreas.metric_process",
       "--path", str(DST),
       "--metrics-name", "simple_structure", "structure_validity", "smact_validity",
       "composition_consistency", "spacegroup_consistency", "stable_unique_novel",
       "--prompt-type", "conditional+thinking", "--num-workers", "4", "--forced"]
print(" ".join(cmd), flush=True)
t0 = time.time()
rc = subprocess.run(cmd).returncode
print(f"rc={rc} in {time.time()-t0:.0f}s", flush=True)

df = pd.read_parquet(DST)
print("columns:", list(df.columns))
import numpy as np
for c in ("structure_validity","smact_validity","composition_consistency",
          "spacegroup_consistency","is_stable","is_novel","is_unique","stable_unique_novel",
          "energy_above_hull"):
    if c in df.columns:
        vals = np.concatenate([np.asarray(v, dtype=object).ravel() for v in df[c]])
        ok = [v for v in vals if v is not None and v == v]
        if c == "energy_above_hull":
            f = [float(v) for v in ok]
            print(f"  {c:24s} n={len(f):4d} mean={np.mean(f):.4f}")
        else:
            print(f"  {c:24s} n={len(ok):4d} true={sum(1 for v in ok if v)}/{len(ok)}")
