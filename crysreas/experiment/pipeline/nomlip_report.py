"""Report the non-MLIP metrics for whatever runs have completed, vs the paper."""
import sys
from pathlib import Path
import pandas as pd
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(ROOT))
from crysreas.experiment.make_graph import compute
from crysreas.metric_process.io import load_metrics_dataframe

METRICS = ["structure_validity", "smact_validity", "composition_consistency",
           "spacegroup_consistency"]
PAPER = {  # Table 1 of the draft
    "main.base":     dict(structure_validity=84.03, smact_validity=90.36,
                          composition_consistency=90.92, spacegroup_consistency=40.96),
    "main.thinking": dict(structure_validity=91.29, smact_validity=91.72,
                          composition_consistency=97.20, spacegroup_consistency=48.27),
    "main.rl":       dict(structure_validity=89.85, smact_validity=91.10,
                          composition_consistency=92.75, spacegroup_consistency=41.44),
    "main.full":     dict(structure_validity=94.92, smact_validity=91.78,
                          composition_consistency=97.32, spacegroup_consistency=50.21),
}
rows = []
for p in sorted((DATA_DIR / "nomlip").glob("*.parquet")):
    tag = p.stem
    try:
        df = load_metrics_dataframe(p)
    except Exception as e:
        print(f"!! {tag}: {e}"); continue
    if METRICS[0] not in df.columns:
        print(f".. {tag}: metrics not written yet"); continue
    r = {"run": tag, "n": len(df)}
    for m in METRICS:
        got = compute(df, m)
        r[m] = got
        w = PAPER.get(tag, {}).get(m)
        if w is not None and got != "N/A":
            r[m + "_d"] = f"{float(got.split('±')[0].strip().rstrip('%')) - w:+.2f}"
    rows.append(r)
if rows:
    out = pd.DataFrame(rows)
    out.to_parquet(DATA_DIR / "nomlip_comparison.parquet")
    pd.set_option("display.width", 220, "display.max_columns", 50)
    print(out.to_string(index=False))
else:
    print("nothing ready yet")
