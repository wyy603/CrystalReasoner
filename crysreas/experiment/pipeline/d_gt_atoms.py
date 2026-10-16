"""Ground-truth atom count per test material, for the complexity curves.

The prompt fixes a reduced formula and a space group, not a cell size, so the
number of atoms that defines "how hard is this prompt" has to come from the
Materials Project record, never from the generated structure. Using the
generated count instead would make the x-axis depend on the model being
measured, and would silently drop every row that failed to parse.
"""
from __future__ import annotations

import json
import pickle
import shelve
from pathlib import Path
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

OUT = DATA_DIR / "gt_atoms.pkl"


def main() -> None:
    ids = json.load(open(DATA_DIR / "splits/split_denovo_8141.json"))["test"]
    db = shelve.open(str(ROOT / "assets/MP/MP_shelve"), flag="r")
    n = {}
    for m in ids:
        try:
            n[m] = len(db[m]["structure"])
        except Exception:
            continue
    db.close()
    with open(OUT, "wb") as f:
        pickle.dump(n, f)
    v = sorted(n.values())
    print(f"{len(n)} / {len(ids)} ids resolved; atoms min={v[0]} median={v[len(v)//2]} max={v[-1]}")


if __name__ == "__main__":
    main()
