"""Uniqueness / diversity of the *unrelaxed* generated structures — no MLIP needed.

Two numbers per run:
  * `uniqueness`  - MatterGen's definition (greedy first-wins within reduced-formula
                    groups) over the whole evaluated set, matching the paper's metric
                    except that it runs on generated structures, before relaxation.
  * `distinct@k`  - mean number of distinct structures among the k samples drawn for
                    one prompt. This is the sample-count-independent version, and is
                    what actually answers "is the model more diverse".
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submodules" / "mattergen"))
from pymatgen.entries.computed_entries import ComputedStructureEntry
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.dataset_matcher import (
    OrderedDatasetUniquenessComputer, get_unique,
)
from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher
from crysreas.metric_process.io import load_metrics_dataframe

MATCHER = DefaultDisorderedStructureMatcher()


def run(path: Path) -> dict | None:
    df = load_metrics_dataframe(path)
    if "simple_structure" not in df.columns:
        return None
    ok = df["simple_structure"].notna()
    structs = df.loc[ok, "simple_structure"].tolist()
    mp_ids = df.loc[ok, "mp_id"].tolist()
    n_total = len(df)

    # (1) set-level uniqueness, MatterGen definition, over parsed structures
    ds = ReferenceDataset.from_entries(
        "generated", [ComputedStructureEntry(s, 0.0) for s in structs])
    mask = OrderedDatasetUniquenessComputer(MATCHER)(ds)
    uniq = np.zeros(n_total, dtype=bool)
    uniq[np.flatnonzero(ok.to_numpy())] = mask

    # (2) per-prompt distinct count
    per_prompt = {}
    for mp, s in zip(mp_ids, structs):
        per_prompt.setdefault(mp, []).append(s)
    distinct = [len(get_unique(MATCHER, v)) for v in per_prompt.values()]
    sizes = [len(v) for v in per_prompt.values()]

    return {
        "run": path.stem,
        "n_structures": n_total,
        "n_parsed": int(ok.sum()),
        "uniqueness": f"{100 * uniq.sum() / n_total:.2f}%",
        "distinct_per_prompt": f"{np.mean(distinct):.2f} / {np.mean(sizes):.1f}",
        "distinct_frac": f"{100 * np.mean(np.array(distinct) / np.array(sizes)):.2f}%",
    }


def main() -> None:
    rows = []
    for p in sorted((DATA_DIR / "nomlip").glob("*.parquet")):
        try:
            r = run(p)
        except Exception as e:
            print(f"!! {p.stem}: {type(e).__name__}: {e}"); continue
        if r:
            rows.append(r); print(r, flush=True)
    if rows:
        out = pd.DataFrame(rows)
        out.to_parquet(DATA_DIR / "uniqueness_nomlip.parquet")
        pd.set_option("display.width", 200)
        print(); print(out.to_string(index=False))


if __name__ == "__main__":
    main()
