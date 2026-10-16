"""Blank the CrystalTextLLM generations the repo's parser cannot round-trip.

One de-novo generation emits a site with only TWO fractional coordinates. The
resulting pymatgen Structure breaks in two independent places:
  * `helpers.structure_validity` -> `crystal.distance_matrix` raises on the
    inhomogeneous coordinate array (`helpers.py:55`);
  * `SimpleCrystal.to_simple_no_sym` raises `IndexError` on `frac[2]`
    (`crysreas/utils/crystal.py:153`) while serialising the result,
so the metric pass dies before writing anything. The 1024x10 run never hit it;
8141 prompts did.

We load with the pipeline's own loader (which explodes n_samples), blank the
offending response, and write the exploded table back. A blank response parses
to None, which is exactly how the pipeline already represents an unusable
generation: the failure propagates as a missing value and `compute()` scores it
as False and keeps it in the denominator, so the row is counted as
a failure. That is the correct accounting and keeps the denominator at 16,282,
comparable with the other seven models.
"""
from __future__ import annotations

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
ROOT = Path("/scratch/yw10199/CrystalReasoner")
sys.path.insert(0, str(ROOT))

from crysreas.metric_process.io import load_metrics_dataframe  # noqa: E402
from crysreas.utils.crystal import SimpleCrystal  # noqa: E402


def parser_for(prompt_type: str):
    from crysreas.metric_process.basic import (
        parse_crystaltext_structure_from_response,
        parse_plaid_wyckoff_structure_from_response,
        parse_simple_structure_from_response,
    )
    p0 = prompt_type.split("+")[0]
    if p0 in ("crystaltextllm_generation", "crystaltextllm_8_generation"):
        return parse_crystaltext_structure_from_response
    if p0 in ("plaid_wyckoff_generation", "plaid_wyckoff_8_generation"):
        return parse_plaid_wyckoff_structure_from_response
    return parse_simple_structure_from_response


def main(path: str, prompt_type: str) -> None:
    df = load_metrics_dataframe(Path(path))
    parse = parser_for(prompt_type)
    bad = []
    for i, resp in enumerate(df["responses"].tolist()):
        try:
            s = parse(resp)
            if s is None:
                continue
            st = s.structure if isinstance(s, SimpleCrystal) else s
            np.asarray(st.frac_coords, dtype=float)   # inhomogeneous coords?
            _ = st.distance_matrix                    # validity checker path
            SimpleCrystal(st).to_simple_no_sym()      # serialiser path
        except Exception as e:
            bad.append((i, type(e).__name__, str(e)[:70]))
    print(f"{path}: {len(df)} exploded rows, {len(bad)} unusable")
    for b in bad[:10]:
        print("   ", b)
    df.loc[[i for i, _, _ in bad], "responses"] = ""
    out = Path(path)
    df.to_parquet(out, index=False)
    print(f"   -> blanked {len(bad)} responses, rewrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
