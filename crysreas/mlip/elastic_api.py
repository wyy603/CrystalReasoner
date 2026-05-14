"""
Elastic / fmax via ``crysreas.mlip.elastic`` (MatterSimModel singleton in that module).

Thin wrappers keep MLIP call sites in ``metric_process.basic`` explicit.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from pymatgen.core.structure import Structure
import ray

def fmax_numpy(structures: list[Structure | None]) -> np.ndarray:
    """Per-structure max force norm; None inputs yield NaN for that row."""
    from crysreas.mlip.elastic import compute_fmax

    n = len(structures)
    out = np.full((n,), np.nan, dtype=np.float64)
    valid_idx = [i for i, s in enumerate(structures) if s is not None]
    if not valid_idx:
        return out
    sub = [structures[i] for i in valid_idx]
    try:
        vals = compute_fmax(sub).detach().cpu().numpy()
        for j, i in enumerate(valid_idx):
            out[i] = float(vals[j])
    except Exception as e:
        print(f"[elastic_api] fmax_numpy failed: {e}")
    return out


def elastic_properties_list(
    structures: list[Structure | None],
    *,
    relaxed: bool = True,
    debug: bool = False,
) -> list[Any] | tuple[list[Any], list[dict[str, Any] | None]]:
    """List aligned with ``structures``; None slots stay None."""
    from crysreas.mlip.elastic import calculate_elastic_properties

    n = len(structures)
    out: list[Any] = [None] * n
    debug_out: list[dict[str, Any] | None] = [None] * n
    valid_idx = [i for i, s in enumerate(structures) if s is not None]
    if not valid_idx:
        return (out, debug_out) if debug else out
    sub = [structures[i] for i in valid_idx]
    n_sub = len(sub)
    # Chunked Ray calls so the driver can print progress (same log as metrics CLI).
    # Set AI4SCI_ELASTIC_CHUNK=1 for per-structure lines; <=0 means one remote call.
    chunk_size = int(os.environ.get("AI4SCI_ELASTIC_CHUNK", "64"))
    if chunk_size <= 0:
        chunk_size = n_sub
    try:
        cursor = 0
        while cursor < n_sub:
            end = min(cursor + chunk_size, n_sub)
            chunk_sub = sub[cursor:end]
            remote_res = ray.get(
                calculate_elastic_properties.remote(
                    chunk_sub, relaxed=relaxed, debug=debug
                )
            )
            if debug:
                results, debug_results = remote_res
            else:
                results = remote_res
                debug_results = [None] * len(results)

            for j, k in enumerate(range(cursor, end)):
                i = valid_idx[k]
                elem = results[j]
                out[i] = elem.model_dump() if elem is not None else None
                debug_out[i] = debug_results[j] if j < len(debug_results) else None
            print(
                f"[elastic_api] elastic_properties: {end}/{n_sub} structures",
                flush=True,
            )
            cursor = end
    except Exception as e:
        print(f"[elastic_api] elastic_properties_list failed: {e}")
    return (out, debug_out) if debug else out
