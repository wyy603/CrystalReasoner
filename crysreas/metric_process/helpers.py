"""Shared helpers for basic metrics (no BaseMetric)."""

from __future__ import annotations

import bisect
import logging
import re
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from crysreas.utils.crystal import SimpleCrystal
from crysreas.utils.crystaltext import parse_crystaltext
from crysreas.utils.plaid_wyckoff import parse_plaid_wyckoff

# Columns stored as SimpleCrystal text in parquet / pickle round-trips (see run_metric).
PARQUET_STRUCTURE_COLUMNS = frozenset(
    ("simple_structure", "relaxed_structures", "relaxed_structures_300")
)

logger = logging.getLogger(__name__)


def find_index(offset: list[tuple[Any, int]], pos: int) -> int:
    t_end = [off[1] for off in offset]
    idx = bisect.bisect_right(t_end, pos)
    if idx < len(t_end):
        return idx
    return len(t_end) - 1


def getseg(x: float, arr: list[float | int]) -> tuple[float | int, float]:
    if x <= arr[0] and arr[0] != 0:
        return 0, arr[0]
    for i in range(0, len(arr) - 1):
        if arr[i] <= x and x <= arr[i + 1]:
            return arr[i], arr[i + 1]
    return arr[-1], np.inf


def structure_validity(crystal, cutoff: float = 0.5) -> bool:
    if crystal is None:
        return False
    if crystal.volume < 1.0:
        return False
    if min(crystal.lattice.abc) < 1.1:
        return False
    for angle in crystal.lattice.angles:
        if angle < 20 or angle > 160:
            return False
    dist_mat = crystal.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    return True


def parse_simple_structure_from_response(response: Any) -> Any:
    """Parse Structure from LLM response; narrow try/except around CIF parsing only."""
    if response is None or (isinstance(response, float) and pd.isna(response)):
        return None
    text = str(response)
    match = re.search(r"<CIF>(.*?)</CIF>", text, re.DOTALL)
    if not match:
        return None
    cif_simple = match.group(1).strip()
    try:
        simple_crystal = SimpleCrystal.from_simple_no_sym(cif_simple)
    except Exception as e:
        logger.debug("CIF parse failed: %s", e)
        return None
    if simple_crystal:
        return simple_crystal.structure
    return None


def parse_crystaltext_structure_from_response(response: Any) -> Any:
    """Parse Structure from LLM response containing CrystalTextLLM inner text."""
    if response is None or (isinstance(response, float) and pd.isna(response)):
        return None
    text = str(response)
    match = re.search(r"<CIF>(.*?)</CIF>", text, re.DOTALL)
    if not match:
        return None
    return parse_crystaltext(match.group(1).strip())


def parse_plaid_wyckoff_structure_from_response(response: Any) -> Any:
    """Parse Structure from LLM response containing PLaID++ Wyckoff inner text."""
    if response is None or (isinstance(response, float) and pd.isna(response)):
        return None
    text = str(response)
    match = re.search(r"<CIF>(.*?)</CIF>", text, re.DOTALL)
    if not match:
        return None
    return parse_plaid_wyckoff(match.group(1).strip())


def return_gt_dict(elem: dict, prompt_type_first: str) -> dict:
    structure = elem.get("structure")
    crystal = SimpleCrystal.from_sym_structure(structure)
    gt: dict = {}
    gt["mp_id"] = str(elem["material_id"])
    gt["comp"] = str(crystal.structure.composition.reduced_composition)
    gt["spg"] = crystal.sga.get_space_group_number()
    if prompt_type_first == "elastic":
        gt["bulk_modulus"] = getseg(elem["bulk_modulus"], [-np.inf, 0, 10, 50, 150, 300, np.inf])
        gt["shear_modulus"] = getseg(elem["shear_modulus"], [-np.inf, 0, 20, 80, 200, np.inf])
    elif prompt_type_first == "cte":
        # Volumetric CTE at 300 K (1/K). Bin edges: MP QHA spans ~(-2e-4, 1e-2); typical bulk ~2e-5–8e-5.
        # Micro-style knots (×10⁻⁶ K⁻¹) with negative tail; last edge catches large outliers.
        _cte_edges_micro = [-np.inf, -10, 0, 5, 15, 40, 100, 200, 300, np.inf]
        _cte_edges = [x * 1e-6 for x in _cte_edges_micro]
        gt["thermal_expansion_300k"] = getseg(float(elem["thermal_expansion_300k"]), _cte_edges)
    return gt


def spacegroup_number_safe(structure: Any) -> int | None:
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5)
        return sga.get_space_group_number()
    except Exception as e:
        logger.debug("SpacegroupAnalyzer failed: %s", e)
        return None


def _serialize_entry(x: Any, key: str) -> Any:
    if x is None or pd.isnull(x):
        return None

    if key in PARQUET_STRUCTURE_COLUMNS:
        if isinstance(x, str):
            return x
        return SimpleCrystal(x).to_simple_no_sym()

    if isinstance(x, dict):
        y = {}
        for k, v in x.items():
            if isinstance(v, np.ndarray):
                y[k] = v.tolist()
            else:
                y[k] = v
        return y

    return x


def _deserialize_entry(x: Any, key: str) -> Any:
    if x is None or pd.isnull(x):
        return None

    if key in PARQUET_STRUCTURE_COLUMNS:
        if isinstance(x, Structure):
            return x
        return SimpleCrystal.from_simple_no_sym(x).structure

    if isinstance(x, dict):
        y = {}
        for k, v in x.items():
            if isinstance(v, list):
                y[k] = np.array(v, dtype=float)
            else:
                y[k] = v
        return y

    return x


def df_serialize(df: pd.DataFrame) -> None:
    for key in df.keys():
        df[key] = df[key].apply(partial(_serialize_entry, key=key))


def df_deserialize(df: pd.DataFrame) -> None:
    for key in df.keys():
        df[key] = df[key].apply(partial(_deserialize_entry, key=key))
