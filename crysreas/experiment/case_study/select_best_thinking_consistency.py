#!/usr/bin/env python3
"""Select thinking-trace case studies whose report best matches the final CIF.

The response format is a natural-language material report followed by a
``<CIF>...</CIF>`` block in the repository's simple CIF format.  This script
parses both parts and scores whether the report is internally consistent with
the generated structure.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spglib
from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import cellpar_to_cell


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "checkpoints_merged" / "thinking" / "conditional+thinking.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = OUTPUT_DIR / "best_thinking_consistency_cases.json"
OUTPUT_CSV = OUTPUT_DIR / "best_thinking_consistency_cases.csv"
OUTPUT_MD = OUTPUT_DIR / "best_thinking_consistency_cases.md"
OUTPUT_CIF_DIR = OUTPUT_DIR / "cif"
OUTPUT_SPACEGROUP_SUMMARY_MD = OUTPUT_DIR / "spacegroup_consistency_summary.md"
OUTPUT_SPACEGROUP_RELATIVE_DIFF_MD = OUTPUT_DIR / "spacegroup_relative_difference_table.md"
PARQUET_METRIC_COLUMNS = [
    "extra_args",
    "gt",
    "smact_validity",
    "structure_validity",
    "composition_consistency",
    "spacegroup_consistency",
    "relaxed_structures",
    "energies",
    "energy_above_hull",
    "is_stable",
    "is_novel",
    "is_unique",
    "stable_unique_novel",
]

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
}
SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
GENERATED_SUFFIX_RE = re.compile(r"<\|im_end\|>|<\|endoftext\|>")


@dataclass(frozen=True)
class SimpleStructure:
    simple: str
    lattice: np.ndarray
    species: list[str]
    frac_coords: np.ndarray
    volume: float
    spacegroup_number: int | None
    spacegroup_symbol: str | None
    equivalent_atoms: list[int] | None
    site_multiplicities_by_element: dict[str, list[int]]
    atom_count_by_element: dict[str, int]


def word_to_int(text: str) -> int | None:
    s = str(text).strip().lower()
    if s.isdigit():
        return int(s)
    return NUMBER_WORDS.get(s)


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def jsonable_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return jsonable_value(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable_value(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_value(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def extract_parquet_metrics(row: pd.Series) -> dict[str, Any]:
    return {
        column: jsonable_value(row.get(column))
        for column in PARQUET_METRIC_COLUMNS
        if column in row.index
    }


def pair_key(left: str, right: str) -> str:
    return "-".join(sorted([left, right]))


def normalize_text(text: Any) -> str:
    return str(text).translate(SUBSCRIPT_DIGITS)


def parse_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return str(prompt.get("content", ""))
    if isinstance(prompt, (list, tuple, np.ndarray)):
        parts = []
        for item in prompt:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(x for x in parts if x)
    return str(prompt)


def extract_final_cif(response: str) -> str | None:
    m = re.search(r"<CIF>(.*?)</CIF>", str(response), flags=re.S)
    if not m:
        return None
    return GENERATED_SUFFIX_RE.sub("", m.group(1)).strip()


def extract_report_text(response: str) -> str:
    text = str(response)
    idx = text.find("<CIF>")
    if idx >= 0:
        text = text[:idx]
    return text.strip()


def parse_simple_structure(simple: str) -> SimpleStructure | None:
    try:
        lines = [line.strip() for line in str(simple).strip().splitlines() if line.strip()]
        lengths = [float(x) for x in lines[1].split()]
        angles = [float(x) for x in lines[2].split()]
        lattice = np.asarray(cellpar_to_cell(lengths + angles), dtype=float)
        species: list[str] = []
        coords: list[list[float]] = []
        for line in lines[3:]:
            parts = line.split()
            species.append(parts[0])
            coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
        frac_coords = np.asarray(coords, dtype=float)
        volume = float(abs(np.linalg.det(lattice)))
        numbers = [atomic_numbers[x] for x in species]
        dataset = spglib.get_symmetry_dataset(
            (lattice, frac_coords, numbers),
            symprec=0.1,
            angle_tolerance=5.0,
        )
        sg_num = int(dataset.number) if dataset is not None else None
        sg_symbol = str(dataset.international) if dataset is not None else None
        eq_atoms = [int(x) for x in dataset.equivalent_atoms] if dataset is not None else None

        atom_counts = dict(Counter(species))
        site_mults: dict[str, list[int]] = defaultdict(list)
        if eq_atoms is None:
            for element, count in atom_counts.items():
                site_mults[element] = [1] * count
        else:
            groups: dict[int, list[int]] = defaultdict(list)
            for idx, group_id in enumerate(eq_atoms):
                groups[group_id].append(idx)
            for indices in groups.values():
                element_counts = Counter(species[i] for i in indices)
                if len(element_counts) != 1:
                    continue
                element = next(iter(element_counts))
                site_mults[element].append(len(indices))
        site_mults = {element: sorted(values) for element, values in sorted(site_mults.items())}

        return SimpleStructure(
            simple=str(simple).strip(),
            lattice=lattice,
            species=species,
            frac_coords=frac_coords,
            volume=volume,
            spacegroup_number=sg_num,
            spacegroup_symbol=sg_symbol,
            equivalent_atoms=eq_atoms,
            site_multiplicities_by_element=site_mults,
            atom_count_by_element=dict(sorted(atom_counts.items())),
        )
    except Exception:
        return None


def _pymatgen_symmetrized_cif(structure: SimpleStructure) -> tuple[str | None, str | None]:
    """Return a pymatgen symmetrized CIF if this environment provides pymatgen core APIs."""
    try:
        from pymatgen.core import Lattice, Structure
        from pymatgen.io.cif import CifWriter
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except Exception as exc:
        return None, f"pymatgen unavailable: {type(exc).__name__}: {exc}"

    try:
        lattice = Lattice(structure.lattice)
        pmg_structure = Structure(
            lattice=lattice,
            species=structure.species,
            coords=structure.frac_coords,
            coords_are_cartesian=False,
        )
        analyzer = SpacegroupAnalyzer(pmg_structure, symprec=0.1, angle_tolerance=5.0)
        symmetrized = analyzer.get_refined_structure()
        return str(CifWriter(symmetrized, symprec=0.1)), None
    except Exception as exc:
        return None, f"pymatgen CIF generation failed: {type(exc).__name__}: {exc}"


def _format_cif_float(value: float) -> str:
    return f"{float(value):.8f}".rstrip("0").rstrip(".")


def _spglib_standardized_cif(structure: SimpleStructure) -> tuple[str, str]:
    """Fallback real CIF for environments without pymatgen's Structure/CifWriter APIs."""
    numbers = [atomic_numbers[x] for x in structure.species]
    standardized = spglib.standardize_cell(
        (structure.lattice, structure.frac_coords, numbers),
        to_primitive=False,
        no_idealize=False,
        symprec=0.1,
    )
    if standardized is None:
        lattice = structure.lattice
        frac_coords = structure.frac_coords
        atomic_nums = numbers
    else:
        lattice, frac_coords, atomic_nums = standardized

    dataset = spglib.get_symmetry_dataset(
        (lattice, frac_coords, atomic_nums),
        symprec=0.1,
        angle_tolerance=5.0,
    )
    atoms = Atoms(numbers=atomic_nums, scaled_positions=frac_coords, cell=lattice, pbc=True)
    cell = atoms.cell.cellpar()
    sg_symbol = str(dataset.international) if dataset is not None else "P1"
    sg_number = int(dataset.number) if dataset is not None else 1
    species = atoms.get_chemical_symbols()

    lines = [
        "data_symmetrized",
        f"_symmetry_space_group_name_H-M   '{sg_symbol}'",
        f"_symmetry_Int_Tables_number      {sg_number}",
        f"_cell_length_a   {_format_cif_float(cell[0])}",
        f"_cell_length_b   {_format_cif_float(cell[1])}",
        f"_cell_length_c   {_format_cif_float(cell[2])}",
        f"_cell_angle_alpha   {_format_cif_float(cell[3])}",
        f"_cell_angle_beta    {_format_cif_float(cell[4])}",
        f"_cell_angle_gamma   {_format_cif_float(cell[5])}",
        "_symmetry_cell_setting   triclinic",
        "",
        "loop_",
        "  _atom_site_label",
        "  _atom_site_type_symbol",
        "  _atom_site_fract_x",
        "  _atom_site_fract_y",
        "  _atom_site_fract_z",
    ]
    counts: Counter[str] = Counter()
    for element, frac in zip(species, np.asarray(frac_coords, dtype=float), strict=True):
        counts[element] += 1
        wrapped = np.mod(frac, 1.0)
        lines.append(
            f"  {element}{counts[element]} {element} "
            f"{_format_cif_float(wrapped[0])} "
            f"{_format_cif_float(wrapped[1])} "
            f"{_format_cif_float(wrapped[2])}"
        )
    return "\n".join(lines) + "\n", "spglib_fallback"


def build_symmetrized_cif(simple_cif: str) -> dict[str, Any]:
    structure = parse_simple_structure(simple_cif)
    if structure is None:
        return {
            "generator": None,
            "cif": None,
            "note": "Could not parse final simple CIF.",
        }

    cif, note = _pymatgen_symmetrized_cif(structure)
    if cif is not None:
        return {
            "generator": "pymatgen",
            "cif": cif,
            "note": "Generated with pymatgen SpacegroupAnalyzer.get_refined_structure() and CifWriter.",
        }

    fallback_cif, fallback_generator = _spglib_standardized_cif(structure)
    return {
        "generator": fallback_generator,
        "cif": fallback_cif,
        "note": (
            "pymatgen symmetrized CIF was requested, but this environment lacks usable "
            f"pymatgen core/CIF modules; fallback reason: {note}"
        ),
    }


def safe_filename_token(text: object) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return token.strip("_") or "unknown"


def export_selected_cif_files(
    *,
    json_path: Path,
    output_dir: Path,
    regenerate: bool,
) -> list[Path]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    selected = data.get("selected_records", [])
    if not selected:
        raise RuntimeError(f"No selected_records found in {json_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rank, record in enumerate(selected, start=1):
        sym_cif = record.get("symmetrized_real_cif") or {}
        if regenerate or not sym_cif.get("cif"):
            sym_cif = build_symmetrized_cif(str(record["final_cif"]))

        cif_text = sym_cif.get("cif")
        if not cif_text:
            raise RuntimeError(
                f"Could not build CIF for row={record.get('row_index')} mp_id={record.get('mp_id')}: "
                f"{sym_cif.get('note')}"
            )

        sg_number = record.get("space_group", {}).get("final_cif_number", "sg")
        sg_symbol = record.get("space_group", {}).get("final_cif_symbol", "unknown")
        filename = (
            f"{rank:02d}_row_{safe_filename_token(record.get('row_index'))}_"
            f"{safe_filename_token(record.get('mp_id'))}_"
            f"sg_{safe_filename_token(sg_number)}_{safe_filename_token(sg_symbol)}.cif"
        )
        path = output_dir / filename
        path.write_text(str(cif_text).strip() + "\n", encoding="utf-8")
        written.append(path)
    return written


def parse_space_group_claim(report: str) -> dict[str, Any]:
    m = re.search(r"space group\s+(.{1,40}?)\s*\(id\s*(\d+)\)", report, flags=re.I)
    if not m:
        return {"symbol": None, "number": None}
    return {"symbol": m.group(1).strip(), "number": int(m.group(2))}


def parse_prompt_formation_energy(prompt: Any) -> float | None:
    m = re.search(r"formation energy per atom is\s*([-+]?\d+(?:\.\d+)?)", parse_prompt_text(prompt), flags=re.I)
    return float(m.group(1)) if m else None


def parse_report_formation_energy(report: str) -> float | None:
    matches = re.findall(r"formation energy per atom is\s*([-+]?\d+(?:\.\d+)?)", report, flags=re.I)
    return float(matches[-1]) if matches else None


def parse_hull_energy(report: str) -> float | None:
    m = re.search(r"lying\s+([-+]?\d+(?:\.\d+)?)\s*eV/atom\s+above the convex hull", report, flags=re.I)
    return float(m.group(1)) if m else None


def parse_fermi_energy(report: str) -> float | None:
    m = re.search(r"Fermi energy.*?of\s+([-+]?\d+(?:\.\d+)?)\s*eV", report, flags=re.I | re.S)
    return float(m.group(1)) if m else None


def parse_volume_claim(report: str) -> float | None:
    matches = re.findall(r"volume\s+([-+]?\d+(?:\.\d+)?)", report, flags=re.I)
    return float(matches[-1]) if matches else None


def parse_site_claims(report: str) -> dict[str, list[int]]:
    claims: dict[str, list[int]] = {}
    normalized = normalize_text(report)
    pat = re.compile(
        r"\b([A-Z][a-z]?)\s+has\s+(\d+)\s+sites?\s*:\s*(.*?)(?=(?:\s+[A-Z][a-z]?\s+has\s+\d+\s+sites?:)|(?:\.\s+Second)|(?:\.\s+Third)|$)",
        flags=re.I | re.S,
    )
    for match in pat.finditer(normalized):
        element = match.group(1)
        chunk = match.group(3)
        values = [int(x) for x in re.findall(r"one site has\s+(\d+)\s+atoms?", chunk, flags=re.I)]
        if not values:
            values = [int(x) for x in re.findall(r"site has\s+(\d+)\s+atoms?", chunk, flags=re.I)]
        claims[element] = sorted(values)
    return dict(sorted(claims.items()))


def bond_values_from_sentence(sentence: str) -> list[float]:
    values: list[float] = []
    occupied: list[tuple[int, int]] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*Å", sentence):
        occupied.append(match.span())
        values.extend([float(match.group(1)), float(match.group(2))])
    weighted_re = re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|\d+)\s+"
        r"(?:equivalent\s+)?(?:shorter|longer)\s*\((\d+(?:\.\d+)?)\s*Å\)",
        flags=re.I,
    )
    for match in weighted_re.finditer(sentence):
        occupied.append(match.span())
        count = word_to_int(match.group(1)) or 1
        values.extend([float(match.group(2))] * count)
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*Å", sentence):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        values.append(float(match.group(1)))
    return [x for x in values if 0.45 <= x <= 5.0]


def parse_bond_claims(report: str) -> dict[str, list[float]]:
    claims: dict[str, list[float]] = defaultdict(list)
    sentences = re.split(r"(?<=[.!?])\s+", normalize_text(report))
    pair_re = re.compile(r"\b([A-Z][a-z]?)-([A-Z][a-z]?)\s+bond (?:lengths?|distances?)\b", flags=re.I)
    for sentence in sentences:
        pairs = [(m.group(1), m.group(2)) for m in pair_re.finditer(sentence)]
        if not pairs:
            continue
        values = bond_values_from_sentence(sentence)
        if not values:
            continue
        for left, right in pairs:
            claims[pair_key(left, right)].extend(values)
    return {key: values for key, values in sorted(claims.items())}


def pair_periodic_distances(structure: SimpleStructure, pair: str, max_distance: float = 5.0) -> list[float]:
    left, right = pair.split("-", 1)
    values: list[float] = []
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    for i, elem_i in enumerate(structure.species):
        for j, elem_j in enumerate(structure.species):
            if pair_key(elem_i, elem_j) != pair_key(left, right):
                continue
            if i > j:
                continue
            if left == right and i == j:
                allowed_shifts = shifts[np.any(shifts != 0, axis=1)]
            elif i == j:
                continue
            else:
                allowed_shifts = shifts
            delta = structure.frac_coords[j] + allowed_shifts - structure.frac_coords[i]
            cart = delta @ structure.lattice
            distances = np.linalg.norm(cart, axis=1)
            for d in distances:
                d = float(d)
                if 0.45 <= d <= max_distance:
                    values.append(d)
    return sorted(values)


def first_shell_mean(distances: list[float]) -> tuple[float | None, list[float], float | None]:
    if not distances:
        return None, [], None
    d_min = min(distances)
    cutoff = d_min + max(0.30, 0.15 * d_min)
    shell = [x for x in distances if x <= cutoff]
    return float(np.mean(shell)), shell, cutoff


def compare_bond_claims(claims: dict[str, list[float]], structure: SimpleStructure | None) -> list[dict[str, Any]]:
    rows = []
    for pair, predicted_values in claims.items():
        predicted_mean = float(np.mean(predicted_values)) if predicted_values else None
        actual_mean = None
        actual_values: list[float] = []
        cutoff = None
        if structure is not None:
            actual_mean, actual_values, cutoff = first_shell_mean(pair_periodic_distances(structure, pair))
        abs_error = abs(predicted_mean - actual_mean) if predicted_mean is not None and actual_mean is not None else None
        rel_error = 100.0 * abs_error / actual_mean if abs_error is not None and actual_mean else None
        rows.append(
            {
                "pair": pair,
                "predicted_mean_angstrom": round_or_none(predicted_mean),
                "final_cif_mean_angstrom": round_or_none(actual_mean),
                "absolute_error_angstrom": round_or_none(abs_error),
                "relative_error_pct": round_or_none(rel_error),
                "predicted_values_angstrom": [round(float(x), 4) for x in predicted_values],
                "final_cif_first_shell_values_sample": [round(float(x), 4) for x in actual_values[:24]],
                "final_cif_first_shell_count": len(actual_values),
                "first_shell_cutoff_angstrom": round_or_none(cutoff),
                "matched_within_0_05_angstrom": bool(abs_error is not None and abs_error <= 0.05),
            }
        )
    return rows


def score_record(record: dict[str, Any]) -> float:
    score = 0.0
    if record["space_group"]["matched"]:
        score += 25.0
    if record["site_multiplicities"]["matched"]:
        score += 25.0
    if record["volume"]["relative_error_pct"] is not None:
        score += max(0.0, 20.0 - record["volume"]["relative_error_pct"])
    if record["formation_energy"]["matched"]:
        score += 10.0
    bond_rows = record["bond_lengths"]
    if bond_rows:
        errors = [x["absolute_error_angstrom"] for x in bond_rows if x["absolute_error_angstrom"] is not None]
        matched = sum(1 for x in bond_rows if x["matched_within_0_05_angstrom"])
        score += 15.0 * matched / len(bond_rows)
        if errors:
            score += max(0.0, 5.0 - 20.0 * float(np.mean(errors)))
    return round(score, 6)


def evaluate_row(row: pd.Series, row_index: int) -> dict[str, Any] | None:
    response = str(row["responses"])
    final_cif = extract_final_cif(response) or str(row.get("simple_structure", "")).strip()
    structure = parse_simple_structure(final_cif)
    if structure is None:
        return None
    report = extract_report_text(response)
    sg_claim = parse_space_group_claim(report)
    site_claims = parse_site_claims(report)
    volume_claim = parse_volume_claim(report)
    formation_pred = parse_report_formation_energy(report)
    formation_real = parse_prompt_formation_energy(row.get("prompt"))
    fermi_pred = parse_fermi_energy(report)
    bond_claims = parse_bond_claims(report)
    bond_rows = compare_bond_claims(bond_claims, structure)

    volume_rel = (
        100.0 * abs(volume_claim - structure.volume) / structure.volume
        if volume_claim is not None and structure.volume
        else None
    )
    formation_abs = (
        abs(formation_pred - formation_real)
        if formation_pred is not None and formation_real is not None
        else None
    )
    record = {
        "row_index": int(row_index),
        "mp_id": str(row.get("mp_id", "")),
        "prompt": parse_prompt_text(row.get("prompt")),
        "response": response,
        "final_cif": final_cif,
        "space_group": {
            "predicted_number": sg_claim["number"],
            "predicted_symbol": sg_claim["symbol"],
            "final_cif_number": structure.spacegroup_number,
            "final_cif_symbol": structure.spacegroup_symbol,
            "matched": bool(sg_claim["number"] == structure.spacegroup_number),
        },
        "volume": {
            "predicted": round_or_none(volume_claim),
            "final_cif": round_or_none(structure.volume),
            "relative_error_pct": round_or_none(volume_rel),
            "matched_within_1_pct": bool(volume_rel is not None and volume_rel <= 1.0),
        },
        "site_multiplicities": {
            "predicted_atoms_per_site_by_element": site_claims,
            "final_cif_atoms_per_site_by_element": structure.site_multiplicities_by_element,
            "predicted_num_sites_by_element": {k: len(v) for k, v in site_claims.items()},
            "final_cif_num_sites_by_element": {
                k: len(v) for k, v in structure.site_multiplicities_by_element.items()
            },
            "matched": bool(site_claims == structure.site_multiplicities_by_element),
        },
        "atom_counts": {
            "final_cif_atom_count_by_element": structure.atom_count_by_element,
            "final_cif_total_atoms": len(structure.species),
        },
        "bond_lengths": bond_rows,
        "formation_energy": {
            "predicted_eV_per_atom": round_or_none(formation_pred),
            "real_from_prompt_eV_per_atom": round_or_none(formation_real),
            "absolute_error_eV_per_atom": round_or_none(formation_abs),
            "matched": bool(formation_abs is not None and formation_abs <= 0.005),
        },
        "energy_above_hull": {
            "predicted_eV_per_atom": round_or_none(parse_hull_energy(report)),
            "metric_pipeline_eV_per_atom": round_or_none(row.get("energy_above_hull")),
        },
        "fermi_energy": {
            "predicted_eV": round_or_none(fermi_pred),
            "real_eV": None,
            "note": "The final CIF and conditional+thinking parquet do not contain a real Fermi energy label.",
        },
        "parquet_metrics": extract_parquet_metrics(row),
    }
    record["score"] = score_record(record)
    return record


def compact_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    bond_errors = [x["absolute_error_angstrom"] for x in record["bond_lengths"] if x["absolute_error_angstrom"] is not None]
    return {
        "rank_score": record["score"],
        "row_index": record["row_index"],
        "mp_id": record["mp_id"],
        "predicted_spacegroup_number": record["space_group"]["predicted_number"],
        "final_cif_spacegroup_number": record["space_group"]["final_cif_number"],
        "final_cif_spacegroup_symbol": record["space_group"]["final_cif_symbol"],
        "space_group_match": record["space_group"]["matched"],
        "site_match": record["site_multiplicities"]["matched"],
        "volume_relative_error_pct": record["volume"]["relative_error_pct"],
        "formation_abs_error_eV_per_atom": record["formation_energy"]["absolute_error_eV_per_atom"],
        "bond_pair_count": len(record["bond_lengths"]),
        "bond_pairs_matched_0_05A": sum(x["matched_within_0_05_angstrom"] for x in record["bond_lengths"]),
        "bond_mean_abs_error_A": round_or_none(float(np.mean(bond_errors)) if bond_errors else None),
        "predicted_fermi_eV": record["fermi_energy"]["predicted_eV"],
        "parquet_energies": record["parquet_metrics"].get("energies"),
        "parquet_energy_above_hull": record["parquet_metrics"].get("energy_above_hull"),
        "parquet_smact_validity": record["parquet_metrics"].get("smact_validity"),
        "parquet_structure_validity": record["parquet_metrics"].get("structure_validity"),
        "parquet_composition_consistency": record["parquet_metrics"].get("composition_consistency"),
        "parquet_spacegroup_consistency": record["parquet_metrics"].get("spacegroup_consistency"),
        "parquet_is_stable": record["parquet_metrics"].get("is_stable"),
        "parquet_is_novel": record["parquet_metrics"].get("is_novel"),
        "parquet_is_unique": record["parquet_metrics"].get("is_unique"),
        "parquet_stable_unique_novel": record["parquet_metrics"].get("stable_unique_novel"),
    }


def markdown_table_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(x) for x in values) + " |"


def markdown_metric_value(value: Any) -> str:
    if value is None:
        return "`None`"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = text.replace("\n", "\\n")
    if len(text) > 180:
        text = text[:177] + "..."
    return f"`{text}`"


def select_records(
    records: list[dict[str, Any]],
    *,
    top_k: int,
    distinct_spacegroups: bool,
) -> list[dict[str, Any]]:
    if not distinct_spacegroups:
        return records[:top_k]

    selected: list[dict[str, Any]] = []
    seen_spacegroups: set[int | None] = set()
    for record in records:
        spacegroup = record["space_group"]["final_cif_number"]
        if spacegroup in seen_spacegroups:
            continue
        selected.append(record)
        seen_spacegroups.add(spacegroup)
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        raise RuntimeError(
            f"Only found {len(selected)} distinct final-CIF space groups after filters; "
            f"requested top_k={top_k}."
        )
    return selected


def pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def summarize_spacegroup_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    site_matches = sum(1 for record in records if record["site_multiplicities"]["matched"])
    volume_matches = sum(1 for record in records if record["volume"]["matched_within_1_pct"])
    spacegroup_matches = sum(1 for record in records if record["space_group"]["matched"])
    volume_relative_errors = [
        record["volume"]["relative_error_pct"]
        for record in records
        if record["volume"]["relative_error_pct"] is not None
    ]
    bond_pair_count = sum(len(record["bond_lengths"]) for record in records)
    bond_pair_matches = sum(
        1
        for record in records
        for bond in record["bond_lengths"]
        if bond["matched_within_0_05_angstrom"]
    )
    bond_relative_errors = [
        bond["relative_error_pct"]
        for record in records
        for bond in record["bond_lengths"]
        if bond["relative_error_pct"] is not None
    ]
    return {
        "n_structures": n,
        "site_match_pct": pct(site_matches, n),
        "volume_match_pct": pct(volume_matches, n),
        "spacegroup_match_pct": pct(spacegroup_matches, n),
        "bond_length_match_pct": pct(bond_pair_matches, bond_pair_count),
        "volume_relative_diff_pct": mean_or_none(volume_relative_errors),
        "bond_length_relative_diff_pct": mean_or_none(bond_relative_errors),
        "site_match_count": site_matches,
        "volume_match_count": volume_matches,
        "spacegroup_match_count": spacegroup_matches,
        "bond_pair_match_count": bond_pair_matches,
        "bond_pair_count": bond_pair_count,
        "volume_relative_diff_count": len(volume_relative_errors),
        "bond_relative_diff_count": len(bond_relative_errors),
    }


def fmt_latex_pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def write_spacegroup_relative_difference_markdown(
    *,
    output_md: Path,
    source_path: Path,
    target_spacegroups: list[str],
    grouped_records: dict[str, list[dict[str, Any]]],
    all_records: list[dict[str, Any]],
) -> None:
    rows: list[tuple[str, dict[str, Any]]] = [
        (spacegroup, summarize_spacegroup_records(grouped_records.get(spacegroup, [])))
        for spacegroup in target_spacegroups
    ]
    rows.append(("All structures", summarize_spacegroup_records(all_records)))

    lines: list[str] = []
    lines.append("# Space-group relative difference table")
    lines.append("")
    lines.append(f"- Source parquet: `{source_path}`")
    lines.append(f"- Target final-CIF space groups: `{', '.join(target_spacegroups)}`")
    lines.append("- Site match: percentage of structures whose predicted atoms-per-site pattern exactly matches the final CIF equivalent-site pattern.")
    lines.append("- Volume rel. diff.: average `abs(predicted - final) / final`, reported as percent.")
    lines.append("- Spacegroup match: percentage of structures whose predicted space-group number matches the final CIF space-group number.")
    lines.append("- Bond rel. diff.: micro-average over extracted X-Y bond-length claims using `abs(predicted mean - final mean) / final mean`, reported as percent.")
    lines.append("- All structures: every row in the parquet that can be parsed and evaluated, not only the target space groups.")
    lines.append("")
    lines.append(markdown_table_row([
        "space group",
        "n structures",
        "site match",
        "volume rel. diff.",
        "spacegroup match",
        "bond rel. diff.",
        "bond pairs",
    ]))
    lines.append(markdown_table_row(["---", "---", "---", "---", "---", "---", "---"]))
    for label, summary in rows:
        lines.append(markdown_table_row([
            label,
            summary["n_structures"],
            f"{fmt_pct(summary['site_match_pct'])} ({summary['site_match_count']}/{summary['n_structures']})",
            f"{fmt_pct(summary['volume_relative_diff_pct'])} (n={summary['volume_relative_diff_count']})",
            f"{fmt_pct(summary['spacegroup_match_pct'])} ({summary['spacegroup_match_count']}/{summary['n_structures']})",
            f"{fmt_pct(summary['bond_length_relative_diff_pct'])} (n={summary['bond_relative_diff_count']})",
            summary["bond_pair_count"],
        ]))
    lines.append("")
    lines.append("## LaTeX table")
    lines.append("")
    lines.append("```latex")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Thinking-trace consistency by final-CIF space group. Site and space-group columns report match percentages; volume and bond columns report average relative differences.}")
    lines.append(r"\label{tab:thinking_trace_relative_difference}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Space group & $N$ & Site match (\%) & Volume rel. diff. (\%) & SG match (\%) & Bond rel. diff. (\%) & Bond pairs \\")
    lines.append(r"\midrule")
    for label, summary in rows:
        lines.append(
            f"{latex_escape(label)} & "
            f"{summary['n_structures']} & "
            f"{fmt_latex_pct(summary['site_match_pct'])} & "
            f"{fmt_latex_pct(summary['volume_relative_diff_pct'])} & "
            f"{fmt_latex_pct(summary['spacegroup_match_pct'])} & "
            f"{fmt_latex_pct(summary['bond_length_relative_diff_pct'])} & "
            f"{summary['bond_pair_count']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("```")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(records: list[dict[str, Any]], investigation_records: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Thinking trace 与最终 CIF 一致性最佳样本")
    lines.append("")
    lines.append("## response 列格式规则")
    lines.append("")
    lines.append("我抽查了前 5 个样本的 `response`，其格式基本一致：先输出 `## Material Report`，其中 `### Crystal Structure` 包含空间群、各元素等价 site 数量与每个 site 的原子数、局部配位和 X-Y 键长；随后 `### Stability` 给出 hull energy 与 formation energy；`### Electronic Properties` 给出 band gap 判定和 Fermi energy；最后用 `<CIF>...</CIF>` 包住最终生成的简化 CIF。")
    lines.append("")
    lines.append("脚本把这些自然语言字段硬编码为正则解析规则：`space group ... (id N)`、`X has n sites: one site has m atoms`、`volume V`、`formation energy per atom is E`、`Fermi energy ... of E eV`，以及包含 `X-Y bond length(s)/distance(s)` 的句子。对于 `shorter/longer` 和 `range` 的键长描述，脚本会保留加权数值或区间端点，再计算预测均值。")
    lines.append("")
    lines.append("最终 CIF 用 ASE 解析晶格、用 spglib 识别空间群和等价原子，体积直接由晶格行列式计算。X-Y 真实键长由周期镜像距离得到，取该元素对首近邻 shell 的平均值与 thinking trace 的 X-Y 预测均值比较。")
    lines.append("")
    lines.append("注意：当前 `conditional+thinking.parquet` 与最终 CIF 不包含真实 Fermi energy 标签，因此 Fermi energy 只能记录 thinking trace 的预测值，不能在这个数据源内计算真实误差。formation energy 的真实值来自 prompt。")
    lines.append("")
    lines.append("## 前 5 个样本抽查摘要")
    lines.append("")
    lines.append(markdown_table_row(["row", "mp_id", "SG预测/真实", "体积预测/真实", "site是否匹配", "键长pair数", "Fermi预测"]))
    lines.append(markdown_table_row(["---", "---", "---", "---", "---", "---", "---"]))
    for rec in investigation_records:
        lines.append(
            markdown_table_row(
                [
                    rec["row_index"],
                    rec["mp_id"],
                    f'{rec["space_group"]["predicted_number"]}/{rec["space_group"]["final_cif_number"]}',
                    f'{rec["volume"]["predicted"]}/{rec["volume"]["final_cif"]}',
                    rec["site_multiplicities"]["matched"],
                    len(rec["bond_lengths"]),
                    rec["fermi_energy"]["predicted_eV"],
                ]
            )
        )
    lines.append("")
    lines.append("## 选出的 3 个最佳样本")
    lines.append("")
    for rank, rec in enumerate(records, start=1):
        lines.append(f"### 样本 {rank}: `{rec['mp_id']}` / row `{rec['row_index']}`")
        lines.append("")
        lines.append(f"- 总分：`{rec['score']}`")
        lines.append(f"- 空间群：预测 `{rec['space_group']['predicted_symbol']}` / `{rec['space_group']['predicted_number']}`；最终 CIF `{rec['space_group']['final_cif_symbol']}` / `{rec['space_group']['final_cif_number']}`；匹配 `{rec['space_group']['matched']}`")
        lines.append(f"- 体积：预测 `{rec['volume']['predicted']}`；最终 CIF `{rec['volume']['final_cif']}`；相对误差 `{rec['volume']['relative_error_pct']}%`")
        lines.append(f"- site：预测 `{rec['site_multiplicities']['predicted_atoms_per_site_by_element']}`；最终 CIF `{rec['site_multiplicities']['final_cif_atoms_per_site_by_element']}`；匹配 `{rec['site_multiplicities']['matched']}`")
        lines.append(f"- formation energy：预测 `{rec['formation_energy']['predicted_eV_per_atom']}` eV/atom；真实 prompt `{rec['formation_energy']['real_from_prompt_eV_per_atom']}` eV/atom；误差 `{rec['formation_energy']['absolute_error_eV_per_atom']}`")
        lines.append(f"- Fermi energy：预测 `{rec['fermi_energy']['predicted_eV']}` eV；真实值 `{rec['fermi_energy']['real_eV']}`；说明 `{rec['fermi_energy']['note']}`")
        lines.append("")
        lines.append("键长逐 pair 对比：")
        lines.append("")
        lines.append(markdown_table_row(["pair", "预测均值 Å", "最终 CIF 均值 Å", "绝对误差 Å", "是否 <=0.05 Å"]))
        lines.append(markdown_table_row(["---", "---", "---", "---", "---"]))
        for bond in rec["bond_lengths"]:
            lines.append(
                markdown_table_row(
                    [
                        bond["pair"],
                        bond["predicted_mean_angstrom"],
                        bond["final_cif_mean_angstrom"],
                        bond["absolute_error_angstrom"],
                        bond["matched_within_0_05_angstrom"],
                    ]
                )
            )
        lines.append("")
        lines.append("最终简化 CIF：")
        lines.append("")
        lines.append("```text")
        lines.append(rec["final_cif"])
        lines.append("```")
        lines.append("")
        sym_cif = rec.get("symmetrized_real_cif", {})
        if sym_cif.get("cif"):
            lines.append("Symmetrized real CIF：")
            lines.append("")
            lines.append(f"- 生成器：`{sym_cif.get('generator')}`")
            lines.append(f"- 说明：{sym_cif.get('note')}")
            lines.append("")
            lines.append("```cif")
            lines.append(str(sym_cif["cif"]).strip())
            lines.append("```")
            lines.append("")
        lines.append("Parquet metrics：")
        lines.append("")
        lines.append(markdown_table_row(["metric", "value"]))
        lines.append(markdown_table_row(["---", "---"]))
        metrics = rec.get("parquet_metrics", {})
        for metric_name in PARQUET_METRIC_COLUMNS:
            if metric_name == "relaxed_structures":
                continue
            if metric_name in metrics:
                lines.append(markdown_table_row([metric_name, markdown_metric_value(metrics[metric_name])]))
        lines.append("")
        if metrics.get("relaxed_structures"):
            lines.append("`relaxed_structures`：")
            lines.append("")
            lines.append("```text")
            lines.append(str(metrics["relaxed_structures"]).strip())
            lines.append("```")
            lines.append("")
        lines.append("完整 response 已保存在 `best_thinking_consistency_cases.json`。")
        lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

from tqdm import tqdm


def run_find(args: argparse.Namespace) -> None:

    df = pd.read_parquet(args.input)
    if args.scan_limit is not None:
        df = df.head(args.scan_limit)

    records: list[dict[str, Any]] = []
    investigation_records: list[dict[str, Any]] = []
    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating records"):
        record = evaluate_row(row, int(row_index))
        if record is None:
            continue
        if (
            args.min_atoms is not None
            and record["atom_counts"]["final_cif_total_atoms"] <= args.min_atoms
        ):
            continue
        if len(investigation_records) < 5:
            investigation_records.append(record)
        records.append(record)

    if not records:
        raise RuntimeError("No parseable records found after applying filters.")

    records = sorted(
        records,
        key=lambda x: (
            x["score"],
            x["space_group"]["matched"],
            x["site_multiplicities"]["matched"],
            -(x["volume"]["relative_error_pct"] or 9999.0),
        ),
        reverse=True,
    )
    selected = select_records(
        records,
        top_k=args.top_k,
        distinct_spacegroups=args.distinct_spacegroups,
    )
    for record in selected:
        record["symmetrized_real_cif"] = build_symmetrized_cif(str(record["final_cif"]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "source": str(args.input),
                "selection_rule": "highest internal-consistency score over thinking response vs final simple CIF",
                "filters": {
                    "final_cif_total_atoms_gt": args.min_atoms,
                    "scan_limit": args.scan_limit,
                    "distinct_final_cif_spacegroups": args.distinct_spacegroups,
                },
                "investigated_first_5_records": investigation_records,
                "selected_records": selected,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame([compact_csv_row(x) for x in selected]).to_csv(OUTPUT_CSV, index=False)
    write_markdown(selected, investigation_records)

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")
    for idx, rec in enumerate(selected, start=1):
        print(
            f"#{idx}: row={rec['row_index']} mp_id={rec['mp_id']} score={rec['score']} "
            f"atoms={rec['atom_counts']['final_cif_total_atoms']} "
            f"sg={rec['space_group']['final_cif_symbol']}({rec['space_group']['final_cif_number']}) "
            f"sg_match={rec['space_group']['matched']} site_match={rec['site_multiplicities']['matched']}"
        )


def run_export_cifs(args: argparse.Namespace) -> None:
    written = export_selected_cif_files(
        json_path=args.json,
        output_dir=args.output_dir,
        regenerate=args.regenerate,
    )
    print(f"Wrote {len(written)} CIF files to {args.output_dir}")
    for path in written:
        print(path)


def run_summarize_spacegroups(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.input)
    target_spacegroups = list(args.spacegroups)
    target_set = set(target_spacegroups)
    grouped_records: dict[str, list[dict[str, Any]]] = {
        spacegroup: [] for spacegroup in target_spacegroups
    }
    all_records: list[dict[str, Any]] = []

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating records"):
        record = evaluate_row(row, int(row_index))
        if record is None:
            continue
        all_records.append(record)
        final_symbol = record["space_group"]["final_cif_symbol"]
        if final_symbol not in target_set:
            continue
        grouped_records[str(final_symbol)].append(record)

    write_spacegroup_relative_difference_markdown(
        output_md=args.output_md,
        source_path=args.input,
        target_spacegroups=target_spacegroups,
        grouped_records=grouped_records,
        all_records=all_records,
    )
    print(f"Wrote {args.output_md}")
    summary_rows = [
        (spacegroup, summarize_spacegroup_records(grouped_records[spacegroup]))
        for spacegroup in target_spacegroups
    ]
    summary_rows.append(("All structures", summarize_spacegroup_records(all_records)))
    for label, summary in summary_rows:
        print(
            f"{label}: n={summary['n_structures']} "
            f"site={fmt_pct(summary['site_match_pct'])} "
            f"volume_rel_diff={fmt_pct(summary['volume_relative_diff_pct'])} "
            f"spacegroup={fmt_pct(summary['spacegroup_match_pct'])} "
            f"bond_rel_diff={fmt_pct(summary['bond_length_relative_diff_pct'])}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    find_parser = subparsers.add_parser(
        "find",
        help="Scan the parquet, select case-study records, and write JSON/CSV/Markdown.",
    )
    find_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    find_parser.add_argument("--top-k", type=int, default=3)
    find_parser.add_argument("--scan-limit", type=int, default=None)
    find_parser.add_argument(
        "--min-atoms",
        type=int,
        default=None,
        help="Keep only structures whose final CIF total atom count is strictly greater than this value.",
    )
    find_parser.add_argument(
        "--distinct-spacegroups",
        action="store_true",
        help="Select at most one structure per final-CIF space group number.",
    )
    find_parser.set_defaults(func=run_find)

    export_parser = subparsers.add_parser(
        "export-cifs",
        help="Read selected records from JSON and write one real CIF file per selected record.",
    )
    export_parser.add_argument("--json", type=Path, default=OUTPUT_JSON)
    export_parser.add_argument("--output-dir", type=Path, default=OUTPUT_CIF_DIR)
    export_parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate symmetrized CIFs from final_cif instead of using CIF text stored in JSON.",
    )
    export_parser.set_defaults(func=run_export_cifs)

    summary_parser = subparsers.add_parser(
        "summarize-spacegroups",
        help="Compute consistency and relative-difference summaries for selected final-CIF space groups.",
    )
    summary_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    summary_parser.add_argument("--output-md", type=Path, default=OUTPUT_SPACEGROUP_RELATIVE_DIFF_MD)
    summary_parser.add_argument(
        "--spacegroups",
        nargs="+",
        default=["Fm-3m", "Fd-3m", "P3m1"],
        help="Final-CIF space-group symbols to include.",
    )
    summary_parser.set_defaults(func=run_summarize_spacegroups)

    return parser


def main() -> None:
    parser = build_arg_parser()
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["find", *argv]
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["find"])
    args.func(args)


if __name__ == "__main__":
    main()
