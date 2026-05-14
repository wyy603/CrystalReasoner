"""
Backend for EXPERIMENT §1.3: compute per-row and aggregated tables only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shelve
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.groups import SpaceGroup

from crysreas.data.prompt_generator.get_info import get_info

from .common import (
    ComparePaths,
    EXPERIMENT_DIR,
    crystal_section,
    ensure_thinking_dirs,
    median_min_rel_pct_error,
    parse_bond_lengths_angstrom,
    parse_instruction_spacegroup_id,
    parse_prompt_text,
    parse_simple_structure,
    parse_space_group_id,
    parse_volume,
    split_thinking_and_tail,
    structure_bond_pool,
    structure_spacegroup_id,
    structure_volume,
)

DATA_1_3_PER_ROW = "thinking_exp_1_3_per_row.parquet"
DATA_1_3_SG_BAR = "thinking_exp_1_3_spacegroup_bar.parquet"
DATA_1_3_PREDICTION_EXAMPLES = "thinking_exp_1_3_prediction_examples.parquet"
DATA_1_3_CONSISTENCY_V2_SUMMARY = "thinking_exp_1_3_consistency_v2_summary.json"

TARGET_SG_SYMBOLS = ["P1", "C2/c", "Amm2", "I4m2", "P3", "P6_3/mmc", "F-43m"]
THINKING_COLUMNS = ["mp_id", "prompt", "responses", "simple_structure"]
DEFAULT_DB_PATH = EXPERIMENT_DIR.parent.parent / "assets" / "MP" / "MP_shelve"
PROMPT_TYPE = "conditional+thinking"
CONSISTENCY_SUBSET_SIZE = 128
RANDOM_EXAMPLE_COUNT = 3
RANDOM_EXAMPLE_SEED = 20260505


def _symbol_to_id(symbol: str) -> int:
    manual = {"I4m2": 119}
    if symbol in manual:
        return manual[symbol]
    return int(SpaceGroup(symbol).int_number)


def _extract_final_cif(response: object) -> str | None:
    if not isinstance(response, str):
        return None
    match = re.search(r"<CIF>(.*?)</CIF>", response, flags=re.S)
    if match:
        return match.group(1).strip()
    _, tail = split_thinking_and_tail(response)
    if not tail:
        return None
    tail = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", tail).strip()
    for prefix in ("## CIF File", "CIF File:"):
        if tail.startswith(prefix):
            return tail[len(prefix) :].strip()
    return tail


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _round_list(values: list[float], *, limit: int = 8, digits: int = 6) -> list[float]:
    return [round(float(x), digits) for x in values[:limit]]


def _pair_key(left: str, right: str) -> str:
    return "-".join(sorted([str(left), str(right)]))


def _site_element_symbol(site: object) -> str | None:
    try:
        return str(site.specie.symbol)
    except Exception:
        return None


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
    "twelve": 12,
}


def _word_to_int(text: str) -> int | None:
    text = str(text).strip().lower()
    if text.isdigit():
        return int(text)
    return NUMBER_WORDS.get(text)


SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUPERSCRIPT_CHARS_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+")
ELEMENT_TOKEN_RE = r"([A-Z][a-z]?)[^A-Za-z0-9]{0,8}"
COUNT_TOKEN_RE = r"(one|two|three|four|five|six|seven|eight|nine|ten|twelve|\d+)"
POLYHEDRA_WORD_RE = r"(?:octahedra|octahedral|tetrahedra|tetrahedral|cuboctahedra|polyhedra|geometry)"


def _normalize_claim_text(text: object) -> str:
    normalized = str(text).translate(SUBSCRIPT_DIGITS)
    return SUPERSCRIPT_CHARS_RE.sub("", normalized)


def _first_shell_counts_for_pair(
    structure: Structure | None,
    center_element: str,
    neighbor_element: str,
    *,
    radius: float = 8.0,
) -> tuple[int | None, list[int], float | None, float | None]:
    if structure is None:
        return None, [], None, None
    center_indices = [
        i for i, site in enumerate(structure) if _site_element_symbol(site) == center_element
    ]
    if not center_indices:
        return None, [], None, None

    neighbor_sets: list[list[float]] = []
    all_distances: list[float] = []
    for i in center_indices:
        distances: list[float] = []
        try:
            neighbors = structure.get_neighbors(structure[i], radius)
        except Exception:
            neighbors = []
        for neighbor in neighbors:
            if _site_element_symbol(neighbor) != neighbor_element:
                continue
            distance = float(getattr(neighbor, "nn_distance", 0.0))
            if distance <= 1e-6:
                continue
            distances.append(distance)
            all_distances.append(distance)
        neighbor_sets.append(distances)

    if not all_distances:
        return None, [0 for _ in center_indices], None, None
    d_min = min(all_distances)
    cutoff = d_min + max(0.30, 0.15 * d_min)
    counts = [sum(1 for distance in distances if distance <= cutoff) for distances in neighbor_sets]
    if not counts:
        return None, [], d_min, cutoff
    mode_count = Counter(counts).most_common(1)[0][0]
    return int(mode_count), [int(x) for x in counts], float(d_min), float(cutoff)


def _first_shell_total_counts(
    structure: Structure | None,
    center_element: str,
    *,
    radius: float = 8.0,
) -> tuple[int | None, list[int], float | None, float | None]:
    if structure is None:
        return None, [], None, None
    center_indices = [
        i for i, site in enumerate(structure) if _site_element_symbol(site) == center_element
    ]
    if not center_indices:
        return None, [], None, None

    neighbor_sets: list[list[float]] = []
    all_distances: list[float] = []
    for i in center_indices:
        distances: list[float] = []
        try:
            neighbors = structure.get_neighbors(structure[i], radius)
        except Exception:
            neighbors = []
        for neighbor in neighbors:
            distance = float(getattr(neighbor, "nn_distance", 0.0))
            if distance <= 1e-6:
                continue
            distances.append(distance)
            all_distances.append(distance)
        neighbor_sets.append(distances)

    if not all_distances:
        return None, [0 for _ in center_indices], None, None
    d_min = min(all_distances)
    cutoff = d_min + max(0.30, 0.15 * d_min)
    counts = [sum(1 for distance in distances if distance <= cutoff) for distances in neighbor_sets]
    mode_count = Counter(counts).most_common(1)[0][0]
    return int(mode_count), [int(x) for x in counts], float(d_min), float(cutoff)


def _parse_coordination_claims(text: object) -> list[dict[str, object]]:
    normalized = _normalize_claim_text(text)
    claims: list[dict[str, object]] = []
    seen: set[tuple[str, str | None, int, str]] = set()

    bonded_pair_re = re.compile(
        rf"\b{ELEMENT_TOKEN_RE}is\s+bonded(?:\s+in\s+a\s+{COUNT_TOKEN_RE}-coordinate\s+geometry)?"
        rf"\s+to\s+{COUNT_TOKEN_RE}(?:\s+equivalent)?\s+{ELEMENT_TOKEN_RE}atoms?\b",
        flags=re.I,
    )
    for match in bonded_pair_re.finditer(normalized):
        center = match.group(1)
        count_text = match.group(3)
        neighbor = match.group(4)
        count = _word_to_int(count_text)
        if count is None:
            continue
        key = (center, neighbor, count, "bonded_to")
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            {
                "source": match.group(0).strip(),
                "center_element": center,
                "neighbor_element": neighbor,
                "pair": _pair_key(center, neighbor),
                "predicted_count": count,
                "claim_type": "bonded_to",
            }
        )

    coordinate_re = re.compile(
        rf"\b{ELEMENT_TOKEN_RE}is\s+bonded\s+in\s+a\s+{COUNT_TOKEN_RE}-coordinate\s+geometry\b",
        flags=re.I,
    )
    for match in coordinate_re.finditer(normalized):
        center = match.group(1)
        count = _word_to_int(match.group(2))
        if count is None:
            continue
        key = (center, None, count, "coordinate_geometry")
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            {
                "source": match.group(0).strip(),
                "center_element": center,
                "neighbor_element": None,
                "pair": None,
                "predicted_count": count,
                "claim_type": "coordinate_geometry",
            }
        )

    polyhedra_re = re.compile(
        rf"\b([A-Z][a-z]?)([A-Z][a-z]?)(\d+)\s+{POLYHEDRA_WORD_RE}\b",
        flags=re.I,
    )
    for match in polyhedra_re.finditer(normalized):
        center = match.group(1)
        neighbor = match.group(2)
        try:
            count = int(match.group(3))
        except ValueError:
            continue
        if center == neighbor:
            continue
        key = (center, neighbor, count, "polyhedra")
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            {
                "source": match.group(0).strip(),
                "center_element": center,
                "neighbor_element": neighbor,
                "pair": _pair_key(center, neighbor),
                "predicted_count": count,
                "claim_type": "polyhedra",
            }
        )
    return claims


def _coordination_consistency(text: object, structure: Structure | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for claim in _parse_coordination_claims(text):
        center = str(claim["center_element"])
        neighbor = claim.get("neighbor_element")
        predicted_count = int(claim["predicted_count"])
        if neighbor:
            actual_count, counts, d_min, cutoff = _first_shell_counts_for_pair(structure, center, str(neighbor))
        else:
            actual_count, counts, d_min, cutoff = _first_shell_total_counts(structure, center)
        rows.append(
            {
                **claim,
                "actual_count": actual_count,
                "exact_match": bool(actual_count == predicted_count) if actual_count is not None else False,
                "absolute_count_difference": (
                    abs(predicted_count - actual_count) if actual_count is not None else None
                ),
                "actual_counts_by_center_site_sample": counts[:12],
                "first_shell_min_distance_angstrom": _round_float(d_min),
                "first_shell_cutoff_angstrom": _round_float(cutoff),
            }
        )
    return rows


def _mentioned_element_pairs(text: object) -> list[str]:
    normalized = _normalize_claim_text(text)
    pairs: set[str] = set()
    for claim in _parse_coordination_claims(normalized):
        pair = claim.get("pair")
        if pair:
            pairs.add(str(pair))
    for left, right in re.findall(r"\b([A-Z][a-z]?)-([A-Z][a-z]?)\s+bond\b", normalized, flags=re.I):
        if left != right:
            pairs.add(_pair_key(left, right))
    return sorted(pairs)


def _present_first_shell_pairs(structure: Structure | None) -> list[str]:
    if structure is None:
        return []
    elements = sorted({elem for site in structure if (elem := _site_element_symbol(site))})
    pairs: set[str] = set()
    for i, left in enumerate(elements):
        for right in elements[i:]:
            actual_count, _, _, _ = _first_shell_counts_for_pair(structure, left, right)
            if actual_count is not None and actual_count > 0:
                pairs.add(_pair_key(left, right))
            if left != right:
                reverse_count, _, _, _ = _first_shell_counts_for_pair(structure, right, left)
                if reverse_count is not None and reverse_count > 0:
                    pairs.add(_pair_key(left, right))
    return sorted(pairs)


def _volume_consistency(claimed: float | None, actual: float | None) -> dict[str, object]:
    abs_error = abs(claimed - actual) if claimed is not None and actual is not None else None
    rel_error = (100.0 * abs_error / actual) if abs_error is not None and actual else None
    return {
        "claimed_volume": _round_float(claimed),
        "final_cif_volume": _round_float(actual),
        "absolute_error": _round_float(abs_error),
        "relative_error_pct": _round_float(rel_error),
        "matched": bool(rel_error <= 10.0) if rel_error is not None else False,
    }


def _element_pair_presence_consistency(text: object, structure: Structure | None) -> dict[str, object]:
    mentioned = _mentioned_element_pairs(text)
    present = _present_first_shell_pairs(structure)
    present_set = set(present)
    matched = [pair for pair in mentioned if pair in present_set]
    missing = [pair for pair in mentioned if pair not in present_set]
    precision = (len(matched) / len(mentioned)) if mentioned else None
    return {
        "mentioned_pairs": mentioned,
        "present_pairs": present,
        "matched_pairs": matched,
        "missing_pairs": missing,
        "pair_presence_precision": _round_float(precision),
    }


def _consistency_v2(
    text: object,
    structure: Structure | None,
    volume_claimed: float | None,
    volume_structure: float | None,
) -> dict[str, object]:
    return {
        "volume_consistency": _volume_consistency(volume_claimed, volume_structure),
        "coordination_consistency": _coordination_consistency(text, structure),
        "element_pair_presence_consistency": _element_pair_presence_consistency(text, structure),
    }


def _bond_values_from_sentence(sentence: str) -> list[float]:
    values: list[float] = []
    occupied_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*Å", sentence):
        occupied_spans.append(match.span())
        for group in (1, 2):
            try:
                value = float(match.group(group))
            except ValueError:
                continue
            if 0.45 <= value <= 5.0:
                values.append(value)

    weighted_re = re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twelve|\d+)\s+"
        r"(?:equivalent\s+)?(?:shorter|longer)\s*\((\d+(?:\.\d+)?)\s*Å\)",
        flags=re.I,
    )
    for match in weighted_re.finditer(sentence):
        occupied_spans.append(match.span())
        count = _word_to_int(match.group(1)) or 1
        value = float(match.group(2))
        if 0.45 <= value <= 5.0:
            values.extend([value] * count)

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*Å", sentence):
        if any(start <= match.start() < end for start, end in occupied_spans):
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if 0.45 <= value <= 5.0:
            values.append(value)
    return values


def _parse_xy_bond_length_claims(text: str) -> dict[str, list[float]]:
    claims: dict[str, list[float]] = {}
    pair_re = re.compile(r"\b([A-Z][a-z]?)-([A-Z][a-z]?)\s+bond (?:lengths?|distances?)\b", flags=re.I)
    sentences = re.split(r"(?<=[.!?])\s+", str(text))
    for sentence in sentences:
        pairs = [(m.group(1), m.group(2)) for m in pair_re.finditer(sentence)]
        if not pairs:
            continue
        values = _bond_values_from_sentence(sentence)
        if not values:
            continue
        for left, right in pairs:
            if left == right:
                continue
            claims.setdefault(_pair_key(left, right), []).extend(values)
    return claims


def _structure_xy_distances(
    structure: Structure | None,
    pair: str,
    *,
    target_count: int,
) -> tuple[list[float], str]:
    if structure is None:
        return [], "missing_structure"
    try:
        left, right = pair.split("-", 1)
    except ValueError:
        return [], "invalid_pair"
    all_distances: list[float] = []
    for i, site_i in enumerate(structure):
        elem_i = _site_element_symbol(site_i)
        if elem_i is None:
            continue
        for j in range(i + 1, len(structure)):
            elem_j = _site_element_symbol(structure[j])
            if elem_j is None or _pair_key(elem_i, elem_j) != _pair_key(left, right):
                continue
            try:
                distance = float(structure.get_distance(i, j))
            except Exception:
                continue
            if distance <= 0:
                continue
            all_distances.append(distance)
    if all_distances:
        n = max(1, min(int(target_count), len(all_distances)))
        return sorted(all_distances)[:n], f"nearest_{n}_{pair}_distances"
    return [], "pair_not_found"


def _xy_bond_length_comparison(text: str, structure: Structure | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair, predicted_values in sorted(_parse_xy_bond_length_claims(text).items()):
        actual_values, actual_source = _structure_xy_distances(
            structure,
            pair,
            target_count=len(predicted_values),
        )
        predicted_mean = sum(predicted_values) / len(predicted_values)
        actual_mean = sum(actual_values) / len(actual_values) if actual_values else None
        abs_error = abs(predicted_mean - actual_mean) if actual_mean is not None else None
        rel_error = (100.0 * abs_error / actual_mean) if abs_error is not None and actual_mean else None
        rows.append(
            {
                "pair": pair,
                "predicted_mean_angstrom": _round_float(predicted_mean),
                "predicted_values_angstrom": _round_list(predicted_values, limit=20),
                "predicted_count": len(predicted_values),
                "actual_mean_angstrom": _round_float(actual_mean),
                "actual_values_angstrom_sample": _round_list(actual_values, limit=20),
                "actual_count": len(actual_values),
                "actual_source": actual_source,
                "absolute_error_angstrom": _round_float(abs_error),
                "relative_error_pct": _round_float(rel_error),
            }
        )
    return rows


def _bond_matches(predicted: list[float], actual: list[float], *, limit: int = 8) -> list[dict[str, float]]:
    matches: list[dict[str, float]] = []
    for pred in predicted[:limit]:
        candidates = [x for x in actual if x > 0]
        if not candidates:
            break
        real = min(candidates, key=lambda x: abs(float(pred) - float(x)))
        matches.append(
            {
                "predicted_angstrom": round(float(pred), 6),
                "real_nearest_angstrom": round(float(real), 6),
                "relative_error_pct": round(100.0 * abs(float(pred) - float(real)) / float(real), 6),
            }
        )
    return matches


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_thinking_frame(paths: ComparePaths) -> pd.DataFrame:
    return pd.read_parquet(paths.thinking_parquet, columns=THINKING_COLUMNS)


def _normalize_for_similarity(text: object) -> str:
    return " ".join(str(text).split())


def _similarity_ratio(left: object, right: object, *, quick: bool = False) -> float:
    matcher = SequenceMatcher(None, _normalize_for_similarity(left), _normalize_for_similarity(right))
    score = matcher.quick_ratio() if quick else matcher.ratio()
    return float(score)


def _ground_truth_response(mp_id: str, db: shelve.Shelf | None) -> str | None:
    if db is None or mp_id not in db:
        return None
    try:
        return str(get_info(db[mp_id], PROMPT_TYPE, seed=mp_id[3:])["answer"])
    except Exception:
        return None


def _attach_ground_truth_responses(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    try:
        db = shelve.open(str(DEFAULT_DB_PATH), flag="r")
    except Exception:
        db = None
    cache: dict[str, str | None] = {}
    try:
        for row in rows:
            mp_id = str(row["mp_id"])
            if mp_id not in cache:
                cache[mp_id] = _ground_truth_response(mp_id, db)
            row["ground_truth_response"] = cache[mp_id]
    finally:
        if db is not None:
            db.close()
    return rows


def _first_unique_mp_id_rows(
    rows: list[dict[str, object]],
    subset_size: int = CONSISTENCY_SUBSET_SIZE,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        mp_id = str(row["mp_id"])
        if mp_id in seen:
            continue
        selected.append(row)
        seen.add(mp_id)
        if len(selected) >= subset_size:
            break
    return selected


def _attach_similarity_scores(rows: list[dict[str, object]], subset_size: int) -> list[dict[str, object]]:
    rows = _attach_ground_truth_responses(rows)
    for row in rows:
        ground_truth = row.get("ground_truth_response")
        row["response_ground_truth_similarity_subset_size"] = subset_size
        row["response_ground_truth_similarity"] = _round_float(
            _similarity_ratio(row["response"], ground_truth, quick=False), digits=8
        ) if ground_truth else None
    return rows


def _prepare_consistency_row(row: dict[str, object]) -> dict[str, object]:
    out = dict(row)
    structure = parse_simple_structure(str(out["final_cif"]))
    consistency_v2 = _consistency_v2(
        out.get("crystal_text"),
        structure,
        out.get("volume_claimed"),
        out.get("volume_structure"),
    )
    out["consistency_v2"] = consistency_v2
    out["consistency_v2_json"] = _json_dumps(consistency_v2)
    return out


def _summarize_consistency_subset(rows: list[dict[str, object]]) -> dict[str, object]:
    total_claims = 0
    exact_claims = 0
    per_structure_rates: list[float] = []
    volume_matches: list[bool] = []
    pair_precisions: list[float] = []
    summary_rows: list[dict[str, object]] = []

    for row in rows:
        consistency_v2 = row["consistency_v2"]
        coordination = consistency_v2["coordination_consistency"]
        exact_count = sum(1 for item in coordination if item.get("exact_match"))
        claim_count = len(coordination)
        if claim_count:
            total_claims += claim_count
            exact_claims += exact_count
            per_structure_rates.append(exact_count / claim_count)

        volume_match = bool(consistency_v2["volume_consistency"].get("matched"))
        volume_matches.append(volume_match)

        pair_precision = consistency_v2["element_pair_presence_consistency"].get("pair_presence_precision")
        if pair_precision is not None:
            pair_precisions.append(float(pair_precision))

        summary_rows.append(
            {
                "mp_id": row["mp_id"],
                "coordination_claim_count": claim_count,
                "coordination_exact_match_count": exact_count,
                "coordination_exact_match_rate": _round_float(exact_count / claim_count) if claim_count else None,
                "volume_match": volume_match,
                "volume_relative_error_pct": consistency_v2["volume_consistency"].get("relative_error_pct"),
                "pair_presence_precision": pair_precision,
            }
        )

    return {
        "subset_size_requested": CONSISTENCY_SUBSET_SIZE,
        "subset_size_actual": len(rows),
        "unique_mp_id_count": len({str(row["mp_id"]) for row in rows}),
        "random_seed": RANDOM_EXAMPLE_SEED,
        "coordination_total_claims": total_claims,
        "coordination_exact_match_claims": exact_claims,
        "coordination_exact_match_rate_micro": _round_float(exact_claims / total_claims)
        if total_claims
        else None,
        "coordination_exact_match_rate_macro": _round_float(
            sum(per_structure_rates) / len(per_structure_rates)
        )
        if per_structure_rates
        else None,
        "structures_with_coordination_claims": len(per_structure_rates),
        "volume_match_rate": _round_float(sum(volume_matches) / len(volume_matches)) if volume_matches else None,
        "volume_match_count": int(sum(volume_matches)),
        "pair_presence_precision_macro": _round_float(sum(pair_precisions) / len(pair_precisions))
        if pair_precisions
        else None,
        "rows": summary_rows,
    }


def _select_random_examples(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(rows) <= RANDOM_EXAMPLE_COUNT:
        selected = list(rows)
    else:
        selected = random.Random(RANDOM_EXAMPLE_SEED).sample(rows, RANDOM_EXAMPLE_COUNT)

    examples: list[dict[str, object]] = []
    for idx, row in enumerate(selected, start=1):
        out = dict(row)
        out["response_ground_truth_similarity_rank"] = f"random_{idx}"
        out["selection_method"] = "random_from_128_unique_mp_id_subset"
        out["selection_seed"] = RANDOM_EXAMPLE_SEED
        out["consistency_subset_size"] = len(rows)
        examples.append(out)
    return examples


def run(paths: ComparePaths) -> None:
    ensure_thinking_dirs(data_dir=paths.data_dir)
    df_think = _load_thinking_frame(paths)

    rows: list[dict[str, object]] = []
    prediction_examples: list[dict[str, object]] = []
    for mp_id, prompt, response, simple in zip(
        df_think["mp_id"].astype(str).tolist(),
        df_think["prompt"].tolist(),
        df_think["responses"].tolist(),
        df_think["simple_structure"].tolist(),
        strict=True,
    ):
        prompt_text = parse_prompt_text(prompt)
        thinking, _ = split_thinking_and_tail(response)
        crystal = crystal_section(thinking)
        final_cif = _extract_final_cif(response)
        structure = parse_simple_structure(final_cif) if final_cif else parse_simple_structure(str(simple))
        n_atoms = int(len(structure)) if structure is not None else None

        sg_instruction = parse_instruction_spacegroup_id(prompt)
        sg_claimed = parse_space_group_id(crystal)
        sg_structure = structure_spacegroup_id(structure)

        volume_claimed = parse_volume(crystal)
        volume_structure = structure_volume(structure)
        volume_rel_abs_err = None
        volume_rel_diff_pct = None
        if volume_claimed is not None and volume_structure is not None and volume_structure > 0:
            volume_rel_abs_err = float(abs(volume_claimed - volume_structure) / volume_structure)
            volume_rel_diff_pct = float(volume_rel_abs_err * 100.0)

        bond_claims = parse_bond_lengths_angstrom(crystal)
        bond_pool = structure_bond_pool(structure)
        bond_median_min_rel_pct = median_min_rel_pct_error(bond_claims, bond_pool)

        rows.append(
            {
                "mp_id": mp_id,
                "n_atoms": n_atoms,
                "sg_id_instruction": sg_instruction,
                "sg_id_claimed": sg_claimed,
                "sg_id_structure": sg_structure,
                "sg_id_claimed_eq_structure": (
                    bool(sg_claimed == sg_structure) if sg_claimed is not None and sg_structure is not None else None
                ),
                "sg_id_instruction_eq_claimed": (
                    bool(sg_instruction == sg_claimed)
                    if sg_instruction is not None and sg_claimed is not None
                    else None
                ),
                "sg_id_instruction_eq_structure": (
                    bool(sg_instruction == sg_structure)
                    if sg_instruction is not None and sg_structure is not None
                    else None
                ),
                "volume_claimed": volume_claimed,
                "volume_structure": volume_structure,
                "volume_rel_abs_err": volume_rel_abs_err,
                "volume_rel_diff_pct": volume_rel_diff_pct,
                "n_bonds_claimed": len(bond_claims),
                "bond_median_min_rel_pct": bond_median_min_rel_pct,
                "structure_ok": structure is not None,
            }
        )
        if final_cif and structure is not None:
            prediction_examples.append(
                {
                    "mp_id": mp_id,
                    "n_atoms": n_atoms,
                    "prompt": prompt_text,
                    "response": response,
                    "final_cif": final_cif,
                    "crystal_text": crystal,
                    "volume_claimed": volume_claimed,
                    "volume_structure": volume_structure,
                }
            )

    per_row = pd.DataFrame(rows)
    sg_id_map = {sym: _symbol_to_id(sym) for sym in TARGET_SG_SYMBOLS}
    target_ids = set(sg_id_map.values())
    sel = per_row[per_row["sg_id_instruction"].isin(target_ids)].copy()

    grp_rows: list[dict[str, object]] = []
    for sym in TARGET_SG_SYMBOLS:
        sid = sg_id_map[sym]
        g = sel[sel["sg_id_instruction"] == sid]
        c1 = g["sg_id_claimed_eq_structure"].dropna()
        c2 = g["sg_id_instruction_eq_claimed"].dropna()
        c3 = g["sg_id_instruction_eq_structure"].dropna()
        grp_rows.append(
            {
                "spacegroup_symbol": sym,
                "spacegroup_id": sid,
                "n_rows": int(len(g)),
                "ratio_claimed_eq_structure": float(c1.mean()) if len(c1) else float("nan"),
                "ratio_instruction_eq_claimed": float(c2.mean()) if len(c2) else float("nan"),
                "ratio_instruction_eq_structure": float(c3.mean()) if len(c3) else float("nan"),
            }
        )
    grp_df = pd.DataFrame(grp_rows)
    consistency_subset = _first_unique_mp_id_rows(prediction_examples)
    consistency_subset = [_prepare_consistency_row(row) for row in consistency_subset]
    consistency_summary = _summarize_consistency_subset(consistency_subset)
    selected_examples = _select_random_examples(consistency_subset)
    selected_examples = _attach_similarity_scores(selected_examples, len(consistency_subset))
    for row in selected_examples:
        row.pop("consistency_v2", None)
        row.pop("crystal_text", None)
        row.pop("volume_claimed", None)
        row.pop("volume_structure", None)
    examples_df = pd.DataFrame(selected_examples)

    p_row = paths.data_dir / DATA_1_3_PER_ROW
    p_bar = paths.data_dir / DATA_1_3_SG_BAR
    p_examples = paths.data_dir / DATA_1_3_PREDICTION_EXAMPLES
    p_summary = paths.data_dir / DATA_1_3_CONSISTENCY_V2_SUMMARY
    per_row.to_parquet(p_row, index=False)
    grp_df.to_parquet(p_bar, index=False)
    examples_df.to_parquet(p_examples, index=False)
    with p_summary.open("w", encoding="utf-8") as handle:
        json.dump(consistency_summary, handle, indent=2, ensure_ascii=False)
    print(f"1.3 backend wrote {p_row}, {p_bar}, {p_examples}, and {p_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.3 backend (data only).")
    parser.add_argument("--thinking", type=str, default=str(ComparePaths().thinking_parquet))
    parser.add_argument("--no-thinking", type=str, default=str(ComparePaths().no_thinking_parquet))
    parser.add_argument("--data-dir", type=str, default=str(ComparePaths().data_dir))
    args = parser.parse_args()
    run(
        ComparePaths(
            thinking_parquet=Path(args.thinking),
            no_thinking_parquet=Path(args.no_thinking),
            data_dir=Path(args.data_dir),
        ),
    )


if __name__ == "__main__":
    main()
