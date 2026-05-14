import random
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from crysreas.metric_process import merge_metric_process_config
from crysreas.metric_process.basic import ensure_gt
from crysreas.utils.crystal import SimpleCrystal
from crysreas.utils.crystaltext import crystaltext_string, crystaltext_string_masked
from crysreas.utils.plaid_wyckoff import plaid_wyckoff_string, plaid_wyckoff_string_masked

from .property_text import (
    electrical_info_prompt,
    mechanical_summary_prompt,
    stability_status_prompt,
)
from .prompt_helpers import getseg, strseg
from .structure_prompt import structure_info_prompt

PROMPT_LOOKUP: Dict[str, str] = {
    "formation_energy_per_atom": "The formation energy per atom is",
    "band_gap": "The band gap is",
    "formula_pretty": "The chemical formula is",
    "energy_above_hull": "The energy above the convex hull is",
    "elements": "The elements are",
    "spacegroup.number": "The spacegroup number is",
    "bulk_modulus": "The bulk_modulus is",
    "shear_modulus": "The shear_modulus is",
    "thermal_expansion_300k": "The thermal expansion at 300 K is",
    "small_atoms": "The generated structure should have less than 10 atoms",
}

CRYSTALTEXT_GENERATION_TYPES = {"crystaltextllm_generation", "crystaltextllm_8_generation"}
CRYSTALTEXT_TRAIN_TYPES = {"crystaltextllm_train", "crystaltextllm_8_train"}
PLAID_WYCKOFF_GENERATION_TYPES = {"plaid_wyckoff_generation", "plaid_wyckoff_8_generation"}
PLAID_WYCKOFF_TRAIN_TYPES = {"plaid_wyckoff_train", "plaid_wyckoff_8_train"}
PRIOR_WORK_TRAIN_TO_GENERATION = {
    "crystaltextllm_train": "crystaltextllm_generation",
    "crystaltextllm_8_train": "crystaltextllm_8_generation",
    "plaid_wyckoff_train": "plaid_wyckoff_generation",
    "plaid_wyckoff_8_train": "plaid_wyckoff_8_generation",
}


def _uses_precision_8(prompt_family: str) -> bool:
    return "_8_" in prompt_family


def _part_prompt_numeric(elem: Any, _crystal: Any, attr: str) -> str:
    return f"{PROMPT_LOOKUP[attr]} {round(float(elem[attr]), 4)}. "


_ATTR_FORMATTERS: Dict[str, Callable[[Any, Any], str]] = {
    "elements": lambda e, c: f"{PROMPT_LOOKUP['elements']} {', '.join(e['elements'])}. ",
    "formation_energy_per_atom": lambda e, c: _part_prompt_numeric(e, c, "formation_energy_per_atom"),
    "band_gap": lambda e, c: _part_prompt_numeric(e, c, "band_gap"),
    "energy_above_hull": lambda e, c: _part_prompt_numeric(e, c, "energy_above_hull"),
    "spacegroup.number": lambda e, c: f"{PROMPT_LOOKUP['spacegroup.number']} {c.sga.get_space_group_number()}. ",
    "formula_pretty": lambda e, c: f"{PROMPT_LOOKUP['formula_pretty']} {e['condensed_structure'].formula}. ",
    "bulk_modulus": lambda e, c: (
        f"{PROMPT_LOOKUP['bulk_modulus']} {strseg(getseg(e['bulk_modulus'], [-np.inf, 0, 10, 50, 150, 300, np.inf]))}. "
    ),
    "shear_modulus": lambda e, c: (
        f"{PROMPT_LOOKUP['shear_modulus']} {strseg(getseg(e['shear_modulus'], [-np.inf, 0, 20, 80, 200, np.inf]))}. "
    ),
    "thermal_expansion_300k": lambda e, c: (
        f"{PROMPT_LOOKUP['thermal_expansion_300k']} {strseg(getseg(e['thermal_expansion_300k'], np.array([-np.inf, -10, 0, 5, 15, 40, 100, 200, 300, np.inf]) * 1e-6), exp=1e6)} 10^-6 K^-1. "
    ),
    "small_atoms": lambda e, c: f"{PROMPT_LOOKUP['small_atoms']}. ",
}


def _conditional_attribute_fragment(attr: str, elem: Any, crystal: Any) -> str:
    if attr in _ATTR_FORMATTERS:
        return _ATTR_FORMATTERS[attr](elem, crystal)
    return f"{PROMPT_LOOKUP[attr]} {elem[attr]}. "


def _crystaltext_condition_prompt(elem: Any, crystal: Any, seed: int | str | None) -> str:
    rng = random.Random(int(seed)) if seed is not None else random
    all_attributes = [
        "formation_energy_per_atom",
        "band_gap",
        "energy_above_hull",
        "spacegroup.number",
    ]
    num_attributes = rng.randint(0, len(all_attributes))
    attributes = rng.sample(all_attributes, num_attributes)
    attributes = ["formula_pretty"] + attributes

    part_prompt = ""
    for attr in attributes:
        part_prompt += _conditional_attribute_fragment(attr, elem, crystal)
    return (
        f"Below is a description of a bulk material. {part_prompt}"
        "Generate a description of the lengths and angles of the lattice vectors and then "
        "the element type and coordinates for each atom within the lattice:\n"
    )


def _plaid_wyckoff_condition_prompt(elem: Any, crystal: Any, seed: int | str | None) -> str:
    rng = random.Random(int(seed)) if seed is not None else random
    all_attributes = [
        "energy_above_hull",
        "formation_energy_per_atom",
        "band_gap",
        "spacegroup.number",
    ]
    num_attributes = rng.randint(0, len(all_attributes))
    attributes = rng.sample(all_attributes, num_attributes)
    attributes = ["formula_pretty"] + attributes

    part_prompt = ""
    for attr in attributes:
        part_prompt += _conditional_attribute_fragment(attr, elem, crystal)
    return (
        f"Below is a description of a bulk material. {part_prompt}"
        "Generate a description of the lengths and angles of the lattice vectors and then "
        "the element type and coordinates for each atom within the lattice:\n"
    )


def get_info(
    elem,
    prompt_type: str,
    debug=True,
    seed=None,
    return_gt=False,
) -> Union[Dict[str, str], Tuple[Dict[str, str], Any]]:
    if seed is not None:
        random.seed(int(seed))
    if debug:
        return get_info(elem, prompt_type, debug=False, seed=seed, return_gt=return_gt)

    description = elem.get("description", "A crystalline material.")
    structure = elem.get("structure")
    crystal = SimpleCrystal.from_sym_structure(structure)
    simple_cif_no_sym = crystal.to_simple_no_sym()
    structure_info = structure_info_prompt(crystal.structure)
    stability_status = stability_status_prompt(elem)
    electrical_info = electrical_info_prompt(elem)
    mechanical_summary = mechanical_summary_prompt(elem)

    prompt_types = prompt_type.split("+")

    question = ""
    if prompt_types[0] == "unconditional":
        question += (
            "Below is a description of a bulk material. Generate a description of the lengths "
            "and angles of the lattice vectors and then the element type and coordinates for "
            "each atom within the lattice:\n"
        )
    elif prompt_types[0] == "default":
        question += (
            f"{description.split('.', 1)[0]}. Please generate a material report of this material. "
            f"The material report should include basic information, electronic properties, stability, "
            f"mechanical properties, and a CIF file.\n"
        )
    elif prompt_types[0] in CRYSTALTEXT_GENERATION_TYPES:
        question += _crystaltext_condition_prompt(elem, crystal, seed)
    elif prompt_types[0] in PLAID_WYCKOFF_GENERATION_TYPES:
        question += _plaid_wyckoff_condition_prompt(elem, crystal, seed)
    elif prompt_types[0] in ("conditional", "elastic", "spacegroup", "cte"):
        if prompt_types[0] == "conditional":
            all_attributes: List[str] = [
                "formation_energy_per_atom",
                "band_gap",
                "energy_above_hull",
                "spacegroup.number",
            ]
            num_attributes = random.randint(0, len(all_attributes))
            attributes = random.sample(all_attributes, num_attributes)
            attributes = ["formula_pretty"] + attributes
        elif prompt_types[0] == "spacegroup":
            attributes = ["formula_pretty", "spacegroup.number"]
        elif prompt_types[0] == "elastic":
            attributes = ["formula_pretty", "bulk_modulus", "shear_modulus"]
        else:
            attributes = ["formula_pretty", "thermal_expansion_300k", "small_atoms"]

        part_prompt = ""
        for attr in attributes:
            part_prompt += _conditional_attribute_fragment(attr, elem, crystal)
        question += (
            f"Below is a description of a bulk material. {part_prompt}"
            "Generate a description of the lengths and angles of the lattice vectors and then "
            "the element type and coordinates for each atom within the lattice:\n"
        )
    else:
        raise Exception("prompt_types[0] not correct.")

    structure_validity = (
        "The structure is reasonable, because the band lengths are all greater than 0.5, "
        f"and the structure's volume {structure.volume:.2f} is larger than 0.1."
    )

    answer = ""
    if prompt_types[1] == "thinking":
        answer += "Let's generate a material report first, according to the given information.\n\n"
        prompt_list = []
        if electrical_info != "":
            prompt_list.append(f"### Electronic Properties\n**Classification:** {electrical_info}\n\n")
        if stability_status != "":
            prompt_list.append(f"### Stability\n**Thermodynamic Status:** {stability_status}\n\n")
        if mechanical_summary != "":
            prompt_list.append(f"### Mechanical Properties\n{mechanical_summary}\n\n")
        answer += (
            "## Material Report:\n\n"
            "### Crystal Structure\n"
            f"First, consider space groups and atom numbers. {structure_info} Second, consider band gaps. "
            f"{description} Third, consider structure validity. {structure_validity}\n\n"
        )
        answer += "".join(random.sample(prompt_list, len(prompt_list)))
        answer += f"## CIF File\n<CIF>{simple_cif_no_sym}</CIF>\n"
    elif prompt_types[1] == "no_thinking":
        if prompt_types[0] in CRYSTALTEXT_GENERATION_TYPES:
            crystal_text = crystaltext_string(
                crystal.structure,
                seed=seed,
                translate=True,
                precision_8=_uses_precision_8(prompt_types[0]),
            )
            answer += f"CIF File: <CIF>{crystal_text}</CIF>\n"
        elif prompt_types[0] in PLAID_WYCKOFF_GENERATION_TYPES:
            crystal_text = plaid_wyckoff_string(
                crystal.structure,
                precision_8=_uses_precision_8(prompt_types[0]),
            )
            answer += f"CIF File: <CIF>{crystal_text}</CIF>\n"
        else:
            answer += f"CIF File: <CIF>{simple_cif_no_sym}</CIF>\n"
    else:
        raise Exception("prompt_types[1] not correct.")

    item = {"question": question, "answer": answer, "task_type": "generation"}
    if return_gt:
        mp_cfg = merge_metric_process_config(None, {"prompt_type": prompt_type.split("+")})
        df_gt = pd.DataFrame({"mp_id": [str(elem["material_id"])]})
        ensure_gt(df_gt, mp_cfg)
        gt = df_gt["gt"].iloc[0]
        return item, gt
    return item


def get_info_infill(elem, prompt_type: str, debug=True, seed=None) -> Dict[str, str]:
    if seed is not None:
        random.seed(int(seed))
    if debug:
        return get_info_infill(elem, prompt_type, debug=False, seed=seed)

    prompt_types = prompt_type.split("+")
    if prompt_types[0] not in PRIOR_WORK_TRAIN_TO_GENERATION or prompt_types[1] != "no_thinking":
        raise ValueError(f"Unsupported infill prompt_type: {prompt_type}")

    structure = elem.get("structure")
    crystal = SimpleCrystal.from_sym_structure(structure)
    if prompt_types[0] in PLAID_WYCKOFF_TRAIN_TYPES:
        partial_crystal_str, species_to_remove = plaid_wyckoff_string_masked(
            crystal.structure,
            seed=seed,
            precision_8=_uses_precision_8(prompt_types[0]),
        )
    else:
        partial_crystal_str, species_to_remove = crystaltext_string_masked(
            crystal.structure,
            seed=seed,
            precision_8=_uses_precision_8(prompt_types[0]),
        )
    question = (
        'Below is a partial description of a bulk material where one '
        'element has been replaced with the string "[MASK]":\n'
        f"{partial_crystal_str}\n"
        "Generate an element that could replace [MASK] in the bulk material:\n"
    )
    answer = str(species_to_remove)
    return {
        "question": question,
        "answer": answer,
        "task_type": "infill",
        "task_span": (0, len(answer)),
    }
