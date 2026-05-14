"""Symmetry, lattice, composition, and Wyckoff-based structure prompts."""

from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def get_symbolic_coord_string(coords, group_id=None, letter=None, precision=3, tol=1e-3):
    """Fractional coordinates string; extra args reserved for symbolic Wyckoff (unused)."""
    return f"({coords[0]:.8f}, {coords[1]:.8f}, {coords[2]:.8f})"


def _symmetry_setup(structure: Structure):
    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    dataset = sga.get_symmetry_dataset()

    if dataset is None:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        dataset = sga.get_symmetry_dataset()

    symmetrized_structure = sga.get_symmetrized_structure()
    formula = structure.composition.reduced_formula
    spg_symbol = dataset.international
    spg_number = dataset.number
    return sga, dataset, symmetrized_structure, formula, spg_symbol, spg_number


def _lattice_string(sga: SpacegroupAnalyzer, structure: Structure) -> str:
    lat = structure.lattice
    a, b, c = lat.a, lat.b, lat.c
    alpha, beta, gamma = lat.alpha, lat.beta, lat.gamma
    vol = lat.volume

    crystal_system = sga.get_crystal_system()
    lattice_type = sga.get_lattice_type()

    if crystal_system == "cubic" or lattice_type == "rhombohedral":
        lattice_str = (
            f"The lattice is {crystal_system}, "
            f"_cell_length_a = _cell_length_b = _cell_length_c = {a:.8f}"
        )
    elif crystal_system in ["tetragonal", "hexagonal"] or (
        crystal_system == "trigonal" and lattice_type == "hexagonal"
    ):
        lattice_str = (
            f"The lattice is {crystal_system}, "
            f"_cell_length_a = _cell_length_b = {a:.8f}, "
            f"_cell_length_c = {c:.8f}"
        )
    else:
        lattice_str = (
            f"The lattice is {crystal_system}, "
            f"_cell_length_a = {a:.8f}, _cell_length_b = {b:.8f}, "
            f"_cell_length_c = {c:.8f}"
        )

    if crystal_system in ["hexagonal", "triclinic", "monoclinic", "trigonal", "rhombohedral"]:
        lattice_str += (
            f", _cell_angle_alpha = {alpha:.2f}, "
            f"_cell_angle_beta = {beta:.2f}, "
            f"_cell_angle_gamma = {gamma:.2f}"
        )
    lattice_str += "."
    return lattice_str


def _composition_block(structure: Structure):
    comp = structure.composition
    reduced_comp, z_factor = comp.get_reduced_composition_and_factor()
    z_val = int(round(z_factor)) if abs(z_factor - round(z_factor)) < 1e-5 else z_factor
    elements = sorted(comp.elements, key=lambda e: e.X if hasattr(e, "X") else 0)
    symbols = []
    raw_counts = []
    reduced_counts = []
    formula_parts = []

    for el in elements:
        c_raw = comp[el]
        c_red = reduced_comp[el]
        fmt = lambda x: str(int(round(x))) if abs(x - round(x)) < 1e-5 else f"{x:.2f}"

        s_raw = fmt(c_raw)
        s_red = fmt(c_red)

        symbols.append(el.symbol)
        raw_counts.append(s_raw)
        reduced_counts.append(s_red)
        formula_parts.append(f"{el.symbol}{s_raw}")

    el_names_colon = ":".join(symbols)
    raw_ratio = ":".join(raw_counts)
    reduced_ratio = ":".join(reduced_counts)
    formula_sum = " ".join(formula_parts)

    composition_str = ""
    if len(raw_counts) > 1:
        composition_str += (
            f"The total atomic ratio of {el_names_colon} in the unit cell is "
            f"{raw_ratio} = {reduced_ratio}, _cell_formula_units_Z = {z_val}. "
        )
    composition_str += f"The _chemical_formula_sum is {formula_sum}."
    return composition_str, elements, comp


def _oxidation_structure(structure: Structure):
    try:
        if hasattr(structure[0].specie, "oxi_state"):
            oxi_structure = structure
        else:
            bva = BVAnalyzer()
            oxi_structure = bva.get_oxi_state_decorated_structure(structure)
        has_oxidation = True
    except Exception:
        oxi_structure = structure
        has_oxidation = False
    return oxi_structure, has_oxidation


def _build_sites_text(
    structure: Structure,
    dataset,
    symmetrized_structure,
    elements,
    comp,
    oxi_structure,
    has_oxidation: bool,
    spg_number: int,
) -> str:
    wyckoff_letters = dataset.wyckoffs
    sites_description_parts = []
    charge_balance_terms = []
    balance_eqs = []

    for el in elements:
        eneg_val = el.X if hasattr(el, "X") else 0.0

        el_sites_groups = [
            group for group in symmetrized_structure.equivalent_sites if group[0].specie == el
        ]

        site_data_list = []

        for group in el_sites_groups:
            representative = group[0]

            original_index = -1
            for i, site in enumerate(structure):
                if site.distance(representative) < 1e-4:
                    original_index = i
                    break

            w_letter = wyckoff_letters[original_index]
            multiplicity = len(group)

            oxi_state = 0
            if has_oxidation:
                try:
                    oxi_state = oxi_structure[original_index].specie.oxi_state
                    if abs(oxi_state - round(oxi_state)) < 1e-5:
                        oxi_state = int(round(oxi_state))
                except Exception:
                    pass

            charge_balance_terms.append(
                {
                    "mult": multiplicity,
                    "charge": oxi_state,
                    "el": el.symbol,
                }
            )

            coords_str = get_symbolic_coord_string(representative.frac_coords, spg_number, w_letter)

            site_data_list.append(
                {
                    "letter": w_letter,
                    "mult": multiplicity,
                    "coords": coords_str,
                    "oxi": oxi_state,
                }
            )

        site_data_list.sort(key=lambda x: x["letter"])

        lines = []
        balance_eq = []

        for i, data in enumerate(site_data_list):
            _idx = i + 1
            oxi_str = f", oxidation state {data['oxi']:+}" if has_oxidation else ""

            line = f"one site has {data['mult']} atoms{oxi_str}"
            lines.append(line)
            balance_eq.append(str(data["mult"]))

        balance_eq = f"for {el}, " + "+".join(balance_eq) + f"={int(comp[el])}"
        balance_eqs.append(balance_eq)

        header = f"{el.symbol} has {len(site_data_list)} sites: "
        body = ", ".join(lines)

        full_block = f"{header}{body}."
        sites_description_parts.append(full_block)

    if has_oxidation:
        balance_parts = []
        total_charge = 0

        for item in charge_balance_terms:
            term_val = item["mult"] * item["charge"]
            total_charge += term_val
            balance_parts.append(f"{item['mult']}*({item['charge']:+})")

        balance_eq = "+".join(balance_parts) + f"={total_charge}"
        balance_eqs.append(balance_eq)

        if abs(total_charge) < 1e-5:
            total_charge = 0

        balance_str = f"Since {', '.join(balance_eqs)}, the structure is like this:"
        sites_description_parts = [balance_str] + sites_description_parts

    return " ".join(sites_description_parts)


def structure_info_prompt(structure: Structure):
    """
    Analyzes ANY pymatgen Structure object and generates the specific text prompt
    requested, including correct Wyckoff letters and symmetry data.
    """
    sga, dataset, symmetrized_structure, formula, spg_symbol, spg_number = _symmetry_setup(structure)

    _lattice_str = _lattice_string(sga, structure)
    _composition_str, elements, comp = _composition_block(structure)

    oxi_structure, has_oxidation = _oxidation_structure(structure)

    sites_text = _build_sites_text(
        structure,
        dataset,
        symmetrized_structure,
        elements,
        comp,
        oxi_structure,
        has_oxidation,
        spg_number,
    )

    full_prompt = (
        f"This material {formula} should have the space group {spg_symbol} (id {spg_number}). "
        f"{sites_text}"
    )

    return full_prompt
