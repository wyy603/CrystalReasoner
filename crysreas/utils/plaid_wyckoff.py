from __future__ import annotations

import re
from typing import Any

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup


_SITE_RE = re.compile(r"^(?P<multiplicity>\d+)(?P<label>[A-Za-z]+)$")


def _formula_string(structure: Structure) -> str:
    return structure.composition.formula.replace(" ", "")


def _wrap_frac(value: float) -> float:
    wrapped = value % 1.0
    return 0.0 if abs(wrapped - 1.0) < 1e-8 else wrapped


def _snap_frac(value: float, *, tol: float = 5e-4) -> float:
    wrapped = _wrap_frac(value)
    candidates = {0.0, 0.25, 1 / 3, 0.5, 2 / 3, 0.75}
    for snapped in candidates:
        if abs(wrapped - snapped) <= tol:
            return _wrap_frac(snapped)
    return wrapped


def _unique_orbit(coords: list[Any], *, tol: float = 1e-4) -> list[list[float]]:
    unique: list[list[float]] = []
    for coord in coords:
        wrapped = [_wrap_frac(float(x)) for x in coord]
        if any(all(abs(a - b) <= tol or abs(abs(a - b) - 1.0) <= tol for a, b in zip(wrapped, old)) for old in unique):
            continue
        unique.append(wrapped)
    return unique


def _format_symbol(symbol: str) -> str:
    return symbol.replace(" ", "")


def plaid_wyckoff_string(
    structure: Structure,
    *,
    symprec: float = 0.01,
    precision_8: bool = False,
) -> str:
    """Serialize a structure using the PLaID++ Wyckoff-style text representation."""
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        sym_structure = sga.get_symmetrized_structure()
        sg_symbol = _format_symbol(sga.get_space_group_symbol())
        wyckoff_by_index: dict[int, str] = {}
        for indices, wyckoff in zip(sym_structure.equivalent_indices, sym_structure.wyckoff_symbols):
            for idx in indices:
                wyckoff_by_index[idx] = wyckoff
        site_groups = [
            (site.specie.symbol, site.frac_coords, wyckoff_by_index.get(idx, "1a"), 1)
            for idx, site in enumerate(structure)
        ]
    except Exception:
        sg_symbol = "P1"
        site_groups = [
            (site.specie.symbol, site.frac_coords, "1a", 1)
            for site in structure
        ]

    lattice = structure.lattice
    lattice_fmt = ".8f" if precision_8 else ".2f"
    coord_fmt = ".8f" if precision_8 else ".3f"
    lines = [
        _formula_string(structure),
        f"Spacegroup: {sg_symbol}",
        f"abc: {format(lattice.a, lattice_fmt)} {format(lattice.b, lattice_fmt)} {format(lattice.c, lattice_fmt)}",
        f"angles: {lattice.alpha:.2f} {lattice.beta:.2f} {lattice.gamma:.2f}",
        f"Sites ({structure.num_sites})",
    ]
    for specie, frac, wyckoff, _multiplicity in site_groups:
        x, y, z = (_wrap_frac(float(v)) for v in frac)
        lines.append(f"{specie} {format(x, coord_fmt)} {format(y, coord_fmt)} {format(z, coord_fmt)} {wyckoff}")
    return "\n".join(lines)


def plaid_wyckoff_string_masked(
    structure: Structure,
    *,
    seed: int | str | None = None,
    precision_8: bool = False,
) -> tuple[str, str]:
    """Return a PLaID++ Wyckoff string with one element symbol masked."""
    import numpy as np

    rng = np.random.default_rng(None if seed is None else int(seed))
    species = [site.specie.symbol for site in structure]
    species_to_remove = str(rng.choice(species))
    crystal_str = plaid_wyckoff_string(structure, precision_8=precision_8)
    partial_crystal_str = re.sub(
        rf"(?m)^{re.escape(species_to_remove)}(?=\s)",
        "[MASK]",
        crystal_str,
    )
    return partial_crystal_str, species_to_remove


def parse_plaid_wyckoff(text: Any) -> Structure | None:
    """Deserialize the PLaID++ Wyckoff-style text representation."""
    if text is None:
        return None
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if len(lines) < 6:
        return None
    try:
        sg_match = re.match(r"^Spacegroup:\s*(.+)$", lines[1])
        abc_match = re.match(r"^abc:\s+(.+)$", lines[2])
        angle_match = re.match(r"^angles:\s+(.+)$", lines[3])
        if not sg_match or not abc_match or not angle_match:
            return None

        sg_token = sg_match.group(1).strip()
        lengths = [float(x) for x in abc_match.group(1).split()]
        angles = [float(x) for x in angle_match.group(1).split()]
        if len(lengths) != 3 or len(angles) != 3:
            return None
        lattice = Lattice.from_parameters(*lengths, *angles)

        try:
            space_group = SpaceGroup(sg_token)
        except Exception:
            space_group = SpaceGroup.from_int_number(int(sg_token))

        species: list[str] = []
        coords: list[list[float]] = []
        site_count_match = re.match(r"^Sites\s*\((\d+)\)", lines[4])
        encoded_site_count = int(site_count_match.group(1)) if site_count_match else None
        site_lines = lines[5:]
        explicit_sites = encoded_site_count == len(site_lines)

        for line in site_lines:
            parts = line.split()
            if len(parts) < 5:
                return None
            specie = parts[0]
            coord = [_snap_frac(float(x)) for x in parts[1:4]]
            site_match = _SITE_RE.match(parts[4])
            expected_multiplicity = int(site_match.group("multiplicity")) if site_match else None
            orbit = [coord] if explicit_sites or expected_multiplicity == 1 else _unique_orbit(space_group.get_orbit(coord))
            if expected_multiplicity is not None and len(orbit) != expected_multiplicity:
                # Keep parsing permissive for generated text, but prefer the encoded symmetry.
                orbit = orbit[:expected_multiplicity]
            for orbit_coord in orbit:
                species.append(specie)
                coords.append([_wrap_frac(float(x)) for x in orbit_coord])

        return Structure(
            lattice=lattice,
            species=species,
            coords=coords,
            coords_are_cartesian=False,
            to_unit_cell=True,
        )
    except Exception:
        return None


def plaid_wyckoff_roundtrip_matches(
    structure: Structure,
    *,
    symprec: float = 0.01,
    matcher: StructureMatcher | None = None,
) -> bool:
    serialized = plaid_wyckoff_string(structure, symprec=symprec)
    parsed = parse_plaid_wyckoff(serialized)
    if parsed is None:
        return False
    matcher = matcher or StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    return bool(matcher.fit(structure, parsed))
