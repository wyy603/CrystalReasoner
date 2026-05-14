from __future__ import annotations

from typing import Any

import numpy as np
from pymatgen.core import Lattice, Structure


def crystaltext_string(
    structure: Structure,
    *,
    seed: int | str | None = None,
    translate: bool = True,
    precision_8: bool = False,
) -> str:
    struct = structure.copy()
    if translate:
        rng = np.random.default_rng(None if seed is None else int(seed))
        struct.translate_sites(indices=range(len(struct.sites)), vector=rng.uniform(size=(3,)))

    lengths = struct.lattice.parameters[:3]
    angles = struct.lattice.parameters[3:]
    atom_ids = struct.species
    frac_coords = struct.frac_coords
    length_fmt = ".8f" if precision_8 else ".1f"
    coord_fmt = ".8f" if precision_8 else ".2f"
    return (
        " ".join(format(x, length_fmt) for x in lengths)
        + "\n"
        + " ".join(str(int(round(x))) for x in angles)
        + "\n"
        + "\n".join(
            str(t) + "\n" + " ".join(format(x, coord_fmt) for x in c)
            for t, c in zip(atom_ids, frac_coords)
        )
    )


def crystaltext_string_masked(
    structure: Structure,
    *,
    seed: int | str | None = None,
    precision_8: bool = False,
) -> tuple[str, str]:
    struct = structure.copy()
    rng = np.random.default_rng(None if seed is None else int(seed))
    struct.translate_sites(indices=range(len(struct.sites)), vector=rng.uniform(size=(3,)))
    species = [str(s) for s in struct.species]
    species_to_remove = str(rng.choice(species))
    crystal_str = crystaltext_string(struct, seed=seed, translate=False, precision_8=precision_8)
    partial_crystal_str = crystal_str.replace(species_to_remove, "[MASK]")
    return partial_crystal_str, species_to_remove


def parse_crystaltext(text: Any) -> Structure | None:
    if text is None:
        return None
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if len(lines) < 4 or (len(lines) - 2) % 2 != 0:
        return None
    try:
        lengths = [float(x) for x in lines[0].split()]
        angles = [float(x) for x in lines[1].split()]
        if len(lengths) != 3 or len(angles) != 3:
            return None
        species = []
        coords = []
        for i in range(2, len(lines), 2):
            species.append(lines[i])
            coords.append([float(x) for x in lines[i + 1].split()])
        return Structure(
            lattice=Lattice.from_parameters(*lengths, *angles),
            species=species,
            coords=coords,
            coords_are_cartesian=False,
        )
    except Exception:
        return None
