"""
Batch structure relaxation via MatterSim BatchRelaxer.

Logic migrated from ``crysreas.metric.metrics.mattergen_relax_structures``; uses
``crysreas.mlip.models.get_potential`` for a single load per process.
"""

from __future__ import annotations

import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import ray

from crysreas.mlip.models import get_potential

MAX_STEPS = 20

@ray.remote(num_gpus=0.5)
def relax_structures_batch(
    structures: list[Structure | None],
    energies: list[float] | None = None,
    potential_load_path: str | None = None,
    device: str | None = None,
    steps: int = MAX_STEPS,
) -> tuple[list[Structure | None], list[float | None], list[bool | None]]:
    """
    Relax each non-None structure; None slots stay None in outputs.

    Returns (relaxed_structures, energies, converged) aligned with input length.
    """

    from mattersim.applications.batch_relax import BatchRelaxer

    n = len(structures)
    relaxed_structures: list[Structure | None] = [None] * n
    out_energies: list[float | None] = [None] * n
    converged: list[bool | None] = [None] * n

    valid_idx = [i for i, s in enumerate(structures) if s is not None]
    if not valid_idx:
        return relaxed_structures, out_energies, converged

    potential = get_potential(device=device, load_path=potential_load_path)
    dev_str = device or ""
    if not dev_str and hasattr(potential, "device"):
        dev_str = str(potential.device)

    # Convert to ASE atoms per item; conversion failures stay as None outputs.
    atom_items: list[tuple[int, object]] = []
    for idx in valid_idx:
        s = structures[idx]
        try:
            atom_items.append((idx, AseAtomsAdaptor.get_atoms(s)))
        except Exception as e:
            print(f"[relax] get_atoms failed for index={idx}: {e}")

    if not atom_items:
        return relaxed_structures, out_energies, converged

    did_batch_fail = False
    try:
        atoms = [a for _, a in atom_items]
        batch_relaxer = BatchRelaxer(potential=potential, filter="EXPCELLFILTER", max_n_steps=steps)
        relaxation_trajectories = batch_relaxer.relax(atoms)

        for batch_i, traj in relaxation_trajectories.items():
            try:
                item_idx = int(batch_i)
                if item_idx < 0 or item_idx >= len(atom_items):
                    continue
                idx = atom_items[item_idx][0]
                final_atoms = traj[-1]
                relaxed_structures[idx] = AseAtomsAdaptor.get_structure(final_atoms)
                out_energies[idx] = final_atoms.info.get("total_energy")
                converged[idx] = len(traj) < steps
            except Exception as e:
                print(f"[relax] parse/get_structure failed for index={batch_i}: {e}")
                continue
    except Exception as e:
        did_batch_fail = True
        print(f"[relax] batch relaxation failed, fallback to per-item: {e}")
        if "cuda" in dev_str.lower():
            torch.cuda.empty_cache()

    if did_batch_fail:
        for idx, atom in atom_items:
            try:
                batch_relaxer = BatchRelaxer(
                    potential=potential, filter="EXPCELLFILTER", max_n_steps=steps
                )
                one_traj = batch_relaxer.relax([atom])
                traj = one_traj.get(0) or one_traj.get("0")
                if traj is None and len(one_traj) > 0:
                    traj = next(iter(one_traj.values()))
                if not traj:
                    continue
                final_atoms = traj[-1]
                relaxed_structures[idx] = AseAtomsAdaptor.get_structure(final_atoms)
                out_energies[idx] = final_atoms.info.get("total_energy")
                converged[idx] = len(traj) < steps
            except Exception as e:
                print(f"[relax] single relaxation failed for index={idx}: {e}")
                if "cuda" in dev_str.lower():
                    torch.cuda.empty_cache()
    return relaxed_structures, out_energies, converged
