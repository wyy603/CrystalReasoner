"""QHA calculations"""

import copy
import colorsys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch_sim as ts
from ase import Atoms
from ase.build import bulk
from phonopy.api_phonopy import Phonopy
from phonopy.api_qha import PhonopyQHA
from phonopy.structure.atoms import PhonopyAtoms
from pydantic import BaseModel, ConfigDict, Field
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_pmg_structure
import plotly.io as pio
from torch_sim.models.mattersim import MatterSimModel
from mattersim.forcefield.potential import Potential
from .models import get_potential

MAX_MEMORY_SCALER = 43000
MEMORY_SCALES_WITH = "n_atoms_x_density"
model = None

class DataQHA(BaseModel):
    """QHA thermal properties.

    Attributes:
        temperatures: Array of temperatures (K).
        bulk_modulus_temperature: Bulk modulus at each temperature (GPa).
        heat_capacity_temperature: Heat capacity at each temperature (J/K/mol).
        volume_temperature: Volume at each temperature (A^3).
        gibbs_temperature: Gibbs free energy at each temperature (kJ/mol).
        thermal_expansion: Volumetric coefficient of thermal expansion (1/K).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    temperatures: np.ndarray
    bulk_modulus_temperature: np.ndarray | None
    heat_capacity_temperature: np.ndarray | None
    volume_temperature: np.ndarray | None
    gibbs_temperature: np.ndarray | None
    thermal_expansion: np.ndarray | None

    # Per-volume data for free energy plotting
    volumes: np.ndarray | None = None
    electronic_energies: np.ndarray | None = None
    phonon_free_energies: np.ndarray | None = None


class ConfigQHA(BaseModel):
    """Configuration for QHA calculation.

    Attributes:
        displacement: Displacement magnitude in Angstrom
        q_point_mesh: Phonon q-point mesh
        supercell_matrix: Supercell matrix for phonon calculations
        temperatures: List of temperatures (K)
        length_factors: Volume scaling factors
        symprec: Symmetry precision
        nsteps: Maximum number of relaxation steps
        fmax: Force convergence tolerance in eV/A
    """

    displacement: float = 0.03
    mesh: list[int] = Field(default_factory=lambda: [20, 20, 20])
    supercell_matrix: list[list[int]] = Field(
        default_factory=lambda: [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    )
    temperatures: list[int] = Field(
        default_factory=lambda: list(range(0, 510, 10))
    )
    length_factors: list[float] = Field(
        default_factory=lambda: np.linspace(0.92, 1.08, 7).round(4).tolist(),
        min_length=1,
        description="List of volume scaling factors",
    )
    symprec: float = 1e-5
    nsteps: int = 300
    fmax: float = 1e-3

import time
from typing import Any

import ray

# Ray-friendly payloads (plain dicts / lists / floats); avoids fragile pickle of Phonopy / pymatgen.
_DTYPE_STRUCTURE = "pymatgen.core.structure.Structure"
_DTYPE_PHONOPY_ATOMS = "phonopy.structure.atoms.PhonopyAtoms"
_DTYPE_PHONOPY = "phonopy.api_phonopy.Phonopy"


def _serialize(obj: Any) -> Any:
    """Recursively serialize to Ray/pickle-safe JSON-like data (plain dict/list/tuple/scalars)."""
    if obj is None:
        return None
    # bool before int (bool is subclass of int); numpy bool_ before int
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {key: _serialize(v) for key, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_serialize(x) for x in obj)
    if isinstance(obj, np.ndarray):
        return _serialize(obj.tolist())
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, Structure):
        return {"_dtype": _DTYPE_STRUCTURE, "struct": obj.as_dict()}
    if isinstance(obj, PhonopyAtoms):
        return {
            "_dtype": _DTYPE_PHONOPY_ATOMS,
            "symbols": list(obj.symbols),
            "cell": np.asarray(obj.cell, dtype=np.float64).tolist(),
            "positions": np.asarray(obj.positions, dtype=np.float64).tolist(),
        }
    if isinstance(obj, Phonopy):
        dist: float | None = None
        ds = obj.dataset
        if ds is not None:
            if "first_atoms" in ds and ds["first_atoms"]:
                dist = float(
                    np.linalg.norm(np.asarray(ds["first_atoms"][0]["displacement"]))
                )
            elif "displacements" in ds:
                disp = np.asarray(ds["displacements"]).reshape(-1, 3)
                dist = float(np.linalg.norm(disp[0]))
        prim = obj.primitive_matrix
        prim_payload: str | list[list[float]]
        if prim is None:
            prim_payload = "auto"
        else:
            prim_payload = np.asarray(prim, dtype=np.float64).tolist()
        return {
            "_dtype": _DTYPE_PHONOPY,
            "unitcell": _serialize(obj.unitcell),
            "supercell_matrix": np.asarray(obj.supercell_matrix, dtype=np.int64).tolist(),
            "primitive_matrix": prim_payload,
            "symprec": float(obj.symmetry.tolerance),
            "displacement_distance": dist,
        }
    raise TypeError(f"_serialize: unsupported type {type(obj)!r}")


def _deserialize_phonopy_from_dict(d: dict[str, Any]) -> Phonopy:
    """Rebuild Phonopy from a tagged dict produced by `_serialize`."""
    uc = _deserialize_any(d["unitcell"])
    if not isinstance(uc, PhonopyAtoms):
        raise TypeError("Phonopy payload: unitcell must deserialize to PhonopyAtoms")
    pm_raw = d["primitive_matrix"]
    if pm_raw == "auto":
        prim_arg: Any = "auto"
    else:
        prim_arg = np.asarray(pm_raw, dtype=np.float64)
    ph = Phonopy(
        unitcell=uc,
        supercell_matrix=np.asarray(d["supercell_matrix"], dtype=np.int64),
        primitive_matrix=prim_arg,
        symprec=float(d["symprec"]),
    )
    dist = d.get("displacement_distance")
    if dist is None:
        raise ValueError("Phonopy payload missing displacement_distance (no dataset?)")
    ph.generate_displacements(distance=float(dist))
    return ph


def _deserialize_any(data: Any) -> Any:
    """Inverse of `_serialize` for nested plain data and tagged Structure / PhonopyAtoms / Phonopy."""
    if data is None:
        return None
    if isinstance(data, bool):
        return data
    if isinstance(data, (int, float, str)):
        return data
    if isinstance(data, list):
        return [_deserialize_any(x) for x in data]
    if isinstance(data, tuple):
        return tuple(_deserialize_any(x) for x in data)
    if isinstance(data, dict):
        tag = data.get("_dtype")
        if tag == _DTYPE_STRUCTURE:
            return Structure.from_dict(data["struct"])
        if tag == _DTYPE_PHONOPY_ATOMS:
            return PhonopyAtoms(
                symbols=data["symbols"],
                positions=np.asarray(data["positions"], dtype=np.float64),
                cell=np.asarray(data["cell"], dtype=np.float64),
            )
        if tag == _DTYPE_PHONOPY:
            return _deserialize_phonopy_from_dict(data)
        return {k: _deserialize_any(v) for k, v in data.items()}
    raise TypeError(f"_deserialize_any: unsupported type {type(data)!r}")


def _deserialize(data: Any, typ: type | None = None) -> Any:
    """Restore values from `_serialize` output.

    If ``typ`` is ``None``, recursively deserialize nested lists/dicts and tagged objects.
    If ``typ`` is ``Structure``, ``PhonopyAtoms``, or ``Phonopy``, ``data`` must be a dict
    with matching ``_dtype`` (same checks as before).
    If ``typ`` is ``np.ndarray``, ``data`` is converted with ``numpy.asarray``.
    """
    if typ is None:
        return _deserialize_any(data)
    if typ is np.ndarray:
        return np.asarray(data)
    if not isinstance(data, dict):
        raise TypeError(f"_deserialize: expected dict for typ={typ!r}, got {type(data)!r}")
    tag = data.get("_dtype")
    if typ is Structure:
        if tag != _DTYPE_STRUCTURE:
            raise TypeError(f"expected {_DTYPE_STRUCTURE}, got {tag!r}")
        return Structure.from_dict(data["struct"])
    if typ is PhonopyAtoms:
        if tag != _DTYPE_PHONOPY_ATOMS:
            raise TypeError(f"expected {_DTYPE_PHONOPY_ATOMS}, got {tag!r}")
        return PhonopyAtoms(
            symbols=data["symbols"],
            positions=np.asarray(data["positions"], dtype=np.float64),
            cell=np.asarray(data["cell"], dtype=np.float64),
        )
    if typ is Phonopy:
        if tag != _DTYPE_PHONOPY:
            raise TypeError(f"expected {_DTYPE_PHONOPY}, got {tag!r}")
        return _deserialize_phonopy_from_dict(data)
    raise TypeError(f"_deserialize: unsupported typ {typ!r}")


def _qha_task_from_wire(task: Any) -> tuple:
    """Turn wire form of a QHA task tuple back into the object expected by `_calculate_qha_for_input`.

    After wire round-trip, each ``Phonopy`` is rebuilt via ``generate_displacements``. The
    independently deserialized ``PhonopyAtoms`` lists may not match the rebuilt ``ph`` in
    count/order; ``_calculate_qha_for_input`` indexes forces using ``len(supercells)`` and
    ``ph`` together, so we must use ``ph.supercells_with_displacements`` as the single source
    of truth (same as an in-process object).
    """
    t = _deserialize_any(task)
    if isinstance(t, list):
        t = tuple(t)
    if not isinstance(t, tuple) or len(t) != 9:
        raise TypeError(f"QHA task must be a 9-tuple after deserialize, got {type(t)!r} len={getattr(t, '__len__', 'n/a')!r}")
    (
        ph_sets,
        relaxed_structs,
        relaxation_energies,
        supercells_per_ph,
        force_block,
        temperatures_arr,
        mesh,
        t_step,
        input_index,
    ) = t
    synced_supercells: list[list[PhonopyAtoms]] = []
    for ph, _scs in zip(ph_sets, supercells_per_ph):
        fresh = ph.supercells_with_displacements
        synced_supercells.append(list(fresh) if fresh is not None else [])
    mesh_list = [int(x) for x in mesh]
    return (
        ph_sets,
        relaxed_structs,
        relaxation_energies,
        synced_supercells,
        np.asarray(force_block, dtype=np.float64),
        np.asarray(temperatures_arr),
        mesh_list,
        int(t_step),
        int(input_index),
    )


def _collect_supercell(
    struct: Structure, config_dict: dict
) -> tuple[Phonopy, list[PhonopyAtoms]]:
    phonopy_atoms = PhonopyAtoms(
        symbols=[str(site.specie) for site in struct],
        positions=struct.cart_coords,
        cell=struct.lattice.matrix,
    )
    ph = Phonopy(
        unitcell=phonopy_atoms,
        supercell_matrix=np.array(config_dict["supercell_matrix"]),
        primitive_matrix="auto",
        symprec=config_dict["symprec"],
    )
    ph.generate_displacements(distance=config_dict["displacement"])
    supercells = ph.supercells_with_displacements
    return ph, list(supercells) if supercells is not None else []

def _calculate_qha_for_input(
    task: tuple[
        list[Phonopy],
        list[Structure],
        list[float],
        list[list[PhonopyAtoms]],
        np.ndarray,
        np.ndarray,
        list[int],
        int,
        int,
    ]
) -> dict | None:
    """Calculate QHA for a single input structure.
    
    Returns None if force constant calculation fails.
    """
    (
        ph_sets,
        relaxed_structs,
        relaxation_energies,
        supercells_per_ph,
        force_block,
        temperatures_arr,
        mesh,
        t_step,
        input_index,
    ) = task

    # Calculate expected total force count
    expected_force_count = 0
    for ph, supercells in zip(ph_sets, supercells_per_ph):
        n_atoms = len(ph.supercell)
        expected_force_count += n_atoms * len(supercells)

    volumes = []
    energies = []
    free_energies_list = []
    entropies_list = []
    heat_capacities_list = []

    offset = 0
    for ph, struct, energy, supercells in zip(
        ph_sets, relaxed_structs, relaxation_energies, supercells_per_ph
    ):
        n_atoms = len(ph.supercell)  # pyright: ignore[reportArgumentType]
        n_displacements = len(supercells)
        n_total = n_atoms * n_displacements
        ph.forces = list(
            force_block[offset : offset + n_total].reshape(
                n_displacements, n_atoms, 3
            )
        )
        offset += n_total

        try:
            ph.produce_force_constants()  # pyright: ignore[reportUnknownMemberType]
        except Exception as e:
            print(f"Error producing force constants for input {input_index}: {e}")
            return None
        
        ph.run_mesh(mesh)
        ph.run_thermal_properties(
            t_min=int(temperatures_arr[0]),
            t_max=int(temperatures_arr[-1]),
            t_step=t_step,
        )
        thermal_props = ph.get_thermal_properties_dict()

        volumes.append(struct.volume)
        energies.append(energy)
        free_energies_list.append(
            thermal_props.get("free_energy", np.array([]))  # pyright: ignore[reportArgumentType]
        )
        entropies_list.append(
            thermal_props.get("entropy", np.array([]))  # pyright: ignore[reportArgumentType]
        )
        heat_capacities_list.append(
            thermal_props.get("heat_capacity", np.array([]))  # pyright: ignore[reportArgumentType]
        )

    qha = PhonopyQHA(
        volumes=volumes,
        electronic_energies=np.tile(energies, (len(temperatures_arr), 1)),
        temperatures=temperatures_arr,
        free_energy=np.array(free_energies_list).T,
        cv=np.array(heat_capacities_list).T,
        entropy=np.array(entropies_list).T,
        eos="vinet",
    )
    return {
        "bulk_modulus_temperature": qha.bulk_modulus_temperature,
        "heat_capacity_temperature": np.array(qha.heat_capacity_P_numerical),
        "volume_temperature": qha.volume_temperature,
        "gibbs_temperature": qha.gibbs_temperature,
        "thermal_expansion": np.array(qha.thermal_expansion),
        "volumes": np.array(volumes),
        "electronic_energies": np.array(energies),
        "phonon_free_energies": np.array(free_energies_list),
    }


def _ensure_ray() -> None:
    if not ray.is_initialized():
        print("ray.init cte")
        ray.init(ignore_reinit_error=True)


@ray.remote(num_cpus=1)
def _ray_collect_supercell(payload: tuple[dict, dict]) -> tuple[dict, list[dict]]:
    struct_d, config_dict = payload
    struct = _deserialize(struct_d, Structure)
    ph, supercells = _collect_supercell(struct, config_dict)
    return _serialize(ph), [_serialize(sc) for sc in supercells]


@ray.remote(num_cpus=1)
def _ray_calculate_qha_for_input(task_serialized: Any) -> dict | None:
    """Runs off the driver; nested inside ``calculate_qha`` Ray task — use ``num_cpus=0`` so
    child tasks do not reserve extra CPU slots (avoids deadlock when cluster CPUs are tight).
    """
    try:
        task = _qha_task_from_wire(task_serialized)
        return _calculate_qha_for_input(task)
    except Exception as e:
        print(f"Ray child QHA task failed: {e}")
        return None

@ray.remote(num_gpus=0.5)
def calculate_qha(
    structures: list[Structure],
    config: ConfigQHA | None = None,
    debug_time = False,
) -> list[DataQHA | None]:
    """Calculate quasi-harmonic thermal properties for a list of structures.

    Args:
        model: MaceModel instance for energy/force evaluation.
        structures: List of pymatgen Structures to compute QHA for.
        config: Configuration for the QHA calculation.

    Returns:
        List of DataQHA (temperature-dependent thermal properties) for each structure.
    """

    n_structures = len(structures)
    if n_structures == 0:
        return []

    t0 = time.time()

    global model
    if model is None:
        model = MatterSimModel(get_potential(device = "cuda"))

    if config is None:
        config = ConfigQHA()

    # Set up autobatchers
    binning_autobatcher = ts.BinningAutoBatcher(
        model=model,
        memory_scales_with=MEMORY_SCALES_WITH,
        max_memory_scaler=MAX_MEMORY_SCALER,
    )
    inflight_autobatcher = ts.InFlightAutoBatcher(
        model=model,
        memory_scales_with=MEMORY_SCALES_WITH,
        max_memory_scaler=MAX_MEMORY_SCALER,
    )

    t1 = time.time()
    if debug_time:
        print("t1", t1 - t0)

    # Relax structures
    try:
        model._compute_forces = True
        state = ts.io.structures_to_state(
            structures, device=model.device, dtype=model.dtype
        )
        relaxed_state = ts.optimize(
            system=state,
            model=model,
            optimizer=ts.optimizers.Optimizer.lbfgs,
            init_kwargs={
                "cell_filter": ts.optimizers.cell_filters.CellFilter.frechet,  # pyright: ignore[reportAttributeAccessIssue]
                "constant_volume": False,
                "hydrostatic_strain": True,
            },
            max_steps=config.nsteps,
            convergence_fn=ts.runners.generate_force_convergence_fn(
                force_tol=config.fmax,
                include_cell_forces=True,
            ),
        )
        relaxed_structures: list[Structure] = ts.io.state_to_structures(relaxed_state)
    except Exception as e:
        print(f"[cte] initial relaxation failed: {e}")
        return [None] * n_structures
    n_inputs = len(relaxed_structures)
    n_volumes = len(config.length_factors)

    t2 = time.time()
    if debug_time:
        print("t2", t2 - t1)

    # Build all scaled structures
    all_scaled_structs: list[Structure] = []
    for relaxed_struct in relaxed_structures:
        for factor in config.length_factors:
            volume_scale = factor ** (1.0 / 3.0)
            target_cell = relaxed_struct.lattice.matrix * volume_scale
            all_scaled_structs.append(
                Structure(
                    lattice=target_cell,
                    species=relaxed_struct.species,
                    coords=relaxed_struct.frac_coords,
                    coords_are_cartesian=False,
                )
            )

    # Relax scaled structures
    # print(f"Relaxing {len(all_scaled_structs)} scaled structures")
    
    try:
        relaxed_scaled_state = ts.optimize(
            system=all_scaled_structs,
            model=model,
            optimizer=ts.optimizers.Optimizer.lbfgs,
            init_kwargs={
                "cell_filter": ts.optimizers.cell_filters.CellFilter.frechet,  # pyright: ignore[reportAttributeAccessIssue]
                "constant_volume": True,
                "hydrostatic_strain": True,
            },
            max_steps=config.nsteps,
            convergence_fn=ts.runners.generate_force_convergence_fn(
                force_tol=config.fmax,
                include_cell_forces=False,
            ),
            autobatcher=inflight_autobatcher,
        )
        all_relaxed_scaled = ts.io.state_to_structures(relaxed_scaled_state)
        scaled_energy = relaxed_scaled_state.energy.detach().cpu()
    except Exception as e:
        print(f"[cte] scaled relaxation failed: {e}")
        return [None] * n_structures

    t3 = time.time()
    if debug_time:
        print("t3", t3 - t2)

    # Group relaxed scaled structs and energies by input structure
    all_relaxed_per_input: list[list[Structure]] = [
        all_relaxed_scaled[i * n_volumes : (i + 1) * n_volumes]
        for i in range(n_inputs)
    ]
    relaxation_energies_per_input: list[list[float]] = [
        scaled_energy[i * n_volumes : (i + 1) * n_volumes].tolist()
        for i in range(n_inputs)
    ]

    # Build Phonopy objects and collect supercells for all inputs
    ph_sets_per_input: list[list[Phonopy]] = []
    supercells_per_ph_per_input: list[list[list[PhonopyAtoms]]] = []
    supercells_flat_per_input: list[list[PhonopyAtoms]] = []
    config_dict = config.model_dump()

    # Per input structure: if any volume point fails, drop this input (append None in final results).
    grouped_refs: list[list[Any]] = []
    for relaxed_structs in all_relaxed_per_input:
        grouped_refs.append(
            [
                _ray_collect_supercell.remote((_serialize(s), config_dict))
                for s in relaxed_structs
            ]
        )

    for i, refs in enumerate(grouped_refs):
        try:
            grouped_results = ray.get(refs)
        except Exception as e:
            print(
                f"Input {i}: _ray_collect_supercell failed for at least one volume point, "
                f"drop this structure. error={e}"
            )
            ph_sets_per_input.append([])
            supercells_per_ph_per_input.append([])
            supercells_flat_per_input.append([])
            continue

        ph_sets: list[Phonopy] = []
        supercells_per_ph: list[list[PhonopyAtoms]] = []
        supercells_flat: list[PhonopyAtoms] = []

        for ph_d, supercells_d in grouped_results:
            try:
                ph = _deserialize(ph_d, Phonopy)
                supercells = [_deserialize(x, PhonopyAtoms) for x in supercells_d]
                ph_sets.append(ph)
                supercells_per_ph.append(supercells)
                supercells_flat.extend(supercells)
            except Exception as e:
                print(f"Input {i}: deserialize Phonopy payload failed: {e}")
                ph_sets = []
                supercells_per_ph = []
                supercells_flat = []
                break

        ph_sets_per_input.append(ph_sets)
        supercells_per_ph_per_input.append(supercells_per_ph)
        supercells_flat_per_input.append(supercells_flat)

    t4 = time.time()
    if debug_time:
        print("t4", t4 - t3)

    # Calculate FC2
    supercells_flat = [
        sc for flat in supercells_flat_per_input for sc in flat
    ]
    fc_structures: list[Structure] = [
        get_pmg_structure(sc) for sc in supercells_flat
    ]

    t5 = time.time()
    if debug_time:
        print("t5", t5 - t4)
    
    # print(f"Calculating {len(fc_structures)} FC2s")
    try:
        fc_results = ts.static(
            system=fc_structures, model=model, autobatcher=binning_autobatcher
        )
        forces: np.ndarray = (
            np.concatenate([r["forces"].detach().cpu().numpy() for r in fc_results])
            if fc_results
            else np.array([])
        )
    except Exception as e:
        print(f"[cte] FC2 force calculation failed: {e}")
        return [None] * n_structures

    t6 = time.time()
    if debug_time:
        print("t6", t6 - t5)

    temperatures_arr = np.array(config.temperatures)
    if len(temperatures_arr) <= 1:
        t_step = 1
    else:
        t_step = int(
            (temperatures_arr[-1] - temperatures_arr[0]) / (len(temperatures_arr) - 1)
        )

    # Calculate QHA properties (parallel over input structures)
    results: list[DataQHA] = []
    force_blocks: list[np.ndarray] = []
    offset = 0
    for i in range(n_inputs):
        per_input_force_count = 0
        for ph, supercells in zip(ph_sets_per_input[i], supercells_per_ph_per_input[i]):
            n_atoms = len(ph.supercell)  # pyright: ignore[reportArgumentType]
            per_input_force_count += n_atoms * len(supercells)
        force_blocks.append(forces[offset : offset + per_input_force_count])
        offset += per_input_force_count

    qha_tasks = [
        (
            ph_sets_per_input[i],
            all_relaxed_per_input[i],
            relaxation_energies_per_input[i],
            supercells_per_ph_per_input[i],
            force_blocks[i],
            temperatures_arr,
            config.mesh,
            t_step,
            i
        )
        for i in range(n_inputs)
    ]

    # In-process (no wire round-trip): [_calculate_qha_for_input(t) for t in qha_tasks]
    # Nested Ray: see _qha_task_from_wire (supercell sync) and _ray_calculate_qha_for_input (num_cpus=0).
    #qha_outputs = [_calculate_qha_for_input(t) for t in qha_tasks]
    qha_outputs: list[dict | None] = [None] * n_inputs
    valid_refs: list[tuple[int, Any]] = []
    for i, task in enumerate(qha_tasks):
        valid_refs.append((i, _ray_calculate_qha_for_input.remote(_serialize(task))))
    if valid_refs:
        try:
            valid_outputs = ray.get([ref for _, ref in valid_refs])
            for (i, _), out in zip(valid_refs, valid_outputs):
                qha_outputs[i] = out
        except Exception as e:
            print(f"[cte] ray.get qha outputs failed: {e}")
            return [None] * n_structures

    for out in qha_outputs:
        if out is None:
            results.append(None)
        else:
            results.append(
                DataQHA(
                    temperatures=temperatures_arr,
                    bulk_modulus_temperature=out["bulk_modulus_temperature"],
                    heat_capacity_temperature=out["heat_capacity_temperature"],
                    volume_temperature=out["volume_temperature"],
                    gibbs_temperature=out["gibbs_temperature"],
                    thermal_expansion=out["thermal_expansion"],
                    volumes=out["volumes"],
                    electronic_energies=out["electronic_energies"],
                    phonon_free_energies=out["phonon_free_energies"],
                )
            )

    # for idx, data in enumerate(results):
    #     print(idx, dir(data))
    #     for key in dir(data):
    #         if key.startswith('_'): continue
    #         value = getattr(data, key)
    #         if callable(value): continue
    #         if isinstance(value, np.ndarray):
    #             print(f"{key:25} | Shape: {value.shape}")
    #         else:
    #             print(f"{key:25} | Value: {value}")
    
    # 2structures: config displacement=0.03 mesh=[20, 20, 20] supercell_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 4]] temperatures=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500] length_factors=[0.85, 0.8714, 0.8929, 0.9143, 0.9357, 0.9571, 0.9786, 1.0, 1.0214, 1.0429, 1.0643, 1.0857, 1.1071, 1.1286, 1.15] symprec=1e-05 nsteps=300 fmax=0.0001
    # bulk_modulus_temperature  | Shape: (50,)
    # electronic_energies       | Shape: (15,)
    # gibbs_temperature         | Shape: (50,)
    # heat_capacity_temperature | Shape: (50,)
    # model_config              | Value: {'arbitrary_types_allowed': True}
    # phonon_free_energies      | Shape: (15, 51)
    # temperatures              | Shape: (51,)
    # thermal_expansion         | Shape: (50,)
    # volume_temperature        | Shape: (50,)
    # volumes                   | Shape: (15,)

    return results



def plot_thermal_expansion(
    data_list: list[DataQHA],
    titles: list[str] | None = None,
) -> go.Figure:
    """Plot CTE vs temperature for all structures in one figure, one subplot per structure.

    Args:
        data_list: List of DataQHA (one per structure).
        titles: Optional subplot titles (default: "Structure 1", "Structure 2", ...).

    Returns:
        Figure with one row per structure.
    """
    n = len(data_list)
    if n == 0:
        return go.Figure()
    if titles is None:
        titles = [f"Structure {i + 1}" for i in range(n)]
    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=titles[:n],
        vertical_spacing=0.08,
        shared_xaxes=True,
    )
    y_ranges: list[tuple[float, float]] = []
    for i, data in enumerate(data_list):
        if data.thermal_expansion is None:
            raise ValueError(
                f"DataQHA for structure {i + 1} does not contain thermal_expansion data."
            )
        temperatures = data.temperatures
        alpha = np.array(data.thermal_expansion)
        n_trim = min(len(temperatures), len(alpha))
        temperatures = temperatures[:n_trim]
        alpha = alpha[:n_trim]
        alpha_scaled = alpha * 1e6
        y_ranges.append((float(np.min(alpha_scaled)), float(np.max(alpha_scaled))))
        fig.add_trace(
            go.Scatter(
                x=temperatures,
                y=alpha_scaled,
                mode="lines",
                line={"color": "#1f77b4", "width": 2.5},
                name="α(T)",
                hovertemplate=(
                    "T = %{x:.0f} K<br>"
                    "α = %{y:.2f} × 10⁻⁶ K⁻¹"
                    "<extra></extra>"
                ),
            ),
            row=i + 1,
            col=1,
        )
        idx_300 = int(np.abs(temperatures - 300).argmin())
        alpha_300 = alpha_scaled[idx_300]
        fig.add_trace(
            go.Scatter(
                x=[temperatures[idx_300]],
                y=[alpha_300],
                mode="markers",
                marker={
                    "color": "red",
                    "size": 10,
                    "symbol": "circle",
                    "line": {"color": "darkred", "width": 1.5},
                },
                name=f"α(300 K) = {alpha_300:.2f} × 10⁻⁶ K⁻¹",
                showlegend=True,
            ),
            row=i + 1,
            col=1,
        )
    fig.update_xaxes(title_text="Temperature (K)")
    fig.update_yaxes(title_text="α  (10⁻⁶ K⁻¹)")
    shapes = []
    for i in range(n):
        xref = "x" if i == 0 else f"x{i + 1}"
        yref = "y" if i == 0 else f"y{i + 1}"
        y_lo, y_hi = y_ranges[i]
        pad = (y_hi - y_lo) * 0.05 or 1.0
        shapes.append(
            {
                "type": "line",
                "xref": xref,
                "yref": yref,
                "x0": 300,
                "y0": y_lo - pad,
                "x1": 300,
                "y1": y_hi + pad,
                "line": {"dash": "dash", "color": "grey", "width": 1.5},
                "opacity": 0.7,
            }
        )
    fig.update_layout(
        shapes=shapes,
        template="plotly_white",
        font={"size": 14},
        height=600 * n,
        width=900,
        showlegend=False,
    )
    return fig
# 从数学和物理角度看，alpha（即 $ \alpha $）通常定义为单位温度变化引起的体积相对变化率：$$\alpha(T) = \frac{1}{V} \left( \frac{\partial V}{\partial T} \right)_P$$
# alpha_300: 室温下的热膨胀参考值

def plot_free_energy_volume(
    data_list: list[DataQHA],
    titles: list[str] | None = None,
    temperature_step: int = 5,
) -> go.Figure:
    """Plot F(V,T) for all structures in one figure, one subplot per structure.

    Args:
        data_list: List of DataQHA (one per structure).
        titles: Optional subplot titles (default: "Structure 1", "Structure 2", ...).
        temperature_step: Plot every Nth temperature per subplot.

    Returns:
        Figure with one row per structure.
    """
    n = len(data_list)
    if n == 0:
        return go.Figure()
    if titles is None:
        titles = [f"Structure {i + 1}" for i in range(n)]
    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=titles[:n],
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    def _hue_to_hex(hue: float) -> str:
        r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.9, 0.95)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    EV_TO_KJ_MOL = 96.4853365

    for idx, data in enumerate(data_list):
        if (
            data.volumes is None
            or data.electronic_energies is None
            or data.phonon_free_energies is None
        ):
            raise ValueError(
                "DataQHA must contain volumes, electronic_energies, and "
                "phonon_free_energies to plot free energy curves."
            )
        temperatures = data.temperatures
        volumes = np.array(data.volumes)
        electronic_energies = np.array(data.electronic_energies)
        phonon_fe = np.array(data.phonon_free_energies)
        e_static_kjmol = electronic_energies * EV_TO_KJ_MOL
        sort_idx = np.argsort(volumes)
        volumes_sorted = volumes[sort_idx]
        e_static_sorted = e_static_kjmol[sort_idx]
        phonon_fe_sorted = phonon_fe[sort_idx, :]
        temp_indices = list(range(0, len(temperatures), temperature_step))
        selected_temps = temperatures[temp_indices]
        n_curves = len(temp_indices)
        all_F_total = []
        for t_idx in temp_indices:
            F_total = e_static_sorted + phonon_fe_sorted[:, t_idx]
            all_F_total.append(F_total)
        global_min = min(F.min() for F in all_F_total)
        colors = [
            _hue_to_hex(240 - 240 * i / max(n_curves - 1, 1))
            for i in range(n_curves)
        ]
        row = idx + 1
        for curve_i, temp in enumerate(selected_temps):
            F_norm = all_F_total[curve_i] - global_min
            fig.add_trace(
                go.Scatter(
                    x=volumes_sorted,
                    y=F_norm,
                    mode="lines+markers",
                    line={"color": colors[curve_i], "width": 2},
                    marker={"color": colors[curve_i], "size": 4},
                    name=f"{temp:.0f} K",
                    showlegend=False,
                    hovertemplate=(
                        f"T = {temp:.0f} K<br>"
                        "V = %{x:.2f} ų<br>"
                        "F − F_min = %{y:.4f} kJ/mol"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
        eq_volumes = []
        eq_energies = []
        for curve_i in range(n_curves):
            F_norm = all_F_total[curve_i] - global_min
            min_idx = int(np.argmin(F_norm))
            eq_volumes.append(volumes_sorted[min_idx])
            eq_energies.append(F_norm[min_idx])
        fig.add_trace(
            go.Scatter(
                x=eq_volumes,
                y=eq_energies,
                mode="markers",
                marker={
                    "color": "red",
                    "size": 8,
                    "symbol": "diamond",
                    "line": {"color": "darkred", "width": 1},
                },
                name="V_eq(T)",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
    fig.update_xaxes(title_text="Volume (ų)")
    fig.update_yaxes(title_text="F(V, T) − F_min  (kJ/mol)")
    fig.update_layout(
        template="plotly_white",
        font={"size": 14},
        height=600 * n,
        width=900,
    )
    return fig
# 这段代码的功能是绘制自由能-体积 (Free Energy vs. Volume, $F-V$) 曲线图




# 3.5min with 2 structures
# now 3min47s for 16 structures (8cores)

if __name__ == "__main__":
    ray.init(num_cpus=16, num_gpus=1)
    
    # Load NequIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Si diamond structures
    si_structure: Atoms = bulk(name="Si", crystalstructure="diamond", a=5.43, cubic=True)
    structure: Structure = AseAtomsAdaptor.get_structure(atoms=si_structure)  # pyright: ignore[reportArgumentType]
    structures = [copy.deepcopy(structure) for _ in range(1)]

    # Calculate QHA properties
    data_qha_list = ray.get(calculate_qha.remote(structures))

    # Plot CTE and F(V,T)
    fig_cte = plot_thermal_expansion(data_qha_list)
    fig_fev = plot_free_energy_volume(data_qha_list)
    cte_div = pio.to_html(fig_cte, full_html=False, include_plotlyjs=True)
    fev_div = pio.to_html(fig_fev, full_html=False, include_plotlyjs=False)
    html = f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"/><title>QHA Results</title></head>
<body>
{cte_div}
{fev_div}
</body>
</html>"""
    with open("cte.html", "w") as f:
        f.write(html)
