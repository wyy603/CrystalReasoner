"""Elastic properties and fmax computation (MatterSim / torch_sim)."""

import copy
import gc

import numpy as np
import torch
import torch_sim as ts
from pydantic import BaseModel, ConfigDict
from pymatgen.core.structure import Structure
from torch_sim.elastic import get_bravais_type
from torch_sim.models.mattersim import MatterSimModel
from torch_sim.optimizers.state import OptimState
from mattersim.forcefield.potential import Potential
from .models import get_potential
import ray

MAX_MEMORY_SCALER = 43000
MEMORY_SCALES_WITH = "n_atoms_x_density"
model = None


class DataElastic(BaseModel):
    """Elastic properties computed from elastic tensor analysis.

    Attributes:
        elastic_tensor: 6x6 elastic tensor in Voigt notation (GPa).
        bulk_modulus: Bulk modulus (GPa).
        shear_modulus: Shear modulus (GPa).
        young_modulus: Young's modulus (GPa).
        poisson_ratio: Poisson's ratio (dimensionless).
        pugh_ratio: Pugh's ratio (dimensionless).
        energy: Total energy of the relaxed structure (eV).
        relaxed_structure: Relaxed pymatgen Structure.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    elastic_tensor: np.ndarray
    bulk_modulus: float
    shear_modulus: float
    young_modulus: float
    poisson_ratio: float
    pugh_ratio: float
    energy: float


class ConfigElastic(BaseModel):
    """Configuration for elastic properties calculation.

    Attributes:
        nsteps: Maximum number of relaxation steps. Defaults to 300.
        fmax: Force convergence tolerance in eV/A. Defaults to 1e-3.
    """

    nsteps: int = 300
    fmax: float = 1e-4


def compute_fmax(structures: list) -> torch.Tensor:
    global model
    num_structures = len(structures)
    try:
        if model is None:
            model = MatterSimModel(get_potential(device = "cuda"))

        model._compute_forces = True
        state = ts.io.structures_to_state(structures, device=model.device, dtype=model.dtype)
        output = model(state)
        forces = output["forces"]
        atom_force_norms = torch.linalg.norm(forces, dim=-1)

        system_indices = state.system_idx
        fmax_per_structure = torch.zeros(num_structures, device=model.device, dtype=model.dtype)
        fmax_per_structure.scatter_reduce_(
            dim=0,
            index=system_indices,
            src=atom_force_norms,
            reduce="amax",
            include_self=False,
        )
        return fmax_per_structure
    except Exception as e:
        print(f"[elastic] compute_fmax failed: {e}")
        return torch.full((num_structures,), float("nan"), dtype=torch.float32)

@ray.remote(num_gpus=0.5)
def calculate_elastic_properties(
    structures: list[Structure],
    config: ConfigElastic | None = None,
    relaxed=True,
    debug: bool = False,
) -> (
    list[DataElastic | None]
    | tuple[list[DataElastic | None], list[dict[str, object] | None]]
):
    """Calculate elastic properties for a list of structures using the provided model.

    Args:
        model: MaceModel instance for energy/force/stress evaluation.
        structures: List of pymatgen Structures to compute elastic properties for.
        config: Configuration for the elastic calculation.

    Returns:
        List of DataElastic (elastic tensor, moduli, relaxed structure) for each structure.
    """

    global model
    if model is None:
        model = MatterSimModel(get_potential(device = "cuda"))

    if config is None:
        config = ConfigElastic()

    debug_info: list[dict[str, object] | None] | None = (
        [None] * len(structures) if debug else None
    )

    try:
        # Set up autobatcher
        autobatcher = ts.BinningAutoBatcher(
            model=model,
            memory_scales_with=MEMORY_SCALES_WITH,
            max_memory_scaler=MAX_MEMORY_SCALER,
        )

        # Relax structures
        if relaxed:
            relaxed_structures = structures
        else:
            model._compute_forces = True
            state = ts.io.structures_to_state(structures, device=model.device, dtype=model.dtype)
            relaxed_state = ts.optimize(
                system=state,
                model=model,
                optimizer=ts.optimizers.Optimizer.fire,
                init_kwargs={
                    "cell_filter": ts.optimizers.cell_filters.CellFilter.frechet,  # pyright: ignore[reportAttributeAccessIssue]
                    "constant_volume": False,
                    "hydrostatic_strain": False,
                },
                max_steps=config.nsteps,
                convergence_fn=ts.runners.generate_force_convergence_fn(
                    force_tol=config.fmax,
                    include_cell_forces=True,
                ),
            )
            relaxed_structures: list[Structure] = ts.io.state_to_structures(relaxed_state)
            if debug and debug_info is not None:
                try:
                    debug_state = ts.io.structures_to_state(
                        relaxed_structures, device=model.device, dtype=model.dtype
                    )
                    debug_output = model(debug_state)
                    debug_forces = debug_output["forces"]
                    atom_force_norms = torch.linalg.norm(debug_forces, dim=-1)
                    system_indices = debug_state.system_idx
                    fmax_per_structure = torch.zeros(
                        len(relaxed_structures), device=model.device, dtype=model.dtype
                    )
                    fmax_per_structure.scatter_reduce_(
                        dim=0,
                        index=system_indices,
                        src=atom_force_norms,
                        reduce="amax",
                        include_self=False,
                    )
                    fmax_values = fmax_per_structure.detach().cpu().numpy().tolist()
                    for idx, fmax in enumerate(fmax_values):
                        fmax_float = float(fmax)
                        debug_info[idx] = {
                            "first_relax_fmax": fmax_float,
                            "first_relax_converged_mlip": fmax_float <= float(config.fmax),
                            "first_relax_force_tol": float(config.fmax),
                        }
                except Exception as debug_e:
                    for idx in range(len(structures)):
                        debug_info[idx] = {
                            "first_relax_debug_error": str(debug_e),
                        }
    except Exception as e:
        print(f"[elastic] batch setup/relax failed: {e}")
        if debug:
            fallback_debug = [
                {"batch_setup_or_relax_error": str(e)} for _ in range(len(structures))
            ]
            return [None] * len(structures), fallback_debug
        return [None] * len(structures)

    # Calculate elastic properties
    results: list[DataElastic | None] = []
    for relaxed_structure in relaxed_structures:
        try:
            with torch.set_grad_enabled(True):
                state_i = ts.io.structures_to_state(
                    [relaxed_structure], device=model.device, dtype=model.dtype
                )
                state_i.positions.requires_grad_(True)
                state_i.cell.requires_grad_(True)
                output = model(state_i)

                energy = output["energy"].detach()
                forces = output["forces"].detach()
                stress = output["stress"].detach()

                optim_state = OptimState.from_state(
                    state_i, forces=forces, energy=energy, stress=stress
                )
                bravais_type = get_bravais_type(optim_state)
                elastic_tensor = ts.elastic.calculate_elastic_tensor(
                    optim_state,
                    model=model,
                    bravais_type=bravais_type,
                    autobatcher=autobatcher,
                )
                elastic_tensor = elastic_tensor * ts.units.UnitConversion.eV_per_Ang3_to_GPa
                elastic_tensor_np = elastic_tensor.cpu().numpy()

                bulk_modulus, shear_modulus, poisson_ratio, pugh_ratio = ts.elastic.calculate_elastic_moduli(elastic_tensor_np)  # pyright: ignore[reportArgumentType]
                young_modulus = 9 * bulk_modulus * shear_modulus / (
                    3 * bulk_modulus + shear_modulus
                )

                results.append(
                    DataElastic(
                        elastic_tensor=elastic_tensor_np,
                        bulk_modulus=bulk_modulus,
                        shear_modulus=shear_modulus,
                        young_modulus=young_modulus,
                        poisson_ratio=poisson_ratio,
                        pugh_ratio=pugh_ratio,
                        energy=energy.item(),
                    )
                )
        except Exception:
            results.append(None)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if debug:
        return results, (debug_info if debug_info is not None else [None] * len(structures))
    return results


if __name__ == "__main__":
    from crysreas.utils.crystal import SimpleCrystal

    structure = SimpleCrystal.from_simple_no_sym("""P1
15.66504400 3.02928400 5.63711369
90 107.847 90
Na 1 0.66651100 0.00000000 0.00178500
Na 1 0.50012300 0.50000000 0.99972500
Na 1 0.83202500 0.50000000 0.99566200
Na 1 0.16651100 0.50000000 0.00178500
Na 1 0.00012300 0.00000000 0.99972500
Na 1 0.33202500 0.00000000 0.99566200
Mn 1 0.99982300 0.50000000 0.50073300
Mn 1 0.49982300 0.00000000 0.50073300
Co 1 0.67373600 0.50000000 0.49349200
Co 1 0.17373600 0.00000000 0.49349200
Ni 1 0.83296400 0.00000000 0.50580000
Ni 1 0.33296400 0.50000000 0.50580000
O 1 0.57231700 0.50000000 0.69008100
O 1 0.91867900 0.50000000 0.69828000
O 1 0.74119500 0.00000000 0.71081800
O 1 0.92924600 0.00000000 0.30547000
O 1 0.74877100 0.50000000 0.29142500
O 1 0.58461100 0.00000000 0.30672800
O 1 0.07231700 0.00000000 0.69008100
O 1 0.41867900 0.00000000 0.69828000
O 1 0.24119500 0.50000000 0.71081800
O 1 0.42924600 0.50000000 0.30547000
O 1 0.24877100 0.00000000 0.29142500
O 1 0.08461100 0.50000000 0.30672800""").structure
    structures = [copy.deepcopy(structure) for _ in range(16)]

    print(compute_fmax(structures))

    # Calculate elastic properties
    props_list = calculate_elastic_properties(structures)

    # Print results
    for i, props in enumerate(props_list):
        print(f"Structure {i + 1}:")
        print("  bulk modulus: ", props.bulk_modulus)
        print("  shear modulus: ", props.shear_modulus)
        print("  young modulus: ", props.young_modulus)
        print("  poisson ratio: ", props.poisson_ratio)
        print("  pugh ratio: ", props.pugh_ratio)

# 2min16s for 16 structures
# 2min for 64 structures 14000MB
