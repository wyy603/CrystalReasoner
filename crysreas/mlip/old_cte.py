"""QHA calculations"""

# pyright: basic

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
import tqdm
import plotly.io as pio


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


def calculate_qha(
    model,
    structures: list[Structure],
    config: ConfigQHA | None = None,
) -> list[DataQHA]:
    """Calculate quasi-harmonic thermal properties for a list of structures.

    Args:
        model: MaceModel instance for energy/force evaluation.
        structures: List of pymatgen Structures to compute QHA for.
        config: Configuration for the QHA calculation.

    Returns:
        List of DataQHA (temperature-dependent thermal properties) for each structure.
    """
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

    # Relax structures
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
    n_inputs = len(relaxed_structures)
    n_volumes = len(config.length_factors)

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
    print(f"Relaxing {len(all_scaled_structs)} scaled structures")
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
    for all_relaxed_structs in tqdm.tqdm(all_relaxed_per_input, desc="Building Phonopy objects"):
        ph_sets: list[Phonopy] = []
        supercells_per_ph: list[list[PhonopyAtoms]] = []
        supercells_flat: list[PhonopyAtoms] = []
        for struct in all_relaxed_structs:
            phonopy_atoms = PhonopyAtoms(
                symbols=[str(site.specie) for site in struct],
                positions=struct.cart_coords,
                cell=struct.lattice.matrix,
            )
            ph = Phonopy(
                unitcell=phonopy_atoms,
                supercell_matrix=np.array(config.supercell_matrix),
                primitive_matrix="auto",
                symprec=config.symprec,
            )
            ph.generate_displacements(distance=config.displacement)
            supercells = ph.supercells_with_displacements  # pyright: ignore[reportAssignmentType]
            supercells_flat.extend(supercells)  # pyright: ignore[reportArgumentType]
            supercells_per_ph.append(supercells)  # pyright: ignore[reportArgumentType]
            ph_sets.append(ph)
        ph_sets_per_input.append(ph_sets)
        supercells_per_ph_per_input.append(supercells_per_ph)
        supercells_flat_per_input.append(supercells_flat)

    # Calculate FC2
    supercells_flat = [
        sc for flat in supercells_flat_per_input for sc in flat
    ]
    fc_structures: list[Structure] = [
        get_pmg_structure(sc) for sc in supercells_flat
    ]
    print(f"Calculating {len(fc_structures)} FC2s")
    fc_results = ts.static(
        system=fc_structures, model=model, autobatcher=binning_autobatcher
    )
    forces: np.ndarray = (
        np.concatenate([r["forces"].detach().cpu().numpy() for r in fc_results])
        if fc_results
        else np.array([])
    )

    temperatures_arr = np.array(config.temperatures)
    t_step = int(
        (temperatures_arr[-1] - temperatures_arr[0]) / (len(temperatures_arr) - 1)
    )

    # Calculate QHA properties
    results: list[DataQHA] = []
    offset = 0
    for i in tqdm.tqdm(range(n_inputs), desc="Calculating QHA properties"):
        ph_sets = ph_sets_per_input[i]
        all_relaxed_structs = all_relaxed_per_input[i]
        relaxation_energies = relaxation_energies_per_input[i]
        supercells_per_ph = supercells_per_ph_per_input[i]

        volumes = []
        energies = []
        free_energies_list = []
        entropies_list = []
        heat_capacities_list = []

        for ph, struct, energy, supercells in zip(
            ph_sets, all_relaxed_structs, relaxation_energies, supercells_per_ph
        ):
            n_atoms = len(ph.supercell)  # pyright: ignore[reportArgumentType]
            n_displacements = len(supercells)
            n_total = n_atoms * n_displacements
            ph.forces = list(
                forces[offset : offset + n_total].reshape(
                    n_displacements, n_atoms, 3
                )
            )
            offset += n_total
            ph.produce_force_constants()  # pyright: ignore[reportUnknownMemberType]
            ph.run_mesh(config.mesh)
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
        results.append(
            DataQHA(
                temperatures=temperatures_arr,
                bulk_modulus_temperature=qha.bulk_modulus_temperature,
                heat_capacity_temperature=np.array(qha.heat_capacity_P_numerical),
                volume_temperature=qha.volume_temperature,
                gibbs_temperature=qha.gibbs_temperature,
                thermal_expansion=np.array(qha.thermal_expansion),
                volumes=np.array(volumes),
                electronic_energies=np.array(energies),
                phonon_free_energies=np.array(free_energies_list),
            )
        )
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

from torch_sim.models.mattersim import MatterSimModel
from mattersim.forcefield.potential import Potential
from .models import get_potential
if __name__ == "__main__":
    
    # Load NequIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MAX_MEMORY_SCALER = 43000
    MEMORY_SCALES_WITH = "n_atoms_x_density"
    model = MatterSimModel(get_potential(device = "cuda"))

    # Build Si diamond structures
    si_structure: Atoms = bulk(name="Si", crystalstructure="diamond", a=5.43, cubic=True)
    structure: Structure = AseAtomsAdaptor.get_structure(atoms=si_structure)  # pyright: ignore[reportArgumentType]
    structures = [copy.deepcopy(structure) for _ in range(1)]

    # Calculate QHA properties
    data_qha_list = calculate_qha(model, structures)

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
    with open("old_cte.html", "w") as f:
        f.write(html)

