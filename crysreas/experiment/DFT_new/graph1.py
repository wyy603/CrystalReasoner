import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crysreas.experiment.assets.plot_helpers import add_panel_label, apply_publication_style

apply_publication_style()
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 18

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "dft_relax"
OUTPUT_PATH = EXPERIMENT_DIR / "graph1.pdf"
MODEL_ORDER = [
    "No_Thinking",
    "With_Thinking",
    "RL_No_Thinking",
    "RL_With_Thinking",
]
MODEL_MATCH_ORDER = sorted(MODEL_ORDER, key=len, reverse=True)

MODEL_STYLES = {
    "No_Thinking": {"color": "#1f77b4", "label": "CrysReas-Base"},
    "With_Thinking": {"color": "#d62728", "label": "CrysReas-Thinking"},
    "RL_No_Thinking": {"color": "#2ca02c", "label": "CrysReas-RL"},
    "RL_With_Thinking": {"color": "#ff7f0e", "label": "CrysReas"},
}


def infer_model_name(file_path: Path) -> str | None:
    """Infer the model name from a JSON file name."""
    stem = file_path.stem
    base_name = stem[:-6] if stem.endswith("_ehull") else stem
    for model_name in MODEL_MATCH_ORDER:
        if base_name.endswith(model_name):
            return model_name
    return None


def load_e_above_hull_values() -> dict[str, np.ndarray]:
    """Load e_above_hull values from JSON files grouped by model."""
    values_by_model: dict[str, list[float]] = {model_name: [] for model_name in MODEL_ORDER}

    for json_path in sorted(DATA_DIR.glob("*_ehull.json")):
        model_name = infer_model_name(json_path)
        if model_name is None:
            continue

        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        e_above_hull = payload.get("e_above_hull")
        if isinstance(e_above_hull, (int, float)):
            values_by_model[model_name].append(float(e_above_hull))

    return {
        model_name: np.asarray(values, dtype=float)
        for model_name, values in values_by_model.items()
    }


def silverman_bandwidth(values: np.ndarray) -> float:
    """Estimate KDE bandwidth with Silverman's rule."""
    if values.size < 2:
        return 0.1

    std = np.std(values, ddof=1)
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    if sigma <= 0:
        sigma = max(abs(values.mean()) * 0.1, 0.1)
    return max(5 * sigma * values.size ** (-1 / 5), 1e-3)


def compute_kde_counts(values: np.ndarray, x_grid: np.ndarray, bin_width: float) -> np.ndarray:
    """Compute a KDE curve scaled to histogram counts."""
    if values.size == 0:
        return np.zeros_like(x_grid)

    bandwidth = silverman_bandwidth(values)
    scaled = (x_grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    density /= values.size * bandwidth * np.sqrt(2 * np.pi)
    return density * values.size * bin_width


def draw(
    ax: plt.Axes,
    values_by_model: dict[str, np.ndarray],
    *,
    panel_label: str | None = None,
) -> None:
    """Draw the combined histogram and KDE figure on a provided axis."""
    non_empty_values = [values for values in values_by_model.values() if values.size > 0]
    if not non_empty_values:
        raise ValueError("No valid e_above_hull values were found in dft_relax JSON files.")

    all_values = np.concatenate(non_empty_values)
    x_min = float(all_values.min())
    x_max = float(all_values.max())
    padding = max((x_max - x_min) * 0.05, 0.05)
    bin_count = 20
    bins = np.linspace(x_min - padding, x_max + padding, bin_count + 1)
    bin_width = bins[1] - bins[0]
    x_grid = np.linspace(bins[0], bins[-1], 800)

    for model_name in MODEL_ORDER:
        values = values_by_model[model_name]
        if values.size == 0:
            continue

        style = MODEL_STYLES[model_name]
        ax.hist(
            values,
            bins=bins,
            alpha=0.4,
            color=style["color"],
            edgecolor=style["color"],
            linewidth=0.8,
        )
        kde_counts = compute_kde_counts(values, x_grid, bin_width)
        mu = float(np.mean(values))
        sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        ax.plot(
            x_grid,
            kde_counts,
            color=style["color"],
            linewidth=2.2,
            label=rf'{style["label"]} $(n={values.size},\ \mu={mu:.3f},\ \sigma={sigma:.3f})$',
        )

    ax.set_xlabel(r"$E_{hull}$ (eV/atom)")
    ax.set_ylabel("Count")
    ax.set_title(r"$E_{hull}$ Distribution for Four Variants")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    if panel_label is not None:
        add_panel_label(ax, panel_label)


def build_plot(values_by_model: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    draw(ax, values_by_model, panel_label="(a)")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    values_by_model = load_e_above_hull_values()
    build_plot(values_by_model)
    print(f"Saved plot to {OUTPUT_PATH}")
    for model_name in MODEL_ORDER:
        values = values_by_model[model_name]
        mu = float(np.mean(values)) if values.size > 0 else float("nan")
        sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        print(f"{model_name}: n={values.size}, mu={mu:.6f}, sigma={sigma:.6f}")


if __name__ == "__main__":
    main()
