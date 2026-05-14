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
OUTPUT_PATH = EXPERIMENT_DIR / "graph2.pdf"
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
PLOT_PAIRS = [
    ("No_Thinking", "With_Thinking"),
    ("No_Thinking", "RL_No_Thinking"),
]


def parse_file_metadata(file_path: Path) -> tuple[str, str] | None:
    """Extract mp_id and model name from a JSON file name."""
    stem = file_path.stem
    base_name = stem[:-6] if stem.endswith("_ehull") else stem
    for model_name in MODEL_MATCH_ORDER:
        suffix = f"_{model_name}"
        if base_name.endswith(suffix):
            return base_name[: -len(suffix)], model_name
    return None


def load_e_above_hull_by_mp_id() -> dict[str, dict[str, float]]:
    """Load numeric e_above_hull values keyed by mp_id and model."""
    values_by_mp_id: dict[str, dict[str, float]] = {}

    for json_path in sorted(DATA_DIR.glob("*_ehull.json")):
        metadata = parse_file_metadata(json_path)
        if metadata is None:
            continue

        mp_id, model_name = metadata
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        e_above_hull = payload.get("e_above_hull")
        if isinstance(e_above_hull, (int, float)):
            values_by_mp_id.setdefault(mp_id, {})[model_name] = float(e_above_hull)

    return values_by_mp_id


def build_pair_arrays(
    values_by_mp_id: dict[str, dict[str, float]], x_model: str, y_model: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect paired x/y arrays for a given model pair."""
    mp_ids: list[str] = []
    x_values: list[float] = []
    y_values: list[float] = []

    for mp_id in sorted(values_by_mp_id):
        model_values = values_by_mp_id[mp_id]
        if x_model in model_values and y_model in model_values:
            mp_ids.append(mp_id)
            x_values.append(model_values[x_model])
            y_values.append(model_values[y_model])

    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float), mp_ids


def add_reference_line(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray) -> None:
    """Draw the x=y line across the visible data range."""
    combined = np.concatenate([x_values, y_values])
    value_min = float(combined.min())
    value_max = float(combined.max())
    padding = max((value_max - value_min) * 0.05, 0.05)
    line_min = value_min - padding
    line_max = value_max + padding

    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        linewidth=2.0,
        color="#4d4d4d",
    )
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)


def draw_subplot(
    ax: plt.Axes,
    values_by_mp_id: dict[str, dict[str, float]],
    x_model: str,
    y_model: str,
    *,
    panel_label: str | None = None,
) -> None:
    """Draw one paired scatter subplot."""
    x_values, y_values, mp_ids = build_pair_arrays(values_by_mp_id, x_model, y_model)
    if x_values.size == 0:
        raise ValueError(f"No paired samples found for {x_model} vs {y_model}.")

    x_style = MODEL_STYLES[x_model]
    y_style = MODEL_STYLES[y_model]
    ax.scatter(
        x_values,
        y_values,
        s=60,
        alpha=0.75,
        color=y_style["color"],
        edgecolors="white",
        linewidths=0.7,
        label=f"Paired points (n={len(mp_ids)})",
    )
    add_reference_line(ax, x_values, y_values)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel(f'{x_style["label"]} ' + r"$E_{hull}$ (eV/atom)")
    ax.set_ylabel(f'{y_style["label"]} ' + r"$E_{hull}$ (eV/atom)")
    ax.set_title(f'{x_style["label"]} vs {y_style["label"]}')
    ax.legend(frameon=False, loc="upper left")
    if panel_label is not None:
        add_panel_label(ax, panel_label)


def draw(axes: np.ndarray, values_by_mp_id: dict[str, dict[str, float]]) -> None:
    """Draw the two-panel scatter comparison plot on provided axes."""
    for ax, (x_model, y_model), label in zip(axes, PLOT_PAIRS, ("(b)", "(c)"), strict=True):
        draw_subplot(ax, values_by_mp_id, x_model, y_model, panel_label=label)


def build_plot(values_by_mp_id: dict[str, dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    draw(axes, values_by_mp_id)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    values_by_mp_id = load_e_above_hull_by_mp_id()
    build_plot(values_by_mp_id)
    print(f"Saved plot to {OUTPUT_PATH}")

    for x_model, y_model in PLOT_PAIRS:
        x_values, y_values, mp_ids = build_pair_arrays(values_by_mp_id, x_model, y_model)
        print(
            f'{MODEL_STYLES[x_model]["label"]} vs {MODEL_STYLES[y_model]["label"]}: '
            f"n={len(mp_ids)}"
        )
        if mp_ids:
            diff = y_values - x_values
            print(
                f"  mean(x)={x_values.mean():.6f}, mean(y)={y_values.mean():.6f}, "
                f"mean(y-x)={diff.mean():.6f}"
            )


if __name__ == "__main__":
    main()
