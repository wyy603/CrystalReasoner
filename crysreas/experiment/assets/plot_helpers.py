"""
Shared plotting and confidence-interval helpers for experiment figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Z_95 = 1.96
PANEL_LABEL_SIZE = 24

MODEL_STYLE = {
    "thinking": {"label": "CrysReas-Thinking", "color": "#4c78a8", "marker": "o"},
    "no_thinking": {"label": "CrysReas-Base", "color": "#f58518", "marker": "s"},
    "with_trace": {"label": "With Thinking Trace", "color": "#4c78a8", "marker": "o"},
    "without_trace": {"label": "Without Thinking Trace", "color": "#f58518", "marker": "s"},
    "rl_thinking_mix": {"label": "CrysReas", "color": "#4c78a8", "marker": "o"},
    "spacegroup_thinking": {"label": "CrysReas-space-group", "color": "#f58518", "marker": "s"},
    "elastic_thinking": {"label": "CrysReas-ElasticProperties", "color": "#f58518", "marker": "s"},
    "rl_cte_thinking": {"label": "CrysReas-ThermalExpansion", "color": "#f58518", "marker": "s"},
    "remove": {"label": "Remove", "color": "#4c78a8"},
    "replace": {"label": "Replace", "color": "#f58518"},
    "original": {"label": "Original", "color": "#54a24b"},
}


def apply_publication_style() -> None:
    """Apply a shared plotting style for all exported figures."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["font.size"] = 27
    plt.rcParams["axes.titlesize"] = 30
    plt.rcParams["axes.labelsize"] = 30
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 22
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mean_ci_95(values: pd.Series | list[float] | np.ndarray) -> dict[str, float]:
    """Return mean and 95% CI using 1.96 * SEM."""
    arr = pd.Series(values, dtype="float64").dropna().to_numpy(dtype=float)
    n_samples = int(arr.size)
    if n_samples == 0:
        return {
            "n_samples": 0,
            "mean": float("nan"),
            "sem": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    mean = float(arr.mean())
    if n_samples < 2:
        sem = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        sem = float(arr.std(ddof=1) / np.sqrt(n_samples))
        delta = Z_95 * sem
        ci_low = mean - delta
        ci_high = mean + delta
    return {
        "n_samples": n_samples,
        "mean": mean,
        "sem": sem,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def wilson_ci_95(true_count: int | float, n_samples: int | float) -> dict[str, float]:
    """Return ratio and Wilson 95% CI."""
    n = int(n_samples)
    k = int(true_count)
    if n <= 0:
        return {
            "n_samples": 0,
            "true_count": 0,
            "ratio": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    p = k / n
    z2 = Z_95**2
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = Z_95 * np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * (n**2)))) / denom
    return {
        "n_samples": n,
        "true_count": k,
        "ratio": float(p),
        "ci_low": float(max(0.0, center - half)),
        "ci_high": float(min(1.0, center + half)),
    }


def aggregate_mean_curve(df: pd.DataFrame, *, group_cols: list[str], value_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys, strict=True)}
        stats = mean_ci_95(group[value_col])
        row.update(
            {
                "n_samples": stats["n_samples"],
                value_col: stats["mean"],
                f"{value_col}_sem": stats["sem"],
                f"{value_col}_ci_low": stats["ci_low"],
                f"{value_col}_ci_high": stats["ci_high"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_binomial_curve(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        valid = pd.Series(group[value_col]).dropna().astype(bool)
        row = {col: key for col, key in zip(group_cols, keys, strict=True)}
        stats = wilson_ci_95(int(valid.sum()), int(valid.size))
        row.update(
            {
                "n_samples": stats["n_samples"],
                value_col: stats["ratio"],
                f"{value_col}_true_count": stats["true_count"],
                f"{value_col}_ci_low": stats["ci_low"],
                f"{value_col}_ci_high": stats["ci_high"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_panel_label(ax: Any, label: str, *, fontsize: int = PANEL_LABEL_SIZE + 10) -> None:
    ax.text(
        -0.22,
        1.11,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        clip_on=False,
    )


def _yerr_from_ci(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    lower = y - low
    upper = high - y
    lower = np.where(np.isfinite(lower) & (lower >= 0), lower, np.nan)
    upper = np.where(np.isfinite(upper) & (upper >= 0), upper, np.nan)
    return np.vstack([lower, upper])


def add_errorbars(ax: Any, x: np.ndarray, y: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray) -> None:
    yerr = _yerr_from_ci(y, ci_low, ci_high)
    finite = np.isfinite(yerr).any(axis=0)
    if not finite.any():
        return
    ax.errorbar(
        x[finite],
        y[finite],
        yerr=yerr[:, finite],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=4,
    )


def plot_curve_with_ci(
    ax: Any,
    *,
    df: pd.DataFrame,
    group_col: str,
    groups: list[str] | tuple[str, ...],
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    xticks: list[int] | np.ndarray | None = None,
    ylim: tuple[float, float] | None = None,
    legend_loc: str = "best",
) -> None:
    for group_name in groups:
        group_df = df[df[group_col] == group_name].sort_values(x_col)
        if len(group_df) == 0:
            continue
        style = MODEL_STYLE[group_name]
        x = group_df[x_col].to_numpy(dtype=float)
        y = group_df[y_col].to_numpy(dtype=float)
        low = group_df.get(f"{y_col}_ci_low", pd.Series(np.nan, index=group_df.index)).to_numpy(dtype=float)
        high = group_df.get(f"{y_col}_ci_high", pd.Series(np.nan, index=group_df.index)).to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=4.5,
            label=style["label"],
            zorder=3,
        )
        add_errorbars(ax, x, y, low, high)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc=legend_loc)


def plot_grouped_bars(
    ax: Any,
    *,
    x: np.ndarray,
    series: list[dict[str, Any]],
    xticks: np.ndarray,
    xtick_labels: list[str],
    ylabel: str,
    xlabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    legend_loc: str = "best",
    rotation: float = 0.0,
    ha: str = "center",
) -> None:
    for item in series:
        ax.bar(
            item["x"],
            item["y"],
            width=item["width"],
            color=item["color"],
            label=item["label"],
            zorder=2,
        )
        add_errorbars(
            ax,
            np.asarray(item["x"], dtype=float),
            np.asarray(item["y"], dtype=float),
            np.asarray(item["ci_low"], dtype=float),
            np.asarray(item["ci_high"], dtype=float),
        )
    ax.set_xticks(xticks, xtick_labels, rotation=rotation, ha=ha)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc=legend_loc)
