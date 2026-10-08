"""Statistics and plots for per-point registration errors."""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_error_statistics(errors_mm: np.ndarray) -> Dict[str, float]:
    """Return descriptive statistics for a non-empty vector of errors in mm.

    ``rms_mm`` uses the same reduction as the project's RMS-TRE implementation:
    ``sqrt(mean(distance ** 2))``.  Whether it is the evaluation RMS-TRE depends
    on which points/distances are supplied (see the visualization README).
    """
    errors = np.asarray(errors_mm, dtype=np.float64).reshape(-1)
    if errors.size == 0:
        raise ValueError("errors_mm must contain at least one value")
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("errors_mm must contain finite, non-negative values")
    return {
        "count": int(errors.size),
        "min_mm": float(np.min(errors)),
        "max_mm": float(np.max(errors)),
        "mean_mm": float(np.mean(errors)),
        "median_mm": float(np.median(errors)),
        "p95_mm": float(np.percentile(errors, 95)),
        "rms_mm": float(np.sqrt(np.mean(np.square(errors)))),
    }


def plot_error_distribution(
    errors_mm: np.ndarray,
    output_path: str,
    *,
    bins: int = 40,
    title: str = "Point-wise registration error distribution",
    color: str = "#3977AF",
    x_range: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
) -> Dict[str, float]:
    """Save a paper-ready histogram and return its descriptive statistics."""
    stats = compute_error_statistics(errors_mm)
    errors = np.asarray(errors_mm, dtype=np.float64).reshape(-1)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    histogram_range = x_range
    if histogram_range is None and np.ptp(errors) <= max(1e-12, abs(stats["mean_mm"]) * 1e-12):
        padding = max(1e-6, abs(stats["mean_mm"]) * 1e-3)
        histogram_range = (stats["mean_mm"] - padding, stats["mean_mm"] + padding)
    ax.hist(errors, bins=bins, range=histogram_range, color=color, alpha=0.9,
            edgecolor="white", linewidth=0.45)
    for value, label, line_color in (
        (stats["mean_mm"], "Mean", "#D62728"),
        (stats["median_mm"], "Median", "#2CA02C"),
        (stats["p95_mm"], "95th percentile", "#9467BD"),
    ):
        ax.axvline(value, color=line_color, linewidth=1.5, linestyle="--",
                   label=f"{label}: {value:.2f} mm")
    ax.set_xlabel("TRE (mm)")
    ax.set_ylabel("Number of points")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stats


def plot_grouped_error_distribution(
    error_groups: Sequence[np.ndarray],
    labels: Sequence[str],
    output_path: str,
    *,
    bins: int = 50,
    dpi: int = 300,
) -> None:
    """Save overlaid, density-normalized distributions for batch selections."""
    if len(error_groups) != len(labels) or not error_groups:
        raise ValueError("error_groups and labels must be non-empty and equally sized")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = ("#2C7BB6", "#D7191C", "#7B3294", "#008837")
    for index, (group, label) in enumerate(zip(error_groups, labels)):
        values = np.asarray(group, dtype=np.float64).reshape(-1)
        if values.size:
            ax.hist(values, bins=bins, histtype="step", density=True,
                    linewidth=1.8, color=colors[index % len(colors)], label=label)
    ax.set_xlabel("TRE (mm)")
    ax.set_ylabel("Probability density")
    ax.set_title("Selected-case point-wise error distributions")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
