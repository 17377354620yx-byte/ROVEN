"""Three-panel, publication-quality rigid-registration error visualization."""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from .render import make_camera, render_point_clouds, resolve_render_backend
from .statistics import compute_error_statistics, plot_error_distribution


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) == 0:
        raise ValueError(f"{name} must be a non-empty array with shape (N, 3)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _validate_transform(transform: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(transform, dtype=np.float64)
    if result.shape == (3, 4):
        result = np.vstack([result, [0.0, 0.0, 0.0, 1.0]])
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4) or (3, 4)")
    if not np.isfinite(result).all() or not np.allclose(result[3], [0, 0, 0, 1]):
        raise ValueError(f"{name} is not a finite homogeneous transform")
    return result


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a source-to-target transform using the project's row-vector rule."""
    points = _validate_points(points, "points")
    transform = _validate_transform(transform, "transform")
    return points @ transform[:3, :3].T + transform[:3, 3]


def compute_registration_error(
    source: np.ndarray,
    T_pred: np.ndarray,
    T_gt: np.ndarray,
    *,
    scale_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``||T_pred(p)-T_gt(p)||`` for each source point, in mm."""
    source = _validate_points(source, "source")
    scale = float(scale_factor)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale_factor must be finite and greater than zero")
    predicted = apply_transform(source, T_pred)
    ground_truth = apply_transform(source, T_gt)
    errors_mm = np.linalg.norm(predicted - ground_truth, axis=1) * scale
    return predicted, ground_truth, errors_mm


def _color_limits(errors: np.ndarray, error_range: Optional[Tuple[float, float]]):
    if error_range is None:
        lower, upper = float(np.min(errors)), float(np.max(errors))
    else:
        lower, upper = map(float, error_range)
        if not np.isfinite([lower, upper]).all() or lower < 0 or upper <= lower:
            raise ValueError("error_range must be a finite (min, max), with 0 <= min < max")
    if upper <= lower or np.isclose(upper, lower, rtol=1e-10, atol=1e-12):
        midpoint = 0.5 * (lower + upper)
        padding = max(1e-6, abs(midpoint) * 1e-3)
        lower, upper = max(0.0, midpoint - padding), midpoint + padding
    return lower, upper


def generate_error_heatmap(
    source: np.ndarray,
    target: np.ndarray,
    T_pred: np.ndarray,
    T_gt: np.ndarray,
    *,
    output_path: str = "output/visualization/error_heatmap/case_0001_heatmap.png",
    error_path: Optional[str] = None,
    distribution_path: Optional[str] = None,
    scale_factor: float = 1.0,
    colormap: str = "jet",
    error_range: Optional[Tuple[float, float]] = None,
    camera_direction: Optional[Sequence[float]] = None,
    point_size: float = 3.0,
    panel_size: Tuple[int, int] = (760, 620),
    dpi: int = 300,
    render_backend: str = "auto",
) -> Dict[str, object]:
    """Generate GT, prediction and error-map panels using one Open3D camera.

    The source and target must be expressed in the same normalized coordinate
    system as both transforms.  ``scale_factor`` converts normalized distances
    back to millimetres.  The returned dictionary is JSON-serializable except
    for the per-point array, which is saved separately as ``.npy``.
    """
    source = _validate_points(source, "source")
    target = _validate_points(target, "target")
    predicted, ground_truth, errors_mm = compute_registration_error(
        source, T_pred, T_gt, scale_factor=scale_factor
    )
    stats = compute_error_statistics(errors_mm)
    lower, upper = _color_limits(errors_mm, error_range)
    norm = Normalize(vmin=lower, vmax=upper, clip=True)
    cmap = matplotlib.colormaps[colormap]
    heat_colors = cmap(norm(errors_mm))[:, :3]

    camera = make_camera(
        [target, ground_truth, predicted], direction=camera_direction
    )
    width, height = map(int, panel_size)
    target_color = np.asarray([0.72, 0.75, 0.78])
    gt_color = np.asarray([0.12, 0.47, 0.71])
    pred_color = np.asarray([0.95, 0.45, 0.12])
    selected_backend = resolve_render_backend(render_backend)
    panels = [
        render_point_clouds([(target, target_color), (ground_truth, gt_color)],
                            camera, width=width, height=height, point_size=point_size,
                            backend=selected_backend),
        render_point_clouds([(target, target_color), (predicted, pred_color)],
                            camera, width=width, height=height, point_size=point_size,
                            backend=selected_backend),
        render_point_clouds([(target, target_color), (predicted, heat_colors)],
                            camera, width=width, height=height, point_size=point_size,
                            backend=selected_backend),
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if error_path is None:
        stem = path.stem[:-8] if path.stem.endswith("_heatmap") else path.stem
        error_path = str(path.with_name(stem + "_error.npy"))
    if distribution_path is None:
        stem = path.stem[:-8] if path.stem.endswith("_heatmap") else path.stem
        distribution_path = str(path.with_name(stem + "_error_distribution.png"))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), constrained_layout=True)
    titles = ("Ground Truth Alignment", "Prediction Alignment", "Error Heatmap")
    for axis, image, title in zip(axes, panels, titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=12, pad=8)
        axis.axis("off")
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axes, orientation="horizontal",
                            fraction=0.055, pad=0.035, aspect=45)
    colorbar.set_label("Error (mm)")
    summary = (
        f"Min {stats['min_mm']:.2f} mm  |  Max {stats['max_mm']:.2f} mm\n"
        f"Mean TRE {stats['mean_mm']:.2f} mm  |  RMS {stats['rms_mm']:.2f} mm"
    )
    axes[2].text(0.5, 0.91, summary, transform=axes[2].transAxes,
                 ha="center", va="top", fontsize=9, color="#222222")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    np.save(error_path, errors_mm.astype(np.float32))
    plot_error_distribution(errors_mm, distribution_path, dpi=dpi)

    return {
        "heatmap_path": str(path.resolve()),
        "error_path": str(Path(error_path).resolve()),
        "distribution_path": str(Path(distribution_path).resolve()),
        "scale_factor": float(scale_factor),
        "render_backend": selected_backend,
        "color_range_mm": [lower, upper],
        "statistics": stats,
    }
