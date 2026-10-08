"""Publication-ready three-stage visualization for P2P liver registration.

Each case is rendered as: (a) raw input, (b) predicted registration, and
(c) a directed nearest-neighbour surface-distance heatmap on the partial area.
The native adapter preserves the complete-source -> partial-target convention
used by ``experiments/geotransformer.p2p_liver/test.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.error_heatmap import apply_transform
from visualization.render import make_camera, render_point_clouds, resolve_render_backend
from visualization.statistics import compute_error_statistics


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
DEFAULT_OUTPUT_DIR = ROOT / "output/visualization/paper_registration"
EXPERIMENT_DIR = ROOT / "experiments/geotransformer.p2p_liver"
SOURCE_COLOR = np.asarray([0.10, 0.42, 0.78], dtype=np.float64)
TARGET_COLOR = np.asarray([0.86, 0.20, 0.20], dtype=np.float64)

# Monotonic single-hue palette: low errors remain visible on white, while high
# errors gain both saturation and visual weight without rainbow hue changes.
MEDICAL_BLUE = LinearSegmentedColormap.from_list(
    "medical_blue",
    ["#DDEFF5", "#B8DCEB", "#86BED8", "#4C96C2", "#1F69A5", "#083B73"],
    N=256,
)


def _points(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty (N, 3) array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _transform(value: ArrayLike, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0.0, 0.0, 0.0, 1.0]])
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4) or (3, 4)")
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError(f"{name} is not a finite homogeneous transform")
    return matrix


def _safe_case_id(case_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_"
                   for char in str(case_id))
    return safe.strip("_") or "case_0000"


def _estimate_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1e-3
    distances, _ = cKDTree(points).query(points, k=2, workers=-1)
    positive = distances[:, 1][distances[:, 1] > 0]
    return float(np.median(positive)) if len(positive) else 1e-3


def extract_partial_source_indices(
    source_complete: ArrayLike,
    target_partial: ArrayLike,
    T_gt: ArrayLike,
    *,
    match_radius: Optional[float] = None,
    radius_multiplier: float = 2.5,
) -> Tuple[np.ndarray, float]:
    """Identify the complete-source region visible in the partial target."""
    source = _points(source_complete, "source_complete")
    target = _points(target_partial, "target_partial")
    gt_source = apply_transform(source, _transform(T_gt, "T_gt"))
    radius = float(match_radius) if match_radius is not None else (
        radius_multiplier * _estimate_spacing(target)
    )
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("match_radius must be finite and greater than zero")
    distances, _ = cKDTree(target).query(gt_source, k=1, workers=-1)
    indices = np.flatnonzero(distances <= radius)
    minimum = min(len(source), max(32, int(np.ceil(0.01 * len(source)))))
    if len(indices) < minimum:
        _, nearest_source = cKDTree(gt_source).query(target, k=1, workers=-1)
        indices = np.unique(nearest_source)
    if len(indices) == 0:
        raise ValueError("Could not extract a partial source region")
    return indices.astype(np.int64), radius


def compute_partial_registration_error(
    source_complete: ArrayLike,
    target_partial: ArrayLike,
    T_pred: ArrayLike,
    T_gt: ArrayLike,
    *,
    scale_factor: float = 1.0,
    partial_indices: Optional[np.ndarray] = None,
    partial_match_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute directed predicted-to-GT partial surface distance in millimetres.

    Returns ``(pred_partial, gt_partial, errors_mm, indices, match_radius)``.
    Each error is measured to the nearest point anywhere on the GT partial
    surface, rather than to the point with the same array index.
    """
    source = _points(source_complete, "source_complete")
    target = _points(target_partial, "target_partial")
    pred_transform = _transform(T_pred, "T_pred")
    gt_transform = _transform(T_gt, "T_gt")
    scale = float(scale_factor)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale_factor must be finite and greater than zero")
    if partial_indices is None:
        indices, radius = extract_partial_source_indices(
            source, target, gt_transform, match_radius=partial_match_radius
        )
    else:
        indices = np.asarray(partial_indices, dtype=np.int64).reshape(-1)
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(source)):
            raise ValueError("partial_indices contains no valid source points")
        indices = np.unique(indices)
        radius = float(partial_match_radius) if partial_match_radius is not None else float("nan")
    source_partial = source[indices]
    pred_partial = apply_transform(source_partial, pred_transform)
    gt_partial = apply_transform(source_partial, gt_transform)
    distances, _ = cKDTree(gt_partial).query(pred_partial, k=1, workers=-1)
    return pred_partial, gt_partial, distances * scale, indices, radius


def _fit_camera(
    point_sets: Sequence[np.ndarray],
    panel_size: Tuple[int, int],
    direction: Optional[Sequence[float]],
    *,
    fill_fraction: float,
) -> Dict[str, np.ndarray]:
    """Create a shared orientation and tightly fit all supplied point sets."""
    camera = make_camera(point_sets, direction=direction)
    points = np.concatenate(point_sets, axis=0)
    center = np.asarray(camera["center"], dtype=np.float64)
    eye = np.asarray(camera["eye"], dtype=np.float64)
    up_hint = np.asarray(camera["up"], dtype=np.float64)
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    offsets = points - center
    x, y, z = offsets @ right, offsets @ up, offsets @ forward
    aspect = float(panel_size[0]) / float(panel_size[1])
    tangent = np.tan(np.deg2rad(float(camera["fov_deg"])) * 0.5)
    required_x = np.abs(x) / (tangent * aspect * fill_fraction) - z
    required_y = np.abs(y) / (tangent * fill_fraction) - z
    distance = max(float(np.max(required_x)), float(np.max(required_y)), 1e-6)
    camera["eye"] = center - forward * distance * 1.02
    camera["up"] = up
    return camera


def _paper_statistics(errors_mm: np.ndarray) -> Dict[str, float]:
    values = compute_error_statistics(errors_mm)
    return {
        "count": values["count"],
        "min_error_mm": values["min_mm"],
        "max_error_mm": values["max_mm"],
        "mean_tre_mm": values["mean_mm"],
        "rms_tre_mm": values["rms_mm"],
        "median_error_mm": values["median_mm"],
        "p95_error_mm": values["p95_mm"],
    }


def _color_normalization(errors_mm: np.ndarray) -> Normalize:
    lower, upper = float(np.min(errors_mm)), float(np.max(errors_mm))
    if np.isclose(lower, upper, rtol=1e-10, atol=1e-12):
        padding = max(1e-6, abs(lower) * 1e-3)
        lower, upper = max(0.0, lower - padding), upper + padding
    return Normalize(vmin=lower, vmax=upper, clip=True)


def _resolve_colormap(name: str):
    if name in ("medical_blue", "blue_sequential"):
        return MEDICAL_BLUE
    return matplotlib.colormaps[name]


def generate_paper_registration_visualization(
    source_point_cloud: ArrayLike,
    target_point_cloud: ArrayLike,
    T_pred: ArrayLike,
    T_gt: ArrayLike,
    *,
    case_id: str = "case_0001",
    output_dir: Union[str, os.PathLike] = DEFAULT_OUTPUT_DIR,
    scale_factor: float = 1.0,
    partial_indices: Optional[np.ndarray] = None,
    partial_match_radius: Optional[float] = None,
    colormap: str = "medical_blue",
    camera_direction: Optional[Sequence[float]] = None,
    point_size: float = 4.5,
    heatmap_point_size: float = 5.5,
    panel_size: Tuple[int, int] = (1200, 900),
    dpi: int = 600,
    render_backend: str = "auto",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Save a compact Input / Registration / Error Heatmap figure."""
    source = _points(source_point_cloud, "source_point_cloud")
    target = _points(target_point_cloud, "target_point_cloud")
    pred_transform = _transform(T_pred, "T_pred")
    gt_transform = _transform(T_gt, "T_gt")
    source_registered = apply_transform(source, pred_transform)
    pred_partial, gt_partial, errors_mm, indices, match_radius = (
        compute_partial_registration_error(
            source, target, pred_transform, gt_transform,
            scale_factor=scale_factor, partial_indices=partial_indices,
            partial_match_radius=partial_match_radius,
        )
    )
    stats = _paper_statistics(errors_mm)
    backend = resolve_render_backend(render_backend)
    width, height = map(int, panel_size)
    comparison_camera = _fit_camera(
        [source, target, source_registered], (width, height), camera_direction,
        fill_fraction=0.82,
    )
    heatmap_camera = _fit_camera(
        [pred_partial, gt_partial], (width, height), camera_direction,
        fill_fraction=0.78,
    )
    input_image = render_point_clouds(
        [(source, SOURCE_COLOR), (target, TARGET_COLOR)], comparison_camera,
        width=width, height=height, point_size=point_size, backend=backend,
    )
    registration_image = render_point_clouds(
        [(source_registered, SOURCE_COLOR), (target, TARGET_COLOR)], comparison_camera,
        width=width, height=height, point_size=point_size, backend=backend,
    )
    normalization = _color_normalization(errors_mm)
    cmap = _resolve_colormap(colormap)
    heatmap_image = render_point_clouds(
        [(pred_partial, cmap(normalization(errors_mm))[:, :3])], heatmap_camera,
        width=width, height=height, point_size=heatmap_point_size, backend=backend,
    )

    fig = plt.figure(figsize=(12.2, 4.15), facecolor="white")
    grid = fig.add_gridspec(1, 3, left=0.012, right=0.992, top=0.91,
                            bottom=0.13, wspace=0.012)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    images = (input_image, registration_image, heatmap_image)
    titles = ("(a) Input", "(b) Registration Result", "(c) Error Heatmap")
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=11, fontweight="semibold", pad=4)
        axis.set_axis_off()

    legend_handles = [
        Line2D([], [], linestyle="", marker="o", markersize=5,
               markerfacecolor=SOURCE_COLOR, markeredgewidth=0, label="Source"),
        Line2D([], [], linestyle="", marker="o", markersize=5,
               markerfacecolor=TARGET_COLOR, markeredgewidth=0, label="Target"),
    ]
    axes[0].legend(handles=legend_handles, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, 0.015), frameon=False, fontsize=8,
                   handletextpad=0.35, columnspacing=0.9)
    annotation = (
        f"[{stats['min_error_mm']:.2f}, {stats['max_error_mm']:.2f}] mm\n"
        f"Mean {stats['mean_tre_mm']:.2f} mm  |  RMS {stats['rms_tre_mm']:.2f} mm"
    )
    axes[2].text(
        0.5, 0.035, annotation, transform=axes[2].transAxes,
        ha="center", va="bottom", fontsize=8.2, color="#111111",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88,
              "pad": 2.0},
    )
    color_axis = fig.add_axes([0.715, 0.075, 0.23, 0.024])
    scalar = ScalarMappable(norm=normalization, cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Error (mm)", fontsize=8, labelpad=1)
    colorbar.ax.tick_params(labelsize=7, length=2, pad=1)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe_id = _safe_case_id(case_id)
    figure_path = output / f"{safe_id}_heatmap.png"
    error_path = output / f"{safe_id}_error.npy"
    indices_path = output / f"{safe_id}_partial_indices.npy"
    stats_path = output / f"{safe_id}_statistics.json"
    fig.savefig(figure_path, dpi=int(dpi), facecolor="white")
    plt.close(fig)
    np.save(error_path, errors_mm.astype(np.float32))
    np.save(indices_path, indices)
    payload: Dict[str, Any] = {
        "case_id": safe_id,
        "figure_path": str(figure_path.resolve()),
        "error_path": str(error_path.resolve()),
        "partial_indices_path": str(indices_path.resolve()),
        "statistics_path": str(stats_path.resolve()),
        "transform_direction": "complete_source_to_partial_target",
        "scale_factor_mm": float(scale_factor),
        "source_point_count": int(len(source)),
        "target_partial_point_count": int(len(target)),
        "extracted_partial_point_count": int(len(indices)),
        "partial_match_radius_normalized": match_radius,
        "surface_distance": "directed nearest-neighbour: predicted partial -> GT partial",
        "colormap": colormap,
        "render_backend": backend,
        "statistics": stats,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    stats_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _load_array(value: Any, key: Optional[str] = None,
                base_dir: Optional[Path] = None) -> np.ndarray:
    if not isinstance(value, (str, os.PathLike)):
        return np.asarray(value)
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            selected = key or (data.files[0] if len(data.files) == 1 else None)
            if selected is None or selected not in data:
                raise KeyError(f"Specify an npz key for {path}; available: {data.files}")
            return np.asarray(data[selected])
    if suffix in (".txt", ".xyz", ".csv"):
        return np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    if suffix in (".ply", ".pcd"):
        import open3d as o3d
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
        if len(points) == 0:
            raise ValueError(f"No points read from {path}")
        return points
    if suffix == ".json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported array file: {path}")


def _load_experiment_module(name: str, filename: str):
    path = EXPERIMENT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_dataset(dataset_name: str, noise_mm: Optional[int]):
    sys.path.insert(0, str(EXPERIMENT_DIR))
    saved = {key: os.environ.pop(key) for key in ("P2P_TEST_INDICES", "P2P_TEST_LIMIT")
             if key in os.environ}
    try:
        config = _load_experiment_module("paper_vis_config", "config.py")
        dataset = _load_experiment_module("paper_vis_dataset", "dataset.py")
        cfg = config.make_cfg()
        if dataset_name == "in_silico":
            return dataset.LiverTask3TestDataset(cfg, noise_mm=noise_mm)
        if dataset_name == "in_vitro":
            if noise_mm is not None:
                raise ValueError("in_vitro does not support synthetic noise")
            return dataset.LiverTask3InVitroTestDataset(cfg)
        raise ValueError(dataset_name)
    finally:
        os.environ.update(saved)
        sys.path.pop(0)


def _dataset_name(summary: Mapping[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    protocol = str(summary.get("protocol", "")).lower()
    if "in-silico" in protocol or "in_silico" in protocol:
        return "in_silico"
    if "in-vitro" in protocol or "in_vitro" in protocol:
        return "in_vitro"
    raise ValueError("Cannot infer dataset; pass --dataset")


def _native_metric(sample: Mapping[str, Any], transform: np.ndarray) -> float:
    distances = np.linalg.norm(
        apply_transform(sample["source_markers"], transform) - sample["target_markers"],
        axis=1,
    ) * float(sample["physical_scale"])
    return float(np.sqrt(np.mean(distances ** 2)))


def _render_native_batch(summary: Mapping[str, Any], args: argparse.Namespace) -> list:
    dataset_name = _dataset_name(summary, args.dataset)
    noise = summary.get("noise_mm")
    dataset = _native_dataset(dataset_name, None if noise is None else int(noise))
    prediction_dir = Path(args.predictions_dir) if args.predictions_dir else None
    if prediction_dir is None:
        checkpoint = Path(str(summary.get("checkpoint", "")))
        prediction_dir = checkpoint.parent.parent / "registration"
    records = list(summary.get("samples", []))
    best = sorted(records, key=lambda row: float(row["rms_tre_mm"]))[:args.best]
    worst = sorted(records, key=lambda row: float(row["rms_tre_mm"]), reverse=True)[:args.worst]
    selections = [("best", rank, row) for rank, row in enumerate(best, 1)]
    selections += [("failure", rank, row) for rank, row in enumerate(worst, 1)]
    outputs = []
    for group, rank, record in selections:
        index = int(record["index"])
        prediction_path = prediction_dir / f"{index:05d}.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Missing {prediction_path}. Run test.py with --save_predictions "
                "or pass --predictions-dir."
            )
        sample = dataset[index]
        with np.load(prediction_path, allow_pickle=False) as prediction:
            pred_transform = np.asarray(prediction["estimated_transform"])
        reported = float(record["rms_tre_mm"])
        recomputed = _native_metric(sample, pred_transform)
        difference = abs(reported - recomputed)
        if difference > args.metric_tolerance:
            raise AssertionError(
                f"case {index}: reported/recomputed RMS-TRE differ by {difference:.6f} mm"
            )
        result = generate_paper_registration_visualization(
            sample["src_points"], sample["ref_points"], pred_transform,
            sample["transform"], case_id=f"case_{index:05d}",
            output_dir=Path(args.output_dir) / group,
            scale_factor=float(sample["physical_scale"]),
            partial_match_radius=args.partial_match_radius,
            colormap=args.colormap, point_size=args.point_size,
            heatmap_point_size=args.heatmap_point_size, dpi=args.dpi,
            render_backend=args.renderer,
            metadata={
                "selection": group, "rank": rank, "dataset_index": index,
                "sample": str(record.get("sample", "")),
                "reported_evaluation_rms_tre_mm": reported,
                "recomputed_evaluation_rms_tre_mm": recomputed,
                "evaluation_abs_difference_mm": difference,
            },
        )
        outputs.append(result)
    return outputs


def _render_manifest(payload: Mapping[str, Any], manifest_path: Path,
                     args: argparse.Namespace) -> list:
    outputs = []
    for position, case in enumerate(payload.get("cases", [payload]), 1):
        source = _load_array(case.get("source", case.get("src_path")),
                             case.get("src_key"), manifest_path.parent)
        target = _load_array(case.get("target", case.get("tgt_path")),
                             case.get("tgt_key"), manifest_path.parent)
        pred = _load_array(case["T_pred"], case.get("T_pred_key"), manifest_path.parent)
        ground_truth = _load_array(case["T_gt"], case.get("T_gt_key"), manifest_path.parent)
        outputs.append(generate_paper_registration_visualization(
            source, target, pred, ground_truth,
            case_id=case.get("case_id", f"case_{position:04d}"),
            output_dir=args.output_dir,
            scale_factor=float(case.get("scale_factor", args.scale_factor)),
            partial_match_radius=case.get("partial_match_radius", args.partial_match_radius),
            colormap=args.colormap, point_size=args.point_size,
            heatmap_point_size=args.heatmap_point_size, dpi=args.dpi,
            render_backend=args.renderer,
        ))
    return outputs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="Native test summary or generic case manifest JSON")
    parser.add_argument("--predictions-dir", help="Native test registration/*.npz directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset", choices=("in_silico", "in_vitro"))
    parser.add_argument("--best", type=int, default=5)
    parser.add_argument("--worst", type=int, default=5)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--partial-match-radius", type=float,
                        help="GT crop radius in normalized coordinates; default is automatic")
    parser.add_argument(
        "--colormap", default="medical_blue",
        help="medical_blue uses one blue hue with increasing saturation/darkness",
    )
    parser.add_argument("--point-size", type=float, default=4.5)
    parser.add_argument("--heatmap-point-size", type=float, default=5.5)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--renderer", choices=("auto", "open3d", "software"), default="auto")
    parser.add_argument(
        "--metric-tolerance", type=float, default=0.05,
        help="Allowed RMS-TRE drift (mm) between an older summary and re-saved prediction",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.best < 0 or args.worst < 0 or args.best + args.worst == 0:
        raise ValueError("--best and --worst must be non-negative and not both zero")
    result_path = Path(args.results).resolve()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "samples" in payload and "method" in payload:
        outputs = _render_native_batch(payload, args)
    elif isinstance(payload, dict):
        outputs = _render_manifest(payload, result_path, args)
    else:
        raise ValueError("results JSON must contain an object")
    manifest_path = Path(args.output_dir) / "paper_visualization_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"source": str(result_path), "cases": outputs}, indent=2,
                   ensure_ascii=False), encoding="utf-8",
    )
    print(f"Generated {len(outputs)} figure(s): {manifest_path}")


if __name__ == "__main__":
    main()
