#!/usr/bin/env python3
"""Render best-three P2P correct-correspondence method comparisons."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.correspondence_cache import read_correspondence_cache


METHOD_ORDER = (
    "ours",
    "geotransformer",
    "castv2",
    "dfat",
    "lepard",
    "predator",
    "parenet",
    "livermatch",
    "livermatch_p2p",
)

METHOD_TITLES = {
    "ours": "Ours",
    "geotransformer": "GeoTransformer",
    "castv2": "CASTV2",
    "dfat": "DFAT",
    "lepard": "Lepard",
    "predator": "Predator",
    "parenet": "PARENet",
    "livermatch": "LiverMatch",
    "livermatch_p2p": "LiverMatch+P2P",
}


@dataclass(frozen=True)
class ViewStyle:
    vis_rx: float = 22.0
    vis_ry: float = -18.0
    vis_rz: float = -8.0
    elev: float = 10.0
    azim: float = -78.0
    layout_right: float = 1.25
    layout_up: float = 0.75
    threshold_mm: float = 5.0
    precision_threshold_mm: float = 2.0
    line_width: float = 0.45
    line_alpha: float = 0.025
    precision_line_width: float = 0.52
    precision_line_alpha: float = 0.28
    color_mode: str = "precision"
    uniform_line_alpha: float = 0.38
    source_size: float = 1.8
    target_size: float = 2.2
    target_halo_factor: float = 2.3
    margin: float = 0.02
    panel_zoom: float = 1.12


def _rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    rx_mat = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    ry_mat = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    rz_mat = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return rz_mat @ ry_mat @ rx_mat


def _rotate(points: np.ndarray, rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - center) @ rotation.T + center


def _camera_basis(elev_deg: float, azim_deg: float) -> tuple[np.ndarray, np.ndarray]:
    elev, azim = np.deg2rad([elev_deg, azim_deg])
    view = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    view /= np.linalg.norm(view)
    right = np.array([-np.sin(azim), np.cos(azim), 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    up /= np.linalg.norm(up)
    return right, up


def _display_offset(
    source: np.ndarray,
    target: np.ndarray,
    style: ViewStyle,
) -> np.ndarray:
    scale = max(float(np.ptp(source, axis=0).max()), float(np.ptp(target, axis=0).max()))
    right, up = _camera_basis(style.elev, style.azim)
    desired = scale * (style.layout_right * right + style.layout_up * up)
    return source.mean(axis=0) + desired - target.mean(axis=0)


def _subsample(points: np.ndarray, limit: int = 8000) -> np.ndarray:
    if len(points) <= limit:
        return points
    return points[np.linspace(0, len(points) - 1, limit, dtype=np.int64)]


def _set_bounds(axes, points: np.ndarray, margin: float, zoom: float = 1.0) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = np.maximum(upper - lower, 1e-6)
    padding = float(margin) * float(span.max())
    lower, upper = lower - padding, upper + padding
    axes.set_xlim(lower[0], upper[0])
    axes.set_ylim(lower[1], upper[1])
    axes.set_zlim(lower[2], upper[2])
    axes.set_box_aspect(upper - lower, zoom=zoom)


def render_panel(axes, cache: Mapping[str, object], style: ViewStyle) -> dict[str, object]:
    """Draw every 5 mm inlier and emphasize the high-precision 2 mm subset."""

    source = np.asarray(cache["source_points_mm"], dtype=np.float64)
    target = np.asarray(cache["target_points_mm"], dtype=np.float64)
    source_corr = np.asarray(cache["source_corr_mm"], dtype=np.float64)
    target_corr = np.asarray(cache["target_corr_mm"], dtype=np.float64)
    mask_key = f"correct_mask_{style.threshold_mm:g}mm"
    correct_mask = np.asarray(cache[mask_key], dtype=bool)
    if correct_mask.shape != (len(source_corr),):
        raise ValueError(f"{mask_key} does not match correspondence count")
    precision_mask_key = f"correct_mask_{style.precision_threshold_mm:g}mm"
    precision_mask = np.asarray(cache[precision_mask_key], dtype=bool)
    if precision_mask.shape != (len(source_corr),):
        raise ValueError(f"{precision_mask_key} does not match correspondence count")
    if np.any(precision_mask & ~correct_mask):
        raise ValueError("precision correspondences must be a subset of 5 mm correspondences")

    rotation = _rotation_matrix_xyz(style.vis_rx, style.vis_ry, style.vis_rz)
    center = np.vstack([source, target]).mean(axis=0)
    source_display = _rotate(source, rotation, center)
    target_display_raw = _rotate(target, rotation, center)
    source_corr_display = _rotate(source_corr, rotation, center)
    target_corr_display = _rotate(target_corr, rotation, center)
    offset = _display_offset(source_display, target_display_raw, style)
    target_display = target_display_raw + offset
    target_corr_display = target_corr_display + offset
    segments = np.stack(
        [source_corr_display[correct_mask], target_corr_display[correct_mask]], axis=1
    )
    precision_segments = np.stack(
        [source_corr_display[precision_mask], target_corr_display[precision_mask]], axis=1
    )

    uniform_mode = style.color_mode == "uniform"
    lines = Line3DCollection(
        segments,
        colors="#32CD32",
        linewidths=style.line_width,
        alpha=style.uniform_line_alpha if uniform_mode else style.line_alpha,
        zorder=1,
    )
    axes.add_collection3d(lines)
    precision_lines = Line3DCollection(
        precision_segments,
        colors="#008A2E",
        linewidths=style.precision_line_width,
        alpha=style.precision_line_alpha,
        zorder=2,
    )
    if not uniform_mode:
        axes.add_collection3d(precision_lines)
    source_plot = _subsample(source_display)
    target_plot = _subsample(target_display)
    source_artist = axes.scatter(
        *source_plot.T,
        s=style.source_size,
        c="#FF2D2D",
        alpha=0.92,
        edgecolors="none",
        depthshade=False,
        rasterized=True,
        zorder=3,
    )
    target_halo = axes.scatter(
        *target_plot.T,
        s=style.target_size * style.target_halo_factor,
        c="white",
        alpha=1.0,
        edgecolors="none",
        depthshade=False,
        rasterized=True,
        zorder=4,
    )
    target_artist = axes.scatter(
        *target_plot.T,
        s=style.target_size,
        c="#1F4DFF",
        alpha=1.0,
        edgecolors="none",
        depthshade=False,
        rasterized=True,
        zorder=5,
    )
    bounds_parts = [source_display, target_display]
    if len(segments):
        bounds_parts.extend([segments[:, 0], segments[:, 1]])
    _set_bounds(axes, np.vstack(bounds_parts), style.margin, style.panel_zoom)
    axes.view_init(elev=style.elev, azim=style.azim)
    axes.set_proj_type("ortho")
    axes.grid(False)
    axes.set_axis_off()
    return {
        "num_rendered_correspondences": int(correct_mask.sum()),
        "num_precision_correspondences": int(precision_mask.sum()),
        "line_collection": lines,
        "precision_line_collection": precision_lines,
        "source_points": source_artist,
        "target_halo": target_halo,
        "target_points": target_artist,
        "target_corr_display": target_corr_display,
        "visualization_offset_mm": offset.tolist(),
    }


def render_standalone_panel(
    cache: Mapping[str, object],
    output_path: str | Path,
    style: ViewStyle = ViewStyle(),
    *,
    title: str | None = None,
    na_message: str | None = None,
    dpi: int = 300,
) -> dict[str, object]:
    """Render one auditable panel with the same geometry and layering as the grid."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(3.0, 3.0), facecolor="white")
    axes = figure.add_subplot(111, projection="3d", computed_zorder=False)
    audit = render_panel(axes, cache, style)
    if title:
        axes.set_title(title, fontsize=10, fontweight="bold", pad=-2)
    if na_message:
        axes.text2D(
            0.5,
            0.48,
            na_message,
            transform=axes.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color="#555555",
        )
    figure.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return audit


def write_metrics_csv(rows: list[dict[str, object]], output_path: str | Path) -> Path:
    """Write one metrics table using the stable field order of the first row."""

    if not rows:
        raise ValueError("Cannot write an empty metrics table")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _sample_slug(sample_id: str) -> str:
    return sample_id.replace("/", "__").removesuffix(".npz")


def _load_method_cache(root: Path, dataset: str, sample_id: str, method: str) -> dict[str, object]:
    return read_correspondence_cache(
        root / "correspondences" / dataset / _sample_slug(sample_id) / f"{method}.npz"
    )


def sort_cases_by_visibility(cases: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Return a copy ordered from highest to lowest visibility."""

    return sorted(cases, key=lambda case: float(case["visibility"]), reverse=True)


def order_methods_by_correct_correspondence_count(
    caches: Mapping[str, Mapping[str, object]],
    threshold_mm: float = 5.0,
) -> list[str]:
    """Order methods left-to-right by GT-correct correspondence count."""

    stable_index = {method: index for index, method in enumerate(METHOD_ORDER)}
    mask_key = f"correct_mask_{threshold_mm:g}mm"
    return sorted(
        METHOD_ORDER,
        key=lambda method: (
            int(np.asarray(caches[method][mask_key], dtype=bool).sum()),
            stable_index[method],
        ),
    )


def order_methods_by_total_correct_correspondence_count(
    caches_by_case: list[Mapping[str, Mapping[str, object]]],
    threshold_mm: float = 5.0,
) -> list[str]:
    """Compute one stable column order shared by every displayed sample."""

    if not caches_by_case:
        raise ValueError("Cannot order methods without any sample caches")
    stable_index = {method: index for index, method in enumerate(METHOD_ORDER)}
    mask_key = f"correct_mask_{threshold_mm:g}mm"
    totals = {
        method: sum(
            int(np.asarray(caches[method][mask_key], dtype=bool).sum())
            for caches in caches_by_case
        )
        for method in METHOD_ORDER
    }
    return sorted(METHOD_ORDER, key=lambda method: (totals[method], stable_index[method]))


def render_dataset_comparison(
    dataset: str,
    selection: Mapping[str, object],
    output_root: str | Path,
    *,
    style: ViewStyle = ViewStyle(),
    dpi: int = 300,
) -> dict[str, object]:
    root = Path(output_root)
    cases = sort_cases_by_visibility(selection["datasets"][dataset]["cases"])
    figure = plt.figure(figsize=(27.0, 3.15 * len(cases)), facecolor="white")
    grid = figure.add_gridspec(len(cases), len(METHOD_ORDER), wspace=0.0, hspace=0.06)
    metrics_rows = []
    panel_paths = []
    display_orders = []
    case_caches = [
        {
            method: _load_method_cache(root, dataset, str(case["sample"]), method)
            for method in METHOD_ORDER
        }
        for case in cases
    ]
    method_order = order_methods_by_total_correct_correspondence_count(
        case_caches, style.threshold_mm
    )
    mask_key = f"correct_mask_{style.threshold_mm:g}mm"
    statistics_key = f"{style.threshold_mm:g}"
    total_correct_by_method = {
        method: sum(
            int(np.asarray(caches[method][mask_key], dtype=bool).sum())
            for caches in case_caches
        )
        for method in method_order
    }

    for row_index, (case, caches) in enumerate(zip(cases, case_caches)):
        sample_id = str(case["sample"])
        display_orders.append(
            {
                "sample_id": sample_id,
                "visibility": float(case["visibility"]),
                "methods": method_order,
                "num_pred_correspondences": {
                    method: len(caches[method]["source_corr_mm"]) for method in method_order
                },
                "num_correct_correspondences": {
                    method: int(
                        np.asarray(
                            caches[method][f"correct_mask_{style.threshold_mm:g}mm"],
                            dtype=bool,
                        ).sum()
                    )
                    for method in method_order
                },
            }
        )
        for column_index, method in enumerate(method_order):
            axes = figure.add_subplot(
                grid[row_index, column_index], projection="3d", computed_zorder=False
            )
            status = "ok"
            cache = caches[method]
            audit = render_panel(axes, cache, style)
            if row_index == 0:
                axes.set_title(METHOD_TITLES[method], fontsize=10, fontweight="bold", pad=-2)
            if column_index == 0:
                axes.text2D(
                    -0.06,
                    0.5,
                    f"Visibility {float(case['visibility']) * 100:.1f}%",
                    transform=axes.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                )
            if style.color_mode != "uniform":
                axes.text2D(
                    0.5,
                    0.015,
                    (
                        f"N$_{{{style.precision_threshold_mm:g}}}$="
                        f"{audit['num_precision_correspondences']}  /  "
                        f"N$_{{{style.threshold_mm:g}}}$="
                        f"{audit['num_rendered_correspondences']}"
                    ),
                    transform=axes.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#303030",
                )
            statistics = cache["threshold_statistics"]
            provenance = cache["provenance"]
            metrics_rows.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "method": method,
                    "display_row": row_index,
                    "display_column": column_index,
                    "status": status,
                    "num_source_points": len(cache["source_points_mm"]),
                    "num_target_points": len(cache["target_points_mm"]),
                    "num_pred_correspondences": len(cache["source_corr_mm"]),
                    "num_correct_correspondences": statistics[statistics_key]["num_correct"],
                    "inlier_ratio": statistics[statistics_key]["inlier_ratio"],
                    "IR_2mm": statistics["2"]["inlier_ratio"],
                    "IR_3mm": statistics["3"]["inlier_ratio"],
                    "IR_4mm": statistics["4"]["inlier_ratio"],
                    "IR_5mm": statistics["5"]["inlier_ratio"],
                    "inlier_threshold_mm": style.threshold_mm,
                    "RRE_deg": provenance["RRE_deg"],
                    "RTE_mm": provenance["RTE_mm"],
                    "RMSE_mm": provenance["RMSE_mm"],
                    "num_rendered_correspondences": audit["num_rendered_correspondences"],
                    "num_precision_correspondences": audit[
                        "num_precision_correspondences"
                    ],
                }
            )
            panel_path = (
                root / "panels" / dataset / _sample_slug(sample_id) / f"{method}.png"
            )
            render_standalone_panel(
                cache, panel_path, style, title=METHOD_TITLES[method], dpi=dpi
            )
            panel_paths.append(str(panel_path.resolve()))

    if style.color_mode != "uniform":
        figure.text(
            0.54,
            0.006,
            "Dark green: residual < 2 mm    Light green: residual 2–5 mm",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#303030",
        )
    bottom = 0.025 if style.color_mode != "uniform" else 0.01
    figure.subplots_adjust(left=0.09, right=0.995, bottom=bottom, top=0.965)
    png_path = root / f"{dataset}_best3_correspondence_comparison.png"
    pdf_path = root / f"{dataset}_best3_correspondence_comparison.pdf"
    figure.savefig(png_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    metrics_path = root / f"{dataset}_correspondence_metrics.csv"
    write_metrics_csv(metrics_rows, metrics_path)
    return {
        "dataset": dataset,
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "metrics": str(metrics_path.resolve()),
        "rows": metrics_rows,
        "panels": panel_paths,
        "column_order": method_order,
        "total_correct_correspondences": total_correct_by_method,
        "display_order": display_orders,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("in_silico", "in_vitro", "all"), default="all")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--panel-zoom",
        type=float,
        default=ViewStyle.panel_zoom,
        help="3D content zoom inside each subplot (default: 1.12)",
    )
    parser.add_argument(
        "--color-mode",
        choices=("precision", "uniform"),
        default="precision",
        help="precision: dark <2 mm plus light 2-5 mm; uniform: all <5 mm light green",
    )
    parser.add_argument(
        "--threshold-mm",
        type=float,
        choices=(2.0, 3.0, 4.0, 5.0),
        default=ViewStyle.threshold_mm,
        help="GT residual threshold used for rendered correct correspondences",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if args.panel_zoom <= 0:
        raise ValueError("--panel-zoom must be positive")
    style = ViewStyle(
        panel_zoom=args.panel_zoom,
        color_mode=args.color_mode,
        threshold_mm=args.threshold_mm,
    )
    datasets = ("in_silico", "in_vitro") if args.dataset == "all" else (args.dataset,)
    manifests = []
    for dataset in datasets:
        manifests.append(
            render_dataset_comparison(
                dataset, selection, args.output_root, style=style, dpi=args.dpi
            )
        )
    combined_metrics_path = args.output_root / "correspondence_metrics.csv"
    write_metrics_csv(
        [row for manifest in manifests for row in manifest["rows"]],
        combined_metrics_path,
    )
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "primary_inlier_threshold_mm": style.threshold_mm,
                "sensitivity_thresholds_mm": [2.0, 3.0, 4.0, 5.0],
                "method_order": list(METHOD_ORDER),
                "visualization": style.__dict__,
                "combined_metrics": str(combined_metrics_path.resolve()),
                "datasets": manifests,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
