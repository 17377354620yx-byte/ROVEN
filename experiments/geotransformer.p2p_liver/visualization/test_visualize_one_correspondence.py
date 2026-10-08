#!/usr/bin/env python3
"""Render one real RTOR correspondence result as a paper-ready 3D figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = Path(
    "/home/yangx/code/new_deform/RTORv6/output/"
    "ircadb1_vis20_30_noise2_rot25_90_seed20260822/dataset/pairs"
)

DEFAULT_CHECKPOINT = (
    ROOT
    / "output"
    / "geotransformer.p2p_liver.rtor_a3_cooperative"
    / "snapshots"
    / "epoch-150.pth.tar"
)

DEFAULT_SAMPLE_ID = "3Dircadb1.19_vis20"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "correspondence_visualization"

P2P_IN_SILICO_ROOT = Path("/mnt/data3/yangx/P2P/in_silico_visibility_0.2_0.4")
P2P_IN_VITRO_ROOT = Path("/mnt/data3/yangx/P2P/in_vitro_visibility_0.2_0.4")
P2P_IN_SILICO_SAMPLE_ID = "00046/00002/00003.npz"
P2P_IN_VITRO_SAMPLE_ID = "5_00020.npz"


def resolve_dataset_defaults(
    dataset_kind: str,
    data_root: Path | None,
    sample_id: str | None,
) -> tuple[Path, str]:
    defaults = {
        "ircadb": (DEFAULT_DATA_ROOT, DEFAULT_SAMPLE_ID),
        "in_silico": (P2P_IN_SILICO_ROOT, P2P_IN_SILICO_SAMPLE_ID),
        "in_vitro": (P2P_IN_VITRO_ROOT, P2P_IN_VITRO_SAMPLE_ID),
    }
    if dataset_kind not in defaults:
        raise ValueError(f"Unknown dataset: {dataset_kind}")
    default_root, default_sample = defaults[dataset_kind]
    return (
        default_root if data_root is None else Path(data_root),
        default_sample if sample_id is None else str(sample_id),
    )


def artifact_paths(dataset_kind: str, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    if dataset_kind == "ircadb":
        return (
            output_dir / "ours_correct_only.png",
            output_dir / "ours_correct_only.pdf",
            output_dir / "correspondence_stats.json",
        )
    if dataset_kind in ("in_silico", "in_vitro"):
        return (
            output_dir / f"ours_{dataset_kind}_correct_only.png",
            output_dir / f"ours_{dataset_kind}_correct_only.pdf",
            output_dir / f"{dataset_kind}_correspondence_stats.json",
        )
    raise ValueError(f"Unknown dataset: {dataset_kind}")


def resolve_offset_z_factor(dataset_kind: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    if dataset_kind == "in_silico":
        return 0.85
    if dataset_kind == "in_vitro":
        return 0.15
    return 0.0


def restore_points_mm(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    return points * float(scale) + center


def correct_correspondence_mask(
    source_corr_mm: np.ndarray,
    target_corr_mm: np.ndarray,
    gt_transform_mm: np.ndarray,
    threshold_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_corr_mm = np.asarray(source_corr_mm, dtype=np.float64)
    target_corr_mm = np.asarray(target_corr_mm, dtype=np.float64)
    gt_transform_mm = np.asarray(gt_transform_mm, dtype=np.float64)
    if (
        source_corr_mm.shape != target_corr_mm.shape
        or source_corr_mm.ndim != 2
        or source_corr_mm.shape[1] != 3
    ):
        raise ValueError(
            "source and target correspondences must have matching (N, 3) shapes"
        )
    if gt_transform_mm.shape != (4, 4):
        raise ValueError("gt_transform_mm must have shape (4, 4)")
    if not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError("threshold_mm must be positive and finite")

    source_aligned = (
        source_corr_mm @ gt_transform_mm[:3, :3].T
        + gt_transform_mm[:3, 3]
    )
    residuals_mm = np.linalg.norm(source_aligned - target_corr_mm, axis=1)
    return residuals_mm < float(threshold_mm), residuals_mm


def select_correct_correspondences(
    correct_mask: np.ndarray,
    confidence: np.ndarray | None,
    *,
    max_lines: int | None = None,
    min_lines: int = 1,
) -> np.ndarray:
    correct_mask = np.asarray(correct_mask, dtype=bool)
    if correct_mask.ndim != 1:
        raise ValueError("correct_mask must be one-dimensional")

    correct_indices = np.flatnonzero(correct_mask)
    if len(correct_indices) < min_lines:
        raise ValueError(f"at least {min_lines} correct correspondences are required")

    if confidence is None:
        ordered = correct_indices
    else:
        confidence = np.asarray(confidence, dtype=np.float64)
        if confidence.shape != correct_mask.shape:
            raise ValueError("confidence and correct_mask must have matching shapes")
        order = np.argsort(-confidence[correct_indices], kind="stable")
        ordered = correct_indices[order]

    if max_lines is None or max_lines <= 0:
        return ordered
    return ordered[:max_lines]


def build_stats(
    *,
    sample_id: str,
    num_source_points: int,
    num_target_points: int,
    correct_mask: np.ndarray,
    threshold_mm: float,
    pose_metrics: dict,
) -> dict:
    correct_mask = np.asarray(correct_mask, dtype=bool)
    if correct_mask.ndim != 1:
        raise ValueError("correct_mask must be one-dimensional")

    count = int(correct_mask.size)
    correct = int(correct_mask.sum())

    return {
        "sample_id": str(sample_id),
        "num_source_points": int(num_source_points),
        "num_target_points": int(num_target_points),
        "num_pred_correspondences": count,
        "num_correct_correspondences": correct,
        "inlier_ratio": float(correct / count) if count else 0.0,
        "inlier_threshold_mm": float(threshold_mm),
        "RRE_deg": float(pose_metrics["RRE"]),
        "RTE_mm": float(pose_metrics["RTE"]),
        "RMSE_mm": float(pose_metrics["RMSE"]),
    }


def _activate_project_imports() -> None:
    for path in (str(EXPERIMENT_DIR), str(ROOT)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def load_p2p_sample(cfg, dataset_kind: str, data_root: Path, sample_id: str) -> dict:
    _activate_project_imports()

    from torch.utils.data import Subset
    from dataset import (
        LiverTask3InVitroTestDataset,
        LiverTask3TestDataset,
        _norm_vox,
    )
    from tools.ircadb_benchmark import NormalizedPair, transform_to_mm

    data_root = Path(data_root).resolve()
    sample_id = str(sample_id)

    if dataset_kind == "in_silico":
        cfg.data.test_root = str(data_root / "Deform_mesh_npz_test" / "Test")
        cfg.data.test_list = str(data_root / "Deform_mesh_npz_test" / "list.npz")
        dataset = LiverTask3TestDataset(cfg, noise_mm=None, rotate_target=True)
    elif dataset_kind == "in_vitro":
        cfg.data.in_vitro_root = str(data_root)
        cfg.data.in_vitro_list = str(data_root / "rigid_list.npy")
        dataset = LiverTask3InVitroTestDataset(cfg)
    else:
        raise ValueError("dataset_kind must be 'in_silico' or 'in_vitro'")

    matches = np.flatnonzero(dataset.samples.astype(str) == sample_id)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one sample named {sample_id!r} in {data_root}, found {len(matches)}"
        )

    dataset_index = int(matches[0])
    network_sample = dataset[dataset_index]

    if dataset_kind == "in_silico":
        sample_path = Path(cfg.data.test_root) / sample_id
        with np.load(sample_path, allow_pickle=True) as entry:
            source_raw_mm = np.asarray(entry["src_pcd"], dtype=np.float64)
            target_raw_mm = np.asarray(entry["tgt_pcd"], dtype=np.float64)
            target_rotation = np.asarray(entry["rot_tgt"], dtype=np.float64)

        target_raw_mm = target_raw_mm @ target_rotation.T
        gt_source = "marker least-squares oracle after rot_tgt (dataset transform)"
    else:
        sample_path = Path(cfg.data.in_vitro_root) / "Rigid_test_data" / sample_id
        with np.load(sample_path, allow_pickle=False) as entry:
            source_raw_mm = np.asarray(entry["src_vs"], dtype=np.float64)
            target_raw_mm = np.asarray(entry["tgt_vs"], dtype=np.float64)

        gt_source = "marker least-squares oracle (dataset transform)"

    source_model_mm, scale_mm = _norm_vox(source_raw_mm, float(cfg.data.voxel_size))
    target_model_mm, _ = _norm_vox(target_raw_mm, float(cfg.data.voxel_size))

    source_center_mm = source_model_mm.mean(axis=0)
    target_center_mm = target_model_mm.mean(axis=0)

    pair = NormalizedPair(
        source=np.asarray(network_sample["src_points"], dtype=np.float64),
        target=np.asarray(network_sample["ref_points"], dtype=np.float64),
        source_center=source_center_mm,
        target_center=target_center_mm,
        scale=float(scale_mm),
    )

    gt_transform_network = np.asarray(network_sample["transform"], dtype=np.float64)
    gt_transform_mm = transform_to_mm(gt_transform_network, pair)

    return {
        "dataset": Subset(dataset, [dataset_index]),
        "dataset_index": dataset_index,
        "sample_id": sample_id,
        "sample_path": sample_path,
        "source_points_mm": source_raw_mm,
        "target_points_mm": target_raw_mm,
        "source_model_points_mm": source_model_mm,
        "target_model_points_mm": target_model_mm,
        "source_center_mm": source_center_mm,
        "target_center_mm": target_center_mm,
        "scale_mm": float(scale_mm),
        "gt_transform_network": gt_transform_network,
        "gt_transform_mm": gt_transform_mm,
        "gt_source": gt_source,
        "normalized_pair": pair,
    }


def infer_one_sample(
    *,
    dataset_kind: str = "ircadb",
    data_root: Path,
    checkpoint_path: Path,
    sample_id: str,
    device_index: int = 0,
    threshold_mm: float = 5.0,
) -> dict:
    _activate_project_imports()

    import torch

    from config import make_cfg
    from dataset import _build_loader
    from ircadb_dataset import IRCADbStackDataset
    from model import create_model
    from tools.ircadb_benchmark import (
        load_samples,
        rigid_metrics_mm,
        transform_to_mm,
    )
    from geotransformer.utils.torch import release_cuda, to_cuda

    data_root = Path(data_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("This checkpoint/model path requires a CUDA device")

    torch.cuda.set_device(int(device_index))
    torch.manual_seed(7351)
    np.random.seed(7351)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    protocol = checkpoint.get("metadata", {}).get("p2p_protocol", {})
    expected = {
        "architecture": "rtor_a3",
        "dual_encoder": False,
        "registration_profile": "legacy",
        "interaction_profile": "cooperative",
    }

    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key, value) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint protocol mismatch: {mismatches}")

    neighbor_limits = protocol.get("neighbor_limits")
    if neighbor_limits is None:
        raise ValueError("Checkpoint metadata does not contain neighbor_limits")

    cfg = make_cfg(
        architecture="rtor_a3",
        registration_profile="legacy",
        dual_encoder=False,
        interaction_profile="cooperative",
    )
    cfg.test.num_workers = 0

    if dataset_kind == "ircadb":
        dataset = IRCADbStackDataset(data_root, cfg.data.voxel_size)
        dataset.records = load_samples(data_root, sample_ids=[sample_id])
        context = dataset.load_context(0)

        pair = context.normalized
        source_points_mm = np.asarray(context.source_mm, dtype=np.float64)
        target_points_mm = np.asarray(context.target_mm, dtype=np.float64)
        gt_transform_mm = np.asarray(context.gt_transform_mm, dtype=np.float64)

        resolved_sample_id = context.record.sample_id
        sample_path = context.record.path
        gt_source = "dataset transform/transform_gt"

        source_variable = "context.source_mm (dataset: src_points/source_points)"
        target_variable = "context.target_mm (dataset: ref_points/target_points)"
        gt_variable = "context.gt_transform_mm (dataset: transform/transform_gt)"

    elif dataset_kind in ("in_silico", "in_vitro"):
        bundle = load_p2p_sample(cfg, dataset_kind, data_root, sample_id)

        dataset = bundle["dataset"]
        pair = bundle["normalized_pair"]
        source_points_mm = bundle["source_points_mm"]
        target_points_mm = bundle["target_points_mm"]
        gt_transform_mm = bundle["gt_transform_mm"]
        resolved_sample_id = bundle["sample_id"]
        sample_path = bundle["sample_path"]
        gt_source = bundle["gt_source"]

        if dataset_kind == "in_silico":
            source_variable = "src_pcd"
            target_variable = "tgt_pcd @ rot_tgt.T"
        else:
            source_variable = "src_vs"
            target_variable = "tgt_vs"

        gt_variable = "data_dict['transform'] restored from normalized marker oracle"

    else:
        raise ValueError(f"Unknown dataset: {dataset_kind}")

    loader = _build_loader(
        cfg,
        dataset,
        list(neighbor_limits),
        train=False,
    )
    data_dict = next(iter(loader))

    model = create_model(cfg).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    with torch.no_grad():
        output_dict = model(to_cuda(data_dict))

    src_corr_network = release_cuda(output_dict["src_corr_points"])
    tgt_corr_network = release_cuda(output_dict["ref_corr_points"])
    confidence = release_cuda(output_dict["corr_scores"])
    estimated_transform_network = release_cuda(output_dict["estimated_transform"])

    estimated_transform_mm = transform_to_mm(
        estimated_transform_network,
        pair,
    )

    source_corr_mm = restore_points_mm(
        src_corr_network,
        pair.source_center,
        pair.scale,
    )

    target_corr_mm = restore_points_mm(
        tgt_corr_network,
        pair.target_center,
        pair.scale,
    )

    correct_mask, residuals_mm = correct_correspondence_mask(
        source_corr_mm,
        target_corr_mm,
        gt_transform_mm,
        threshold_mm,
    )

    metrics = rigid_metrics_mm(
        source_points_mm,
        target_points_mm,
        estimated_transform_mm,
        gt_transform_mm,
    )

    stats = build_stats(
        sample_id=resolved_sample_id,
        num_source_points=len(source_points_mm),
        num_target_points=len(target_points_mm),
        correct_mask=correct_mask,
        threshold_mm=threshold_mm,
        pose_metrics=metrics,
    )

    stats.update(
        {
            "checkpoint": str(checkpoint_path),
            "dataset": dataset_kind,
            "data_root": str(data_root),
            "sample_path": str(Path(sample_path).resolve()),
            "coordinate_units": "mm",
            "network_normalization": (
                "source and target independently centered; "
                "both divided by complete-source radius"
            ),
            "normalization_scale_mm": float(pair.scale),
            "gt_source": gt_source,
            "confidence_available": True,
            "num_rendered_correspondences": 0,
            "variables": {
                "source_points": source_variable,
                "target_points": target_variable,
                "gt_transform": gt_variable,
                "predicted_source_correspondences": "output_dict['src_corr_points']",
                "predicted_target_correspondences": "output_dict['ref_corr_points']",
                "correspondence_confidence": "output_dict['corr_scores']",
                "predicted_transform": "output_dict['estimated_transform']",
            },
        }
    )

    return {
        "source_points_mm": source_points_mm,
        "target_points_mm": target_points_mm,
        "source_corr_mm": source_corr_mm,
        "target_corr_mm": target_corr_mm,
        "confidence": np.asarray(confidence, dtype=np.float64),
        "correct_mask": correct_mask,
        "residuals_mm": residuals_mm,
        "gt_transform_mm": gt_transform_mm,
        "estimated_transform_mm": estimated_transform_mm,
        "stats": stats,
    }


def rotation_matrix_xyz(
    rx_deg: float = 0.0,
    ry_deg: float = 0.0,
    rz_deg: float = 0.0,
) -> np.ndarray:
    rx = np.deg2rad(float(rx_deg))
    ry = np.deg2rad(float(ry_deg))
    rz = np.deg2rad(float(rz_deg))

    rx_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx), np.cos(rx)],
        ],
        dtype=np.float64,
    )

    ry_mat = np.array(
        [
            [np.cos(ry), 0.0, np.sin(ry)],
            [0.0, 1.0, 0.0],
            [-np.sin(ry), 0.0, np.cos(ry)],
        ],
        dtype=np.float64,
    )

    rz_mat = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0.0],
            [np.sin(rz), np.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return rz_mat @ ry_mat @ rx_mat


def rotate_for_visualization(
    points: np.ndarray,
    rotation: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")

    return (points - center) @ rotation.T + center


def camera_screen_basis(
    view_elev: float,
    view_azim: float,
) -> tuple[np.ndarray, np.ndarray]:
    elev = np.deg2rad(float(view_elev))
    azim = np.deg2rad(float(view_azim))

    view_vec = np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ],
        dtype=np.float64,
    )
    view_vec /= np.linalg.norm(view_vec)

    right = np.array(
        [
            -np.sin(azim),
            np.cos(azim),
            0.0,
        ],
        dtype=np.float64,
    )
    right /= np.linalg.norm(right)

    up = np.cross(view_vec, right)
    up /= np.linalg.norm(up)

    return right, up


def compute_paper_layout_offset(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    view_elev: float,
    view_azim: float,
    right_factor: float = 1.25,
    up_factor: float = 0.75,
) -> np.ndarray:
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    src_center = source_points.mean(axis=0)
    tgt_center = target_points.mean(axis=0)

    src_span = np.maximum(np.ptp(source_points, axis=0), 1e-6)
    tgt_span = np.maximum(np.ptp(target_points, axis=0), 1e-6)

    scale = max(
        float(src_span.max()),
        float(tgt_span.max()),
    )

    right, up = camera_screen_basis(
        view_elev,
        view_azim,
    )

    desired_center_delta = (
        float(right_factor) * scale * right
        + float(up_factor) * scale * up
    )

    return src_center + desired_center_delta - tgt_center


def _deterministic_subsample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    indices = np.linspace(
        0,
        len(points) - 1,
        limit,
        dtype=np.int64,
    )
    return points[indices]


def _set_data_proportional_bounds(
    axes,
    points: np.ndarray,
    margin_factor: float = 0.02,
) -> None:
    lower = points.min(axis=0).astype(np.float64)
    upper = points.max(axis=0).astype(np.float64)

    span = np.maximum(upper - lower, 1e-6)
    margin = float(margin_factor) * float(span.max())

    lower -= margin
    upper += margin

    axes.set_xlim(lower[0], upper[0])
    axes.set_ylim(lower[1], upper[1])
    axes.set_zlim(lower[2], upper[2])

    axes.set_box_aspect(upper - lower)


def render_correct_only(
    *,
    source_points_mm: np.ndarray,
    target_points_mm: np.ndarray,
    source_corr_mm: np.ndarray,
    target_corr_mm: np.ndarray,
    selected_indices: np.ndarray,
    png_path: Path,
    pdf_path: Path,
    dpi: int = 400,
    vis_rx_deg: float = 0.0,
    vis_ry_deg: float = 0.0,
    vis_rz_deg: float = 0.0,
    view_elev: float = 10.0,
    view_azim: float = -78.0,
    view_roll: float = 0.0,
    layout_right_factor: float = 1.25,
    layout_up_factor: float = 0.75,
    target_line_gap_fraction: float = 0.08,
    target_halo_factor: float = 2.2,
    fig_width: float = 6.0,
    fig_height: float = 6.0,
    margin_factor: float = 0.02,
    source_point_size: float = 3.0,
    target_point_size: float = 4.0,
    line_width: float = 0.45,
    line_alpha: float = 0.38,
):
    import matplotlib

    matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    source_points_mm = np.asarray(source_points_mm, dtype=np.float64)
    target_points_mm = np.asarray(target_points_mm, dtype=np.float64)
    source_corr_mm = np.asarray(source_corr_mm, dtype=np.float64)
    target_corr_mm = np.asarray(target_corr_mm, dtype=np.float64)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)

    if np.any(selected_indices < 0) or np.any(
        selected_indices >= len(source_corr_mm)
    ):
        raise IndexError("selected correspondence index is out of range")

    if not (0.0 <= float(target_line_gap_fraction) < 1.0):
        raise ValueError("target_line_gap_fraction must be in [0, 1)")

    if float(target_halo_factor) < 1.0:
        raise ValueError("target_halo_factor must be >= 1.0")

    vis_rotation = rotation_matrix_xyz(
        vis_rx_deg,
        vis_ry_deg,
        vis_rz_deg,
    )

    common_center = np.vstack(
        [
            source_points_mm,
            target_points_mm,
        ]
    ).mean(axis=0)

    source_display_full = rotate_for_visualization(
        source_points_mm,
        vis_rotation,
        common_center,
    )

    target_display_raw = rotate_for_visualization(
        target_points_mm,
        vis_rotation,
        common_center,
    )

    source_endpoints = rotate_for_visualization(
        source_corr_mm[selected_indices],
        vis_rotation,
        common_center,
    )

    target_endpoints_raw = rotate_for_visualization(
        target_corr_mm[selected_indices],
        vis_rotation,
        common_center,
    )

    target_offset_mm = compute_paper_layout_offset(
        source_display_full,
        target_display_raw,
        view_elev=view_elev,
        view_azim=view_azim,
        right_factor=layout_right_factor,
        up_factor=layout_up_factor,
    )

    target_display_full = target_display_raw + target_offset_mm
    target_endpoints = target_endpoints_raw + target_offset_mm

    # Shorten green lines before they reach the blue target points.
    line_vectors = target_endpoints - source_endpoints

    target_endpoints_for_lines = (
        target_endpoints
        - float(target_line_gap_fraction) * line_vectors
    )

    segments = np.stack(
        [
            source_endpoints,
            target_endpoints_for_lines,
        ],
        axis=1,
    )

    figure = plt.figure(
        figsize=(float(fig_width), float(fig_height)),
        facecolor="white",
    )

    # Critical: turn off automatic 3D z-order sorting.
    axes = figure.add_subplot(
        111,
        projection="3d",
        computed_zorder=False,
    )

    axes.set_facecolor("white")

    source_display = _deterministic_subsample(
        source_display_full,
        8000,
    )

    target_display = _deterministic_subsample(
        target_display_full,
        8000,
    )

    # 1) Green correspondences: bottom.
    lines = Line3DCollection(
        segments,
        colors="#32CD32",
        linewidths=float(line_width),
        alpha=float(line_alpha),
        zorder=1,
    )
    axes.add_collection3d(lines)

    # 2) Red source: above green.
    axes.scatter(
        source_display[:, 0],
        source_display[:, 1],
        source_display[:, 2],
        s=float(source_point_size),
        c="#FF2D2D",
        alpha=0.90,
        edgecolors="none",
        depthshade=False,
        rasterized=True,
        zorder=3,
    )

    # 3) White halo: masks green around blue points.
    axes.scatter(
        target_display[:, 0],
        target_display[:, 1],
        target_display[:, 2],
        s=float(target_point_size) * float(target_halo_factor),
        c="white",
        alpha=1.0,
        edgecolors="none",
        depthshade=False,
        rasterized=True,
        zorder=4,
    )

    # 4) Blue target: top.
    axes.scatter(
        target_display[:, 0],
        target_display[:, 1],
        target_display[:, 2],
        s=float(target_point_size),
        c="#1F4DFF",
        alpha=1.0,
        edgecolors="none",
        depthshade=False,
        rasterized=True,
        zorder=5,
    )

    bounds_points = np.vstack(
        [
            source_display_full,
            target_display_full,
            source_endpoints,
            target_endpoints,
        ]
    )

    _set_data_proportional_bounds(
        axes,
        bounds_points,
        margin_factor=margin_factor,
    )

    try:
        axes.view_init(
            elev=float(view_elev),
            azim=float(view_azim),
            roll=float(view_roll),
        )
    except TypeError:
        axes.view_init(
            elev=float(view_elev),
            azim=float(view_azim),
        )

    axes.grid(False)
    axes.set_axis_off()

    figure.subplots_adjust(
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
    )

    png_path = Path(png_path)
    pdf_path = Path(pdf_path)

    png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    pdf_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        png_path,
        dpi=int(dpi),
        facecolor="white",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    figure.savefig(
        pdf_path,
        facecolor="white",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={
            "Creator": "RTORv2 correspondence visualization"
        },
    )

    vis_meta = {
        "visualization_offset_mm": target_offset_mm.tolist(),
        "vis_rotation_deg": [
            float(vis_rx_deg),
            float(vis_ry_deg),
            float(vis_rz_deg),
        ],
        "view_elev_deg": float(view_elev),
        "view_azim_deg": float(view_azim),
        "view_roll_deg": float(view_roll),
        "layout_right_factor": float(layout_right_factor),
        "layout_up_factor": float(layout_up_factor),
        "target_line_gap_fraction": float(target_line_gap_fraction),
        "target_halo_factor": float(target_halo_factor),
    }

    return figure, axes, lines, vis_meta


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset",
        choices=("ircadb", "in_silico", "in_vitro"),
        default="ircadb",
    )

    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--threshold-mm", type=float, default=5.0)

    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help=(
            "Maximum number of GT-correct correspondences to render. "
            "Use 0 or a negative value to render all correct correspondences."
        ),
    )

    parser.add_argument("--dpi", type=int, default=400)

    parser.add_argument("--vis-rx", type=float, default=22.0)
    parser.add_argument("--vis-ry", type=float, default=-18.0)
    parser.add_argument("--vis-rz", type=float, default=-8.0)

    parser.add_argument("--view-elev", type=float, default=10.0)
    parser.add_argument("--view-azim", type=float, default=-78.0)
    parser.add_argument("--view-roll", type=float, default=0.0)

    parser.add_argument("--layout-right", type=float, default=1.25)
    parser.add_argument("--layout-up", type=float, default=0.75)

    parser.add_argument(
        "--target-line-gap",
        type=float,
        default=0.08,
        help=(
            "Stop green lines before the target endpoint. "
            "0.08 means stop 8%% before target."
        ),
    )

    parser.add_argument(
        "--target-halo-factor",
        type=float,
        default=2.2,
        help=(
            "White halo size multiplier behind blue target points."
        ),
    )

    parser.add_argument("--fig-width", type=float, default=6.0)
    parser.add_argument("--fig-height", type=float, default=6.0)
    parser.add_argument("--margin-factor", type=float, default=0.02)

    parser.add_argument("--source-point-size", type=float, default=3.0)
    parser.add_argument("--target-point-size", type=float, default=4.0)
    parser.add_argument("--line-width", type=float, default=0.45)
    parser.add_argument("--line-alpha", type=float, default=0.38)

    parser.add_argument(
        "--offset-z-factor",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser


def main() -> None:
    args = make_parser().parse_args()

    data_root, sample_id = resolve_dataset_defaults(
        args.dataset,
        args.data_root,
        args.sample_id,
    )

    result = infer_one_sample(
        dataset_kind=args.dataset,
        data_root=data_root,
        checkpoint_path=args.checkpoint,
        sample_id=sample_id,
        device_index=args.device_index,
        threshold_mm=args.threshold_mm,
    )

    selected = select_correct_correspondences(
        result["correct_mask"],
        result["confidence"],
        max_lines=args.max_lines,
        min_lines=1,
    )

    output_dir = args.output_dir.resolve()

    png_path, pdf_path, stats_path = artifact_paths(
        args.dataset,
        output_dir,
    )

    figure, _, _, vis_meta = render_correct_only(
        source_points_mm=result["source_points_mm"],
        target_points_mm=result["target_points_mm"],
        source_corr_mm=result["source_corr_mm"],
        target_corr_mm=result["target_corr_mm"],
        selected_indices=selected,
        png_path=png_path,
        pdf_path=pdf_path,
        dpi=args.dpi,
        vis_rx_deg=args.vis_rx,
        vis_ry_deg=args.vis_ry,
        vis_rz_deg=args.vis_rz,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
        view_roll=args.view_roll,
        layout_right_factor=args.layout_right,
        layout_up_factor=args.layout_up,
        target_line_gap_fraction=args.target_line_gap,
        target_halo_factor=args.target_halo_factor,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        margin_factor=args.margin_factor,
        source_point_size=args.source_point_size,
        target_point_size=args.target_point_size,
        line_width=args.line_width,
        line_alpha=args.line_alpha,
    )

    import matplotlib.pyplot as plt

    plt.close(figure)

    stats = result["stats"]

    stats.update(
        {
            "num_rendered_correspondences": int(len(selected)),
            "visualization_offset_mm": vis_meta["visualization_offset_mm"],
            "visualization_rotation_deg": vis_meta["vis_rotation_deg"],
            "visualization_view": {
                "elev_deg": vis_meta["view_elev_deg"],
                "azim_deg": vis_meta["view_azim_deg"],
                "roll_deg": vis_meta["view_roll_deg"],
            },
            "visualization_layout": {
                "right_factor": vis_meta["layout_right_factor"],
                "up_factor": vis_meta["layout_up_factor"],
                "target_line_gap_fraction": vis_meta["target_line_gap_fraction"],
                "target_halo_factor": vis_meta["target_halo_factor"],
            },
            "correctness_definition": (
                "||R_gt @ p_src + t_gt - p_tgt||_2 "
                f"< {float(args.threshold_mm):g} mm"
            ),
            "correctness_computed_before_visualization_transform": True,
            "selection": (
                "highest corr_scores among GT-correct predictions"
                if args.max_lines > 0
                else "all GT-correct predictions"
            ),
        }
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats_path.write_text(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"JSON: {stats_path}")


if __name__ == "__main__":
    main()
