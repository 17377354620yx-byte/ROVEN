"""Auditable cache utilities for P2P predicted correspondences."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np


DEFAULT_THRESHOLDS_MM = (2.0, 3.0, 4.0, 5.0)


def _scalar_text(value: object, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    text = str(array.item())
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def validate_raw_correspondence_export(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate a model-native export before any coordinate restoration."""

    required = (
        "sample_id",
        "source_points",
        "target_points",
        "source_corr_points",
        "target_corr_points",
        "gt_transform",
        "estimated_transform",
        "physical_scale_mm",
        "coordinate_frame",
        "source_variable",
        "target_variable",
    )
    for key in required:
        if key not in payload:
            raise KeyError(f"raw correspondence export is missing {key}")
    source_points = _points(payload["source_points"], "source_points")
    target_points = _points(payload["target_points"], "target_points")
    source_corr = _points(payload["source_corr_points"], "source_corr_points")
    target_corr = _points(payload["target_corr_points"], "target_corr_points")
    if source_corr.shape != target_corr.shape:
        raise ValueError("source and target correspondence exports must have matching shapes")
    _transform(payload["gt_transform"], "gt_transform")
    _transform(payload["estimated_transform"], "estimated_transform")
    scale = float(payload["physical_scale_mm"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("physical_scale_mm must be positive and finite")
    coordinate_frame = _scalar_text(payload["coordinate_frame"], "coordinate_frame")
    if coordinate_frame not in {"network_normalized", "original_mm"}:
        raise ValueError("coordinate_frame must be network_normalized or original_mm")
    _scalar_text(payload["sample_id"], "sample_id")
    _scalar_text(payload["source_variable"], "source_variable")
    _scalar_text(payload["target_variable"], "target_variable")

    scores_value = payload.get("corr_scores")
    if scores_value is None:
        has_confidence = False
    else:
        scores = np.asarray(scores_value, dtype=np.float64)
        if scores.shape != (len(source_corr),) or not np.isfinite(scores).all():
            raise ValueError("corr_scores must be finite and match correspondence count")
        has_confidence = True
    return {
        "num_source_points": int(len(source_points)),
        "num_target_points": int(len(target_points)),
        "num_pred_correspondences": int(len(source_corr)),
        "has_confidence": has_confidence,
    }


def unsupported_correspondence_record(
    method: str,
    sample_id: str,
    reason: str,
) -> dict[str, object]:
    """Describe a transform-only method without inventing point matches."""

    if not method or not sample_id or not reason:
        raise ValueError("method, sample_id, and reason must not be empty")
    return {
        "method": method,
        "sample_id": sample_id,
        "supports_correspondences": False,
        "status": "not_applicable",
        "reason": reason,
    }


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _transform(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} must be a homogeneous transform")
    return array


def restore_points_to_mm(points: np.ndarray, metadata: Mapping[str, object]) -> np.ndarray:
    """Restore points from metres, millimetres, or normalized coordinates."""

    array = _points(points, "points")
    unit = str(metadata.get("coordinate_unit", ""))
    if unit == "mm":
        return array.copy()
    if unit == "m":
        return array * 1000.0
    if unit != "normalized":
        raise ValueError("coordinate_unit must be one of: mm, m, normalized")

    scale = float(metadata.get("normalization_scale_mm", np.nan))
    center = np.asarray(metadata.get("normalization_center_mm"), dtype=np.float64)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization_scale_mm must be positive and finite")
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("normalization_center_mm must be a finite length-3 vector")
    return array * scale + center


def compute_gt_inlier_mask(
    src_corr_mm: np.ndarray,
    tgt_corr_mm: np.ndarray,
    gt_transform_mm: np.ndarray,
    threshold_mm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify pairs using only the GT source-to-target transform."""

    source = _points(src_corr_mm, "src_corr_mm")
    target = _points(tgt_corr_mm, "tgt_corr_mm")
    if source.shape != target.shape:
        raise ValueError("source and target correspondences must have matching shapes")
    transform = _transform(gt_transform_mm, "gt_transform_mm")
    threshold = float(threshold_mm)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_mm must be positive and finite")

    source_gt = source @ transform[:3, :3].T + transform[:3, 3]
    residuals = np.linalg.norm(source_gt - target, axis=1)
    return residuals < threshold, residuals


def compute_threshold_statistics(
    residuals_mm: np.ndarray,
    thresholds_mm: Sequence[float] = DEFAULT_THRESHOLDS_MM,
) -> dict[str, dict[str, float | int]]:
    residuals = np.asarray(residuals_mm, dtype=np.float64)
    if residuals.ndim != 1 or not np.isfinite(residuals).all():
        raise ValueError("residuals_mm must be a finite one-dimensional array")
    count = int(residuals.size)
    statistics: dict[str, dict[str, float | int]] = {}
    for value in thresholds_mm:
        threshold = float(value)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("thresholds_mm must be positive and finite")
        num_correct = int(np.count_nonzero(residuals < threshold))
        statistics[f"{threshold:g}"] = {
            "num_correct": num_correct,
            "inlier_ratio": float(num_correct / count) if count else 0.0,
        }
    return statistics


def write_correspondence_cache(
    path: str | Path,
    payload: Mapping[str, object],
    provenance: Mapping[str, object],
) -> None:
    """Validate and atomically write one method/sample correspondence cache."""

    output_path = Path(path)
    source_points = _points(payload["source_points_mm"], "source_points_mm")
    target_points = _points(payload["target_points_mm"], "target_points_mm")
    source_corr = _points(payload["source_corr_mm"], "source_corr_mm")
    target_corr = _points(payload["target_corr_mm"], "target_corr_mm")
    if source_corr.shape != target_corr.shape:
        raise ValueError("source and target correspondences must have matching shapes")
    gt_transform = _transform(payload["gt_transform_mm"], "gt_transform_mm")
    estimated = payload.get("estimated_transform_mm")
    estimated_transform = (
        np.empty((0, 4), dtype=np.float64)
        if estimated is None
        else _transform(estimated, "estimated_transform_mm")
    )

    scores_value = payload.get("corr_scores")
    if scores_value is None:
        scores = np.empty((0,), dtype=np.float64)
        has_scores = False
    else:
        scores = np.asarray(scores_value, dtype=np.float64)
        if scores.shape != (source_corr.shape[0],) or not np.isfinite(scores).all():
            raise ValueError("corr_scores must be finite and match correspondence count")
        has_scores = True

    _, residuals = compute_gt_inlier_mask(source_corr, target_corr, gt_transform, 5.0)
    stats = compute_threshold_statistics(residuals)
    arrays: dict[str, np.ndarray] = {
        "source_points_mm": source_points,
        "target_points_mm": target_points,
        "source_corr_mm": source_corr,
        "target_corr_mm": target_corr,
        "corr_scores": scores,
        "has_corr_scores": np.asarray(has_scores),
        "gt_transform_mm": gt_transform,
        "estimated_transform_mm": estimated_transform,
        "residuals_mm": residuals,
        "thresholds_mm": np.asarray(DEFAULT_THRESHOLDS_MM, dtype=np.float64),
        "threshold_statistics_json": np.asarray(json.dumps(stats, sort_keys=True)),
        "provenance_json": np.asarray(json.dumps(dict(provenance), sort_keys=True)),
    }
    for threshold in DEFAULT_THRESHOLDS_MM:
        arrays[f"correct_mask_{threshold:g}mm"] = residuals < threshold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def read_correspondence_cache(path: str | Path) -> dict[str, object]:
    """Read a cache and reject stale or internally inconsistent masks."""

    with np.load(Path(path), allow_pickle=False) as archive:
        result: dict[str, object] = {key: archive[key].copy() for key in archive.files}

    provenance = json.loads(str(result.pop("provenance_json")))
    statistics = json.loads(str(result.pop("threshold_statistics_json")))
    _, residuals = compute_gt_inlier_mask(
        result["source_corr_mm"],
        result["target_corr_mm"],
        result["gt_transform_mm"],
        5.0,
    )
    if not np.allclose(result["residuals_mm"], residuals, atol=1e-9):
        raise ValueError("cached residuals do not match GT recomputation")
    expected_statistics = compute_threshold_statistics(residuals)
    if statistics != expected_statistics:
        raise ValueError("cached threshold statistics do not match GT recomputation")
    for threshold in DEFAULT_THRESHOLDS_MM:
        key = f"correct_mask_{threshold:g}mm"
        if not np.array_equal(result[key], residuals < threshold):
            raise ValueError(f"cached {key} does not match GT recomputation")

    result["provenance"] = provenance
    result["threshold_statistics"] = statistics
    return result
