"""Shared, model-agnostic metrics for the P2P liver benchmark.

The pose convention is source-to-target.  The dataset transform is the rigid
oracle fitted from the paired liver data; it is not a claim that the liver
deformation itself is rigid.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


METRIC_NAMES = (
    "PIR", "IR", "RRE", "RTE", "RMSE", "RR", "r_rmse", "r_mae",
    "t_rmse", "t_mae", "chamfer_dist",
)

METRIC_DEFINITIONS = {
    "PIR": "Precision of predicted coarse patch/node correspondences; N/A for models without a coarse correspondence stage",
    "IR": "Fraction of predicted point correspondences whose oracle residual is below the model's common acceptance radius",
    "RRE": "Isotropic relative rotation error, degrees",
    "RTE": "L2 relative translation error, millimetres",
    "RMSE": "Root mean square displacement between source points transformed by the estimate and rigid oracle, millimetres",
    "RR": "Strict registration recall: RRE < 5 degrees and RTE < 5 millimetres",
    "r_rmse": "RMSE of wrapped XYZ Euler-angle component errors, degrees",
    "r_mae": "MAE of wrapped XYZ Euler-angle component errors, degrees",
    "t_rmse": "RMSE of XYZ translation-component errors, millimetres",
    "t_mae": "MAE of XYZ translation-component errors, millimetres",
    "chamfer_dist": "Mean bidirectional nearest-neighbour L2 Chamfer distance between target and transformed source, millimetres",
}


def as_numpy(value):
    """Detach a tensor-like object and return a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def apply_transform(points, transform):
    points = as_numpy(points)
    transform = as_numpy(transform)
    return points @ transform[:3, :3].T + transform[:3, 3]


def strict_success_metrics(rms_tre_mm, rre_deg, rte_mm):
    """Return the primary strict success criteria (strict inequalities)."""
    return {
        "SR_5mm": float(float(rms_tre_mm) < 5.0),
        "RR_5deg_5mm": float(float(rre_deg) < 5.0 and float(rte_mm) < 5.0),
    }


def pose_metrics(
    source,
    target,
    estimated_transform,
    oracle_transform,
    physical_scale,
    *,
    rr_rre_threshold_deg=5.0,
    rr_rte_threshold_mm=5.0,
):
    """Compute comparable pose/point-cloud metrics for one sample."""
    source = as_numpy(source).astype(np.float64, copy=False)
    target = as_numpy(target).astype(np.float64, copy=False)
    estimated = as_numpy(estimated_transform).astype(np.float64, copy=False)
    oracle = as_numpy(oracle_transform).astype(np.float64, copy=False)
    scale = float(physical_scale)

    relative_rotation = estimated[:3, :3].T @ oracle[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    rre = float(np.degrees(np.arccos(cosine)))
    translation_delta = estimated[:3, 3] - oracle[:3, 3]
    rte_normalized = float(np.linalg.norm(translation_delta))

    estimated_source = apply_transform(source, estimated)
    oracle_source = apply_transform(source, oracle)
    rmse = float(
        np.sqrt(np.mean(np.sum((estimated_source - oracle_source) ** 2, axis=1)))
        * scale
    )

    estimated_euler = Rotation.from_matrix(
        np.array(estimated[:3, :3], copy=True)
    ).as_euler("xyz", degrees=True)
    oracle_euler = Rotation.from_matrix(
        np.array(oracle[:3, :3], copy=True)
    ).as_euler("xyz", degrees=True)
    rotation_delta = (estimated_euler - oracle_euler + 180.0) % 360.0 - 180.0
    translation_delta_mm = translation_delta * scale

    if len(source) and len(target):
        forward = cKDTree(target).query(estimated_source, k=1)[0].mean()
        backward = cKDTree(estimated_source).query(target, k=1)[0].mean()
        chamfer = float(0.5 * (forward + backward) * scale)
    else:
        chamfer = math.nan

    rte_mm = rte_normalized * scale
    rr_strict = float(
        rre < rr_rre_threshold_deg and rte_mm < rr_rte_threshold_mm
    )
    return {
        "RRE": rre,
        "RTE": rte_mm,
        "RMSE": rmse,
        "RR": rr_strict,
        "RR_5deg_5mm": rr_strict,
        "RR_legacy_15deg_rte0.3": float(rre < 15.0 and rte_normalized < 0.3),
        "r_rmse": float(np.sqrt(np.mean(rotation_delta ** 2))),
        "r_mae": float(np.mean(np.abs(rotation_delta))),
        "t_rmse": float(np.sqrt(np.mean(translation_delta_mm ** 2))),
        "t_mae": float(np.mean(np.abs(translation_delta_mm))),
        "chamfer_dist": chamfer,
        "RTE_normalized": rte_normalized,
    }


def correspondence_inlier_ratio(source_corr, target_corr, oracle_transform, radius):
    """Point-correspondence inlier ratio under the rigid oracle."""
    source_corr = as_numpy(source_corr)
    target_corr = as_numpy(target_corr)
    if len(source_corr) == 0:
        return math.nan
    residuals = np.linalg.norm(
        apply_transform(source_corr, oracle_transform) - target_corr, axis=1
    )
    return float(np.mean(residuals < float(radius)))


def summarize_metrics(records, names=METRIC_NAMES):
    """Mean/std/count summary that treats None and non-finite values as N/A."""
    summary = {}
    for name in names:
        values = np.asarray(
            [np.nan if row.get(name) is None else row.get(name, np.nan) for row in records],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        summary[name] = float(values.mean()) if len(values) else None
        summary[f"{name}_std"] = float(values.std()) if len(values) else None
        summary[f"{name}_valid_samples"] = int(len(values))
    return summary
