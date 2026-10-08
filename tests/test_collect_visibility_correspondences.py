import numpy as np
import pytest
from pathlib import Path
import subprocess
import sys

from tools.collect_visibility_correspondences import (
    network_transform_to_mm,
    normalize_raw_export,
)


def _context():
    gt_mm = np.eye(4)
    gt_mm[:3, 3] = [5.0, 20.0, 30.0]
    return {
        "sample_id": "fixture.npz",
        "source_points_mm": np.array([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]]),
        "target_points_mm": np.array([[5.0, 20.0, 30.0], [9.0, 20.0, 30.0]]),
        "source_center_mm": np.array([10.0, 0.0, 0.0]),
        "target_center_mm": np.array([5.0, 20.0, 30.0]),
        "scale_mm": 2.0,
        "gt_transform_mm": gt_mm,
    }


def _raw():
    return {
        "sample_id": "fixture.npz",
        "source_points": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "target_points": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "source_corr_points": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "target_corr_points": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "corr_scores": np.array([0.8, 0.7]),
        "gt_transform": np.eye(4),
        "estimated_transform": np.eye(4),
        "physical_scale_mm": 2.0,
        "coordinate_frame": "network_normalized",
        "source_variable": "model.source_corr",
        "target_variable": "model.target_corr",
    }


def test_network_transform_to_mm_accounts_for_separate_centres():
    transform = np.eye(4)
    transform[:3, 3] = [0.5, 0.0, 0.0]

    restored = network_transform_to_mm(
        transform,
        source_center_mm=np.array([10.0, 0.0, 0.0]),
        target_center_mm=np.array([5.0, 20.0, 30.0]),
        scale_mm=2.0,
    )

    np.testing.assert_allclose(restored[:3, 3], [-4.0, 20.0, 30.0])


def test_normalize_raw_export_restores_pairs_and_uses_central_gt():
    normalized = normalize_raw_export(
        _raw(), _context(), method="ours", checkpoint="/ours.pth"
    )

    np.testing.assert_allclose(normalized["source_corr_mm"], [[10, 0, 0], [12, 0, 0]])
    np.testing.assert_allclose(normalized["target_corr_mm"], [[5, 20, 30], [9, 20, 30]])
    np.testing.assert_allclose(normalized["gt_transform_mm"], _context()["gt_transform_mm"])
    assert normalized["provenance"]["correctness_transform"] == "central_dataset_gt"


def test_normalize_raw_export_rejects_sample_or_scale_mismatch():
    raw = _raw()
    raw["sample_id"] = "another.npz"
    with pytest.raises(ValueError, match="sample"):
        normalize_raw_export(raw, _context(), method="ours", checkpoint="/ours.pth")

    raw = _raw()
    raw["physical_scale_mm"] = 3.0
    with pytest.raises(ValueError, match="scale"):
        normalize_raw_export(raw, _context(), method="ours", checkpoint="/ours.pth")


def test_normalize_raw_export_does_not_create_missing_correspondences():
    raw = _raw()
    raw.pop("source_corr_points")
    with pytest.raises(KeyError):
        normalize_raw_export(raw, _context(), method="goicp", checkpoint=None)


def test_collector_cli_can_be_executed_by_file_path():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "tools/collect_visibility_correspondences.py"), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--raw-dir" in completed.stdout
