import json

import numpy as np
import pytest

from tools.correspondence_cache import (
    compute_gt_inlier_mask,
    compute_threshold_statistics,
    read_correspondence_cache,
    restore_points_to_mm,
    write_correspondence_cache,
)


def test_restore_points_to_mm_handles_metric_and_normalized_coordinates():
    points = np.array([[0.001, -0.002, 0.003]], dtype=np.float64)
    np.testing.assert_allclose(
        restore_points_to_mm(points, {"coordinate_unit": "m"}),
        [[1.0, -2.0, 3.0]],
    )
    np.testing.assert_allclose(
        restore_points_to_mm(points * 1000.0, {"coordinate_unit": "mm"}),
        [[1.0, -2.0, 3.0]],
    )
    np.testing.assert_allclose(
        restore_points_to_mm(
            np.array([[1.0, 2.0, 3.0]]),
            {
                "coordinate_unit": "normalized",
                "normalization_scale_mm": 2.0,
                "normalization_center_mm": [10.0, 20.0, 30.0],
            },
        ),
        [[12.0, 24.0, 36.0]],
    )


def test_gt_inlier_mask_uses_source_to_target_and_strict_threshold():
    gt = np.eye(4)
    gt[:3, 3] = [10.0, 0.0, 0.0]
    source = np.zeros((4, 3), dtype=np.float64)
    target = np.array(
        [
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [14.999, 0.0, 0.0],
            [15.0, 0.0, 0.0],
        ]
    )

    mask, residuals = compute_gt_inlier_mask(source, target, gt, threshold_mm=5.0)

    np.testing.assert_allclose(residuals, [0.0, 2.0, 4.999, 5.0])
    assert mask.tolist() == [True, True, True, False]


def test_visualization_offset_cannot_change_gt_mask():
    source = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[1.0, 2.0, 3.0]])
    gt = np.eye(4)
    mask_before, _ = compute_gt_inlier_mask(source, target, gt, threshold_mm=5.0)

    display_target = target + np.array([[100.0, -20.0, 5.0]])

    assert mask_before.tolist() == [True]
    assert not np.array_equal(display_target, target)


def test_threshold_statistics_use_same_predictions_for_all_thresholds():
    residuals = np.array([1.0, 2.0, 2.999, 4.0, 5.0])

    stats = compute_threshold_statistics(residuals, thresholds_mm=(2.0, 3.0, 4.0, 5.0))

    assert stats == {
        "2": {"num_correct": 1, "inlier_ratio": 0.2},
        "3": {"num_correct": 3, "inlier_ratio": 0.6},
        "4": {"num_correct": 3, "inlier_ratio": 0.6},
        "5": {"num_correct": 4, "inlier_ratio": 0.8},
    }


def test_cache_round_trip_recomputes_auditable_masks(tmp_path):
    path = tmp_path / "ours.npz"
    gt = np.eye(4)
    payload = {
        "source_points_mm": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "target_points_mm": np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        "source_corr_mm": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "target_corr_mm": np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        "corr_scores": np.array([0.9, 0.2]),
        "gt_transform_mm": gt,
        "estimated_transform_mm": gt,
    }
    provenance = {
        "method": "ours",
        "sample_id": "fixture.npz",
        "checkpoint": "/checkpoint.pth",
        "coordinate_frame": "original_mm",
    }

    write_correspondence_cache(path, payload, provenance)
    loaded = read_correspondence_cache(path)

    assert loaded["correct_mask_5mm"].tolist() == [True, False]
    assert loaded["threshold_statistics"]["5"] == {
        "num_correct": 1,
        "inlier_ratio": 0.5,
    }
    assert loaded["provenance"] == provenance
    assert json.loads(str(np.load(path, allow_pickle=False)["provenance_json"])) == provenance


@pytest.mark.parametrize(
    "points,metadata",
    [
        (np.zeros((2, 2)), {"coordinate_unit": "mm"}),
        (np.array([[np.nan, 0.0, 0.0]]), {"coordinate_unit": "mm"}),
        (np.zeros((1, 3)), {"coordinate_unit": "unknown"}),
        (
            np.zeros((1, 3)),
            {"coordinate_unit": "normalized", "normalization_scale_mm": 0.0, "normalization_center_mm": [0, 0, 0]},
        ),
    ],
)
def test_restore_points_rejects_invalid_inputs(points, metadata):
    with pytest.raises(ValueError):
        restore_points_to_mm(points, metadata)


def test_cache_rejects_mismatched_correspondence_shapes(tmp_path):
    payload = {
        "source_points_mm": np.zeros((1, 3)),
        "target_points_mm": np.zeros((1, 3)),
        "source_corr_mm": np.zeros((2, 3)),
        "target_corr_mm": np.zeros((1, 3)),
        "gt_transform_mm": np.eye(4),
    }
    with pytest.raises(ValueError, match="matching"):
        write_correspondence_cache(tmp_path / "bad.npz", payload, {"method": "ours"})
