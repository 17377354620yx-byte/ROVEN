import numpy as np

from visualization.error_heatmap import apply_transform, compute_registration_error
from visualization.statistics import compute_error_statistics


def test_transform_and_scale_follow_project_convention():
    source = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    ground_truth = np.eye(4)
    prediction = np.eye(4)
    prediction[:3, 3] = [0.1, 0.0, 0.0]
    predicted, expected, errors = compute_registration_error(
        source, prediction, ground_truth, scale_factor=10.0
    )
    np.testing.assert_allclose(predicted, source + [0.1, 0.0, 0.0])
    np.testing.assert_allclose(expected, source)
    np.testing.assert_allclose(errors, [1.0, 1.0])


def test_rms_reduction_matches_evaluation_formula():
    errors = np.asarray([3.0, 4.0])
    stats = compute_error_statistics(errors)
    assert stats["mean_mm"] == 3.5
    assert stats["rms_mm"] == np.sqrt((9.0 + 16.0) / 2.0)


def test_row_vector_rotation():
    transform = np.eye(4)
    transform[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    np.testing.assert_allclose(apply_transform([[1.0, 0.0, 0.0]], transform),
                               [[0.0, 1.0, 0.0]], atol=1e-12)
