import numpy as np

from tools.p2p_benchmark_metrics import pose_metrics, strict_success_metrics
from scripts.summarize_visibility_02_04 import strict_rates


def test_strict_success_uses_five_mm_and_strict_boundary():
    assert strict_success_metrics(4.999, 4.999, 4.999) == {
        "SR_5mm": 1.0,
        "RR_5deg_5mm": 1.0,
    }
    assert strict_success_metrics(5.0, 4.999, 4.999)["SR_5mm"] == 0.0
    assert strict_success_metrics(4.999, 5.0, 4.999)["RR_5deg_5mm"] == 0.0
    assert strict_success_metrics(4.999, 4.999, 5.0)["RR_5deg_5mm"] == 0.0


def test_pose_rr_uses_translation_mm_not_normalized_translation():
    source = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    target = source.copy()
    oracle = np.eye(4)
    estimated = np.eye(4)
    estimated[0, 3] = 0.051

    metrics = pose_metrics(source, target, estimated, oracle, physical_scale=100.0)

    assert metrics["RTE"] == 5.1
    assert metrics["RR"] == 0.0
    assert metrics["RR_5deg_5mm"] == 0.0


def test_summary_recomputes_strict_rates_instead_of_trusting_legacy_fields():
    samples = [
        {"rms_tre_mm": 4.0, "RRE": 4.0, "RTE": 4.0, "RR": 0.0},
        {"rms_tre_mm": 5.0, "RRE": 5.0, "RTE": 5.0, "RR": 1.0},
    ]

    assert strict_rates(samples) == {
        "success_rate_5mm": 0.5,
        "rr_5deg_5mm": 0.5,
    }
