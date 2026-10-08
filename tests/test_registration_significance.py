import math
import json
from pathlib import Path

import numpy as np

from scripts.analyze_registration_significance import (
    build_analysis,
    cluster_id,
    cluster_bootstrap_ci,
    exact_mcnemar,
    format_p_for_sentence,
    holm_adjust,
    paired_continuous_summary,
    rank_biserial_from_paired_differences,
)


def test_holm_adjust_preserves_order_and_monotonicity():
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == [0.03, 0.06, 0.06]


def test_sentence_p_value_uses_inequality_without_equals_sign():
    assert format_p_for_sentence(0.0002) == "p<0.001"
    assert format_p_for_sentence(0.02) == "p=0.020"


def test_rank_biserial_is_positive_when_lower_ours_values_dominate():
    # Positive differences mean baseline error minus Ours error is positive.
    effect = rank_biserial_from_paired_differences(np.array([1.0, 3.0, -2.0]))

    assert effect == pytest.approx(1.0 / 3.0)


def test_paired_continuous_summary_uses_all_pairs_and_lower_is_better():
    result = paired_continuous_summary(
        ours=np.array([1.0, 2.0, 5.0, 4.0]),
        baseline=np.array([2.0, 3.0, 4.0, 8.0]),
    )

    assert result["n_pairs"] == 4
    assert result["median_ours"] == 3.0
    assert result["median_baseline"] == 3.5
    assert result["median_paired_difference"] == -1.0
    assert result["ours_win_rate"] == 0.75
    assert 0.0 <= result["p_raw"] <= 1.0
    assert result["rank_biserial"] > 0.0


def test_exact_mcnemar_reports_discordant_pairs_and_rate_difference():
    result = exact_mcnemar(
        ours=np.array([1, 1, 1, 0], dtype=bool),
        baseline=np.array([0, 0, 1, 0], dtype=bool),
    )

    assert result["n_pairs"] == 4
    assert result["ours_only_success"] == 2
    assert result["baseline_only_success"] == 0
    assert result["success_rate_difference"] == 0.5
    assert result["p_raw"] == 0.5


def test_cluster_id_uses_anatomical_case_not_individual_registration():
    assert cluster_id("in_silico_none", "00012/00003/00004.npz") == "00012"
    assert cluster_id("in_silico_noise4", "liver_3/00004/00001.npz") == "liver_3"
    assert cluster_id("in_vitro", "5_00173.npz") == "5"


def test_cluster_bootstrap_ci_respects_constant_paired_difference():
    low, high = cluster_bootstrap_ci(
        values=np.array([-2.0, -2.0, -2.0, -2.0]),
        clusters=np.array(["a", "a", "b", "b"]),
        statistic="median",
        repetitions=200,
        seed=7,
    )

    assert low == -2.0
    assert high == -2.0


def _write_toy_result(
    root: Path, model: str, condition: str, values: list[float], *, lepard_alias=False
) -> None:
    samples = []
    for index, value in enumerate(values):
        sample = {
            "sample": f"case{index // 2}/pair/{index}.npz",
            "visibility": 0.25 if index % 2 == 0 else 0.35,
            "RRE": value + 0.1,
            "RTE": value + 0.2,
            "RMSE": value + 0.3,
            "chamfer_dist": value + 0.4,
            "SR_5mm": float(value < 5.0),
            "RR_5deg_5mm": float(value + 0.2 < 5.0),
        }
        sample["RMS_TRE_mm" if lepard_alias else "rms_tre_mm"] = value
        samples.append(sample)
    path = root / model / f"{condition}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checkpoint": f"{model}.pth", "samples": samples}))


def test_build_analysis_writes_paired_tables_and_validation_report(tmp_path):
    result_root = tmp_path / "results"
    output_dir = tmp_path / "analysis"
    _write_toy_result(result_root, "ours", "clean", [1.0, 2.0, 3.0, 6.0])
    _write_toy_result(
        result_root, "baseline", "clean", [2.0, 3.0, 7.0, 8.0], lepard_alias=True
    )

    build_analysis(
        result_root,
        output_dir,
        model_specs=(("ours", "Ours"), ("baseline", "Baseline")),
        condition_specs=(("clean", "Clean"),),
        cluster_resolver=lambda _condition, sample: sample.split("/", 1)[0],
        bootstrap_repetitions=200,
    )

    per_sample = (output_dir / "per_sample_metrics_core4.csv").read_text()
    continuous = (output_dir / "paired_continuous_tests.csv").read_text()
    binary = (output_dir / "paired_binary_tests.csv").read_text()
    cluster = (output_dir / "cluster_sensitivity_tests.csv").read_text()
    report = (output_dir / "STATISTICAL_ANALYSIS.md").read_text()
    assert len(per_sample.splitlines()) == 9  # header + 2 methods x 4 samples
    assert "RMS-TRE" in continuous and "p_holm" in continuous
    assert "SR@5 mm" in binary and "ours_only_success" in binary
    assert "n_clusters" in cluster and ",2," in cluster
    assert "11/11" in report
    assert "Verification Status: ANALYZED" in report
    assert "p=<" not in report
    assert "Holm-adjusted p" in report


# Imported last so a missing pytest dependency cannot mask a missing production module.
import pytest
