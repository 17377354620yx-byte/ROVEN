from pathlib import Path
import subprocess
import sys

import pytest

from tools.select_visibility_qualitative_cases import (
    normalize_summary,
    select_benchmark_candidates,
    select_verified_cases,
)


BENCHMARK_METHODS = (
    "ours",
    "geotransformer",
    "castv2",
    "dfat",
    "lepard",
    "parenet",
    "livermatch",
    "livermatch_p2p",
)

CORRESPONDENCE_METHODS = BENCHMARK_METHODS + ("predator",)


def test_selector_cli_can_be_invoked_by_file_path():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "tools/select_visibility_qualitative_cases.py"), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--finalize-root" in completed.stdout


def _benchmark_row(sample, rms, ir, index, count=100):
    return {
        "sample": sample,
        "index": index,
        "visibility": 0.25,
        "rms_tre_mm": rms,
        "IR": ir,
        "fine_corr_count": count,
    }


def test_normalize_summary_accepts_livermatch_field_names_and_rejects_duplicates():
    summary = {
        "samples": [
            {
                "sample_name": "case.npz",
                "index": 4,
                "visibility": 0.23,
                "RMS_TRE_mm": 1.25,
                "IR": 0.6,
                "correspondences": 90,
            }
        ]
    }

    rows = normalize_summary(summary)

    assert rows["case.npz"]["rms_tre_mm"] == 1.25
    assert rows["case.npz"]["num_correspondences"] == 90
    with pytest.raises(ValueError, match="duplicate sample"):
        normalize_summary({"samples": summary["samples"] * 2})


def test_normalize_summary_rejects_nonfinite_and_missing_rows():
    with pytest.raises(ValueError, match="finite"):
        normalize_summary(
            {"samples": [{"sample": "bad.npz", "index": 0, "rms_tre_mm": float("nan"), "IR": 0.5}]}
        )
    with pytest.raises(ValueError, match="sample"):
        normalize_summary({"samples": [{"index": 0, "rms_tre_mm": 1.0, "IR": 0.5}]})


def test_select_benchmark_candidates_requires_strict_rms_win_and_is_deterministic():
    samples = [f"case_{i}.npz" for i in range(6)]
    summaries = {}
    for method_index, method in enumerate(BENCHMARK_METHODS):
        rows = []
        for i, sample in enumerate(samples):
            rms = 1.0 + i * 0.1 if method == "ours" else 2.0 + method_index * 0.1
            ir = 0.4 + i * 0.05 if method == "ours" else 0.45
            count = 200 + i * 10 if method == "ours" else 100
            rows.append(_benchmark_row(sample, rms, ir, i, count))
        summaries[method] = {"samples": rows}
    # Equality is not a strict win and must be removed.
    summaries["castv2"]["samples"][5]["rms_tre_mm"] = summaries["ours"]["samples"][5]["rms_tre_mm"]

    first = select_benchmark_candidates(summaries, count=4, per_metric=1)
    second = select_benchmark_candidates(summaries, count=4, per_metric=1)

    assert first == second
    assert len(first) == 4
    assert "case_5.npz" not in {row["sample"] for row in first}
    assert [row["candidate_rank"] for row in first] == [0, 1, 2, 3]
    assert all(row["rms_margin_mm"] > 0 for row in first)
    assert all("correct_proxy_margin" in row for row in first)


def test_select_benchmark_candidates_requires_all_methods_and_enough_winners():
    summaries = {
        method: {"samples": [_benchmark_row("only.npz", 1.0 if method == "ours" else 2.0, 0.5, 0)]}
        for method in BENCHMARK_METHODS
    }
    del summaries["dfat"]
    with pytest.raises(ValueError, match="dfat"):
        select_benchmark_candidates(summaries, count=1)

    summaries["dfat"] = {"samples": []}
    with pytest.raises(ValueError, match="Need 1"):
        select_benchmark_candidates(summaries, count=1)


def _corr_summary(method, samples):
    return {
        "method": method,
        "samples": [
            {
                "sample_id": sample,
                "RMSE_mm": values[method][0],
                "num_pred_correspondences": values[method][1] + 10,
                "threshold_statistics": {
                    "5": {
                        "num_correct": values[method][1],
                        "inlier_ratio": values[method][2],
                    }
                },
            }
            for sample, values in samples.items()
        ],
    }


def test_select_verified_cases_requires_best_correct_count_and_audits_rmse():
    candidates = [{"sample": f"case_{i}.npz", "subset_index": i} for i in range(5)]
    samples = {}
    for i in range(5):
        values = {}
        for method in CORRESPONDENCE_METHODS:
            values[method] = [2.0, 70, 0.5]
        values["ours"] = [1.0, 100 + i * 5, 0.7]
        samples[f"case_{i}.npz"] = values
    # RMSE is audited but does not control a correct-only visual; a count tie is rejected.
    samples["case_0.npz"]["predator"][0] = 0.9
    samples["case_1.npz"]["parenet"][1] = samples["case_1.npz"]["ours"][1]
    summaries = {method: _corr_summary(method, samples) for method in CORRESPONDENCE_METHODS}

    selected, evaluations = select_verified_cases(candidates, summaries, top_k=3)

    assert [row["sample"] for row in selected] == ["case_4.npz", "case_3.npz", "case_2.npz"]
    audited = {row["sample"]: row for row in evaluations}
    assert audited["case_0.npz"]["eligible"] is True
    assert audited["case_0.npz"]["ours_rmse_strictly_best"] is False
    rejected = {row["sample"]: row["reasons"] for row in evaluations if not row["eligible"]}
    assert "ours_correct_count_not_strictly_best" in rejected["case_1.npz"]


def test_select_verified_cases_rejects_missing_method_and_fails_if_underfilled():
    candidates = [{"sample": f"case_{i}.npz", "subset_index": i} for i in range(3)]
    samples = {}
    for i in range(3):
        samples[f"case_{i}.npz"] = {
            method: [1.0 if method == "ours" else 2.0, 100 if method == "ours" else 50, 0.7]
            for method in CORRESPONDENCE_METHODS
        }
    summaries = {method: _corr_summary(method, samples) for method in CORRESPONDENCE_METHODS}
    summaries.pop("predator")

    with pytest.raises(ValueError, match="predator"):
        select_verified_cases(candidates, summaries, top_k=3)

    summaries["predator"] = _corr_summary("predator", samples)
    samples["case_2.npz"]["predator"][1] = 100
    summaries["predator"] = _corr_summary("predator", samples)
    with pytest.raises(ValueError, match="Need 3 strictly eligible"):
        select_verified_cases(candidates, summaries, top_k=3)
