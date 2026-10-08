import json
from pathlib import Path

import pytest

from scripts.plot_registration_boxplots import (
    ALL_MODEL_SPECS,
    MODEL_SPECS,
    build_outputs,
    collect_records,
    normalize_sample,
    visibility_bin,
)


def test_default_figure_models_are_the_four_main_text_methods():
    assert MODEL_SPECS == (
        ("ours", "Ours"),
        ("geotransformer", "GeoTransformer"),
        ("dfat", "DFAT"),
        ("parenet", "PARENet"),
    )
    assert len(ALL_MODEL_SPECS) == 10


def test_normalize_sample_accepts_native_and_lepard_field_names():
    native = normalize_sample(
        {
            "sample": "case/001.npz",
            "visibility": 0.25,
            "rms_tre_mm": 4.5,
            "RRE": 2.0,
            "RTE": 1.0,
            "RMSE": 3.0,
            "chamfer_dist": 8.0,
        }
    )
    lepard = normalize_sample(
        {
            "sample_name": "case/001.npz",
            "visibility": 0.25,
            "RMS_TRE_mm": 4.5,
            "RRE": 2.0,
            "RTE": 1.0,
            "RMSE": 3.0,
            "chamfer_dist": 8.0,
        }
    )

    assert native == lepard
    assert native["sample_id"] == "case/001.npz"
    assert native["RMS-TRE"] == 4.5


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.2, "0.2–0.3"),
        (0.299999, "0.2–0.3"),
        (0.3, "0.3–0.4"),
        (0.399999, "0.3–0.4"),
        (0.4, "0.3–0.4"),
        (0.199999, None),
        (0.400001, None),
    ],
)
def test_visibility_bin_has_nonoverlapping_closed_final_boundary(value, expected):
    assert visibility_bin(value) == expected


def _sample(sample_id: str, rms_tre: float) -> dict:
    return {
        "sample": sample_id,
        "visibility": 0.25,
        "rms_tre_mm": rms_tre,
        "RRE": 2.0,
        "RTE": 1.0,
        "RMSE": 3.0,
        "chamfer_dist": 8.0,
    }


def _write_result(root: Path, model: str, condition: str, samples: list[dict]) -> None:
    path = root / model / f"{condition}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"checkpoint": f"{model}.pth", "samples": samples}),
        encoding="utf-8",
    )


def test_collect_records_rejects_cross_model_sample_mismatch(tmp_path):
    _write_result(tmp_path, "ours", "clean", [_sample("a", 1.0), _sample("b", 2.0)])
    _write_result(tmp_path, "baseline", "clean", [_sample("a", 3.0), _sample("c", 4.0)])

    with pytest.raises(ValueError, match="sample IDs/order mismatch"):
        collect_records(
            tmp_path,
            model_specs=(("ours", "Ours"), ("baseline", "Baseline")),
            condition_specs=(("clean", "Clean"),),
        )


def test_collect_records_preserves_every_individual_observation(tmp_path):
    samples_a = [_sample("a", 1.0), _sample("b", 2.0)]
    samples_b = [_sample("a", 3.0), _sample("b", 4.0)]
    _write_result(tmp_path, "ours", "clean", samples_a)
    _write_result(tmp_path, "baseline", "clean", samples_b)

    records, provenance = collect_records(
        tmp_path,
        model_specs=(("ours", "Ours"), ("baseline", "Baseline")),
        condition_specs=(("clean", "Clean"),),
    )

    assert len(records) == 4
    assert [row["RMS-TRE"] for row in records] == [1.0, 2.0, 3.0, 4.0]
    assert provenance[0]["sample_count"] == 2
    assert provenance[1]["checkpoint"] == "baseline.pth"


def test_collect_records_audits_solver_status_counts(tmp_path):
    samples = [_sample("a", 1.0), _sample("b", 2.0)]
    samples[0]["solver_status"] = "ok"
    samples[1]["solver_status"] = "timeout"
    samples[1]["fallback_identity"] = True
    _write_result(tmp_path, "goicp", "clean", samples)

    _, provenance = collect_records(
        tmp_path,
        model_specs=(("goicp", "Go-ICP"),),
        condition_specs=(("clean", "Clean"),),
    )

    assert provenance[0]["solver_status_counts"] == {"ok": 1, "timeout": 1}
    assert provenance[0]["identity_fallback_count"] == 1


def test_collect_records_uses_reference_visibility_for_paired_bin_assignment(tmp_path):
    reference = [_sample("a", 1.0), _sample("b", 2.0)]
    baseline = [_sample("a", 3.0), _sample("b", 4.0)]
    reference[0]["visibility"] = 0.25
    baseline[0]["visibility"] = 0.35
    _write_result(tmp_path, "ours", "clean", reference)
    _write_result(tmp_path, "baseline", "clean", baseline)

    records, _ = collect_records(
        tmp_path,
        model_specs=(("ours", "Ours"), ("baseline", "Baseline")),
        condition_specs=(("clean", "Clean"),),
    )

    baseline_a = next(
        row for row in records if row["model"] == "Baseline" and row["sample_id"] == "a"
    )
    assert baseline_a["visibility"] == 0.25
    assert baseline_a["reported_visibility"] == 0.35
    assert baseline_a["visibility_bin"] == "0.2–0.3"


def test_build_outputs_writes_figures_and_unaggregated_data(tmp_path):
    result_root = tmp_path / "results"
    output_dir = tmp_path / "figures"
    samples_a = [_sample("a", 1.0), _sample("b", 3.0)]
    samples_b = [_sample("a", 2.0), _sample("b", 4.0)]
    samples_a[1]["visibility"] = 0.35
    samples_b[1]["visibility"] = 0.35
    _write_result(result_root, "ours", "clean", samples_a)
    _write_result(result_root, "baseline", "clean", samples_b)

    build_outputs(
        result_root,
        output_dir,
        model_specs=(("ours", "Ours"), ("baseline", "Baseline")),
        condition_specs=(("clean", "Clean"),),
        formats=("svg",),
        dpi=72,
    )

    assert (output_dir / "boxplot_rms_tre.svg").is_file()
    assert (output_dir / "boxplot_registration_metrics.svg").is_file()
    assert (output_dir / "boxplot_visibility_sensitivity.svg").is_file()
    assert (output_dir / "violin_rms_tre_linear.svg").is_file()
    assert (output_dir / "violin_rms_tre_log.svg").is_file()
    rows = (output_dir / "per_sample_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 5  # header plus all four individual observations
    summary = (output_dir / "summary_statistics.csv").read_text(encoding="utf-8")
    assert "1.5,2.0,2.5" in summary  # linear Q1, median, Q3 for Ours RMS-TRE
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert len(provenance) == 2
