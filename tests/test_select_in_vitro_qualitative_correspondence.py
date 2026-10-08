from tools.select_in_vitro_qualitative_correspondence import (
    METHODS,
    select_cases,
    select_contrast_cases,
)


def _rows():
    rows = {method: {} for method in METHODS}
    samples = (("low.npz", 0.23), ("mid.npz", 0.30), ("high.npz", 0.38))
    for index, (sample, visibility) in enumerate(samples):
        for method_index, method in enumerate(METHODS):
            rows[method][sample] = {
                "index": index,
                "sample": sample,
                "visibility": visibility,
                "deformation_mm": 1.0,
                "rms_tre_mm": 2.0 + method_index,
                "IR": 0.7 - 0.01 * method_index,
                "fine_corr_count": 100 - 5 * method_index,
            }
    return rows


def test_select_cases_returns_one_case_per_visibility_band():
    selected = select_cases(_rows())

    assert [row["sample"] for row in selected] == ["low.npz", "mid.npz", "high.npz"]
    assert all(row["ours_rms_tre_mm"] == 2.0 for row in selected)
    assert all(row["selection_score"] > 0 for row in selected)


def test_contrast_selection_maximizes_expected_inlier_margin_in_each_band():
    rows = _rows()
    for method in METHODS:
        rows[method]["low_better.npz"] = dict(rows[method]["low.npz"])
        rows[method]["low_better.npz"]["sample"] = "low_better.npz"
        rows[method]["low_better.npz"]["index"] = 10
    rows["ours"]["low_better.npz"]["fine_corr_count"] = 200

    selected = select_contrast_cases(rows)

    assert selected[0]["sample"] == "low_better.npz"
    assert all(
        row["selection_strategy"] == "visibility_stratified_max_inlier_margin"
        for row in selected
    )
