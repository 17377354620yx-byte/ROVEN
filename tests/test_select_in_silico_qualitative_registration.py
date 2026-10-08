import numpy as np

from tools.select_in_silico_qualitative_registration import (
    METHODS,
    representative_candidates,
    select_low_visibility_advantage,
    select_cases,
)


def _rows(profiles):
    result = {method: {} for method in METHODS}
    for index, (sample, values) in enumerate(profiles.items()):
        for method, value in zip(METHODS, values):
            result[method][sample] = {
                "index": index,
                "sample": sample,
                "visibility": 0.2 + index * 0.01,
                "deformation_mm": 4.0,
                "rms_tre_mm": value,
            }
    return result


def test_representative_candidates_enforce_method_tiers():
    rows = _rows(
        {
            "good.npz": [5.0, 5.5, 5.8, 6.0, 6.2, 7.0, 9.0, 10.0, 8.0, 40.0],
            "ours_not_best.npz": [6.0, 5.5, 6.2, 7.0, 7.1, 8.0, 9.0, 10.0, 8.5, 40.0],
            "goicp_not_worst.npz": [5.0, 5.5, 5.8, 6.0, 6.2, 7.0, 9.0, 10.0, 8.0, 10.5],
        }
    )

    selected = representative_candidates(rows)

    assert [row["sample"] for row in selected] == ["good.npz"]
    assert all(selected[0]["checks"].values())


def test_select_cases_is_deterministic_and_preserves_metrics():
    rows = _rows(
        {
            "a.npz": [5.0, 5.5, 5.8, 6.0, 6.2, 7.0, 9.0, 10.0, 8.0, 40.0],
            "b.npz": [4.0, 4.5, 4.8, 5.0, 5.2, 6.0, 8.0, 9.0, 7.0, 35.0],
            "c.npz": [6.0, 6.5, 6.8, 7.0, 7.2, 8.0, 10.0, 11.0, 9.0, 45.0],
        }
    )

    selected, candidates = select_cases(rows, np.asarray([10, 11, 12]), top_k=2)

    assert len(candidates) == 3
    assert [row["subset_index"] for row in selected] == [0, 1]
    assert all(row["original_index"] in {10, 11, 12} for row in selected)
    assert all(set(row["rms_tre_mm"]) == set(METHODS) for row in selected)


def test_low_visibility_selector_prefers_clear_ours_advantage():
    rows = _rows(
        {
            "small_gap.npz": [2.0, 2.4, 2.5, 2.6, 2.7, 3.0, 4.0, 5.0, 4.5, 8.0],
            "large_gap.npz": [1.5, 3.5, 3.0, 4.0, 2.5, 3.0, 5.0, 6.0, 5.5, 8.0],
        }
    )
    for method in METHODS:
        rows[method]["small_gap.npz"]["IR"] = 0.50
        rows[method]["large_gap.npz"]["IR"] = 0.30
    rows["ours"]["small_gap.npz"]["IR"] = 0.55
    rows["geotransformer"]["small_gap.npz"]["IR"] = 0.50
    rows["ours"]["large_gap.npz"]["IR"] = 0.70
    rows["geotransformer"]["large_gap.npz"]["IR"] = 0.30

    selected = select_low_visibility_advantage(rows, 0.19, 0.22)

    assert selected["sample"] == "large_gap.npz"
    assert selected["selection_strategy"] == "low_visibility_ours_advantage"
