import numpy as np

from tools.select_representative_median_correspondence import select_stratified_medians


def test_selection_uses_each_visibility_third_and_is_deterministic():
    names = np.asarray([f"case_{index}" for index in range(9)])
    visibility = np.asarray([0.20, 0.21, 0.22, 0.25, 0.26, 0.27, 0.35, 0.36, 0.37])
    deformation = np.arange(9, dtype=np.float64)

    first = select_stratified_medians(names, visibility, deformation, seed=7)
    second = select_stratified_medians(names, visibility, deformation, seed=7)

    assert first == second
    assert [row["sample"] for row in first] == ["case_1", "case_4", "case_7"]
    assert [row["stratum"] for row in first] == [1, 2, 3]
    assert all(row["selection_strategy"] == "equal_count_visibility_stratum_median" for row in first)
