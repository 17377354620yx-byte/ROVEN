import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from visualization.correspondence_method_comparison import (
    METHOD_ORDER,
    ViewStyle,
    order_methods_by_correct_correspondence_count,
    order_methods_by_total_correct_correspondence_count,
    render_panel,
    render_standalone_panel,
    sort_cases_by_visibility,
    write_metrics_csv,
)


def _cache():
    return {
        "source_points_mm": np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        "target_points_mm": np.array([[0, 1, 0], [1, 1, 0]], dtype=float),
        "source_corr_mm": np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], dtype=float),
        "target_corr_mm": np.array([[0, 1, 0], [1, 1, 0], [0.5, 1, 0]], dtype=float),
        "correct_mask_2mm": np.array([True, False, False]),
        "correct_mask_5mm": np.array([True, False, True]),
        "provenance": {"method": "ours", "sample_id": "fixture.npz"},
    }


def test_method_order_matches_requested_comparison():
    assert METHOD_ORDER == (
        "ours", "geotransformer", "castv2", "dfat", "lepard", "predator",
        "parenet", "livermatch", "livermatch_p2p",
    )


def test_cases_are_sorted_by_descending_visibility_without_mutating_input():
    cases = [
        {"sample": "low.npz", "visibility": 0.2},
        {"sample": "high.npz", "visibility": 0.4},
        {"sample": "mid.npz", "visibility": 0.3},
    ]

    ordered = sort_cases_by_visibility(cases)

    assert [case["sample"] for case in ordered] == ["high.npz", "mid.npz", "low.npz"]
    assert [case["sample"] for case in cases] == ["low.npz", "high.npz", "mid.npz"]


def test_methods_are_sorted_by_gt_correct_correspondence_count():
    counts = {
        "ours": 30,
        "geotransformer": 90,
        "castv2": 10,
        "dfat": 70,
        "lepard": 60,
        "predator": 50,
        "parenet": 80,
        "livermatch": 20,
        "livermatch_p2p": 40,
    }
    caches = {
        method: {
            "source_corr_mm": np.zeros((count + 3, 3)),
            "correct_mask_5mm": np.r_[np.ones(count, dtype=bool), np.zeros(3, dtype=bool)],
        }
        for method, count in counts.items()
    }

    assert order_methods_by_correct_correspondence_count(caches) == [
        "castv2", "livermatch", "ours", "livermatch_p2p", "predator",
        "lepard", "dfat", "parenet", "geotransformer",
    ]


def test_one_global_method_order_uses_total_correct_count_across_cases():
    first = {
        method: {"correct_mask_5mm": np.ones(index + 1, dtype=bool)}
        for index, method in enumerate(METHOD_ORDER)
    }
    second = {
        method: {"correct_mask_5mm": np.ones(20 - 2 * index, dtype=bool)}
        for index, method in enumerate(METHOD_ORDER)
    }

    order = order_methods_by_total_correct_correspondence_count([first, second])

    assert order == list(reversed(METHOD_ORDER))


def test_render_panel_draws_every_correct_line_without_cap():
    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d", computed_zorder=False)

    audit = render_panel(axes, _cache(), ViewStyle())

    assert audit["num_rendered_correspondences"] == 2
    assert audit["num_precision_correspondences"] == 1
    assert len(audit["line_collection"]._segments3d) == 2
    assert len(audit["precision_line_collection"]._segments3d) == 1
    plt.close(figure)


def test_render_panel_offsets_only_display_target_and_keeps_line_ends_attached():
    cache = _cache()
    original_target = cache["target_corr_mm"].copy()
    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d", computed_zorder=False)

    audit = render_panel(axes, cache, ViewStyle())

    segments = np.asarray(audit["line_collection"]._segments3d)
    np.testing.assert_allclose(
        segments[:, 1], audit["target_corr_display"][cache["correct_mask_5mm"]]
    )
    np.testing.assert_allclose(cache["target_corr_mm"], original_target)
    assert not np.allclose(segments[:, 1], original_target[cache["correct_mask_5mm"]])
    plt.close(figure)


def test_target_halo_and_blue_points_are_above_green_lines():
    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d", computed_zorder=False)

    audit = render_panel(axes, _cache(), ViewStyle())

    assert audit["line_collection"].get_zorder() < audit["target_halo"].get_zorder()
    assert audit["target_halo"].get_zorder() < audit["target_points"].get_zorder()
    assert audit["target_points"].get_facecolor()[0, 2] > 0.9
    assert axes.axison is False
    plt.close(figure)


def test_uniform_mode_draws_all_5mm_lines_without_precision_overlay():
    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d", computed_zorder=False)

    audit = render_panel(axes, _cache(), ViewStyle(color_mode="uniform"))

    assert len(audit["line_collection"]._segments3d) == 2
    assert audit["line_collection"].get_alpha() == 0.38
    assert audit["precision_line_collection"] not in axes.collections
    plt.close(figure)


def test_view_style_keeps_approved_camera_defaults():
    style = ViewStyle()
    assert (style.vis_rx, style.vis_ry, style.vis_rz) == (22.0, -18.0, -8.0)
    assert (style.elev, style.azim) == (10.0, -78.0)
    assert (style.layout_right, style.layout_up) == (1.25, 0.75)
    assert style.threshold_mm == 5.0
    assert style.precision_threshold_mm == 2.0
    assert style.panel_zoom == 1.12


def test_renderer_cli_can_be_invoked_by_file_path():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "visualization" / "correspondence_method_comparison.py"),
            "--help",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--selection" in result.stdout
    assert "--panel-zoom" in result.stdout
    assert "--color-mode" in result.stdout
    assert "--threshold-mm" in result.stdout


def test_standalone_panel_writes_png_and_keeps_all_correct_lines(tmp_path):
    output_path = tmp_path / "panel.png"

    audit = render_standalone_panel(_cache(), output_path, ViewStyle(), dpi=72)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert audit["num_rendered_correspondences"] == 2


def test_write_metrics_csv_combines_dataset_rows(tmp_path):
    output_path = tmp_path / "correspondence_metrics.csv"
    rows = [
        {"dataset": "in_silico", "method": "ours", "IR_5mm": 0.8},
        {"dataset": "in_vitro", "method": "ours", "IR_5mm": 0.7},
    ]

    write_metrics_csv(rows, output_path)

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "dataset,method,IR_5mm",
        "in_silico,ours,0.8",
        "in_vitro,ours,0.7",
    ]
