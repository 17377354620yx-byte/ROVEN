import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "geotransformer.p2p_liver"
    / "visualization"
    / "visualize_one_correspondence.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("correspondence_paper_view", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_preserve_the_approved_paper_view():
    module = load_module()

    args = module.make_parser().parse_args(["--dataset", "in_silico"])

    assert (args.vis_rx, args.vis_ry, args.vis_rz) == (22.0, -18.0, -8.0)
    assert (args.view_elev, args.view_azim) == (10.0, -78.0)
    assert (args.layout_right, args.layout_up) == (1.25, 0.75)
    assert (args.line_width, args.line_alpha) == (0.45, 0.38)


def test_target_points_are_drawn_opaque_above_exact_correspondence_endpoints(tmp_path):
    module = load_module()
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    figure, axes, lines, vis_meta = module.render_correct_only(
        source_points_mm=source,
        target_points_mm=target,
        source_corr_mm=source,
        target_corr_mm=target,
        selected_indices=np.array([0, 1]),
        png_path=tmp_path / "layering.png",
        pdf_path=tmp_path / "layering.pdf",
        dpi=40,
        vis_rx_deg=0.0,
        vis_ry_deg=0.0,
        vis_rz_deg=0.0,
    )

    from matplotlib.colors import to_hex

    target_collection = next(
        collection
        for collection in axes.collections
        if len(collection.get_facecolor())
        and to_hex(collection.get_facecolor()[0], keep_alpha=False) == "#1f4dff"
    )
    displayed_target = target + np.asarray(vis_meta["visualization_offset_mm"])

    assert axes.computed_zorder is False
    assert target_collection.get_zorder() > lines.get_zorder()
    assert target_collection.get_alpha() == 1.0
    np.testing.assert_allclose(np.asarray(lines._segments3d)[:, 1], displayed_target)

    import matplotlib.pyplot as plt

    plt.close(figure)
