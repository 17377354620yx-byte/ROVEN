import json
from pathlib import Path

import numpy as np
import pytest

from tools.prepare_visibility_fps import (
    build_fps_selection,
    farthest_point_indices,
    materialize_sample,
    remap_edges,
    remap_faces,
)


def test_farthest_point_indices_is_deterministic_unique_and_centroid_seeded():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    )

    first = farthest_point_indices(points, 3)
    second = farthest_point_indices(points, 3)

    assert first.tolist() == [3, 0, 2]
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 3


def test_farthest_point_indices_keeps_small_cloud_without_upsampling():
    points = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    indices = farthest_point_indices(points, 1024)

    assert indices.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("points", "count", "message"),
    [
        (np.empty((0, 3)), 1, "at least one"),
        (np.zeros((2, 2)), 1, "shape"),
        (np.array([[np.nan, 0.0, 0.0]]), 1, "finite"),
        (np.zeros((2, 3)), 0, "positive"),
    ],
)
def test_farthest_point_indices_rejects_invalid_inputs(points, count, message):
    with pytest.raises(ValueError, match=message):
        farthest_point_indices(points, count)


def test_topology_remapping_drops_unselected_vertices_and_stays_in_bounds():
    selected = np.array([3, 0, 2])
    faces = np.array([[0, 2, 3], [0, 1, 2], [3, 2, 0]])
    edges = np.array([[0, 0, 3, 2], [2, 1, 2, 3]])

    mapped_faces = remap_faces(faces, selected)
    mapped_edges = remap_edges(edges, selected)

    assert mapped_faces.tolist() == [[1, 2, 0], [0, 2, 1]]
    assert mapped_edges.tolist() == [[1, 0, 2], [2, 2, 0]]
    assert mapped_faces.max() < len(selected)
    assert mapped_edges.max() < len(selected)


def test_materialize_in_silico_subsets_aligned_arrays_and_preserves_gt(tmp_path: Path):
    source = tmp_path / "input.npz"
    output = tmp_path / "nested" / "output.npz"
    src = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0]], dtype=float)
    tgt = src + np.array([0, 1, 0])
    flow = np.arange(12).reshape(4, 3)
    noise = np.arange(12, 24).reshape(4, 3)
    src_vol = np.array([[20.0, 0, 0], [21.0, 0, 0]])
    tgt_vol = src_vol + 1
    np.savez(
        source,
        src_pcd=src,
        tgt_pcd=tgt,
        flow=flow,
        src_f=np.array([[0, 2, 3], [0, 1, 2]]),
        src_edges=np.array([[0, 0, 3], [2, 1, 2]]),
        tgt_f=np.array([3, 0, 2, 3, 3, 0, 1, 2]),
        **{"1": noise},
        src_vol=src_vol,
        tgt_vol=tgt_vol,
        rot_tgt=np.eye(3),
        R_gt=np.eye(3),
        t_gt=np.zeros((3, 1)),
    )

    audit = materialize_sample("in_silico", source, output, 3)

    with np.load(output, allow_pickle=False) as result:
        assert result["src_pcd"].shape == (3, 3)
        assert result["tgt_pcd"].shape == (3, 3)
        np.testing.assert_array_equal(result["flow"], flow[[3, 0, 2]])
        np.testing.assert_array_equal(result["1"], noise[[3, 0, 2]])
        np.testing.assert_array_equal(result["src_vol"], src_vol)
        np.testing.assert_array_equal(result["tgt_vol"], tgt_vol)
        assert result["src_f"].max() < 3
        assert result["src_edges"].max() < 3
        assert result["tgt_f"].reshape(-1, 4)[:, 0].tolist() == [3]
        assert result["tgt_f"].reshape(-1, 4)[:, 1:].max() < 3
        assert len(np.unique(result["fps_source_indices"])) == 3
        assert len(np.unique(result["fps_target_indices"])) == 3
    assert audit["source_original_points"] == 4
    assert audit["source_fps_points"] == 3


def test_materialize_in_vitro_keeps_small_target_and_markers(tmp_path: Path):
    source = tmp_path / "input.npz"
    output = tmp_path / "output.npz"
    markers = np.array([[100.0, 0, 0], [101.0, 0, 0]])
    np.savez(
        source,
        src_vs=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0]]),
        tgt_vs=np.array([[0, 1, 0], [10, 1, 0]]),
        src_marker=markers,
        tgt_marker=markers + 1,
        vis=np.asarray(0.5),
        angles=np.zeros(3),
    )

    audit = materialize_sample("in_vitro", source, output, 3)

    with np.load(output, allow_pickle=False) as result:
        assert result["src_vs"].shape == (3, 3)
        assert result["tgt_vs"].shape == (2, 3)
        np.testing.assert_array_equal(result["src_marker"], markers)
        assert result["fps_target_indices"].tolist() == [0, 1]
    assert audit["target_original_points"] == 2
    assert audit["target_fps_points"] == 2


def test_build_fps_selection_materializes_layout_metadata_and_manifest(tmp_path: Path):
    silico = tmp_path / "silico"
    vitro = tmp_path / "vitro"
    (silico / "Deform_mesh_npz_test" / "Test" / "case").mkdir(parents=True)
    (vitro / "Rigid_test_data").mkdir(parents=True)
    np.savez(
        silico / "Deform_mesh_npz_test" / "Test" / "case" / "s.npz",
        src_pcd=np.arange(15).reshape(5, 3), tgt_pcd=np.arange(12).reshape(4, 3),
        flow=np.zeros((5, 3)), src_vol=np.zeros((2, 3)), tgt_vol=np.ones((2, 3)),
        rot_tgt=np.eye(3), R_gt=np.eye(3), t_gt=np.zeros((3, 1)),
    )
    np.savez(
        vitro / "Rigid_test_data" / "v.npz",
        src_vs=np.arange(15).reshape(5, 3), tgt_vs=np.arange(6).reshape(2, 3),
        src_marker=np.zeros((2, 3)), tgt_marker=np.ones((2, 3)), vis=0.4, angles=np.zeros(3),
    )
    np.savez(silico / "Deform_mesh_npz_test" / "list.npz", test=["case/s.npz"])
    np.savez(silico / "Deform_mesh_npz_test" / "stat_svd.npz", vis=[0.3], deform=[2.0])
    np.save(silico / "original_indices.npy", [7])
    np.save(vitro / "rigid_list.npy", ["v.npz"])
    np.savez(vitro / "stat.npz", vis=[0.4], deform=[3.0])
    np.save(vitro / "original_indices.npy", [9])
    selection = {
        "datasets": {
            "in_silico": {"cases": [{"sample": "case/s.npz"}], "subset": {"source_root": str(silico)}},
            "in_vitro": {"cases": [{"sample": "v.npz"}], "subset": {"source_root": str(vitro)}},
        }
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    output_root = tmp_path / "fps"

    result = build_fps_selection(selection_path, output_root, 3)

    assert (output_root / "subsets/in_silico/Deform_mesh_npz_test/Test/case/s.npz").is_file()
    assert (output_root / "subsets/in_vitro/Rigid_test_data/v.npz").is_file()
    assert np.load(output_root / "subsets/in_silico/original_indices.npy").tolist() == [7]
    assert np.load(output_root / "subsets/in_vitro/original_indices.npy").tolist() == [9]
    assert np.load(output_root / "subsets/in_silico/Deform_mesh_npz_test/stat_svd.npz")["vis"].tolist() == [0.3]
    assert result["sampling"]["method"] == "deterministic_farthest_point"
    assert result["sampling"]["requested_points_per_cloud"] == 3
    assert result["datasets"]["in_vitro"]["cases"][0]["fps"]["target_fps_points"] == 2
    assert json.loads((output_root / "selection.json").read_text()) == result
