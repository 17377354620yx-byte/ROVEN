#!/usr/bin/env python3
"""Materialize selected visibility samples after deterministic FPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Return deterministic FPS indices, without duplicating small clouds."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(points) == 0:
        raise ValueError("points must contain at least one point")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if int(count) != count or count <= 0:
        raise ValueError("count must be a positive integer")
    count = min(int(count), len(points))
    if count == len(points):
        return np.arange(len(points), dtype=np.int64)

    selected = np.empty(count, dtype=np.int64)
    centroid = points.mean(axis=0)
    squared_to_centroid = np.sum((points - centroid) ** 2, axis=1)
    selected[0] = int(np.argmax(squared_to_centroid))
    minimum_squared_distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
    minimum_squared_distance[selected[0]] = -1.0
    for position in range(1, count):
        selected[position] = int(np.argmax(minimum_squared_distance))
        candidate_distance = np.sum((points - points[selected[position]]) ** 2, axis=1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, candidate_distance)
        minimum_squared_distance[selected[: position + 1]] = -1.0
    return selected


def _index_map(selected: np.ndarray) -> np.ndarray:
    selected = np.asarray(selected, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("selected must be a non-empty one-dimensional array")
    if np.any(selected < 0) or len(np.unique(selected)) != len(selected):
        raise ValueError("selected indices must be unique and non-negative")
    mapping = np.full(int(selected.max()) + 1, -1, dtype=np.int64)
    mapping[selected] = np.arange(len(selected), dtype=np.int64)
    return mapping


def remap_faces(faces: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Keep faces fully contained in selected vertices and remap indices."""

    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    mapping = _index_map(selected)
    valid_range = np.all((faces >= 0) & (faces < len(mapping)), axis=1)
    kept = faces[valid_range]
    if len(kept):
        kept = kept[np.all(mapping[kept] >= 0, axis=1)]
    return mapping[kept].astype(np.int64, copy=False)


def remap_edges(edges: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Keep edges fully contained in selected vertices and remap indices."""

    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edges must have shape (2, E)")
    mapping = _index_map(selected)
    valid_range = np.all((edges >= 0) & (edges < len(mapping)), axis=0)
    kept = edges[:, valid_range]
    if kept.shape[1]:
        kept = kept[:, np.all(mapping[kept] >= 0, axis=0)]
    return mapping[kept].astype(np.int64, copy=False)


def materialize_sample(
    dataset: str,
    source_path: str | Path,
    output_path: str | Path,
    count: int,
) -> dict[str, int | str]:
    """Write one FPS sample while keeping every point-aligned field consistent."""

    source_path = Path(source_path)
    output_path = Path(output_path)
    with np.load(source_path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}

    if dataset == "in_silico":
        source_key, target_key = "src_pcd", "tgt_pcd"
    elif dataset == "in_vitro":
        source_key, target_key = "src_vs", "tgt_vs"
    else:
        raise ValueError("dataset must be in_silico or in_vitro")
    if source_key not in payload or target_key not in payload:
        raise KeyError(f"sample is missing {source_key} or {target_key}")

    source_points = np.asarray(payload[source_key])
    target_points = np.asarray(payload[target_key])
    source_indices = farthest_point_indices(source_points, count)
    target_indices = farthest_point_indices(target_points, count)
    payload[source_key] = source_points[source_indices]
    payload[target_key] = target_points[target_indices]

    if dataset == "in_silico":
        if "flow" in payload:
            if len(payload["flow"]) != len(source_points):
                raise ValueError("flow length must match src_pcd")
            payload["flow"] = payload["flow"][source_indices]
        for key in tuple(payload):
            if key.isdigit() and np.asarray(payload[key]).shape[:1] == (len(target_points),):
                payload[key] = payload[key][target_indices]
        if "src_f" in payload:
            payload["src_f"] = remap_faces(payload["src_f"], source_indices)
        if "src_edges" in payload:
            payload["src_edges"] = remap_edges(payload["src_edges"], source_indices)
        if "tgt_f" in payload:
            encoded_faces = np.asarray(payload["tgt_f"])
            if (
                encoded_faces.ndim == 1
                and encoded_faces.size % 4 == 0
                and np.all(encoded_faces.reshape(-1, 4)[:, 0] == 3)
            ):
                faces = remap_faces(encoded_faces.reshape(-1, 4)[:, 1:], target_indices)
                payload["tgt_f"] = np.column_stack(
                    [np.full(len(faces), 3, dtype=encoded_faces.dtype), faces]
                ).reshape(-1)
            else:
                original_shape = encoded_faces.shape
                faces = remap_faces(encoded_faces.reshape(-1, 3), target_indices)
                payload["tgt_f"] = faces.reshape(-1) if len(original_shape) == 1 else faces

    payload["fps_source_indices"] = source_indices
    payload["fps_target_indices"] = target_indices
    payload["fps_requested_points"] = np.asarray(int(count), dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {
        "source_path": str(source_path.resolve()),
        "output_path": str(output_path.resolve()),
        "source_original_points": int(len(source_points)),
        "target_original_points": int(len(target_points)),
        "source_fps_points": int(len(source_indices)),
        "target_fps_points": int(len(target_indices)),
    }


def _source_layout(dataset: str, root: Path) -> tuple[np.ndarray, Path, Path, Path]:
    if dataset == "in_silico":
        metadata = root / "Deform_mesh_npz_test"
        with np.load(metadata / "list.npz", allow_pickle=False) as archive:
            names = np.asarray(archive["test"]).astype(str)
        return names, metadata / "stat_svd.npz", metadata / "Test", root / "original_indices.npy"
    names = np.load(root / "rigid_list.npy", allow_pickle=False).astype(str)
    return names, root / "stat.npz", root / "Rigid_test_data", root / "original_indices.npy"


def _write_subset_metadata(
    dataset: str,
    output_root: Path,
    names: np.ndarray,
    statistics: Mapping[str, np.ndarray],
    original_indices: np.ndarray,
) -> Path:
    if dataset == "in_silico":
        metadata = output_root / "Deform_mesh_npz_test"
        data_dir = metadata / "Test"
        metadata.mkdir(parents=True, exist_ok=True)
        np.savez(metadata / "list.npz", test=names)
        np.savez(metadata / "stat_svd.npz", **statistics)
    else:
        data_dir = output_root / "Rigid_test_data"
        output_root.mkdir(parents=True, exist_ok=True)
        np.save(output_root / "rigid_list.npy", names)
        np.savez(output_root / "stat.npz", **statistics)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_root / "original_indices.npy", original_indices)
    return data_dir


def build_fps_selection(
    selection_path: str | Path,
    output_root: str | Path,
    count: int = 1024,
) -> dict[str, object]:
    """Materialize both selected datasets and write a runner-compatible selection."""

    selection_path = Path(selection_path)
    output_root = Path(output_root)
    source_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(source_selection))
    result["sampling"] = {
        "method": "deterministic_farthest_point",
        "requested_points_per_cloud": int(count),
        "coordinate_frame": "original_mm_before_normalization",
        "small_cloud_policy": "keep_all_without_duplication",
        "source_selection": str(selection_path.resolve()),
    }

    for dataset in ("in_silico", "in_vitro"):
        block = result["datasets"][dataset]
        source_root = Path(block["subset"]["source_root"])
        names, statistics_path, source_data, original_path = _source_layout(dataset, source_root)
        positions_by_name = {name: index for index, name in enumerate(names.tolist())}
        requested = [str(case["sample"]) for case in block["cases"]]
        missing = [name for name in requested if name not in positions_by_name]
        if missing:
            raise ValueError(f"Samples not found in {source_root}: {', '.join(missing)}")
        positions = np.asarray([positions_by_name[name] for name in requested], dtype=np.int64)
        with np.load(statistics_path, allow_pickle=False) as archive:
            statistics = {key: archive[key][positions] for key in archive.files}
        original = np.load(original_path, allow_pickle=False)[positions]
        subset_root = output_root / "subsets" / dataset
        output_data = _write_subset_metadata(
            dataset, subset_root, names[positions], statistics, original
        )
        for case in block["cases"]:
            sample_name = str(case["sample"])
            case["fps"] = materialize_sample(
                dataset,
                source_data / sample_name,
                output_data / sample_name,
                count,
            )
        block["subset"] = {
            "dataset": dataset,
            "source_root": str(source_root.resolve()),
            "output_root": str(subset_root.resolve()),
            "samples": requested,
            "source_positions": positions.tolist(),
            "original_indices": original.astype(int).tolist(),
            "sampling": result["sampling"],
        }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    build_fps_selection(args.selection, args.output_root, args.count)
    print(f"Wrote {args.output_root / 'selection.json'}")


if __name__ == "__main__":
    main()
