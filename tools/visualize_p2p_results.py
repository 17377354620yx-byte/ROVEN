#!/usr/bin/env python3
"""Generate single-case or top/bottom P2P registration visualizations."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.error_heatmap import apply_transform, generate_error_heatmap
from visualization.statistics import plot_grouped_error_distribution


DEFAULT_OUTPUT = ROOT / "output/visualization/error_heatmap"
EXPERIMENT_DIR = ROOT / "experiments/geotransformer.p2p_liver"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_array(value: Any, *, key: Optional[str] = None) -> np.ndarray:
    """Load an inline array or .npy/.npz/.txt point/transform file."""
    if not isinstance(value, (str, os.PathLike)):
        return np.asarray(value)
    path = Path(value).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            selected = key or (data.files[0] if len(data.files) == 1 else None)
            if selected is None or selected not in data:
                raise KeyError(f"Specify a valid npz key for {path}; available: {data.files}")
            return np.asarray(data[selected])
    if suffix in (".txt", ".xyz", ".csv"):
        return np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    if suffix in (".ply", ".pcd"):
        import open3d as o3d
        cloud = o3d.io.read_point_cloud(str(path))
        result = np.asarray(cloud.points)
        if not len(result):
            raise ValueError(f"No points read from {path}")
        return result
    if suffix == ".json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported array file: {path}")


def _case_id(case: Mapping[str, Any], fallback: int) -> str:
    raw = case.get("case_id", case.get("id", case.get("sample", f"case_{fallback:04d}")))
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(raw))
    return safe.strip("_") or f"case_{fallback:04d}"


def _generic_cases(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and "cases" in payload:
        return [dict(item) for item in payload["cases"]]
    if isinstance(payload, dict) and "src_path" in payload:
        return [dict(payload)]
    raise ValueError("JSON must be one case, a list, {'cases': [...]}, or a native test summary")


def _render_generic(case: Mapping[str, Any], output_dir: Path, index: int,
                    args: argparse.Namespace) -> Dict[str, Any]:
    source = _load_array(case.get("source", case.get("src_path")), key=case.get("src_key"))
    target = _load_array(case.get("target", case.get("tgt_path")), key=case.get("tgt_key"))
    pred = _load_array(case.get("T_pred"), key=case.get("T_pred_key"))
    ground_truth = _load_array(case.get("T_gt"), key=case.get("T_gt_key"))
    identifier = _case_id(case, index)
    result = generate_error_heatmap(
        source, target, pred, ground_truth,
        output_path=str(output_dir / f"{identifier}_heatmap.png"),
        scale_factor=float(case.get("scale_factor", args.scale_factor)),
        colormap=args.colormap,
        error_range=tuple(args.error_range) if args.error_range else None,
        point_size=args.point_size,
        dpi=args.dpi,
        render_backend=args.renderer,
    )
    result.update(case_id=identifier)
    return result


def _load_project_dataset(dataset_name: str, noise: Optional[int]):
    # Use the experiment's adapter itself so centering, rotations and oracle GT
    # are byte-for-byte consistent with test.py.
    sys.path.insert(0, str(EXPERIMENT_DIR))
    saved_selection = {
        key: os.environ.pop(key) for key in ("P2P_TEST_INDICES", "P2P_TEST_LIMIT")
        if key in os.environ
    }
    try:
        config = _load_module("p2p_vis_config", EXPERIMENT_DIR / "config.py")
        dataset_module = _load_module("p2p_vis_dataset", EXPERIMENT_DIR / "dataset.py")
        cfg = config.make_cfg()
        if dataset_name == "in_silico":
            return dataset_module.LiverTask3TestDataset(cfg, noise_mm=noise)
        if dataset_name == "in_vitro":
            if noise is not None:
                raise ValueError("in_vitro does not support synthetic noise")
            return dataset_module.LiverTask3InVitroTestDataset(cfg)
        raise ValueError(f"Unsupported native dataset: {dataset_name}")
    finally:
        os.environ.update(saved_selection)
        sys.path.pop(0)


def _infer_dataset(payload: Mapping[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    protocol = str(payload.get("protocol", "")).lower()
    if "in-vitro" in protocol or "in_vitro" in protocol:
        return "in_vitro"
    if "in-silico" in protocol or "in_silico" in protocol:
        return "in_silico"
    raise ValueError("Cannot infer dataset from summary; pass --dataset")


def _select_native(records: Sequence[Mapping[str, Any]], top_k: int):
    successes = sorted((r for r in records if float(r["rms_tre_mm"]) < 20.0),
                       key=lambda r: float(r["rms_tre_mm"]))[:top_k]
    failures = sorted((r for r in records if float(r["rms_tre_mm"]) >= 20.0),
                      key=lambda r: float(r["rms_tre_mm"]), reverse=True)[:top_k]
    return (
        [("success", rank, row) for rank, row in enumerate(successes, start=1)]
        + [("failure", rank, row) for rank, row in enumerate(failures, start=1)]
    )


def _render_native(payload: Mapping[str, Any], output_dir: Path,
                   args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset_name = _infer_dataset(payload, args.dataset)
    noise = payload.get("noise_mm")
    noise = None if noise is None else int(noise)
    dataset = _load_project_dataset(dataset_name, noise)
    prediction_dir = Path(args.predictions_dir) if args.predictions_dir else None
    if prediction_dir is None:
        checkpoint = Path(str(payload.get("checkpoint", "")))
        prediction_dir = checkpoint.parent.parent / "registration"
    selected = _select_native(payload.get("samples", []), args.top_k)
    if not selected:
        raise ValueError("No success/failure records available for selection")

    results, grouped = [], {"success": [], "failure": []}
    for group, rank, record in selected:
        index = int(record["index"])
        prediction_path = prediction_dir / f"{index:05d}.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Missing {prediction_path}. Re-run test.py with --save_predictions "
                "or pass --predictions-dir."
            )
        sample = dataset[index]
        with np.load(prediction_path, allow_pickle=False) as prediction:
            pred = np.asarray(prediction["estimated_transform"])
        identifier = f"{group}_{rank:02d}_case_{index:05d}"
        rendered = generate_error_heatmap(
            sample["src_points"], sample["ref_points"], pred, sample["transform"],
            output_path=str(output_dir / f"{identifier}_heatmap.png"),
            scale_factor=float(sample["physical_scale"]),
            colormap=args.colormap,
            error_range=tuple(args.error_range) if args.error_range else None,
            point_size=args.point_size,
            dpi=args.dpi,
            render_backend=args.renderer,
        )
        # Exact project metric: predicted source markers vs paired target markers.
        marker_distances = np.linalg.norm(
            apply_transform(sample["source_markers"], pred) - sample["target_markers"], axis=1
        ) * float(sample["physical_scale"])
        recomputed = float(np.sqrt(np.mean(marker_distances ** 2)))
        reported = float(record["rms_tre_mm"])
        delta = abs(recomputed - reported)
        if delta > args.metric_tolerance:
            raise AssertionError(
                f"case {index}: recomputed RMS-TRE {recomputed:.6f} mm differs "
                f"from summary {reported:.6f} mm by {delta:.6f} mm"
            )
        rendered.update(
            case_id=identifier, dataset_index=index, selection=group,
            reported_rms_tre_mm=reported, recomputed_rms_tre_mm=recomputed,
            rms_tre_abs_diff_mm=delta,
        )
        results.append(rendered)
        grouped[group].append(np.load(rendered["error_path"]))

    nonempty_groups, labels = [], []
    for name in ("success", "failure"):
        if grouped[name]:
            nonempty_groups.append(np.concatenate(grouped[name]))
            labels.append(f"Top-{len(grouped[name])} {name} cases")
    plot_grouped_error_distribution(nonempty_groups, labels,
                                    str(output_dir / "error_distribution.png"))
    return results


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", help="Generic case manifest or native test summary JSON")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--predictions-dir", help="Native registration/*.npz directory")
    parser.add_argument("--dataset", choices=("in_silico", "in_vitro"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--scale-factor", type=float, default=1.0,
                        help="Default normalized-to-mm factor for generic manifests")
    parser.add_argument("--colormap", default="jet")
    parser.add_argument("--error-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--renderer", choices=("auto", "open3d", "software"), default="auto",
                        help="auto avoids the known Open3D 0.16/NumPy 2 renderer crash")
    parser.add_argument("--metric-tolerance", type=float, default=1e-3)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "samples" in payload and "method" in payload:
        results = _render_native(payload, output_dir, args)
    else:
        results = [
            _render_generic(case, output_dir, index, args)
            for index, case in enumerate(_generic_cases(payload), start=1)
        ]
    manifest = {
        "input_json": str(Path(args.result_json).resolve()),
        "case_count": len(results),
        "cases": results,
    }
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(results)} case(s). Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
