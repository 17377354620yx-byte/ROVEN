#!/usr/bin/env python3
"""Normalize model-native correspondence exports into auditable mm caches."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.correspondence_cache import (
    validate_raw_correspondence_export,
    write_correspondence_cache,
)


def _text(value: object) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("expected a scalar string")
    return str(array.item())


def network_transform_to_mm(
    transform: np.ndarray,
    *,
    source_center_mm: np.ndarray,
    target_center_mm: np.ndarray,
    scale_mm: float,
) -> np.ndarray:
    """Undo independent source/target centering for a source-to-target pose."""

    transform = np.asarray(transform, dtype=np.float64)
    source_center = np.asarray(source_center_mm, dtype=np.float64)
    target_center = np.asarray(target_center_mm, dtype=np.float64)
    scale = float(scale_mm)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be finite and have shape (4, 4)")
    if source_center.shape != (3,) or target_center.shape != (3,):
        raise ValueError("source and target centers must have shape (3,)")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale_mm must be positive and finite")
    result = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    result[:3, :3] = rotation
    result[:3, 3] = (
        scale * transform[:3, 3] + target_center - rotation @ source_center
    )
    return result


def _pose_metrics(
    estimated_transform_mm: np.ndarray,
    gt_transform_mm: np.ndarray,
    source_points_mm: np.ndarray,
) -> dict[str, float]:
    estimated = np.asarray(estimated_transform_mm, dtype=np.float64)
    gt = np.asarray(gt_transform_mm, dtype=np.float64)
    relative = estimated[:3, :3].T @ gt[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    rre = math.degrees(math.acos(cosine))
    rte = float(np.linalg.norm(estimated[:3, 3] - gt[:3, 3]))
    source = np.asarray(source_points_mm, dtype=np.float64)
    estimated_points = source @ estimated[:3, :3].T + estimated[:3, 3]
    gt_points = source @ gt[:3, :3].T + gt[:3, 3]
    rmse = float(np.sqrt(np.mean(np.sum((estimated_points - gt_points) ** 2, axis=1))))
    return {"RRE_deg": rre, "RTE_mm": rte, "RMSE_mm": rmse}


def normalize_raw_export(
    raw: Mapping[str, object],
    context: Mapping[str, object],
    *,
    method: str,
    checkpoint: str | None,
    raw_path: str | None = None,
) -> dict[str, object]:
    """Restore one real model export and bind it to the common dataset GT."""

    validate_raw_correspondence_export(raw)
    sample_id = _text(raw["sample_id"])
    if sample_id != str(context["sample_id"]):
        raise ValueError(
            f"sample mismatch: export={sample_id!r}, context={context['sample_id']!r}"
        )
    raw_scale = float(np.asarray(raw["physical_scale_mm"]).item())
    scale = float(context["scale_mm"])
    if not np.isclose(raw_scale, scale, rtol=1e-5, atol=1e-5):
        raise ValueError(f"scale mismatch: export={raw_scale}, context={scale}")

    frame = _text(raw["coordinate_frame"])
    source_center = np.asarray(context["source_center_mm"], dtype=np.float64)
    target_center = np.asarray(context["target_center_mm"], dtype=np.float64)
    if frame == "network_normalized":
        source_corr_mm = np.asarray(raw["source_corr_points"], dtype=np.float64) * scale + source_center
        target_corr_mm = np.asarray(raw["target_corr_points"], dtype=np.float64) * scale + target_center
        estimated_mm = network_transform_to_mm(
            np.asarray(raw["estimated_transform"], dtype=np.float64),
            source_center_mm=source_center,
            target_center_mm=target_center,
            scale_mm=scale,
        )
    elif frame == "original_mm":
        source_corr_mm = np.asarray(raw["source_corr_points"], dtype=np.float64)
        target_corr_mm = np.asarray(raw["target_corr_points"], dtype=np.float64)
        estimated_mm = np.asarray(raw["estimated_transform"], dtype=np.float64)
    else:
        raise ValueError(f"unsupported coordinate frame: {frame}")

    scores_value = raw.get("corr_scores")
    if "has_corr_scores" in raw and not bool(np.asarray(raw["has_corr_scores"]).item()):
        scores_value = None
    payload = {
        "source_points_mm": np.asarray(context["source_points_mm"], dtype=np.float64),
        "target_points_mm": np.asarray(context["target_points_mm"], dtype=np.float64),
        "source_corr_mm": source_corr_mm,
        "target_corr_mm": target_corr_mm,
        "corr_scores": None if scores_value is None else np.asarray(scores_value, dtype=np.float64),
        "gt_transform_mm": np.asarray(context["gt_transform_mm"], dtype=np.float64),
        "estimated_transform_mm": estimated_mm,
    }
    metrics = _pose_metrics(
        estimated_mm,
        payload["gt_transform_mm"],
        payload["source_points_mm"],
    )
    provenance = {
        "method": method,
        "sample_id": sample_id,
        "checkpoint": None if checkpoint is None else str(Path(checkpoint).resolve()),
        "raw_export_path": raw_path,
        "raw_coordinate_frame": frame,
        "restored_coordinate_frame": "original_mm",
        "normalization_scale_mm": scale,
        "source_variable": _text(raw["source_variable"]),
        "target_variable": _text(raw["target_variable"]),
        "gt_variable": "central P2P dataset transform restored to original mm",
        "correctness_transform": "central_dataset_gt",
        "visualization_offset_used_for_correctness": False,
        **metrics,
    }
    return {**payload, "provenance": provenance, "metrics": metrics}


def load_raw_export(path: str | Path) -> dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as archive:
        raw = {key: archive[key].copy() for key in archive.files}
    if "sample" in raw:
        raw["sample_id"] = raw.pop("sample")
    if "has_corr_scores" in raw and not bool(np.asarray(raw["has_corr_scores"]).item()):
        raw["corr_scores"] = None
    return raw


def _find_export(raw_dir: Path, case: Mapping[str, object]) -> Path:
    candidates = [
        raw_dir / f"{int(case['subset_index']):05d}.npz",
        raw_dir / f"{int(case['original_index']):05d}.npz",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            raw = load_raw_export(candidate)
            if _text(raw["sample_id"]) == str(case["sample"]):
                return candidate
        except (KeyError, ValueError):
            continue
    raise FileNotFoundError(
        f"No raw export for {case['sample']} in {raw_dir}; tried {candidates}"
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", choices=("in_silico", "in_vitro"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    dataset_block = selection["datasets"][args.dataset]
    subset_root = Path(dataset_block["subset"]["output_root"])

    project_root = Path(__file__).resolve().parents[1]
    experiment_dir = project_root / "experiments" / "geotransformer.p2p_liver"
    if str(experiment_dir) not in sys.path:
        sys.path.insert(0, str(experiment_dir))
    module_path = experiment_dir / "visualization" / "visualize_one_correspondence.py"
    spec = importlib.util.spec_from_file_location("rtor_one_correspondence", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    visualization_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visualization_module)
    from config import make_cfg

    load_p2p_sample = visualization_module.load_p2p_sample

    rows = []
    for case in dataset_block["cases"]:
        raw_path = _find_export(args.raw_dir, case)
        raw = load_raw_export(raw_path)
        context = load_p2p_sample(make_cfg(), args.dataset, subset_root, case["sample"])
        normalized = normalize_raw_export(
            raw,
            context,
            method=args.method,
            checkpoint=str(args.checkpoint),
            raw_path=str(raw_path.resolve()),
        )
        slug = str(case["sample"]).replace("/", "__").removesuffix(".npz")
        output_path = (
            args.output_root / "correspondences" / args.dataset / slug / f"{args.method}.npz"
        )
        write_correspondence_cache(
            output_path,
            normalized,
            normalized["provenance"],
        )
        with np.load(output_path, allow_pickle=False) as cache:
            stats = json.loads(str(cache["threshold_statistics_json"]))
            count = int(cache["source_corr_mm"].shape[0])
        rows.append(
            {
                "sample_id": case["sample"],
                "prediction_path": str(output_path.resolve()),
                "num_pred_correspondences": count,
                "threshold_statistics": stats,
                **normalized["metrics"],
            }
        )
        print(
            f"COLLECTED {args.dataset} {case['sample']} {args.method} "
            f"pred={count} correct5={stats['5']['num_correct']}"
        )
    summary_path = args.output_root / "correspondences" / args.dataset / f"{args.method}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "method": args.method,
                "checkpoint": str(args.checkpoint.resolve()),
                "samples": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
