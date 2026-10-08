#!/usr/bin/env python3
"""Select three representative In Vitro cases for correspondence figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_visibility_best3 import build_subset


METHODS = (
    "ours",
    "geotransformer",
    "castv2",
    "dfat",
    "lepard",
    "parenet",
    "livermatch",
    "livermatch_p2p",
)
VISIBILITY_BINS = ((0.20, 0.267), (0.267, 0.334), (0.334, 0.401))


def _sample(row: Mapping[str, object]) -> str:
    return str(row.get("sample", row.get("sample_name", "")))


def _rms_tre(row: Mapping[str, object]) -> float:
    return float(row.get("rms_tre_mm", row.get("RMS_TRE_mm")))


def load_rows(result_root: Path) -> tuple[dict, dict]:
    rows = {}
    sources = {}
    for method in METHODS:
        source = result_root / method / "in_vitro.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        indexed = {_sample(row): dict(row) for row in payload["samples"]}
        rows[method] = indexed
        sources[method] = str(source.resolve())
    common = set.intersection(*(set(indexed) for indexed in rows.values()))
    if not common or any(set(indexed) != common for indexed in rows.values()):
        raise ValueError("In Vitro summaries do not contain the same samples")
    return rows, sources


def select_cases(rows: Mapping[str, Mapping[str, Mapping]]) -> list[dict]:
    """Select one moderate, Ours-best case from each visibility band."""

    selected = []
    for visibility_min, visibility_max in VISIBILITY_BINS:
        candidates = []
        for sample in sorted(rows["ours"]):
            ours_row = rows["ours"][sample]
            visibility = float(ours_row["visibility"])
            if not visibility_min <= visibility < visibility_max:
                continue
            metrics = {method: _rms_tre(rows[method][sample]) for method in METHODS}
            if not all(math.isfinite(value) for value in metrics.values()):
                continue
            ours = metrics["ours"]
            baseline_best = min(metrics[m] for m in METHODS if m != "ours")
            geo = metrics["geotransformer"]
            if ours >= baseline_best or ours > 5.0 or geo > ours + 3.0:
                continue
            ours_ir = float(ours_row["IR"])
            geo_row = rows["geotransformer"][sample]
            geo_ir = float(geo_row["IR"])
            ours_expected_inliers = ours_ir * int(ours_row["fine_corr_count"])
            geo_expected_inliers = geo_ir * int(geo_row["fine_corr_count"])
            if ours_expected_inliers <= geo_expected_inliers:
                continue
            advantage = (
                (baseline_best - ours)
                + 0.25 * (ours_ir - geo_ir)
                + (ours_expected_inliers - geo_expected_inliers) / 1000.0
            )
            candidates.append(
                {
                    "sample": sample,
                    "source_index": int(ours_row["index"]),
                    "visibility": visibility,
                    "deformation_mm": float(ours_row["deformation_mm"]),
                    "ours_rms_tre_mm": ours,
                    "selection_score": float(advantage),
                    "expected_inlier_margin_vs_geotransformer": float(
                        ours_expected_inliers - geo_expected_inliers
                    ),
                    "rms_tre_mm": metrics,
                    "selection_strategy": "visibility_stratified_ours_best_moderate_gap",
                }
            )
        if not candidates:
            raise ValueError(
                f"No eligible In Vitro case in [{visibility_min}, {visibility_max})"
            )
        selected.append(
            max(candidates, key=lambda row: (row["selection_score"], -row["source_index"]))
        )
    return selected


def select_contrast_cases(rows: Mapping[str, Mapping[str, Mapping]]) -> list[dict]:
    """Select Ours-best cases with the largest expected inlier-count margin vs GeoTransformer."""

    selected = []
    for visibility_min, visibility_max in VISIBILITY_BINS:
        candidates = []
        for sample in sorted(rows["ours"]):
            ours_row = rows["ours"][sample]
            visibility = float(ours_row["visibility"])
            if not visibility_min <= visibility < visibility_max:
                continue
            metrics = {method: _rms_tre(rows[method][sample]) for method in METHODS}
            if not all(math.isfinite(value) for value in metrics.values()):
                continue
            ours = metrics["ours"]
            baseline_best = min(metrics[m] for m in METHODS if m != "ours")
            if ours >= baseline_best or ours > 5.0:
                continue
            geo_row = rows["geotransformer"][sample]
            ours_expected_inliers = float(ours_row["IR"]) * int(
                ours_row["fine_corr_count"]
            )
            geo_expected_inliers = float(geo_row["IR"]) * int(
                geo_row["fine_corr_count"]
            )
            margin = ours_expected_inliers - geo_expected_inliers
            if margin <= 0:
                continue
            candidates.append(
                {
                    "sample": sample,
                    "source_index": int(ours_row["index"]),
                    "visibility": visibility,
                    "deformation_mm": float(ours_row["deformation_mm"]),
                    "ours_rms_tre_mm": ours,
                    "selection_score": float(margin),
                    "expected_inlier_margin_vs_geotransformer": float(margin),
                    "rms_tre_mm": metrics,
                    "selection_strategy": "visibility_stratified_max_inlier_margin",
                }
            )
        if not candidates:
            raise ValueError(
                f"No contrast case available in [{visibility_min}, {visibility_max})"
            )
        selected.append(
            max(candidates, key=lambda row: (row["selection_score"], -row["source_index"]))
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--selection-strategy",
        choices=("representative", "max_visual_margin"),
        default="representative",
    )
    args = parser.parse_args()

    rows, sources = load_rows(args.result_root)
    if args.selection_strategy == "max_visual_margin":
        selected = select_contrast_cases(rows)
        protocol = "contrast_in_vitro_correspondence_v1"
    else:
        selected = select_cases(rows)
        protocol = "representative_in_vitro_correspondence_v1"
    original_indices = np.load(args.data_root / "original_indices.npy", allow_pickle=False)
    for subset_index, case in enumerate(selected):
        case["subset_index"] = subset_index
        case["original_index"] = int(original_indices[case["source_index"]])
    subset = build_subset(
        "in_vitro", args.data_root, selected, args.output_root / "subsets/in_vitro"
    )
    selection = {
        "selection_protocol": protocol,
        "dataset": "in_vitro",
        "metric": "RMS-TRE (mm)",
        "methods": list(METHODS),
        "datasets": {
            "in_vitro": {
                "cases": selected,
                "subset": subset,
                "source_summaries": sources,
            }
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "selection.json"
    output_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_path}")
    for case in selected:
        print(
            case["sample"],
            f"visibility={case['visibility']:.6f}",
            f"ours_rms_tre_mm={case['ours_rms_tre_mm']:.4f}",
        )


if __name__ == "__main__":
    main()
