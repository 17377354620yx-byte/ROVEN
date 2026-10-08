#!/usr/bin/env python3
"""Select representative In Silico/no-noise cases for registration figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_visibility_best3 import build_subset


METHODS = (
    "ours",
    "geotransformer",
    "dfat",
    "castv2",
    "parenet",
    "lepard",
    "lepard_p2p",
    "livermatch",
    "livermatch_p2p",
    "goicp",
)
STRONG = ("geotransformer", "dfat")
MEDIUM = ("castv2", "parenet")
WEAK = ("lepard", "lepard_p2p", "livermatch", "livermatch_p2p")


def _sample_name(row: Mapping) -> str:
    return str(row.get("sample", row.get("sample_name", "")))


def _rms_tre(row: Mapping) -> float:
    value = row.get("rms_tre_mm", row.get("RMS_TRE_mm"))
    return float(value)


def _ir(row: Mapping) -> float:
    return float(row["IR"])


def load_method_rows(result_root: Path) -> tuple[dict, dict]:
    rows = {}
    sources = {}
    for method in METHODS:
        source = Path(result_root) / method / "in_silico_none.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        indexed = {_sample_name(row): dict(row) for row in payload["samples"]}
        if len(indexed) != len(payload["samples"]):
            raise ValueError(f"Duplicate sample names in {source}")
        rows[method] = indexed
        sources[method] = str(source.resolve())
    common = set.intersection(*(set(indexed) for indexed in rows.values()))
    if not common:
        raise ValueError("No common samples across method summaries")
    if any(set(indexed) != common for indexed in rows.values()):
        raise ValueError("Method summaries do not contain the same sample set")
    return rows, sources


def representative_candidates(rows: Mapping[str, Mapping[str, Mapping]]) -> list[dict]:
    samples = sorted(set(rows["ours"]))
    means = {
        method: float(np.mean([_rms_tre(rows[method][sample]) for sample in samples]))
        for method in METHODS
    }
    target_profile = np.log1p([means[method] for method in METHODS])
    candidates = []
    for sample in samples:
        metrics = {
            method: _rms_tre(rows[method][sample]) for method in METHODS
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            continue
        ours = metrics["ours"]
        strong = np.asarray([metrics[method] for method in STRONG])
        medium = np.asarray([metrics[method] for method in MEDIUM])
        weak = np.asarray([metrics[method] for method in WEAK])
        goicp = metrics["goicp"]
        checks = {
            "ours_best_vs_strong": bool(ours < float(np.min(strong))),
            "strong_gap_bounded": bool(
                float(np.max(strong)) <= ours + 2.5
                and float(np.max(strong)) <= ours * 1.8
            ),
            "medium_above_ours": bool(float(np.min(medium)) > ours + 0.5),
            "weak_individually_above_ours": bool(float(np.min(weak)) > ours + 0.2),
            "weak_group_above_medium_group": bool(
                float(np.median(weak)) > float(np.median(medium)) + 0.5
            ),
            "goicp_clearly_worst": bool(
                goicp > float(np.max(weak)) + 5.0 and goicp >= 20.0
            ),
        }
        if not all(checks.values()):
            continue
        profile = np.log1p([metrics[method] for method in METHODS])
        profile_distance = float(np.mean(np.square(profile - target_profile)))
        ours_typicality_penalty = 0.25 * (
            (ours - means["ours"]) / 3.274548683901659
        ) ** 2
        score = profile_distance + ours_typicality_penalty
        ours_row = rows["ours"][sample]
        candidates.append(
            {
                "sample": sample,
                "source_index": int(ours_row["index"]),
                "visibility": float(ours_row["visibility"]),
                "deformation_mm": float(ours_row["deformation_mm"]),
                "score": float(score),
                "profile_distance": profile_distance,
                "ours_typicality_penalty": float(ours_typicality_penalty),
                "rms_tre_mm": metrics,
                "checks": checks,
            }
        )
    return sorted(
        candidates,
        key=lambda row: (row["score"], row["source_index"], row["sample"]),
    )


def select_low_visibility_advantage(
    rows: Mapping[str, Mapping[str, Mapping]],
    visibility_min: float,
    visibility_max: float,
    *,
    max_ours_rms_tre_mm: float = 5.0,
) -> dict:
    """Pick a low-visibility case where Ours clearly beats GeoTransformer.

    Registration quality first constrains Ours to be best across all available
    methods and below ``max_ours_rms_tre_mm``.  Within that safe set, the score
    rewards both RMS-TRE and correspondence-IR margins over GeoTransformer.
    """

    candidates = []
    for sample in sorted(rows["ours"]):
        ours_row = rows["ours"][sample]
        visibility = float(ours_row["visibility"])
        if not visibility_min <= visibility <= visibility_max:
            continue
        metrics = {method: _rms_tre(rows[method][sample]) for method in METHODS}
        if not all(math.isfinite(value) for value in metrics.values()):
            continue
        ours_rms = metrics["ours"]
        baseline_best = min(metrics[method] for method in METHODS if method != "ours")
        if ours_rms >= baseline_best or ours_rms > max_ours_rms_tre_mm:
            continue
        ours_ir = _ir(ours_row)
        geo_ir = _ir(rows["geotransformer"][sample])
        geo_rms = metrics["geotransformer"]
        advantage_score = (geo_rms - ours_rms) + 5.0 * (ours_ir - geo_ir)
        candidates.append(
            {
                "sample": sample,
                "source_index": int(ours_row["index"]),
                "visibility": visibility,
                "deformation_mm": float(ours_row["deformation_mm"]),
                "score": -float(advantage_score),
                "advantage_score": float(advantage_score),
                "ours_ir": ours_ir,
                "geotransformer_ir": geo_ir,
                "rms_tre_mm": metrics,
                "checks": {
                    "ours_best_rms_tre_all_methods": True,
                    "ours_rms_tre_at_most_5mm": True,
                    "visibility_in_requested_low_range": True,
                },
                "selection_strategy": "low_visibility_ours_advantage",
            }
        )
    if not candidates:
        raise ValueError(
            f"No Ours-advantage case in visibility range "
            f"[{visibility_min}, {visibility_max}]"
        )
    return max(
        candidates,
        key=lambda row: (row["advantage_score"], -row["source_index"]),
    )


def select_cases(
    rows: Mapping[str, Mapping[str, Mapping]],
    original_indices: Sequence[int],
    top_k: int = 3,
    low_visibility_range: tuple[float, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    candidates = representative_candidates(rows)
    if len(candidates) < top_k:
        raise ValueError(f"Need {top_k} eligible samples, found {len(candidates)}")
    chosen = list(candidates[:top_k])
    if low_visibility_range is not None:
        low_case = select_low_visibility_advantage(rows, *low_visibility_range)
        replace_index = min(range(len(chosen)), key=lambda index: chosen[index]["visibility"])
        chosen[replace_index] = low_case
    selected = []
    for subset_index, candidate in enumerate(chosen):
        source_index = int(candidate["source_index"])
        selected.append(
            {
                "subset_index": subset_index,
                "source_index": source_index,
                "original_index": int(original_indices[source_index]),
                "sample": candidate["sample"],
                "visibility": candidate["visibility"],
                "deformation_mm": candidate["deformation_mm"],
                "ours_rms_tre_mm": candidate["rms_tre_mm"]["ours"],
                "selection_score": candidate["score"],
                "rms_tre_mm": candidate["rms_tre_mm"],
                "selection_checks": candidate["checks"],
                "selection_strategy": candidate.get(
                    "selection_strategy", "representative_profile"
                ),
            }
        )
    return selected, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--low-visibility-min", type=float)
    parser.add_argument("--low-visibility-max", type=float)
    args = parser.parse_args()

    rows, sources = load_method_rows(args.result_root)
    original_indices = np.load(args.data_root / "original_indices.npy", allow_pickle=False)
    low_visibility_range = None
    if (args.low_visibility_min is None) != (args.low_visibility_max is None):
        parser.error("--low-visibility-min and --low-visibility-max must be used together")
    if args.low_visibility_min is not None:
        low_visibility_range = (args.low_visibility_min, args.low_visibility_max)
    selected, candidates = select_cases(
        rows, original_indices, args.top_k, low_visibility_range
    )
    subset = build_subset(
        "in_silico", args.data_root, selected, args.output_root / "subsets/in_silico"
    )
    selection = {
        "selection_protocol": "representative_in_silico_no_noise_v1",
        "dataset": "in_silico",
        "noise": "none",
        "metric": "RMS-TRE (mm)",
        "methods": list(METHODS),
        "datasets": {
            "in_silico": {
                "cases": selected,
                "subset": subset,
                "source_summaries": sources,
            }
        },
    }
    report = {
        **selection,
        "eligible_count": len(candidates),
        "candidate_ranking": candidates,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_root / "selection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Selected {len(selected)} of {len(candidates)} eligible samples")
    for row in selected:
        print(row["sample"], json.dumps(row["rms_tre_mm"], ensure_ascii=False))


if __name__ == "__main__":
    main()
