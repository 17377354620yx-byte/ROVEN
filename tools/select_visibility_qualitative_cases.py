#!/usr/bin/env python3
"""Select and audit qualitative P2P correspondence advantage cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prepare_visibility_best3 import build_subset


BENCHMARK_METHODS = (
    "ours",
    "geotransformer",
    "castv2",
    "dfat",
    "lepard",
    "parenet",
    "livermatch",
    "livermatch_p2p",
)

CORRESPONDENCE_METHODS = BENCHMARK_METHODS + ("predator",)

SUMMARY_FILES = {
    "in_silico": "in_silico_none.json",
    "in_vitro": "in_vitro.json",
}


def _finite_float(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def normalize_summary(summary: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Normalize benchmark spelling variants and require one finite row per sample."""

    normalized: dict[str, dict[str, object]] = {}
    for raw_value in summary.get("samples", []):
        raw = dict(raw_value)
        sample_value = raw.get("sample", raw.get("sample_name"))
        if not sample_value:
            raise ValueError("benchmark row is missing sample/sample_name")
        sample = str(sample_value)
        if sample in normalized:
            raise ValueError(f"duplicate sample in benchmark summary: {sample}")
        rms_value = raw.get("rms_tre_mm", raw.get("RMS_TRE_mm"))
        if rms_value is None:
            raise ValueError(f"benchmark row {sample} is missing rms_tre_mm")
        if "IR" not in raw or raw["IR"] is None:
            raise ValueError(f"benchmark row {sample} is missing IR")
        count_value = raw.get("fine_corr_count", raw.get("correspondences"))
        row = {
            **raw,
            "sample": sample,
            "rms_tre_mm": _finite_float(rms_value, f"{sample} rms_tre_mm"),
            "IR": _finite_float(raw["IR"], f"{sample} IR"),
            "index": int(raw.get("index", 0)),
            "visibility": _finite_float(raw.get("visibility", 0.0), f"{sample} visibility"),
            "num_correspondences": None if count_value is None else int(count_value),
        }
        normalized[sample] = row
    return normalized


def _descending_ranks(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (-float(row[key]), str(row["sample"])))
    return {str(row["sample"]): rank for rank, row in enumerate(ordered)}


def select_benchmark_candidates(
    summaries: Mapping[str, Mapping[str, object]],
    count: int = 12,
    per_metric: int = 4,
) -> list[dict[str, object]]:
    """Build a deterministic diverse pool of strict Ours RMS-TRE winners."""

    if count <= 0 or per_metric <= 0:
        raise ValueError("count and per_metric must be positive")
    missing_methods = [method for method in BENCHMARK_METHODS if method not in summaries]
    if missing_methods:
        raise ValueError(f"Missing benchmark methods: {', '.join(missing_methods)}")
    rows_by_method = {
        method: normalize_summary(summaries[method]) for method in BENCHMARK_METHODS
    }
    eligible: list[dict[str, object]] = []
    for sample, ours in rows_by_method["ours"].items():
        if any(sample not in rows_by_method[method] for method in BENCHMARK_METHODS[1:]):
            continue
        baselines = [rows_by_method[method][sample] for method in BENCHMARK_METHODS[1:]]
        ours_rms = float(ours["rms_tre_mm"])
        best_baseline_rms = min(float(row["rms_tre_mm"]) for row in baselines)
        if not ours_rms < best_baseline_rms:
            continue
        proxy_methods = ("ours", "geotransformer", "parenet")
        proxy = {}
        proxy_complete = True
        for method in proxy_methods:
            row = rows_by_method[method][sample]
            if row["num_correspondences"] is None:
                proxy_complete = False
                break
            proxy[method] = float(row["IR"]) * int(row["num_correspondences"])
        if not proxy_complete:
            continue
        original_index = int(ours.get("original_index", ours["index"]))
        eligible.append(
            {
                "sample": sample,
                "source_index": int(ours["index"]),
                "original_index": original_index,
                "visibility": float(ours["visibility"]),
                "ours_rms_tre_mm": ours_rms,
                "rms_margin_mm": best_baseline_rms - ours_rms,
                "ours_IR": float(ours["IR"]),
                "ir_gap": float(ours["IR"])
                - max(float(row["IR"]) for row in baselines),
                "correct_proxy": proxy["ours"],
                "correct_proxy_margin": proxy["ours"]
                - max(proxy["geotransformer"], proxy["parenet"]),
            }
        )
    if len(eligible) < count:
        raise ValueError(f"Need {count} strict benchmark winners, found {len(eligible)}")

    metric_keys = ("rms_margin_mm", "ir_gap", "correct_proxy_margin")
    ranks = {key: _descending_ranks(eligible, key) for key in metric_keys}
    selected_names: set[str] = set()
    for key in metric_keys:
        ordered = sorted(eligible, key=lambda row: (-float(row[key]), str(row["sample"])))
        selected_names.update(str(row["sample"]) for row in ordered[:per_metric])
    ordered_all = sorted(
        eligible,
        key=lambda row: (
            sum(ranks[key][str(row["sample"])] for key in metric_keys),
            -float(row["rms_margin_mm"]),
            str(row["sample"]),
        ),
    )
    for row in ordered_all:
        if len(selected_names) >= count:
            break
        selected_names.add(str(row["sample"]))
    selected = [row for row in ordered_all if str(row["sample"]) in selected_names]
    selected = selected[:count]
    for candidate_rank, row in enumerate(selected):
        row["candidate_rank"] = candidate_rank
        row["subset_index"] = candidate_rank
        row["metric_ranks"] = {
            key: ranks[key][str(row["sample"])] for key in metric_keys
        }
    return selected


def _normalize_correspondence_summary(summary: Mapping[str, object]) -> dict[str, dict]:
    normalized = {}
    for raw_value in summary.get("samples", []):
        raw = dict(raw_value)
        sample = str(raw.get("sample_id", ""))
        if not sample:
            raise ValueError("correspondence summary row is missing sample_id")
        if sample in normalized:
            raise ValueError(f"duplicate sample in correspondence summary: {sample}")
        statistics = raw.get("threshold_statistics", {})
        threshold = statistics.get("5", {})
        try:
            rmse = _finite_float(raw["RMSE_mm"], f"{sample} RMSE_mm")
            num_correct = int(threshold["num_correct"])
            inlier_ratio = _finite_float(
                threshold["inlier_ratio"], f"{sample} inlier_ratio"
            )
            num_pred = int(raw["num_pred_correspondences"])
        except KeyError as error:
            raise ValueError(f"correspondence row {sample} is missing {error.args[0]}") from error
        if num_correct < 0 or num_pred < num_correct:
            raise ValueError(f"invalid correspondence counts for {sample}")
        normalized[sample] = {
            "RMSE_mm": rmse,
            "num_correct_correspondences": num_correct,
            "num_pred_correspondences": num_pred,
            "inlier_ratio": inlier_ratio,
        }
    return normalized


def select_verified_cases(
    candidate_cases: list[Mapping[str, object]],
    method_summaries: Mapping[str, Mapping[str, object]],
    top_k: int = 3,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Select correct-only visual winners and retain RMSE as an audit field."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    missing_methods = [
        method for method in CORRESPONDENCE_METHODS if method not in method_summaries
    ]
    if missing_methods:
        raise ValueError(f"Missing correspondence methods: {', '.join(missing_methods)}")
    rows_by_method = {
        method: _normalize_correspondence_summary(method_summaries[method])
        for method in CORRESPONDENCE_METHODS
    }
    evaluations = []
    eligible_pairs = []
    for candidate in candidate_cases:
        sample = str(candidate["sample"])
        absent = [method for method in CORRESPONDENCE_METHODS if sample not in rows_by_method[method]]
        if absent:
            evaluations.append(
                {
                    "sample": sample,
                    "eligible": False,
                    "reasons": [f"missing_results:{','.join(absent)}"],
                    "methods": {},
                }
            )
            continue
        methods = {method: rows_by_method[method][sample] for method in CORRESPONDENCE_METHODS}
        ours = methods["ours"]
        baselines = [methods[method] for method in CORRESPONDENCE_METHODS[1:]]
        rmse_margin = min(float(row["RMSE_mm"]) for row in baselines) - float(
            ours["RMSE_mm"]
        )
        correct_margin = int(ours["num_correct_correspondences"]) - max(
            int(row["num_correct_correspondences"]) for row in baselines
        )
        reasons = []
        if correct_margin <= 0:
            reasons.append("ours_correct_count_not_strictly_best")
        evaluation = {
            "sample": sample,
            "eligible": not reasons,
            "reasons": reasons,
            "ours_rmse_margin_mm": rmse_margin,
            "ours_rmse_strictly_best": rmse_margin > 0,
            "ours_correct_count_margin": correct_margin,
            "ours_inlier_ratio": float(ours["inlier_ratio"]),
            "methods": methods,
        }
        evaluations.append(evaluation)
        if not reasons:
            eligible_pairs.append((dict(candidate), evaluation))
    eligible_pairs.sort(
        key=lambda pair: (
            -int(pair[1]["ours_rmse_strictly_best"]),
            -int(pair[1]["ours_correct_count_margin"]),
            -float(pair[1]["ours_rmse_margin_mm"]),
            -float(pair[1]["ours_inlier_ratio"]),
            str(pair[0]["sample"]),
        )
    )
    if len(eligible_pairs) < top_k:
        raise ValueError(
            f"Need {top_k} strictly eligible qualitative samples, found {len(eligible_pairs)}"
        )
    selected = []
    for final_rank, (candidate, evaluation) in enumerate(eligible_pairs[:top_k]):
        candidate["final_rank"] = final_rank
        candidate["verified_ours_rmse_margin_mm"] = evaluation["ours_rmse_margin_mm"]
        candidate["verified_ours_correct_count_margin"] = evaluation[
            "ours_correct_count_margin"
        ]
        selected.append(candidate)
    return selected, evaluations


def _load_benchmark_summaries(result_root: Path, filename: str) -> dict[str, Mapping]:
    return {
        method: json.loads((result_root / method / filename).read_text(encoding="utf-8"))
        for method in BENCHMARK_METHODS
    }


def build_candidate_selection(args: argparse.Namespace) -> Path:
    roots = {"in_silico": args.in_silico_root, "in_vitro": args.in_vitro_root}
    selection: dict[str, object] = {
        "selection_type": "predeclared_qualitative_candidate_pool",
        "candidate_count_per_dataset": args.candidate_count,
        "per_metric": args.per_metric,
        "datasets": {},
    }
    for dataset, filename in SUMMARY_FILES.items():
        summaries = _load_benchmark_summaries(args.result_root, filename)
        original_indices = np.load(roots[dataset] / "original_indices.npy", allow_pickle=False)
        ours = summaries["ours"]
        for row in ours.get("samples", []):
            index = int(row["index"])
            if index < 0 or index >= len(original_indices):
                raise ValueError(f"Summary index {index} is outside original_indices.npy")
            row["original_index"] = int(original_indices[index])
        cases = select_benchmark_candidates(
            summaries, count=args.candidate_count, per_metric=args.per_metric
        )
        subset_root = args.output_root / "subsets" / dataset
        subset = build_subset(dataset, roots[dataset], cases, subset_root)
        selection["datasets"][dataset] = {
            "summary_sources": {
                method: str((args.result_root / method / filename).resolve())
                for method in BENCHMARK_METHODS
            },
            "cases": cases,
            "subset": subset,
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "selection.json"
    output_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path


def finalize_selection(root: Path, top_k: int) -> tuple[Path, Path]:
    source_path = root / "selection.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    final_selection = json.loads(json.dumps(source))
    final_selection["selection_type"] = "strict_verified_qualitative_advantage"
    report = {
        "selection_type": "selected_qualitative_examples_not_global_performance",
        "criteria": {
            "coordinate_frame": "original_mm",
            "correctness": "GT residual < 5 mm",
            "ours_num_correct_strictly_best": True,
            "fps1024_rmse_role": "audited_and_preferred_but_not_visual_eligibility",
            "goicp_correspondence_status": "not_applicable",
        },
        "datasets": {},
    }
    for dataset in SUMMARY_FILES:
        summaries = {
            method: json.loads(
                (root / "correspondences" / dataset / f"{method}_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            for method in CORRESPONDENCE_METHODS
        }
        cases = source["datasets"][dataset]["cases"]
        selected, evaluations = select_verified_cases(cases, summaries, top_k=top_k)
        final_selection["datasets"][dataset]["cases"] = selected
        report["datasets"][dataset] = {
            "candidate_count": len(cases),
            "selected": selected,
            "evaluations": evaluations,
        }
    final_path = root / "final_selection.json"
    report_path = root / "qualitative_selection_report.json"
    final_path.write_text(
        json.dumps(final_selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return final_path, report_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--in-silico-root", type=Path)
    parser.add_argument("--in-vitro-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--per-metric", type=int, default=4)
    parser.add_argument("--finalize-root", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.finalize_root is not None:
        final_path, report_path = finalize_selection(args.finalize_root, args.top_k)
        print(f"Wrote {final_path}")
        print(f"Wrote {report_path}")
        return
    required = ("result_root", "in_silico_root", "in_vitro_root", "output_root")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"candidate mode requires: {', '.join('--' + x.replace('_', '-') for x in missing)}")
    print(f"Wrote {build_candidate_selection(args)}")


if __name__ == "__main__":
    main()
