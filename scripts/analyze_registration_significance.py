#!/usr/bin/env python3
"""Paired, cluster-aware significance analysis for registration benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


MODEL_SPECS = (
    ("ours", "Ours"),
    ("geotransformer", "GeoTransformer"),
    ("dfat", "DFAT"),
    ("parenet", "PARENet"),
)

CONDITION_SPECS = (
    ("in_silico_none", "In Silico — Clean"),
    ("in_silico_noise4", "In Silico — Noise 4"),
    ("in_vitro", "In Vitro"),
)

CONTINUOUS_METRICS = (
    ("rms_tre_mm", "RMS-TRE", "mm"),
    ("RRE", "RRE", "degrees"),
    ("RTE", "RTE", "mm"),
    ("RMSE", "RMSE", "mm"),
    ("chamfer_dist", "Chamfer", "mm"),
)

BINARY_METRICS = (
    ("SR_5mm", "SR@5 mm"),
    ("RR_5deg_5mm", "RR@5°/5 mm"),
)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm-adjusted p-values in their original order."""
    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("p-values must be a finite one-dimensional sequence")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(len(values), dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def rank_biserial_from_paired_differences(improvements: np.ndarray) -> float:
    """Paired rank-biserial effect; positive values favor the lower-error method."""
    values = np.asarray(improvements, dtype=np.float64)
    values = values[np.isfinite(values) & (values != 0.0)]
    if values.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0.0].sum())
    negative = float(ranks[values < 0.0].sum())
    return (positive - negative) / (positive + negative)


def paired_continuous_summary(ours: np.ndarray, baseline: np.ndarray) -> dict:
    """Summarize a lower-is-better paired comparison without dropping valid pairs."""
    ours = np.asarray(ours, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if ours.shape != baseline.shape or ours.ndim != 1:
        raise ValueError("ours and baseline must be same-length one-dimensional arrays")
    if ours.size == 0 or np.any(~np.isfinite(ours)) or np.any(~np.isfinite(baseline)):
        raise ValueError("paired values must be non-empty and finite")
    paired_difference = ours - baseline
    improvements = -paired_difference
    if np.all(paired_difference == 0.0):
        p_raw = 1.0
    else:
        p_raw = float(
            stats.wilcoxon(
                ours,
                baseline,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    return {
        "n_pairs": int(ours.size),
        "median_ours": float(np.median(ours)),
        "median_baseline": float(np.median(baseline)),
        "median_paired_difference": float(np.median(paired_difference)),
        "ours_win_rate": float(np.mean(ours < baseline)),
        "tie_rate": float(np.mean(ours == baseline)),
        "rank_biserial": rank_biserial_from_paired_differences(improvements),
        "p_raw": p_raw,
    }


def exact_mcnemar(ours: np.ndarray, baseline: np.ndarray) -> dict:
    """Exact two-sided McNemar comparison for paired binary endpoints."""
    ours = np.asarray(ours, dtype=bool)
    baseline = np.asarray(baseline, dtype=bool)
    if ours.shape != baseline.shape or ours.ndim != 1 or ours.size == 0:
        raise ValueError("ours and baseline must be non-empty paired vectors")
    ours_only = int(np.count_nonzero(ours & ~baseline))
    baseline_only = int(np.count_nonzero(~ours & baseline))
    discordant = ours_only + baseline_only
    p_raw = (
        1.0
        if discordant == 0
        else float(
            stats.binomtest(
                min(ours_only, baseline_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )
    return {
        "n_pairs": int(ours.size),
        "ours_success_rate": float(np.mean(ours)),
        "baseline_success_rate": float(np.mean(baseline)),
        "success_rate_difference": float(np.mean(ours) - np.mean(baseline)),
        "ours_only_success": ours_only,
        "baseline_only_success": baseline_only,
        "discordant_pairs": discordant,
        "p_raw": p_raw,
    }


def cluster_id(condition: str, sample_id: str) -> str:
    """Map a registration sample to its anatomical liver/case cluster."""
    if condition.startswith("in_silico"):
        return sample_id.split("/", 1)[0]
    if condition == "in_vitro":
        return sample_id.split("_", 1)[0]
    raise ValueError(f"Unsupported condition for clustering: {condition}")


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    statistic: str,
    repetitions: int = 5000,
    seed: int = 20260928,
) -> tuple[float, float]:
    """Percentile CI from resampling whole anatomical clusters with replacement."""
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters, dtype=str)
    if values.ndim != 1 or clusters.ndim != 1 or values.shape != clusters.shape:
        raise ValueError("values and clusters must be paired one-dimensional arrays")
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("bootstrap values must be non-empty and finite")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    reducer = {"median": np.median, "mean": np.mean}.get(statistic)
    if reducer is None:
        raise ValueError("statistic must be 'median' or 'mean'")
    unique_clusters = np.unique(clusters)
    members = [values[clusters == name] for name in unique_clusters]
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        selected = rng.integers(0, len(members), size=len(members))
        resample = np.concatenate([members[index] for index in selected])
        estimates[repetition] = float(reducer(resample))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _first_present(sample: dict, keys: tuple[str, ...]):
    for key in keys:
        if sample.get(key) is not None:
            return sample[key]
    raise KeyError(f"Missing required sample field; expected one of {keys}")


def _load_records(
    result_root: Path,
    model_specs,
    condition_specs,
    cluster_resolver,
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    provenance: list[dict] = []
    for condition_slug, condition_name in condition_specs:
        reference_ids = None
        reference_visibility = None
        for model_slug, model_name in model_specs:
            path = Path(result_root) / model_slug / f"{condition_slug}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            samples = payload.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"Missing non-empty samples list: {path}")
            sample_ids = [
                str(_first_present(sample, ("sample", "sample_name")))
                for sample in samples
            ]
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError(f"Duplicate sample IDs: {path}")
            visibilities = [float(sample["visibility"]) for sample in samples]
            if reference_ids is None:
                reference_ids = sample_ids
                reference_visibility = visibilities
            elif sample_ids != reference_ids:
                raise ValueError(
                    f"Sample IDs/order mismatch for {model_name} in {condition_name}"
                )
            for index, (sample, sample_id) in enumerate(zip(samples, sample_ids)):
                rms_tre = float(
                    _first_present(sample, ("rms_tre_mm", "RMS_TRE_mm"))
                )
                row = {
                    "model": model_name,
                    "model_slug": model_slug,
                    "condition": condition_name,
                    "condition_slug": condition_slug,
                    "sample_id": sample_id,
                    "cluster_id": str(cluster_resolver(condition_slug, sample_id)),
                    "visibility": float(reference_visibility[index]),
                    "reported_visibility": visibilities[index],
                    "RMS-TRE": rms_tre,
                    "RRE": float(sample["RRE"]),
                    "RTE": float(sample["RTE"]),
                    "RMSE": float(sample["RMSE"]),
                    "Chamfer": float(sample["chamfer_dist"]),
                    "SR@5 mm": float(sample.get("SR_5mm", rms_tre < 5.0)),
                    "RR@5°/5 mm": float(
                        sample.get(
                            "RR_5deg_5mm",
                            float(sample["RRE"]) < 5.0 and float(sample["RTE"]) < 5.0,
                        )
                    ),
                }
                numeric = [
                    row[name]
                    for name in (
                        "visibility",
                        "reported_visibility",
                        "RMS-TRE",
                        "RRE",
                        "RTE",
                        "RMSE",
                        "Chamfer",
                        "SR@5 mm",
                        "RR@5°/5 mm",
                    )
                ]
                if not all(math.isfinite(value) for value in numeric):
                    raise ValueError(f"Non-finite metric for {model_name}/{sample_id}")
                records.append(row)
            provenance.append(
                {
                    "model": model_name,
                    "condition": condition_name,
                    "sample_count": len(samples),
                    "cluster_count": len(
                        {cluster_resolver(condition_slug, item) for item in sample_ids}
                    ),
                    "checkpoint": payload.get("checkpoint"),
                    "source_json": str(path.resolve()),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "max_abs_visibility_delta": max(
                        abs(a - b)
                        for a, b in zip(visibilities, reference_visibility)
                    ),
                }
            )
    return records, provenance


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select(records, condition_slug: str, model_slug: str) -> list[dict]:
    return [
        row
        for row in records
        if row["condition_slug"] == condition_slug
        and row["model_slug"] == model_slug
    ]


def _apply_holm(rows: list[dict], family_key: str = "multiplicity_family") -> None:
    families: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        families.setdefault(str(row[family_key]), []).append(index)
    for indices in families.values():
        adjusted = holm_adjust([float(rows[index]["p_raw"]) for index in indices])
        for index, value in zip(indices, adjusted):
            rows[index]["p_holm"] = value


def _descriptive_rows(records, model_specs, condition_specs) -> list[dict]:
    rows = []
    metric_names = [name for _, name, _ in CONTINUOUS_METRICS]
    for condition_slug, condition_name in condition_specs:
        for model_slug, model_name in model_specs:
            selected = _select(records, condition_slug, model_slug)
            for metric in metric_names:
                values = np.asarray([row[metric] for row in selected], dtype=float)
                q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                rows.append(
                    {
                        "condition": condition_name,
                        "model": model_name,
                        "metric": metric,
                        "n": len(values),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)),
                        "median": float(median),
                        "q1": float(q1),
                        "q3": float(q3),
                        "iqr": float(q3 - q1),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                )
            for metric in (name for _, name in BINARY_METRICS):
                values = np.asarray([row[metric] for row in selected], dtype=float)
                rows.append(
                    {
                        "condition": condition_name,
                        "model": model_name,
                        "metric": metric,
                        "n": len(values),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)),
                        "median": float(np.median(values)),
                        "q1": float(np.quantile(values, 0.25)),
                        "q3": float(np.quantile(values, 0.75)),
                        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                )
    return rows


def _paired_test_rows(
    records,
    model_specs,
    condition_specs,
    *,
    bootstrap_repetitions: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    continuous_rows: list[dict] = []
    binary_rows: list[dict] = []
    cluster_rows: list[dict] = []
    seed = 20260928
    for condition_slug, condition_name in condition_specs:
        ours_rows = _select(records, condition_slug, "ours")
        if not ours_rows:
            raise ValueError("model_specs must contain the 'ours' reference method")
        clusters = np.asarray([row["cluster_id"] for row in ours_rows])
        for baseline_slug, baseline_name in model_specs:
            if baseline_slug == "ours":
                continue
            baseline_rows = _select(records, condition_slug, baseline_slug)
            for _, metric, unit in CONTINUOUS_METRICS:
                ours = np.asarray([row[metric] for row in ours_rows], dtype=float)
                baseline = np.asarray([row[metric] for row in baseline_rows], dtype=float)
                summary = paired_continuous_summary(ours, baseline)
                difference = ours - baseline
                ci_low, ci_high = cluster_bootstrap_ci(
                    difference,
                    clusters,
                    statistic="median",
                    repetitions=bootstrap_repetitions,
                    seed=seed,
                )
                seed += 1
                family = (
                    "primary_rms_tre" if metric == "RMS-TRE" else "secondary_continuous"
                )
                continuous_rows.append(
                    {
                        "condition": condition_name,
                        "metric": metric,
                        "unit": unit,
                        "comparison": f"Ours vs {baseline_name}",
                        "baseline": baseline_name,
                        **summary,
                        "paired_difference_ci95_low": ci_low,
                        "paired_difference_ci95_high": ci_high,
                        "multiplicity_family": family,
                    }
                )

                unique_clusters = np.unique(clusters)
                ours_cluster = np.asarray(
                    [np.median(ours[clusters == item]) for item in unique_clusters]
                )
                baseline_cluster = np.asarray(
                    [np.median(baseline[clusters == item]) for item in unique_clusters]
                )
                cluster_summary = paired_continuous_summary(
                    ours_cluster, baseline_cluster
                )
                cluster_rows.append(
                    {
                        "condition": condition_name,
                        "metric": metric,
                        "unit": unit,
                        "comparison": f"Ours vs {baseline_name}",
                        "baseline": baseline_name,
                        "n_clusters": len(unique_clusters),
                        **cluster_summary,
                        "multiplicity_family": (
                            "cluster_primary_rms_tre"
                            if metric == "RMS-TRE"
                            else "cluster_secondary_continuous"
                        ),
                    }
                )
            for _, metric in BINARY_METRICS:
                ours = np.asarray([row[metric] for row in ours_rows], dtype=bool)
                baseline = np.asarray(
                    [row[metric] for row in baseline_rows], dtype=bool
                )
                summary = exact_mcnemar(ours, baseline)
                ci_low, ci_high = cluster_bootstrap_ci(
                    ours.astype(float) - baseline.astype(float),
                    clusters,
                    statistic="mean",
                    repetitions=bootstrap_repetitions,
                    seed=seed,
                )
                seed += 1
                binary_rows.append(
                    {
                        "condition": condition_name,
                        "metric": metric,
                        "comparison": f"Ours vs {baseline_name}",
                        "baseline": baseline_name,
                        **summary,
                        "rate_difference_ci95_low": ci_low,
                        "rate_difference_ci95_high": ci_high,
                        "multiplicity_family": "binary_endpoints",
                    }
                )
    _apply_holm(continuous_rows)
    _apply_holm(binary_rows)
    _apply_holm(cluster_rows)
    return continuous_rows, binary_rows, cluster_rows


def _omnibus_rows(records, model_specs, condition_specs) -> list[dict]:
    if len(model_specs) < 3:
        return []
    rows = []
    for condition_slug, condition_name in condition_specs:
        for _, metric, unit in CONTINUOUS_METRICS:
            arrays = [
                np.asarray(
                    [row[metric] for row in _select(records, condition_slug, slug)],
                    dtype=float,
                )
                for slug, _ in model_specs
            ]
            result = stats.friedmanchisquare(*arrays)
            n = len(arrays[0])
            kendalls_w = float(result.statistic) / (n * (len(arrays) - 1))
            rows.append(
                {
                    "condition": condition_name,
                    "metric": metric,
                    "unit": unit,
                    "n_blocks": n,
                    "methods": len(arrays),
                    "friedman_chi2": float(result.statistic),
                    "degrees_of_freedom": len(arrays) - 1,
                    "p_raw": float(result.pvalue),
                    "kendalls_w": kendalls_w,
                    "analysis_level": "paired_sample",
                }
            )
    return rows


def _format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def format_p_for_sentence(value: float) -> str:
    """Format a p-value as a complete manuscript expression."""
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def _write_report(
    path: Path,
    records: list[dict],
    provenance: list[dict],
    continuous_rows: list[dict],
    binary_rows: list[dict],
    cluster_rows: list[dict],
    model_specs,
    condition_specs,
) -> None:
    primary = [row for row in continuous_rows if row["metric"] == "RMS-TRE"]
    cluster_primary = {
        (row["condition"], row["baseline"]): row
        for row in cluster_rows
        if row["metric"] == "RMS-TRE"
    }
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-09-28",
        "- Verification Status: ANALYZED",
        "- Version Label: registration_significance_v1",
        "",
        "# Registration Significance Analysis",
        "",
        "## Analysis population",
        "",
        f"The analysis contains {len(records):,} individual method–sample records from "
        f"{len(model_specs)} methods and {len(condition_specs)} conditions. All comparisons are paired by exact sample ID; no observations were reconstructed from aggregate means or standard deviations.",
        "",
        "## Statistical protocol",
        "",
        "- Primary endpoint: per-sample RMS-TRE (mm); lower values are better.",
        "- Main-text pairwise comparisons: Ours versus GeoTransformer, DFAT, and PARENet.",
        "- Continuous endpoints: two-sided paired Wilcoxon signed-rank tests; paired rank-biserial effects are positive when Ours has lower error.",
        "- Binary endpoints: exact two-sided McNemar tests.",
        "- Multiplicity: Holm correction across the nine primary RMS-TRE comparisons; secondary continuous and binary endpoints form separate prespecified families.",
        "- Confidence intervals: percentile bootstrap intervals resampling anatomical liver/case clusters.",
        "- Cluster sensitivity: each liver/case contributes its median error before paired Wilcoxon testing. In Vitro has only four clusters, so this sensitivity analysis has low power.",
        "",
        "## Primary RMS-TRE findings",
        "",
        "| Condition | Comparison | Median Ours | Median baseline | Median paired Δ (Ours−baseline) | 95% cluster-bootstrap CI | Win rate | Rank-biserial | Holm p | Cluster-level Holm p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        robust = cluster_primary[(row["condition"], row["baseline"])]
        lines.append(
            f"| {row['condition']} | {row['comparison']} | {row['median_ours']:.3f} | "
            f"{row['median_baseline']:.3f} | {row['median_paired_difference']:.3f} | "
            f"[{row['paired_difference_ci95_low']:.3f}, {row['paired_difference_ci95_high']:.3f}] | "
            f"{100 * row['ours_win_rate']:.1f}% | {row['rank_biserial']:.3f} | "
            f"{_format_p(row['p_holm'])} | {_format_p(robust['p_holm'])} |"
        )
    lines.extend(
        [
            "",
            "Interpret negative paired differences as lower RMS-TRE for Ours. Sample-level p-values quantify evidence over the benchmark registrations; cluster-level results are the robustness check for repeated registrations from the same anatomy.",
            "",
            "## Manuscript-ready Results text",
            "",
        ]
    )
    by_condition: dict[str, list[dict]] = {}
    for row in primary:
        by_condition.setdefault(row["condition"], []).append(row)
    for condition, condition_rows in by_condition.items():
        fragments = []
        for row in condition_rows:
            direction = "lower" if row["median_paired_difference"] < 0 else "higher"
            fragments.append(
                f"{direction} than {row['baseline']} (median paired difference "
                f"{row['median_paired_difference']:.2f} mm, 95% cluster-bootstrap CI "
                f"{row['paired_difference_ci95_low']:.2f} to {row['paired_difference_ci95_high']:.2f} mm, "
                f"Holm-adjusted {format_p_for_sentence(row['p_holm'])}, paired rank-biserial "
                f"r={row['rank_biserial']:.2f})"
            )
        lines.append(
            f"For {condition}, Ours had a median RMS-TRE of {condition_rows[0]['median_ours']:.2f} mm and was "
            + "; ".join(fragments)
            + "."
        )
        lines.append("")
    lines.extend(
        [
            "## Warnings",
            "",
            "- Statistical significance does not by itself establish clinical relevance; effect sizes and confidence intervals should accompany p-values.",
            "- In Vitro contains four anatomical clusters. Treat cross-case generalization as uncertain even when sample-level tests are small.",
            "- The available benchmark covers visibility 0.2–0.4 only. It cannot support claims about bins from 0.4–1.0.",
            "- The current result files are single checkpoint evaluations; they do not quantify training-seed variability.",
            "",
            "## Fallacy scan",
            "",
            "- Coverage: 11/11 statistical fallacy types checked.",
            "",
            "| Fallacy | Status | Assessment |",
            "|---|---|---|",
            "| Simpson's paradox | CAUTION | Aggregate and visibility-stratified results are both retained; direction changes must be inspected before making global claims. |",
            "| Ecological fallacy | CAUTION | Sample-level findings are separated from case-cluster sensitivity and are not presented as patient-population effects. |",
            "| Berkson's paradox | NOTE | The benchmark is restricted to visibility 0.2–0.4; conclusions apply to this selected low-visibility subset. |",
            "| Collider bias | NOTE | No covariate-adjusted causal model is fitted. |",
            "| Base-rate neglect | NOTE | SR and RR are reported with their benchmark denominators and paired discordances. |",
            "| Regression to the mean | NOTE | This is not a pre/post design selected on baseline error. |",
            "| Survivorship bias | NOTE | Failed cases remain in the benchmark denominator; no success-only filtering is performed. |",
            "| Look-elsewhere effect | CAUTION | Primary and secondary families are declared and Holm-corrected separately. |",
            "| Garden of forking paths | CAUTION | The analysis is post hoc unless the protocol was preregistered; retain this limitation in the manuscript. |",
            "| Correlation ≠ causation | NOTE | Method comparisons are benchmark associations, not causal clinical claims. |",
            "| Reverse causality | NOTE | Not applicable to paired algorithm performance endpoints. |",
            "",
            "## Reproducibility",
            "",
            "- Method: deterministic re-analysis of immutable JSON sources recorded by SHA-256.",
            "- Verdict: REPRODUCIBLE for the statistical re-analysis; model inference was not rerun.",
            "",
            f"Source files audited: {len(provenance)}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_analysis(
    result_root: Path,
    output_dir: Path,
    *,
    model_specs=MODEL_SPECS,
    condition_specs=CONDITION_SPECS,
    cluster_resolver=cluster_id,
    bootstrap_repetitions: int = 5000,
) -> None:
    """Validate paired records and write publication-oriented statistical outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, provenance = _load_records(
        result_root, model_specs, condition_specs, cluster_resolver
    )
    descriptive = _descriptive_rows(records, model_specs, condition_specs)
    continuous, binary, cluster = _paired_test_rows(
        records,
        model_specs,
        condition_specs,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    omnibus = _omnibus_rows(records, model_specs, condition_specs)

    record_fields = [
        "model",
        "model_slug",
        "condition",
        "condition_slug",
        "sample_id",
        "cluster_id",
        "visibility",
        "reported_visibility",
        "RMS-TRE",
        "RRE",
        "RTE",
        "RMSE",
        "Chamfer",
        "SR@5 mm",
        "RR@5°/5 mm",
    ]
    _write_csv(output_dir / "per_sample_metrics_core4.csv", records, record_fields)
    _write_csv(output_dir / "descriptive_statistics.csv", descriptive, list(descriptive[0]))
    continuous_fields = list(continuous[0])
    if "p_holm" in continuous_fields:
        continuous_fields.remove("p_holm")
    continuous_fields.append("p_holm")
    _write_csv(output_dir / "paired_continuous_tests.csv", continuous, continuous_fields)
    binary_fields = list(binary[0])
    if "p_holm" in binary_fields:
        binary_fields.remove("p_holm")
    binary_fields.append("p_holm")
    _write_csv(output_dir / "paired_binary_tests.csv", binary, binary_fields)
    cluster_fields = list(cluster[0])
    if "p_holm" in cluster_fields:
        cluster_fields.remove("p_holm")
    cluster_fields.append("p_holm")
    _write_csv(output_dir / "cluster_sensitivity_tests.csv", cluster, cluster_fields)
    omnibus_fields = [
        "condition",
        "metric",
        "unit",
        "n_blocks",
        "methods",
        "friedman_chi2",
        "degrees_of_freedom",
        "p_raw",
        "kendalls_w",
        "analysis_level",
    ]
    _write_csv(output_dir / "omnibus_friedman_tests.csv", omnibus, omnibus_fields)
    (output_dir / "analysis_provenance.json").write_text(
        json.dumps(
            {
                "model_specs": model_specs,
                "condition_specs": condition_specs,
                "bootstrap_repetitions": bootstrap_repetitions,
                "sources": provenance,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / "STATISTICAL_ANALYSIS.md",
        records,
        provenance,
        continuous,
        binary,
        cluster,
        model_specs,
        condition_specs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_analysis(args.result_root, args.output_dir)
    print(f"Wrote paired registration analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
