#!/usr/bin/env python3
"""Aggregate the unified visibility-[0.2, 0.4] benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.p2p_benchmark_metrics import METRIC_NAMES


MODELS = (
    "ours",
    "geotransformer",
    "castv2",
    "dfat",
    "lepard",
    "lepard_p2p",
    "parenet",
    "livermatch",
    "livermatch_p2p",
    "rtorv8_no_proposal",
    "goicp",
)
CONDITIONS = ("in_silico_none", "in_silico_noise4", "in_vitro")
EXPECTED_COUNTS = {"in_silico_none": 1538, "in_silico_noise4": 1538, "in_vitro": 204}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metric_value(sample: dict) -> float:
    for key in ("rms_tre_mm", "RMS_TRE_mm"):
        if key in sample:
            return float(sample[key])
    raise KeyError("Sample has neither rms_tre_mm nor RMS_TRE_mm")


def strict_rates(samples: list[dict]) -> dict[str, float]:
    """Recompute primary binary endpoints from continuous per-sample errors."""
    sr = [metric_value(sample) < 5.0 for sample in samples]
    rr = [
        float(sample["RRE"]) < 5.0 and float(sample["RTE"]) < 5.0
        for sample in samples
    ]
    return {
        "success_rate_5mm": statistics.fmean(sr),
        "rr_5deg_5mm": statistics.fmean(rr),
    }


def read_result(path: Path, model: str, condition: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Missing non-empty samples list: {path}")
    values = [metric_value(sample) for sample in samples]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite RMS-TRE value: {path}")
    expected = EXPECTED_COUNTS[condition]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} samples but found {len(values)}: {path}")
    strict = strict_rates(samples)
    row = {
        "model": model,
        "condition": condition,
        "count": len(values),
        "mean_rms_tre_mm": statistics.fmean(values),
        "std_rms_tre_mm": statistics.pstdev(values),
        "success_rate_5mm": strict["success_rate_5mm"],
        "success_rate_20mm": statistics.fmean(value < 20.0 for value in values),
        "result_json": str(path.resolve()),
    }
    for name in METRIC_NAMES:
        metric_values = []
        for sample in samples:
            value = sample.get(name)
            if value is not None and math.isfinite(float(value)):
                metric_values.append(float(value))
        row[name] = statistics.fmean(metric_values) if metric_values else None
        row[f"{name}_std"] = statistics.pstdev(metric_values) if metric_values else None
        row[f"{name}_valid_samples"] = len(metric_values)
    strict_rr_values = [
        float(sample["RRE"]) < 5.0 and float(sample["RTE"]) < 5.0
        for sample in samples
    ]
    row["RR"] = strict["rr_5deg_5mm"]
    row["RR_std"] = statistics.pstdev(strict_rr_values)
    row["RR_valid_samples"] = len(strict_rr_values)
    return row


def format_metric(value, scale=1.0) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * scale:.4f}"


def main() -> None:
    args = parse_args()
    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            path = args.result_root / model / f"{condition}.json"
            if path.is_file():
                rows.append(read_result(path, model, condition))
    if not rows:
        raise FileNotFoundError(f"No benchmark JSON files found under {args.result_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Visibility [0.2, 0.4] benchmark",
        "",
        "| Model | Condition | N | RMS-TRE↓ | SR@5mm↑ | PIR↑ | IR↑ | RRE↓ | RTE↓ | RMSE↓ | RR@5°/5mm↑ | r_rmse↓ | r_mae↓ | t_rmse↓ | t_mae↓ | Chamfer↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['model']} | {row['condition']} | {row['count']} | "
            f"{row['mean_rms_tre_mm']:.4f} | "
            f"{100.0 * row['success_rate_5mm']:.2f}% | "
            + " | ".join(format_metric(row[name]) for name in METRIC_NAMES)
            + " |"
        )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output} and {markdown_path}")


if __name__ == "__main__":
    main()
