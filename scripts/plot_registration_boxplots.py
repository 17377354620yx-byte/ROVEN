#!/usr/bin/env python3
"""Build publication-ready boxplots from per-sample registration results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


MODEL_SPECS = (
    ("ours", "Ours"),
    ("geotransformer", "GeoTransformer"),
    ("dfat", "DFAT"),
    ("parenet", "PARENet"),
)

ALL_MODEL_SPECS = MODEL_SPECS + (
    ("castv2", "CASTV2"),
    ("lepard", "Lepard"),
    ("lepard_p2p", "Lepard+P2P"),
    ("livermatch", "LiverMatch"),
    ("livermatch_p2p", "LiverMatch+P2P"),
    ("goicp", "Go-ICP"),
)

CONDITION_SPECS = (
    ("in_silico_none", "In Silico — Clean"),
    ("in_silico_noise4", "In Silico — Noise 4"),
    ("in_vitro", "In Vitro"),
)

METRIC_ALIASES = {
    "RMS-TRE": ("rms_tre_mm", "RMS_TRE_mm"),
    "RRE": ("RRE",),
    "RTE": ("RTE",),
    "RMSE": ("RMSE",),
    "Chamfer": ("chamfer_dist",),
}

METRIC_LABELS = {
    "RMS-TRE": "RMS-TRE (mm)",
    "RRE": "RRE (degrees)",
    "RTE": "RTE (mm)",
    "RMSE": "RMSE (mm)",
    "Chamfer": "Chamfer distance (mm)",
}

LOG_METRICS = {"RMS-TRE", "RRE", "RTE", "RMSE"}

MODEL_COLORS = (
    "#D55E00",  # Ours: vermillion highlight
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#8C564B",
    "#9467BD",
    "#7F7F7F",
    "#000000",
)


def _first_present(sample: dict, names: tuple[str, ...]):
    for name in names:
        if name in sample and sample[name] is not None:
            return sample[name]
    raise KeyError(f"Missing required field; expected one of {names}")


def normalize_sample(sample: dict) -> dict:
    """Normalize evaluator-specific field aliases without aggregating samples."""
    sample_id = _first_present(sample, ("sample", "sample_name"))
    normalized = {
        "sample_id": str(sample_id),
        "visibility": float(_first_present(sample, ("visibility",))),
    }
    for display_name, aliases in METRIC_ALIASES.items():
        value = float(_first_present(sample, aliases))
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {display_name} for sample {sample_id}")
        normalized[display_name] = value
    return normalized


def visibility_bin(value: float) -> str | None:
    """Assign the two low-visibility bins with 0.4 included exactly once."""
    value = float(value)
    if 0.2 <= value < 0.3:
        return "0.2–0.3"
    if 0.3 <= value <= 0.4:
        return "0.3–0.4"
    return None


def collect_records(
    result_root: Path,
    *,
    model_specs=MODEL_SPECS,
    condition_specs=CONDITION_SPECS,
) -> tuple[list[dict], list[dict]]:
    """Load per-sample records and fail if methods do not evaluate identical cases."""
    result_root = Path(result_root)
    records: list[dict] = []
    provenance: list[dict] = []
    for condition_slug, condition_name in condition_specs:
        reference_ids = None
        reference_visibility = None
        for model_slug, model_name in model_specs:
            path = result_root / model_slug / f"{condition_slug}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            samples = [normalize_sample(sample) for sample in payload.get("samples", [])]
            if not samples:
                raise ValueError(f"No per-sample records in {path}")
            sample_ids = [sample["sample_id"] for sample in samples]
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError(f"Duplicate sample IDs in {path}")
            if reference_ids is None:
                reference_ids = sample_ids
                reference_visibility = [sample["visibility"] for sample in samples]
            elif sample_ids != reference_ids:
                raise ValueError(
                    f"sample IDs/order mismatch for {model_name} in {condition_name}"
                )
            visibility_deltas = []
            for sample_index, sample in enumerate(samples):
                reported_visibility = sample["visibility"]
                canonical_visibility = reference_visibility[sample_index]
                visibility_deltas.append(abs(reported_visibility - canonical_visibility))
                records.append(
                    {
                        "model": model_name,
                        "model_slug": model_slug,
                        "condition": condition_name,
                        "condition_slug": condition_slug,
                        **sample,
                        "reported_visibility": reported_visibility,
                        "visibility": canonical_visibility,
                        "visibility_delta": reported_visibility - canonical_visibility,
                        "visibility_bin": visibility_bin(canonical_visibility),
                    }
                )
            provenance.append(
                {
                    "model": model_name,
                    "condition": condition_name,
                    "sample_count": len(samples),
                    "checkpoint": payload.get("checkpoint"),
                    "source_json": str(path.resolve()),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "max_abs_visibility_delta": max(visibility_deltas),
                    "solver_status_counts": {
                        status: sum(
                            str(sample.get("solver_status")) == status
                            for sample in payload.get("samples", [])
                        )
                        for status in sorted(
                            {
                                str(sample.get("solver_status"))
                                for sample in payload.get("samples", [])
                                if sample.get("solver_status") is not None
                            }
                        )
                    },
                    "identity_fallback_count": sum(
                        bool(sample.get("fallback_identity", False))
                        for sample in payload.get("samples", [])
                    ),
                }
            )
    return records, provenance


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(records: list[dict], model_specs, condition_specs) -> list[dict]:
    rows = []
    for condition_slug, condition_name in condition_specs:
        for model_slug, model_name in model_specs:
            selected = [
                row
                for row in records
                if row["condition_slug"] == condition_slug
                and row["model_slug"] == model_slug
            ]
            for metric in METRIC_ALIASES:
                values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                iqr = q3 - q1
                rows.append(
                    {
                        "model": model_name,
                        "condition": condition_name,
                        "metric": metric,
                        "n": len(values),
                        "q1": float(q1),
                        "median": float(median),
                        "q3": float(q3),
                        "iqr": float(iqr),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "tukey_outliers": int(
                            np.count_nonzero(
                                (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
                            )
                        ),
                    }
                )
    return rows


def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _boxplot(ax, datasets, labels, colors, *, metric: str):
    artists = ax.boxplot(
        datasets,
        tick_labels=labels,
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
        medianprops={"color": "white", "linewidth": 1.25},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.75},
        capprops={"linewidth": 0.75},
        flierprops={
            "marker": "o",
            "markersize": 1.7,
            "markerfacecolor": "#555555",
            "markeredgecolor": "none",
            "alpha": 0.32,
            "rasterized": True,
        },
    )
    for patch, color in zip(artists["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.92)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    if metric in LOG_METRICS:
        positive = all(np.all(np.asarray(values) > 0) for values in datasets)
        if positive:
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=0.1)
    ax.tick_params(axis="x", rotation=38)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    return artists


def _save_figure(fig, output_dir: Path, stem: str, formats, dpi: int) -> None:
    for extension in formats:
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if extension.lower() == "png":
            kwargs["dpi"] = dpi
        fig.savefig(output_dir / f"{stem}.{extension}", **kwargs)


def _plot_metric_panels(
    plt,
    records: list[dict],
    model_specs,
    condition_specs,
    *,
    metric: str,
):
    model_names = [name for _, name in model_specs]
    colors = list(MODEL_COLORS[: len(model_specs)])
    fig, axes = plt.subplots(
        1,
        len(condition_specs),
        figsize=(5.1 * len(condition_specs), 4.5),
        squeeze=False,
        constrained_layout=True,
    )
    for column, (condition_slug, condition_name) in enumerate(condition_specs):
        ax = axes[0, column]
        datasets = [
            [
                row[metric]
                for row in records
                if row["condition_slug"] == condition_slug
                and row["model_slug"] == model_slug
            ]
            for model_slug, _ in model_specs
        ]
        _boxplot(ax, datasets, model_names, colors, metric=metric)
        ax.set_title(f"{condition_name}\n$n$ = {len(datasets[0]):,} per method")
        ax.set_ylabel(METRIC_LABELS[metric])
        if metric == "RMS-TRE":
            ax.axhline(5.0, color="#B2182B", linestyle="--", linewidth=1.0, zorder=0)
            ax.text(
                0.99,
                5.0,
                " 5 mm threshold",
                color="#B2182B",
                fontsize=7.5,
                ha="right",
                va="bottom",
                transform=ax.get_yaxis_transform(),
            )
        elif metric in {"RRE", "RTE"}:
            ax.axhline(5.0, color="#B2182B", linestyle="--", linewidth=0.9, zorder=0)
    fig.suptitle(
        f"Per-sample {METRIC_LABELS[metric]} distributions",
        fontsize=11,
        fontweight="bold",
    )
    return fig


def _plot_registration_grid(plt, records, model_specs, condition_specs):
    metrics = ("RRE", "RTE", "RMSE", "Chamfer")
    model_names = [name for _, name in model_specs]
    colors = list(MODEL_COLORS[: len(model_specs)])
    fig, axes = plt.subplots(
        len(metrics),
        len(condition_specs),
        figsize=(5.1 * len(condition_specs), 3.55 * len(metrics)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, metric in enumerate(metrics):
        for column, (condition_slug, condition_name) in enumerate(condition_specs):
            ax = axes[row_index, column]
            datasets = [
                [
                    item[metric]
                    for item in records
                    if item["condition_slug"] == condition_slug
                    and item["model_slug"] == model_slug
                ]
                for model_slug, _ in model_specs
            ]
            _boxplot(ax, datasets, model_names, colors, metric=metric)
            if row_index == 0:
                ax.set_title(f"{condition_name}\n$n$ = {len(datasets[0]):,} per method")
            ax.set_ylabel(METRIC_LABELS[metric])
            if metric in {"RRE", "RTE"}:
                ax.axhline(5.0, color="#B2182B", linestyle="--", linewidth=0.9, zorder=0)
    fig.suptitle(
        "Per-sample rigid-registration metric distributions",
        fontsize=12,
        fontweight="bold",
    )
    return fig


def _plot_visibility_sensitivity(plt, records, model_specs, condition_specs):
    from matplotlib.patches import Patch

    bin_specs = (("0.2–0.3", "//"), ("0.3–0.4", ""))
    model_names = [name for _, name in model_specs]
    colors = list(MODEL_COLORS[: len(model_specs)])
    fig, axes = plt.subplots(
        1,
        len(condition_specs),
        figsize=(5.1 * len(condition_specs), 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    offsets = (-0.19, 0.19)
    for column, (condition_slug, condition_name) in enumerate(condition_specs):
        ax = axes[0, column]
        sample_counts = []
        for model_index, (model_slug, _) in enumerate(model_specs, start=1):
            for bin_index, (bin_name, hatch) in enumerate(bin_specs):
                values = [
                    row["RMS-TRE"]
                    for row in records
                    if row["condition_slug"] == condition_slug
                    and row["model_slug"] == model_slug
                    and row["visibility_bin"] == bin_name
                ]
                sample_counts.append(len(values))
                artists = ax.boxplot(
                    [values],
                    positions=[model_index + offsets[bin_index]],
                    widths=0.32,
                    patch_artist=True,
                    showfliers=True,
                    whis=1.5,
                    medianprops={"color": "white", "linewidth": 1.05},
                    boxprops={"linewidth": 0.7},
                    whiskerprops={"linewidth": 0.7},
                    capprops={"linewidth": 0.7},
                    flierprops={
                        "marker": "o",
                        "markersize": 1.3,
                        "markerfacecolor": "#555555",
                        "markeredgecolor": "none",
                        "alpha": 0.25,
                        "rasterized": True,
                    },
                )
                box = artists["boxes"][0]
                box.set_facecolor(colors[model_index - 1])
                box.set_edgecolor("#333333")
                box.set_hatch(hatch)
                box.set_alpha(0.92)
        ax.set_xticks(range(1, len(model_names) + 1), model_names, rotation=38, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("RMS-TRE (mm)")
        ax.set_title(
            f"{condition_name}\n$n$ = {min(sample_counts):,}–{max(sample_counts):,} per box"
        )
        ax.axhline(5.0, color="#B2182B", linestyle="--", linewidth=1.0, zorder=0)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
    axes[0, -1].legend(
        handles=[
            Patch(facecolor="#999999", edgecolor="#333333", hatch=hatch, label=name)
            for name, hatch in bin_specs
        ],
        title="Visibility",
        loc="upper right",
        frameon=False,
    )
    fig.suptitle(
        "Low-visibility sensitivity of per-sample RMS-TRE",
        fontsize=11,
        fontweight="bold",
    )
    return fig


def _plot_rms_tre_violins(
    plt,
    records: list[dict],
    model_specs,
    condition_specs,
    *,
    scale: str,
    linear_upper_mm: float = 32.0,
):
    """Plot raw-sample RMS-TRE densities with median and IQR overlays."""
    if scale not in {"linear", "log"}:
        raise ValueError(f"Unsupported violin scale: {scale}")
    model_names = [name for _, name in model_specs]
    colors = list(MODEL_COLORS[: len(model_specs)])
    positions = np.arange(1, len(model_specs) + 1)
    fig, axes = plt.subplots(
        1,
        len(condition_specs),
        figsize=(5.1 * len(condition_specs), 4.7),
        squeeze=False,
        constrained_layout=True,
    )
    for column, (condition_slug, condition_name) in enumerate(condition_specs):
        ax = axes[0, column]
        datasets = [
            np.asarray(
                [
                    row["RMS-TRE"]
                    for row in records
                    if row["condition_slug"] == condition_slug
                    and row["model_slug"] == model_slug
                ],
                dtype=np.float64,
            )
            for model_slug, _ in model_specs
        ]
        violins = ax.violinplot(
            datasets,
            positions=positions,
            widths=0.9,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            points=240,
            bw_method=0.22,
        )
        for body, color in zip(violins["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.86)

        q1 = np.asarray([np.quantile(values, 0.25) for values in datasets])
        medians = np.asarray([np.median(values) for values in datasets])
        q3 = np.asarray([np.quantile(values, 0.75) for values in datasets])
        ax.vlines(positions, q1, q3, color="#333333", linewidth=3.0, zorder=3)
        ax.scatter(
            positions,
            medians,
            s=18,
            facecolor="white",
            edgecolor="#333333",
            linewidth=0.65,
            zorder=4,
        )
        ax.axhline(5.0, color="#B2182B", linestyle="--", linewidth=0.95, zorder=0)
        ax.set_xticks(positions, model_names, rotation=42, ha="right")
        ax.set_ylabel("RMS-TRE (mm)")
        ax.set_title(f"{condition_name}\n$n$ = {len(datasets[0]):,} per method")
        ax.grid(axis="both", color="#D9D9D9", linewidth=0.55, alpha=0.72)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if scale == "log":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.0, linear_upper_mm)
            clipped_counts = [int(np.count_nonzero(values > linear_upper_mm)) for values in datasets]
            for position, count in zip(positions, clipped_counts):
                if count:
                    ax.scatter(
                        [position],
                        [linear_upper_mm - 0.65],
                        marker="^",
                        s=13,
                        color="#444444",
                        clip_on=True,
                        zorder=5,
                    )
                    ax.text(
                        position,
                        linear_upper_mm - 1.15,
                        str(count),
                        ha="center",
                        va="top",
                        fontsize=6.2,
                        color="#444444",
                    )
    scale_note = (
        f"Linear display; triangles report counts above {linear_upper_mm:g} mm"
        if scale == "linear"
        else "Logarithmic display; all observed errors shown"
    )
    fig.suptitle(
        f"Per-sample RMS-TRE distributions — {scale_note}",
        fontsize=11,
        fontweight="bold",
    )
    return fig


def _write_audit_report(
    path: Path,
    records: list[dict],
    provenance: list[dict],
    model_specs,
    condition_specs,
) -> None:
    lines = [
        "# Registration boxplot data audit",
        "",
        "All boxplots were computed from individual sample records. No synthetic samples were reconstructed from aggregate means or standard deviations.",
        "",
        "## Coverage",
        "",
        "| Model | Condition | N | Source |",
        "|---|---|---:|---|",
    ]
    for item in provenance:
        lines.append(
            f"| {item['model']} | {item['condition']} | {item['sample_count']} | `{item['source_json']}` |"
        )
    solver_items = [item for item in provenance if item["solver_status_counts"]]
    if solver_items:
        lines.extend(
            [
                "",
                "## Solver failures retained in the denominator",
                "",
                "| Model | Condition | Status counts | Identity fallbacks |",
                "|---|---|---|---:|",
            ]
        )
        for item in solver_items:
            statuses = ", ".join(
                f"{name}: {count}"
                for name, count in item["solver_status_counts"].items()
            )
            lines.append(
                f"| {item['model']} | {item['condition']} | {statuses} | "
                f"{item['identity_fallback_count']} |"
            )
    lines.extend(
        [
            "",
            "## Plot conventions",
            "",
            "- Boxes show Q1–Q3; center lines show medians; whiskers extend to 1.5×IQR.",
            "- Observations beyond Tukey whiskers are retained as outlier markers.",
            "- RMS-TRE, RRE, RTE, and RMSE use logarithmic display axes because genuine registration failures create long right tails.",
            "- The 5 mm line in RMS-TRE panels marks the primary success threshold; 5-degree and 5-mm lines are shown for RRE and RTE.",
            "- Visibility bins are [0.2, 0.3) and [0.3, 0.4], so the 0.4 boundary is included exactly once.",
            "- Sensitivity panels use the Ours result file as the shared sample-level visibility reference for every paired method; evaluator-reported visibility and its difference from the reference remain in `per_sample_metrics.csv`.",
            "- Violin widths encode kernel density estimated from individual samples; white dots and dark vertical bars show medians and IQRs.",
            "- The linear violin view is limited to 0–32 mm for readability and marks the number of observations above 32 mm with triangles; the companion log view shows the complete range.",
            "- No hypothesis-test stars are shown; samples may be clustered within anatomical cases, so naive independent-sample tests would overstate evidence.",
            "",
            f"Validated records: {len(records):,}",
            f"Models: {len(model_specs)}",
            f"Conditions: {len(condition_specs)}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_outputs(
    result_root: Path,
    output_dir: Path,
    *,
    model_specs=MODEL_SPECS,
    condition_specs=CONDITION_SPECS,
    formats=("pdf", "svg", "png"),
    dpi: int = 600,
) -> None:
    """Validate, export, summarize, and plot all requested per-sample results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, provenance = collect_records(
        result_root,
        model_specs=model_specs,
        condition_specs=condition_specs,
    )
    record_fields = [
        "model",
        "model_slug",
        "condition",
        "condition_slug",
        "sample_id",
        "visibility",
        "reported_visibility",
        "visibility_delta",
        "visibility_bin",
        *METRIC_ALIASES,
    ]
    _write_csv(output_dir / "per_sample_metrics.csv", records, record_fields)
    summary_rows = _summary_rows(records, model_specs, condition_specs)
    _write_csv(
        output_dir / "summary_statistics.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_audit_report(
        output_dir / "DATA_AUDIT.md",
        records,
        provenance,
        model_specs,
        condition_specs,
    )

    plt = _configure_matplotlib()
    rms_figure = _plot_metric_panels(
        plt,
        records,
        model_specs,
        condition_specs,
        metric="RMS-TRE",
    )
    _save_figure(rms_figure, output_dir, "boxplot_rms_tre", formats, dpi)
    plt.close(rms_figure)

    registration_figure = _plot_registration_grid(
        plt, records, model_specs, condition_specs
    )
    _save_figure(
        registration_figure,
        output_dir,
        "boxplot_registration_metrics",
        formats,
        dpi,
    )
    plt.close(registration_figure)

    sensitivity_figure = _plot_visibility_sensitivity(
        plt, records, model_specs, condition_specs
    )
    _save_figure(
        sensitivity_figure,
        output_dir,
        "boxplot_visibility_sensitivity",
        formats,
        dpi,
    )
    plt.close(sensitivity_figure)

    for scale in ("linear", "log"):
        violin_figure = _plot_rms_tre_violins(
            plt,
            records,
            model_specs,
            condition_specs,
            scale=scale,
        )
        _save_figure(
            violin_figure,
            output_dir,
            f"violin_rms_tre_{scale}",
            formats,
            dpi,
        )
        plt.close(violin_figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Include all completed methods instead of the four main-text methods.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_specs = ALL_MODEL_SPECS if args.all_models else MODEL_SPECS
    build_outputs(args.result_root, args.output_dir, model_specs=model_specs)
    records, provenance = collect_records(args.result_root, model_specs=model_specs)
    print(
        f"Wrote figures and tables for {len(records)} per-sample rows "
        f"from {len(provenance)} result files to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
