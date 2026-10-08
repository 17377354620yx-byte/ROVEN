#!/usr/bin/env python3
"""Build model-independent, visibility-stratified median cases for both P2P datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    "predator",
    "parenet",
    "livermatch",
    "livermatch_p2p",
)


def load_metadata(dataset: str, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "in_silico":
        metadata = root / "Deform_mesh_npz_test"
        with np.load(metadata / "list.npz", allow_pickle=False) as payload:
            names = np.asarray(payload["test"]).astype(str)
        statistics_path = metadata / "stat_svd.npz"
    elif dataset == "in_vitro":
        names = np.load(root / "rigid_list.npy", allow_pickle=False).astype(str)
        statistics_path = root / "stat.npz"
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    with np.load(statistics_path, allow_pickle=False) as statistics:
        visibility = np.asarray(statistics["vis"], dtype=np.float64)
        deformation = np.asarray(statistics["deform"], dtype=np.float64)
    if not (len(names) == len(visibility) == len(deformation)):
        raise ValueError(f"{dataset}: metadata lengths differ")
    if len(names) < 3 or not np.isfinite(visibility).all():
        raise ValueError(f"{dataset}: need at least three finite samples")
    return names, visibility, deformation


def select_stratified_medians(
    names: np.ndarray,
    visibility: np.ndarray,
    deformation: np.ndarray,
    *,
    seed: int,
) -> list[dict]:
    """Pick the visibility median of each equal-count third without using model results."""

    order = np.lexsort((names.astype(str), visibility))
    strata = np.array_split(order, 3)
    rng = np.random.default_rng(seed)
    selected = []
    for stratum_index, indices in enumerate(strata):
        target = float(np.median(visibility[indices]))
        distance = np.abs(visibility[indices] - target)
        tied = indices[np.isclose(distance, distance.min(), rtol=0.0, atol=1e-12)]
        source_index = int(rng.choice(np.sort(tied)))
        selected.append(
            {
                "subset_index": stratum_index,
                "source_index": source_index,
                "sample": str(names[source_index]),
                "visibility": float(visibility[source_index]),
                "deformation_mm": float(deformation[source_index]),
                "stratum": stratum_index + 1,
                "stratum_count": int(len(indices)),
                "stratum_visibility_min": float(visibility[indices].min()),
                "stratum_visibility_max": float(visibility[indices].max()),
                "stratum_median_visibility": target,
                "selection_strategy": "equal_count_visibility_stratum_median",
                "tie_break_seed": int(seed),
            }
        )
    return selected


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-silico-root", type=Path, required=True)
    parser.add_argument("--in-vitro-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    roots = {
        "in_silico": args.in_silico_root.resolve(),
        "in_vitro": args.in_vitro_root.resolve(),
    }
    selection = {
        "selection_protocol": "model_independent_visibility_stratified_median_v1",
        "selection_note": (
            "Each low-visibility dataset is sorted by visibility, split into three "
            "equal-count strata, and represented by the sample nearest each stratum median. "
            "No model metric is used."
        ),
        "seed": args.seed,
        "methods": list(METHODS),
        "datasets": {},
    }
    for dataset, source_root in roots.items():
        names, visibility, deformation = load_metadata(dataset, source_root)
        cases = select_stratified_medians(
            names, visibility, deformation, seed=args.seed
        )
        original_indices = np.load(
            source_root / "original_indices.npy", allow_pickle=False
        )
        for case in cases:
            case["original_index"] = int(original_indices[case["source_index"]])
        subset = build_subset(
            dataset,
            source_root,
            cases,
            args.output_root / "subsets" / dataset,
        )
        selection["datasets"][dataset] = {
            "cases": cases,
            "subset": subset,
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "selection.json"
    output_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    for dataset, block in selection["datasets"].items():
        for case in block["cases"]:
            print(
                dataset,
                case["sample"],
                f"visibility={case['visibility']:.6f}",
                f"stratum={case['stratum']}",
            )


if __name__ == "__main__":
    main()
