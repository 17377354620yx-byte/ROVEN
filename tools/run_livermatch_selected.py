#!/usr/bin/env python3
"""Run the legacy LiverMatch evaluators on selected P2P test indices.

This adapter leaves the model/data conventions in the original project intact
and only adds reproducible per-case transform export for visualization.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT = Path("/home/yangx/code/point_clould/LiverMatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("base", "p2p"), required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT))
    if args.mode == "p2p":
        # The released evaluator refers to an unshipped ``lib.utils`` module;
        # the two required functions are present verbatim in ``p2p_util``.
        sys.modules["lib.utils"] = importlib.import_module("p2p_util")
    module = importlib.import_module(
        "eval_in_silico" if args.mode == "base" else "eval_in_silico_p2p"
    )
    config_path = PROJECT / "configs/new_task3_004_002.yaml"
    checkpoint_path = (
        PROJECT
        / "snapshot/liver_new_task3_004_002/checkpoints/model_best_loss.pth"
    )
    test_root = "/mnt/data3/yangx/P2P/Dataset/Deform_mesh_npz_test/Test/"
    test_list = "/mnt/data3/yangx/P2P/Dataset/Deform_mesh_npz_test/list.npz"

    config = module.edict(module.load_config(str(config_path)))
    config.architecture = module.architectures[config.model_name]
    config.device = torch.device("cuda:0")
    module.config = config
    module.model = module.KPFCNN(config, **({"K": 5} if args.mode == "p2p" else {}))
    module.model = module.model.to(config.device).eval()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module.model.load_state_dict(state["state_dict"], strict=True)
    module.demo_set = module.LiverDemo(
        config, test_root, test_list, voxel_size=0.04, sigma=None, rot=True
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        kwargs = {"debug": False, "use_SVD": True, "return_transform": True}
        if args.mode == "p2p":
            kwargs.update(cluster=True, use_corr_cl=True, use_all_matches=True)
        _, _, rms_tre, transform = module.eva_one(index, **kwargs)
        output_path = args.output_dir / f"{index:05d}.npz"
        np.savez_compressed(
            output_path,
            estimated_transform=transform,
            rms_tre_mm=float(rms_tre),
        )
        records.append({"index": index, "rms_tre_mm": float(rms_tre)})
        print(f"[{args.mode}] case {index:05d}: RMS-TRE={rms_tre:.4f} mm", flush=True)

    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "LiverMatch" if args.mode == "base" else "LiverMatch+P2P",
                "checkpoint": str(checkpoint_path),
                "samples": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
