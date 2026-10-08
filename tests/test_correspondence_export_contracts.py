import numpy as np
import pytest
from pathlib import Path
import subprocess

from tools.correspondence_cache import (
    unsupported_correspondence_record,
    validate_raw_correspondence_export,
)


def _valid_export():
    return {
        "sample_id": "00008/00002/00000.npz",
        "source_points": np.zeros((4, 3)),
        "target_points": np.ones((3, 3)),
        "source_corr_points": np.zeros((2, 3)),
        "target_corr_points": np.ones((2, 3)),
        "corr_scores": np.array([0.9, 0.4]),
        "gt_transform": np.eye(4),
        "estimated_transform": np.eye(4),
        "physical_scale_mm": 100.0,
        "coordinate_frame": "network_normalized",
        "source_variable": "output['src_corr_points']",
        "target_variable": "output['ref_corr_points']",
    }


def test_raw_export_contract_accepts_real_model_point_pairs():
    checked = validate_raw_correspondence_export(_valid_export())

    assert checked["num_pred_correspondences"] == 2
    assert checked["has_confidence"] is True


def test_raw_export_contract_accepts_scalar_strings_loaded_from_npz():
    payload = _valid_export()
    payload["sample_id"] = np.asarray(payload["sample_id"])
    payload["coordinate_frame"] = np.asarray(payload["coordinate_frame"])
    payload["source_variable"] = np.asarray(payload["source_variable"])
    payload["target_variable"] = np.asarray(payload["target_variable"])

    checked = validate_raw_correspondence_export(payload)

    assert checked["num_pred_correspondences"] == 2


def test_raw_export_contract_allows_explicitly_absent_confidence():
    payload = _valid_export()
    payload["corr_scores"] = None

    checked = validate_raw_correspondence_export(payload)

    assert checked["has_confidence"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("source_variable"),
        lambda value: value.update(source_corr_points=np.zeros((1, 3))),
        lambda value: value.update(corr_scores=np.ones(1)),
        lambda value: value.update(physical_scale_mm=0.0),
        lambda value: value.update(coordinate_frame="unknown"),
    ],
)
def test_raw_export_contract_rejects_untraceable_or_mismatched_data(mutation):
    payload = _valid_export()
    mutation(payload)
    with pytest.raises((KeyError, ValueError)):
        validate_raw_correspondence_export(payload)


def test_goicp_is_explicitly_unsupported_instead_of_synthesizing_matches():
    record = unsupported_correspondence_record(
        method="goicp",
        sample_id="5_00086.npz",
        reason="rigid solver does not predict correspondences",
    )

    assert record == {
        "method": "goicp",
        "sample_id": "5_00086.npz",
        "supports_correspondences": False,
        "status": "not_applicable",
        "reason": "rigid solver does not predict correspondences",
    }


def test_predator_cli_accepts_explicit_visibility_dataset_root():
    script = Path("/home/yangx/code/OverlapPredator/test_p2p_liver.py")
    python = Path("/home/yangx/miniconda3/envs/overpredator_py310/bin/python")
    completed = subprocess.run(
        [str(python), str(script), "--help"],
        cwd=script.parent,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--data-root" in completed.stdout
