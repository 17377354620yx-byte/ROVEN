import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_lists_two_datasets_and_exact_nine_methods_without_goicp():
    env = dict(os.environ, P2P_DRY_RUN="1")
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/run_visibility_best3_correspondences.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    rows = [line.split() for line in completed.stdout.splitlines() if line.startswith("MODEL_CMD ")]
    assert len(rows) == 18
    methods = {row[1] for row in rows}
    datasets = {row[2] for row in rows}
    assert methods == {
        "ours", "geotransformer", "castv2", "dfat", "lepard", "predator",
        "parenet", "livermatch", "livermatch_p2p",
    }
    assert datasets == {"in_silico", "in_vitro"}
    assert "lepard_p2p" not in methods
    assert "goicp" not in methods
    assert all("checkpoint=" in completed.stdout for _ in [0])


def test_dry_run_uses_isolated_output_and_correspondence_flags():
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/run_visibility_best3_correspondences.sh")],
        cwd=ROOT,
        env=dict(os.environ, P2P_DRY_RUN="1"),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "visibility_0.2_0.4_correspondence_best3" in completed.stdout
    assert "--save-correspondences-dir" in completed.stdout
    assert "goicp" not in completed.stdout


def test_predator_limit_follows_each_dataset_selection_size(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        '{"datasets":{"in_silico":{"cases":[{},{},{},{},{}]},'
        '"in_vitro":{"cases":[{},{},{},{}]}}}',
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/run_visibility_best3_correspondences.sh")],
        cwd=ROOT,
        env=dict(
            os.environ,
            P2P_DRY_RUN="1",
            P2P_CORRESPONDENCE_ROOT=str(tmp_path),
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    predator_rows = [
        line for line in completed.stdout.splitlines() if line.startswith("MODEL_CMD predator ")
    ]

    assert len(predator_rows) == 2
    assert "--test-limit 5" in predator_rows[0]
    assert "--test-limit 4" in predator_rows[1]
