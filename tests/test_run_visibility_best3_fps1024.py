import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_visibility_best3_fps1024_correspondences.sh"


def _dry_run(**extra_env):
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=dict(os.environ, P2P_DRY_RUN="1", **extra_env),
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def test_fps1024_dry_run_isolated_and_reuses_all_model_commands():
    output = _dry_run()
    model_rows = [line for line in output.splitlines() if line.startswith("MODEL_CMD ")]

    assert "PREPARE_FPS count=1024" in output
    assert "visibility_0.2_0.4_correspondence_best3_fps1024" in output
    assert len(model_rows) == 18
    assert {line.split()[1] for line in model_rows} == {
        "ours", "geotransformer", "castv2", "dfat", "lepard", "predator",
        "parenet", "livermatch", "livermatch_p2p",
    }
    assert "RENDER_FPS dataset=all" in output


def test_fps1024_dry_run_honors_count_and_force_without_changing_base_root():
    output = _dry_run(P2P_FPS_POINTS="512", P2P_FORCE="1")

    assert "PREPARE_FPS count=512 force=1" in output
    assert "correspondence_best3_fps512" in output
    assert "base_selection=" in output
    assert "prepared_selection=1" in output
