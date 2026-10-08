import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_visibility_qualitative_candidates.sh"


def test_qualitative_wrapper_dry_run_declares_complete_pipeline():
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=dict(os.environ, P2P_DRY_RUN="1"),
        text=True,
        capture_output=True,
        check=True,
    )

    output = completed.stdout
    assert "visibility_0.2_0.4_qualitative_candidates60_fps1024" in output
    assert "SELECT_CANDIDATES count=60 per_metric=20" in output
    assert "PREPARE_FPS count=1024" in output
    assert "P2P_PREPARED_SELECTION=1" in output
    assert output.count("MODEL_CMD ") == 18
    assert "MODEL_CMD predator " in output
    assert "MODEL_CMD goicp " not in output
    assert "MODEL_CMD lepard_p2p " not in output
    assert "FINALIZE_STRICT top_k=3" in output
    assert "RENDER_FINAL dataset=all" in output
