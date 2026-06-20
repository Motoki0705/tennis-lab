"""knowledge-control registration tests (issue #533).

Runs kg_register.py against a fake training-queue repro bundle (isolated via
KNOWLEDGE_DIR), then kg_validate.py, asserting the promoted bundle + node with
the new provider/session/repro/artifacts fields are produced and validate clean.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / ".agents/skills/knowledge-control/scripts"


def _make_repro_bundle(rd: Path) -> None:
    (rd / "predictions").mkdir(parents=True)
    (rd / "run.json").write_text(
        json.dumps(
            {
                "run_id": "123_smoke",
                "name": "i999_smoke",
                "command": "python -m src.tasks.plcs.scripts.train "
                "model=multiview_axial_split loss=canonical_rot data=chunked",
                "provider": "claude",
                "session": "d22b7d68-test",
                "issue": "999",
                "commit": "abc123def456",
                "branch": "feat/x",
                "remote": "git@github.com:Motoki0705/tennis-lab.git",
                "captured_at": "2026-06-19T17:30:00+09:00",
            }
        )
    )
    (rd / "repro.sh").write_text("#!/usr/bin/env bash\necho repro\n")
    (rd / "uncommitted.patch").write_text("diff --git a/x b/x\n")
    np.savez_compressed(
        rd / "predictions" / "pred_test.npz",
        scene_ids=np.array(["scene_000686", "scene_000214"]),
        pred_position=np.zeros((2, 5, 3)),
        target_position=np.zeros((2, 5, 3)),
    )
    (rd / "predictions" / "metrics.json").write_text(
        json.dumps({"position_error_m": 0.42, "angular_error_deg": 9.7})
    )


def test_register_then_validate(tmp_path: Path) -> None:
    rd = tmp_path / "repro" / "123_smoke"
    _make_repro_bundle(rd)
    kb = tmp_path / "kb"
    env = {**os.environ, "KNOWLEDGE_DIR": str(kb)}

    reg = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "kg_register.py"),
            "--repro-dir",
            str(rd),
            "--id",
            "run-i999-test",
            "--issue",
            "999",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert reg.returncode == 0, reg.stderr

    node = kb / "nodes" / "run-i999-test.md"
    assert node.exists()
    text = node.read_text()
    assert "provider: claude" in text
    assert "session: d22b7d68-test" in text
    assert "commit: abc123def456" in text
    assert "run_dir:" in text
    assert "predictions:" in text
    assert "position_error_m: 0.42" in text

    bundle = kb / "runs" / "run-i999-test"
    assert (bundle / "pred_test.npz").exists()
    assert (bundle / "repro.sh").exists()
    assert (bundle / "uncommitted.patch").exists()

    val = subprocess.run(
        [sys.executable, str(SCRIPTS / "kg_validate.py")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert val.returncode == 0, val.stdout + val.stderr
    assert "0 error(s)" in val.stdout
