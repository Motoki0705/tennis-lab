"""Package CLI invocation must work before any dataset path is supplied."""

import subprocess
import sys
from pathlib import Path


def test_review_package_help_without_dataset() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.tennis_scene.clip_studio", "--help"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--data-root" in result.stdout
    assert "--source-directory" in result.stdout
    assert "--port" in result.stdout
