"""E2E tests for WASB dataset generation scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.skip(
    reason="Requires meta.json file setup; test fixture incomplete"
)
def test_wasb_download_videos_status(tmp_path: Path) -> None:
    """Test WASB video download status mode.

    This test verifies that the status mode runs without actual downloads.

    """
    # Create minimal urls.yaml file
    urls_path = tmp_path / "urls.yaml"
    urls_path.write_text("urls: []\n")

    meta_path = tmp_path / "meta.json"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.generate_dataset.download_videos",
            "mode=status",
            f"urls_path={urls_path}",
            f"download.meta_path={meta_path}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"Status check failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"


@pytest.mark.e2e
@pytest.mark.skip(reason="Batch mode requires video processing - complex to test in e2e")
def test_wasb_generate_dataset_batch_mode(tmp_path: Path) -> None:
    """Test WASB dataset generation in batch mode.

    Skipped because batch processing requires extensive video processing
    and model inference, which is too complex for quick e2e tests.

    """
    pass


@pytest.mark.e2e
@pytest.mark.skip(reason="Clip sampling requires existing dataset structure - skipped for e2e")
def test_wasb_clip_sampling_generate_samples(tmp_path: Path) -> None:
    """Test WASB clip sampling sample generation.

    Skipped because clip sampling requires a properly structured dataset
    with processed clips, which is complex to set up for e2e tests.

    """
    pass
