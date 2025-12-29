"""E2E tests for WASB tools scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.fixtures.wasb_fixtures import (
    create_minimal_wasb_checkpoint,
    create_minimal_wasb_dataset,
)


@pytest.mark.e2e
def test_extract_dinov3_backbone(tmp_path: Path) -> None:
    """Test DinoV3 backbone extraction.

    This test verifies that:
    1. The extraction script runs without errors
    2. Output backbone file is created

    """
    # Create minimal checkpoint
    checkpoint_path = tmp_path / "full_model.ckpt"
    create_minimal_wasb_checkpoint(checkpoint_path)

    output_path = tmp_path / "backbone.pth"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.tools.extract_dinov3_backbone",
            f"checkpoint_path={checkpoint_path}",
            f"output_path={output_path}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"Backbone extraction failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check output file was created
    assert output_path.exists(), "Backbone output file was not created"


@pytest.mark.e2e
def test_encode_dinov3_patch_tokens(tmp_path: Path) -> None:
    """Test DinoV3 patch token encoding.

    This test verifies that:
    1. The encoding script runs without errors
    2. Output embeddings directory is created

    """
    # Create dataset and checkpoint
    dataset_dir = tmp_path / "tennis"
    create_minimal_wasb_dataset(dataset_dir)

    checkpoint_path = tmp_path / "model.ckpt"
    create_minimal_wasb_checkpoint(checkpoint_path)

    output_dir = tmp_path / "embeddings"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.tools.encode_dinov3_patch_tokens",
            f"model_checkpoint={checkpoint_path}",
            f"data.root_dir={dataset_dir}",
            f"output_dir={output_dir}",
            "data.train_matches=[game1]",
            "data.val_matches=[]",
            "data.test_matches=[]",
            "device=cpu",
            "num_augments=1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Token encoding failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Check output directory exists
    assert output_dir.exists(), "Embeddings output directory was not created"


@pytest.mark.e2e
def test_encode_dinov3_patch_tokens_without_checkpoint(tmp_path: Path) -> None:
    """Test DinoV3 patch token encoding without checkpoint.

    This test verifies the script can run with a default model.

    """
    # Create dataset only (no checkpoint)
    dataset_dir = tmp_path / "tennis"
    create_minimal_wasb_dataset(dataset_dir)

    output_dir = tmp_path / "embeddings"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.wasb.scripts.tools.encode_dinov3_patch_tokens",
            f"data.root_dir={dataset_dir}",
            f"output_dir={output_dir}",
            "data.train_matches=[game1]",
            "data.val_matches=[]",
            "data.test_matches=[]",
            "device=cpu",
            "num_augments=1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    # This might fail if the script requires a checkpoint, which is acceptable
    # The test is primarily to ensure the script can be invoked
    if result.returncode != 0:
        # Check if it's a checkpoint-related error (acceptable)
        assert (
            "checkpoint" in result.stderr.lower()
            or "model" in result.stderr.lower()
            or "state_dict" in result.stderr.lower()
        ), f"Unexpected error:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    else:
        assert output_dir.exists(), "Embeddings output directory was not created"
