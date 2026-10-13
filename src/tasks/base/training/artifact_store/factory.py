"""Construct artifact stores from the strict shared training config."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.tasks.base.configuration import ArtifactStoreConfig
from src.utils.artifact_store import (
    ArtifactStore,
    LocalArtifactStore,
    RcloneArtifactStore,
)


def build_artifact_store(
    config: ArtifactStoreConfig,
    *,
    local_root: Path,
) -> ArtifactStore:
    """Build the explicitly selected store without fallback between backends."""
    expected_value = os.environ.get("TENNIS_COLAB_ARTIFACT_ROOTS")
    if expected_value is not None:
        expected = json.loads(expected_value)
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(item, str) for item in expected)
        ):
            raise RuntimeError(
                "TENNIS_COLAB_ARTIFACT_ROOTS must name the declared Drive output roots"
            )
        actual = f"{config.remote}:{config.remote_root}"
        if config.mode != "rclone" or actual not in expected:
            raise RuntimeError(
                "Colab runner output requires rclone persistence to a declared Drive root; "
                "set run.artifact_store before starting training"
            )
    if config.mode == "local":
        return LocalArtifactStore(local_root)
    config_value = os.environ.get("RCLONE_CONFIG")
    if not config_value:
        raise RuntimeError(
            "run.artifact_store.mode=rclone requires the RCLONE_CONFIG environment variable"
        )
    assert config.remote is not None
    assert config.remote_root is not None
    return RcloneArtifactStore(
        local_root,
        remote=config.remote,
        remote_root=config.remote_root,
        config_path=Path(config_value),
    )
