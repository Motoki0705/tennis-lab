"""Artifact-bound asset snapshots are parsed without reopening any weights."""

from pathlib import Path

import pytest

from src.tennis_scene.pipeline.model_assets import asset_hashes, file_identity


@pytest.mark.parametrize("conflict", [False, True])
def test_asset_snapshots_are_consistent_across_nested_settings(tmp_path: Path, conflict: bool) -> None:
    path = tmp_path / "weights"
    path.write_bytes(b"original")
    snapshot = file_identity(path)
    duplicate = {**snapshot, "sha256": "0" * 64} if conflict else snapshot
    settings = {"first": snapshot, "nested": [{"second": duplicate}], "unhashed": {"path": "media.mp4"}}
    path.unlink()  # Parsing snapshots for cache/load does not access the assets.
    if conflict:
        with pytest.raises(ValueError, match="Conflicting artifact asset identities"):
            asset_hashes(settings)
    else:
        assert asset_hashes(settings) == {path: snapshot["sha256"]}
