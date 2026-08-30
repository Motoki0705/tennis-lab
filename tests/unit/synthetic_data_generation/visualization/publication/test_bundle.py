"""Unit tests for publication bundle inventory and content validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.synthetic_data_generation.visualization.publication.bundle import (
    validate_publication_bundle,
)


def test_bundle_validator_accepts_complete_fixture(valid_publication_bundle: Path) -> None:
    manifest = validate_publication_bundle(valid_publication_bundle)

    assert manifest.scene_id == "scene-0"
    assert len(manifest.artifacts) == 10


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_bundle_validator_rejects_missing_or_extra_media(
    valid_publication_bundle: Path,
    mutation: str,
) -> None:
    if mutation == "missing":
        (valid_publication_bundle / "dataset-court.gif").unlink()
    else:
        (valid_publication_bundle / "foreign-media.bin").write_bytes(b"foreign")

    with pytest.raises(ValueError, match="inventory differs"):
        validate_publication_bundle(valid_publication_bundle)


def test_bundle_validator_rejects_tampered_manifest(valid_publication_bundle: Path) -> None:
    manifest_path = valid_publication_bundle / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["scene_id"] = "foreign-scene"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Every source owner must bind"):
        validate_publication_bundle(valid_publication_bundle)


def test_bundle_validator_rejects_tampered_media_digest(
    valid_publication_bundle: Path,
) -> None:
    media_path = valid_publication_bundle / "dataset-court.gif"
    data = bytearray(media_path.read_bytes())
    data[-1] ^= 1
    media_path.write_bytes(data)

    with pytest.raises(ValueError, match="content digest changed"):
        validate_publication_bundle(valid_publication_bundle)
