"""Shared local Gaussian fixtures for 3DGS-native BLCS unit tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.synthetic_data_generation.blcs.assets import (
    BallAssetEntry,
    BallAssetRegistry,
)
from src.synthetic_data_generation.composition.contracts import (
    ASSET_COORDINATE_FRAME,
    GAUSSIAN_ASSET_SCHEMA,
    METRE_UNIT,
    NHT_APPEARANCE_MODEL,
    NHT_TENSOR_ENCODING,
    GaussianAsset,
)
from src.synthetic_data_generation.scene_contract import ArtifactRef


def _local_artifact(path: Path, artifact_id: str, payload: bytes) -> ArtifactRef:
    path.write_bytes(payload)
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=path.resolve().as_uri(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


@pytest.fixture
def ball_registry(tmp_path: Path) -> BallAssetRegistry:
    """Return two verified local variants in one explicit appearance space."""
    appearance = _local_artifact(
        tmp_path / "appearance.pt",
        "shared-appearance",
        b"shared frozen NHT appearance",
    )
    appearance_space = hashlib.sha256(b"appearance-space-v1").hexdigest()
    entries = []
    for index, variant_id in enumerate(("ball-felt-green", "ball-felt-yellow")):
        tensors = _local_artifact(
            tmp_path / f"{variant_id}.pt",
            f"{variant_id}-tensors",
            f"metric gaussian tensor {index}".encode(),
        )
        provenance = _local_artifact(
            tmp_path / f"{variant_id}-source.bin",
            f"{variant_id}-source",
            f"user asset source {index}".encode(),
        )
        asset = GaussianAsset(
            schema=GAUSSIAN_ASSET_SCHEMA,
            asset_id=variant_id,
            asset_class="tennis-ball",
            role="movable",
            coordinate_frame=ASSET_COORDINATE_FRAME,
            unit=METRE_UNIT,
            metres_per_unit=1.0,
            gaussian_count=64 + index,
            feature_dim=48,
            tensor_encoding=NHT_TENSOR_ENCODING,
            tensors=tensors,
            appearance_model=NHT_APPEARANCE_MODEL,
            appearance_space_sha256=appearance_space,
            appearance_payload=appearance,
            provenance=(provenance,),
        )
        entries.append(
            BallAssetEntry(
                variant_id=variant_id,
                asset=asset,
                nominal_diameter_m=0.067,
            )
        )
    return BallAssetRegistry.create(
        registry_id="unit-ball-assets",
        appearance_space_sha256=appearance_space,
        entries=entries,
    )
