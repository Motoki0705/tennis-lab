"""Tests for fail-closed production ball Gaussian asset ingestion."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.synthetic_data_generation.blcs.asset_ingestion import (
    BALL_ASSET_CONVERSION_REPORT_SCHEMA,
    BALL_ASSET_INGESTION_SCHEMA,
    FROZEN_TARGET_NHT_FIT,
    IDENTITY_SHARED_NHT,
    INDEPENDENT_NHT_SOURCE,
    SHARED_NHT_SOURCE,
    VANILLA_3DGS_SOURCE,
    BallAssetIngestionSpec,
    BallSourceFormat,
    publish_ball_asset_registry_from_sources,
)
from src.synthetic_data_generation.blcs.assets import load_ball_asset_registry
from src.synthetic_data_generation.composition.contracts import (
    GAUSSIAN_ASSET_SCHEMA,
    NHT_APPEARANCE_MODEL,
    NHT_TENSOR_ENCODING,
    SCENE_COORDINATE_FRAME,
    SCENE_UNIT,
    GaussianAsset,
)
from src.synthetic_data_generation.scene_contract import (
    ArtifactRef,
    SimilarityTransform,
)

_IDENTITY_ROTATION = (
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def _artifact(path: Path, artifact_id: str) -> ArtifactRef:
    payload = path.read_bytes()
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=path.resolve().as_uri(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _write_bytes(path: Path, artifact_id: str, payload: bytes) -> ArtifactRef:
    path.write_bytes(payload)
    return _artifact(path, artifact_id)


def _prepared_tensor_artifact(path: Path) -> ArtifactRef:
    centre = torch.tensor([10.0, -5.0, 2.0], dtype=torch.float32)
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    means = centre + axes.repeat(10, 1)
    count = means.shape[0]
    payload = {
        "means": means,
        "quats": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).repeat(count, 1),
        "scales": torch.full(
            (count, 3),
            math.log(0.02),
            dtype=torch.float32,
        ),
        "opacities": torch.zeros(count, dtype=torch.float32),
        "features": torch.linspace(
            -0.25,
            0.25,
            count * 48,
            dtype=torch.float32,
        ).reshape(count, 48),
        "instance_ids": torch.zeros(count, dtype=torch.int64),
    }
    torch.save(payload, path)
    return _artifact(path, "prepared-target-nht-tensors")


def _background(
    tmp_path: Path,
) -> tuple[GaussianAsset, ArtifactRef, str]:
    appearance = _write_bytes(
        tmp_path / "appearance.pt",
        "target-frozen-appearance",
        b"frozen target NHT shader",
    )
    appearance_space = hashlib.sha256(b"target-appearance-space").hexdigest()
    tensors = _write_bytes(
        tmp_path / "background.pt",
        "target-background-tensors",
        b"background tensor provenance fixture",
    )
    checkpoint = _write_bytes(
        tmp_path / "checkpoint.pt",
        "target-nht-checkpoint",
        b"background checkpoint fixture",
    )
    return (
        GaussianAsset(
            schema=GAUSSIAN_ASSET_SCHEMA,
            asset_id="target-background",
            asset_class="court-scene",
            role="background",
            coordinate_frame=SCENE_COORDINATE_FRAME,
            unit=SCENE_UNIT,
            metres_per_unit=None,
            gaussian_count=100,
            feature_dim=48,
            tensor_encoding=NHT_TENSOR_ENCODING,
            tensors=tensors,
            appearance_model=NHT_APPEARANCE_MODEL,
            appearance_space_sha256=appearance_space,
            appearance_payload=appearance,
            provenance=(checkpoint,),
        ),
        appearance,
        appearance_space,
    )


def _identity_spec(tmp_path: Path) -> tuple[GaussianAsset, BallAssetIngestionSpec]:
    background, appearance, appearance_space = _background(tmp_path)
    tensors = _prepared_tensor_artifact(tmp_path / "ball-source.pt")
    transform = SimilarityTransform(
        scale=0.032,
        rotation=_IDENTITY_ROTATION,
        translation=(-0.32, 0.16, -0.064),
    )
    return (
        background,
        BallAssetIngestionSpec(
            schema=BALL_ASSET_INGESTION_SCHEMA,
            variant_id="optic-yellow",
            asset_id="user-ball-optic-yellow",
            nominal_diameter_m=0.067,
            source_format=SHARED_NHT_SOURCE,
            source_artifacts=(tensors,),
            prepared_tensors=tensors,
            prepared_appearance_space_sha256=appearance_space,
            prepared_appearance_payload=appearance,
            conversion_method=IDENTITY_SHARED_NHT,
            conversion_report=None,
            asset_from_prepared=transform,
        ),
    )


def _conversion_report(
    path: Path,
    *,
    source_format: BallSourceFormat,
    prepared_tensors: ArtifactRef,
    background: GaussianAsset,
    psnr_db: float = 31.5,
    gaussian_count: int = 60,
) -> ArtifactRef:
    path.write_text(
        json.dumps(
            {
                "schema": BALL_ASSET_CONVERSION_REPORT_SCHEMA,
                "status": "passed",
                "method": FROZEN_TARGET_NHT_FIT,
                "source_format": source_format,
                "target_appearance_space_sha256": (background.appearance_space_sha256),
                "target_appearance_payload_sha256": (
                    background.appearance_payload.sha256
                ),
                "prepared_tensors_sha256": prepared_tensors.sha256,
                "gaussian_count": gaussian_count,
                "feature_dim": 48,
                "optimization_steps": 1000,
                "validation_views": 12,
                "validation_psnr_db": psnr_db,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return _artifact(path, "frozen-target-conversion-report")


def test_identity_publication_is_metric_local_and_reproducible(
    tmp_path: Path,
) -> None:
    background, spec = _identity_spec(tmp_path)
    first_path = publish_ball_asset_registry_from_sources(
        tmp_path / "publication-a",
        registry_id="user-ball-assets",
        target_background=background,
        sources=(spec,),
    )
    second_path = publish_ball_asset_registry_from_sources(
        tmp_path / "publication-b",
        registry_id="user-ball-assets",
        target_background=background,
        sources=(spec,),
    )

    first = load_ball_asset_registry(first_path)
    second = load_ball_asset_registry(second_path)
    assert first.registry_fingerprint == second.registry_fingerprint
    assert (
        first.entries[0].asset.tensors.sha256 == second.entries[0].asset.tensors.sha256
    )
    first_ingestion = next(
        item
        for item in first.entries[0].asset.provenance
        if item.artifact_id.endswith("-ingestion")
    )
    second_ingestion = next(
        item
        for item in second.entries[0].asset.provenance
        if item.artifact_id.endswith("-ingestion")
    )
    assert first_ingestion.sha256 == second_ingestion.sha256
    tensor_path = Path(first.entries[0].asset.tensors.uri.removeprefix("file://"))
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    midpoint = 0.5 * (tensors["means"].amin(dim=0) + tensors["means"].amax(dim=0))
    assert torch.allclose(midpoint, torch.zeros(3), atol=1.0e-7, rtol=0.0)
    assert tensors["instance_ids"].eq(0).all()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_ball_asset_registry_from_sources(
            tmp_path / "publication-a",
            registry_id="user-ball-assets",
            target_background=background,
            sources=(spec,),
        )


def test_ingestion_spec_round_trip_is_strict(tmp_path: Path) -> None:
    _, spec = _identity_spec(tmp_path)
    assert BallAssetIngestionSpec.from_dict(spec.to_dict()) == spec

    payload = spec.to_dict()
    payload["fallback_asset"] = "fixture"
    with pytest.raises(ValueError, match="fallback_asset"):
        BallAssetIngestionSpec.from_dict(payload)
    with pytest.raises(ValueError, match="Invalid asset_id"):
        replace(spec, asset_id=".hidden")


@pytest.mark.parametrize(
    "source_format",
    [VANILLA_3DGS_SOURCE, INDEPENDENT_NHT_SOURCE],
)
def test_non_shared_sources_require_declared_conversion(
    tmp_path: Path,
    source_format: BallSourceFormat,
) -> None:
    _, shared = _identity_spec(tmp_path)
    with pytest.raises(ValueError, match="frozen-target conversion path"):
        replace(shared, source_format=source_format)
    with pytest.raises(ValueError, match="require a conversion report"):
        replace(
            shared,
            source_format=source_format,
            conversion_method=FROZEN_TARGET_NHT_FIT,
        )


def test_converted_vanilla_source_requires_matching_passing_report(
    tmp_path: Path,
) -> None:
    background, shared = _identity_spec(tmp_path)
    ply = _write_bytes(
        tmp_path / "user-ball.ply",
        "original-user-ball-ply",
        b"ply\nformat binary_little_endian 1.0\nend_header\n",
    )
    report = _conversion_report(
        tmp_path / "conversion.json",
        source_format=VANILLA_3DGS_SOURCE,
        prepared_tensors=shared.prepared_tensors,
        background=background,
    )
    converted = replace(
        shared,
        source_format=VANILLA_3DGS_SOURCE,
        source_artifacts=(ply,),
        conversion_method=FROZEN_TARGET_NHT_FIT,
        conversion_report=report,
    )

    path = publish_ball_asset_registry_from_sources(
        tmp_path / "converted",
        registry_id="converted-user-ball",
        target_background=background,
        sources=(converted,),
    )

    registry = load_ball_asset_registry(path)
    provenance_ids = {
        artifact.artifact_id for artifact in registry.entries[0].asset.provenance
    }
    assert "original-user-ball-ply" in provenance_ids
    assert "frozen-target-conversion-report" in provenance_ids

    failing_report = _conversion_report(
        tmp_path / "conversion-low-psnr.json",
        source_format=VANILLA_3DGS_SOURCE,
        prepared_tensors=shared.prepared_tensors,
        background=background,
        psnr_db=19.99,
    )
    with pytest.raises(ValueError, match="below 20"):
        publish_ball_asset_registry_from_sources(
            tmp_path / "rejected-low-psnr",
            registry_id="rejected-user-ball",
            target_background=background,
            sources=(replace(converted, conversion_report=failing_report),),
        )
    assert not (tmp_path / "rejected-low-psnr").exists()

    wrong_count_report = _conversion_report(
        tmp_path / "conversion-wrong-count.json",
        source_format=VANILLA_3DGS_SOURCE,
        prepared_tensors=shared.prepared_tensors,
        background=background,
        gaussian_count=59,
    )
    with pytest.raises(ValueError, match="gaussian_count differs"):
        publish_ball_asset_registry_from_sources(
            tmp_path / "rejected-wrong-count",
            registry_id="rejected-user-ball",
            target_background=background,
            sources=(replace(converted, conversion_report=wrong_count_report),),
        )
    assert not (tmp_path / "rejected-wrong-count").exists()


def test_ingestion_rejects_wrong_appearance_and_metric_scale(
    tmp_path: Path,
) -> None:
    background, spec = _identity_spec(tmp_path)
    with pytest.raises(ValueError, match="do not declare"):
        publish_ball_asset_registry_from_sources(
            tmp_path / "wrong-appearance",
            registry_id="wrong-appearance",
            target_background=background,
            sources=(
                replace(
                    spec,
                    prepared_appearance_space_sha256="f" * 64,
                ),
            ),
        )
    assert not (tmp_path / "wrong-appearance").exists()

    wrong_payload = _write_bytes(
        tmp_path / "independent-appearance.pt",
        "independent-appearance",
        b"independently trained NHT shader",
    )
    with pytest.raises(ValueError, match="payload differs"):
        publish_ball_asset_registry_from_sources(
            tmp_path / "wrong-payload",
            registry_id="wrong-payload",
            target_background=background,
            sources=(
                replace(
                    spec,
                    prepared_appearance_payload=wrong_payload,
                ),
            ),
        )
    assert not (tmp_path / "wrong-payload").exists()

    wrong_scale = replace(
        spec,
        asset_from_prepared=SimilarityTransform(
            scale=0.32,
            rotation=_IDENTITY_ROTATION,
            translation=(-3.2, 1.6, -0.64),
        ),
    )
    with pytest.raises(ValueError, match="diameter differs"):
        publish_ball_asset_registry_from_sources(
            tmp_path / "wrong-scale",
            registry_id="wrong-scale",
            target_background=background,
            sources=(wrong_scale,),
        )
    assert not (tmp_path / "wrong-scale").exists()
