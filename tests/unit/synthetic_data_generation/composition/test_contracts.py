"""Tests for semantic Gaussian asset and scene-composition contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.synthetic_data_generation.composition.contracts import (
    GAUSSIAN_ASSET_SCHEMA,
    GAUSSIAN_SCENE_SCHEMA,
    GaussianAsset,
    GaussianAssetRole,
    GaussianCoordinates,
    GaussianDeformationKind,
    GaussianForegroundComposition,
    GaussianFrame,
    GaussianInstance,
    GaussianSceneComposition,
    GaussianSceneObject,
    GaussianTransform,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def _asset(
    *,
    asset_id: str,
    role: GaussianAssetRole,
    appearance_space: str = "b00-deferred-space",
) -> GaussianAsset:
    is_background = role == GaussianAssetRole.BACKGROUND
    return GaussianAsset(
        asset_id=asset_id,
        asset_class="court" if is_background else "player",
        role=role,
        coordinates=(
            GaussianCoordinates.scene()
            if is_background
            else GaussianCoordinates.asset_local_metres()
        ),
        gaussian_count=100 if is_background else 20,
        feature_dim=48,
        floating_dtype="float32",
        appearance_model="nht-deferred",
        appearance_space=appearance_space,
    )


def _translation(x: float, y: float, z: float) -> GaussianTransform:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    return GaussianTransform(
        scale=0.25,
        rigid=RigidTransform.from_matrix(matrix),
    )


def _composition() -> GaussianSceneComposition:
    return GaussianSceneComposition(
        scene_id="b00",
        composition_id="plcs-main",
        background=_asset(
            asset_id="background",
            role=GaussianAssetRole.BACKGROUND,
        ),
        assets=(
            _asset(asset_id="player-surface", role=GaussianAssetRole.MOVABLE),
        ),
        objects=(
            GaussianSceneObject(
                object_id="player-07",
                instance_id=7,
                asset_id="player-surface",
                deformation_kind=GaussianDeformationKind.ARTICULATED,
            ),
        ),
        frames=(
            GaussianFrame(
                frame_index=0,
                instances=(
                    GaussianInstance(
                        object_id="player-07",
                        source_frame_index=13,
                        scene_from_asset=_translation(1.0, 2.0, 3.0),
                    ),
                ),
            ),
            GaussianFrame(
                frame_index=1,
                instances=(
                    GaussianInstance(
                        object_id="player-07",
                        source_frame_index=14,
                        scene_from_asset=_translation(1.1, 2.0, 3.0),
                    ),
                ),
            ),
        ),
    )


def test_semantic_composition_round_trip_has_no_artifact_identity_fields() -> None:
    composition = _composition()

    payload = composition.to_dict()
    loaded = GaussianSceneComposition.from_dict(payload)

    assert loaded == composition
    assert payload["schema"] == GAUSSIAN_SCENE_SCHEMA
    assert payload["background"]["schema"] == GAUSSIAN_ASSET_SCHEMA  # type: ignore[index]
    serialized = repr(payload).lower()
    for forbidden in ("sha256", "fingerprint", "renderer_commit", "provenance"):
        assert forbidden not in serialized


def test_foreground_composition_is_strictly_articulated_and_background_free() -> None:
    asset = _asset(asset_id="player-surface", role=GaussianAssetRole.MOVABLE)
    transform = _translation(0.0, 0.0, 0.0)
    foreground = GaussianForegroundComposition(
        scene_id="b00",
        composition_id="plcs-foreground",
        assets=(asset,),
        objects=(
            GaussianSceneObject("left-player", 7, asset.asset_id, GaussianDeformationKind.ARTICULATED),
            GaussianSceneObject("right-player", 8, asset.asset_id, GaussianDeformationKind.ARTICULATED),
        ),
        frames=(
            GaussianFrame(
                0,
                (
                    GaussianInstance("left-player", 0, transform),
                    GaussianInstance("right-player", 10, transform),
                ),
            ),
            GaussianFrame(
                1,
                (
                    GaussianInstance("left-player", 1, transform),
                    GaussianInstance("right-player", 11, transform),
                ),
            ),
        ),
    )

    payload = foreground.to_dict()
    assert "background" not in payload
    assert GaussianForegroundComposition.from_dict(payload) == foreground
    assert {item.instance_id for item in foreground.objects} == {7, 8}
    with pytest.raises(ValueError, match="declared articulated"):
        replace(
            foreground,
            objects=(
                replace(
                    foreground.objects[0],
                    deformation_kind=GaussianDeformationKind.RIGID,
                ),
                foreground.objects[1],
            ),
        )


def test_composition_rejects_incompatible_appearance_without_hashes() -> None:
    composition = _composition()
    mismatched = replace(
        composition.assets[0],
        appearance_space="independently-trained-space",
    )

    with pytest.raises(ValueError, match="different appearance space"):
        replace(composition, assets=(mismatched,))


def test_asset_contract_requires_explicit_coordinate_convention_and_dtype() -> None:
    movable = _asset(asset_id="ball", role=GaussianAssetRole.MOVABLE)

    with pytest.raises(ValueError, match="movable assets must use"):
        replace(movable, coordinates=GaussianCoordinates.scene())
    with pytest.raises(ValueError, match="floating_dtype"):
        replace(movable, floating_dtype="float16")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="deformation_kind"):
        GaussianSceneObject(
            object_id="player",
            instance_id=1,
            asset_id=movable.asset_id,
            deformation_kind="unknown",  # type: ignore[arg-type]
        )


def test_composition_enforces_unique_object_identity_and_complete_frames() -> None:
    composition = _composition()
    duplicate = replace(
        composition.objects[0],
        object_id="player-08",
    )

    with pytest.raises(ValueError, match="instance ids contains duplicates"):
        replace(composition, objects=(*composition.objects, duplicate))
    with pytest.raises(ValueError, match="exactly equal 0..T-1"):
        replace(
            composition,
            frames=(composition.frames[0], replace(composition.frames[1], frame_index=2)),
        )
    with pytest.raises(ValueError, match="source frames must be consecutive"):
        replace(
            composition,
            frames=(
                composition.frames[0],
                replace(
                    composition.frames[1],
                    instances=(
                        replace(
                            composition.frames[1].instances[0],
                            source_frame_index=15,
                        ),
                    ),
                ),
            ),
        )


def test_composition_supports_background_only_court_frames() -> None:
    composition = GaussianSceneComposition(
        scene_id="b00",
        composition_id="court-frame",
        background=_asset(
            asset_id="background",
            role=GaussianAssetRole.BACKGROUND,
        ),
        assets=(),
        objects=(),
        frames=(GaussianFrame(frame_index=0, instances=()),),
    )

    assert composition.frame(0).instances == ()
    with pytest.raises(KeyError, match="Unknown Gaussian frame"):
        composition.frame(1)


def test_transform_rejects_nonpositive_scale_and_improper_rotation() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        GaussianTransform(scale=0.0, rigid=RigidTransform.identity())

    reflection = np.eye(4, dtype=np.float64)
    reflection[0, 0] = -1.0
    with pytest.raises(ValueError, match="proper rotation"):
        RigidTransform.from_matrix(reflection)


def test_composition_parser_rejects_unknown_fields_without_compatibility() -> None:
    payload = _composition().to_dict()
    payload["composition_fingerprint"] = "legacy"

    with pytest.raises(ValueError, match="extra=.*composition_fingerprint"):
        GaussianSceneComposition.from_dict(payload)
