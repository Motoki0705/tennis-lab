"""Tests for strict Gaussian asset and scene-composition manifests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from src.synthetic_data_generation.composition.contracts import (
    ASSET_COORDINATE_FRAME,
    GAUSSIAN_ASSET_SCHEMA,
    METRE_UNIT,
    NHT_APPEARANCE_MODEL,
    NHT_TENSOR_ENCODING,
    SCENE_COORDINATE_FRAME,
    SCENE_UNIT,
    GaussianAsset,
    GaussianInstance,
    GaussianSceneComposition,
    load_gaussian_scene_manifest,
    write_gaussian_scene_manifest,
)
from src.synthetic_data_generation.scene_contract import (
    ArtifactRef,
    SimilarityTransform,
)


def _artifact(artifact_id: str, digest: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=f"artifact://composition-test/{artifact_id}",
        sha256=digest * 64,
        size_bytes=123,
    )


def _asset(
    *,
    asset_id: str,
    role: Literal["background", "movable"],
    appearance_digest: str = "a" * 64,
) -> GaussianAsset:
    is_background = role == "background"
    return GaussianAsset(
        schema=GAUSSIAN_ASSET_SCHEMA,
        asset_id=asset_id,
        asset_class="court" if is_background else "ball",
        role=role,
        coordinate_frame=(
            SCENE_COORDINATE_FRAME if is_background else ASSET_COORDINATE_FRAME
        ),
        unit=SCENE_UNIT if is_background else METRE_UNIT,
        metres_per_unit=None if is_background else 1.0,
        gaussian_count=100 if is_background else 20,
        feature_dim=48,
        tensor_encoding=NHT_TENSOR_ENCODING,
        tensors=_artifact(f"{asset_id}-tensors", "b"),
        appearance_model=NHT_APPEARANCE_MODEL,
        appearance_space_sha256=appearance_digest,
        appearance_payload=_artifact(f"{asset_id}-appearance", "c"),
        provenance=(_artifact(f"{asset_id}-source", "d"),),
    )


def _composition() -> GaussianSceneComposition:
    return GaussianSceneComposition.create(
        composition_id="composition-0001",
        scene_source=_artifact("scene-source", "e"),
        background=_asset(asset_id="background", role="background"),
        instances=(
            GaussianInstance(
                instance_id=7,
                asset=_asset(asset_id="ball-red", role="movable"),
                scene_from_asset=SimilarityTransform(
                    scale=0.25,
                    rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                    translation=(1.0, 2.0, 3.0),
                ),
            ),
        ),
        renderer_backend="nht-gsplat",
        renderer_commit="1" * 40,
    )


def test_composition_round_trip_is_fingerprinted_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    composition = _composition()
    path = tmp_path / "composition.json"

    write_gaussian_scene_manifest(path, composition)
    loaded = load_gaussian_scene_manifest(path)

    assert loaded == composition
    assert loaded.composition_fingerprint == composition.composition_fingerprint
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_gaussian_scene_manifest(path, composition)


def test_composition_rejects_independent_nht_appearance_spaces() -> None:
    composition = _composition()
    mismatched_asset = _asset(
        asset_id="ball-independent",
        role="movable",
        appearance_digest="f" * 64,
    )

    with pytest.raises(ValueError, match="share one exact appearance space"):
        GaussianSceneComposition.create(
            composition_id="mismatched-appearance",
            scene_source=composition.scene_source,
            background=composition.background,
            instances=(
                GaussianInstance(
                    instance_id=1,
                    asset=mismatched_asset,
                    scene_from_asset=composition.instances[0].scene_from_asset,
                ),
            ),
            renderer_backend=composition.renderer_backend,
            renderer_commit=composition.renderer_commit,
        )


def test_composition_fingerprint_is_independent_of_publication_uri() -> None:
    composition = _composition()
    relocated_background = replace(
        composition.background,
        tensors=replace(
            composition.background.tensors,
            uri="artifact://relocated/background",
        ),
        appearance_payload=replace(
            composition.background.appearance_payload,
            uri="artifact://relocated/appearance",
        ),
    )
    original_instance = composition.instances[0]
    relocated_instance = replace(
        original_instance,
        asset=replace(
            original_instance.asset,
            tensors=replace(
                original_instance.asset.tensors,
                uri="artifact://relocated/asset",
            ),
            appearance_payload=replace(
                original_instance.asset.appearance_payload,
                uri="artifact://relocated/appearance",
            ),
        ),
    )

    relocated = GaussianSceneComposition.create(
        composition_id=composition.composition_id,
        scene_source=replace(
            composition.scene_source,
            uri="artifact://relocated/scene",
        ),
        background=relocated_background,
        instances=(relocated_instance,),
        renderer_backend=composition.renderer_backend,
        renderer_commit=composition.renderer_commit,
    )

    assert relocated.composition_fingerprint == composition.composition_fingerprint


def test_asset_contract_requires_metric_movable_coordinates() -> None:
    movable = _asset(asset_id="ball", role="movable")

    with pytest.raises(ValueError, match="expressed directly in metres"):
        replace(movable, metres_per_unit=0.01)


def test_composition_parser_rejects_unknown_fields() -> None:
    payload = _composition().to_dict()
    payload["silent_fallback"] = True

    with pytest.raises(ValueError, match="extra=.*silent_fallback"):
        GaussianSceneComposition.from_dict(payload)
