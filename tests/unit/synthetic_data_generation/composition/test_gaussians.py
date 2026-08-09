"""Tests for Gaussian geometry, composition, and deformation semantics."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.composition.contracts import (
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
from src.synthetic_data_generation.composition.gaussians import (
    GaussianTensorSet,
    assign_instance_id,
    compose_foreground_frame_gaussians,
    compose_frame_gaussians,
    compose_gaussians,
    gaussian_covariances,
    transform_gaussians,
    validate_articulated_deformation,
    validate_asset_tensors,
)
from src.synthetic_data_generation.scene_contract import RigidTransform

APPEARANCE_MODEL = "nht-deferred"
APPEARANCE_SPACE = "b00-deferred-space"


def _gaussians(
    *,
    coordinates: GaussianCoordinates,
    instance_id: int = 0,
    dtype: torch.dtype = torch.float64,
) -> GaussianTensorSet:
    return GaussianTensorSet(
        means=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=dtype,
        ),
        quaternions_wxyz=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 4,
            dtype=dtype,
        ),
        log_scales=torch.log(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [0.5, 1.0, 2.0],
                    [0.25, 0.5, 1.0],
                    [0.75, 1.25, 1.5],
                ],
                dtype=dtype,
            )
        ),
        opacity_logits=torch.arange(4, dtype=dtype),
        features=torch.arange(16, dtype=dtype).reshape(4, 4),
        instance_ids=torch.full((4,), instance_id, dtype=torch.int64),
        coordinates=coordinates,
        appearance_model=APPEARANCE_MODEL,
        appearance_space=APPEARANCE_SPACE,
    )


def _asset(
    *,
    asset_id: str,
    role: GaussianAssetRole,
    dtype: str = "float64",
) -> GaussianAsset:
    return GaussianAsset(
        asset_id=asset_id,
        asset_class="court" if role == GaussianAssetRole.BACKGROUND else "player",
        role=role,
        coordinates=(
            GaussianCoordinates.scene()
            if role == GaussianAssetRole.BACKGROUND
            else GaussianCoordinates.asset_local_metres()
        ),
        gaussian_count=4,
        feature_dim=4,
        floating_dtype=dtype,  # type: ignore[arg-type]
        appearance_model=APPEARANCE_MODEL,
        appearance_space=APPEARANCE_SPACE,
    )


def _rotation_z_90_transform(*, scale: float = 2.0) -> GaussianTransform:
    matrix = np.asarray(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return GaussianTransform(
        scale=scale,
        rigid=RigidTransform.from_matrix(matrix),
    )


def _composition(kind: GaussianDeformationKind) -> GaussianSceneComposition:
    transform = GaussianTransform.identity()
    return GaussianSceneComposition(
        scene_id="b00",
        composition_id="plcs-test",
        background=_asset(asset_id="background", role=GaussianAssetRole.BACKGROUND),
        assets=(_asset(asset_id="avatar", role=GaussianAssetRole.MOVABLE),),
        objects=(
            GaussianSceneObject(
                object_id="player-1",
                instance_id=9,
                asset_id="avatar",
                deformation_kind=kind,
            ),
        ),
        frames=(
            GaussianFrame(
                frame_index=0,
                instances=(GaussianInstance("player-1", 0, transform),),
            ),
            GaussianFrame(
                frame_index=1,
                instances=(GaussianInstance("player-1", 1, transform),),
            ),
        ),
    )


def test_transform_preserves_anisotropic_covariance_geometry() -> None:
    source = assign_instance_id(
        _gaussians(coordinates=GaussianCoordinates.asset_local_metres()),
        3,
    )
    transform = _rotation_z_90_transform()

    transformed = transform_gaussians(source, transform)

    torch.testing.assert_close(
        transformed.means[:2],
        torch.tensor(
            [[10.0, 1.0, 0.5], [6.0, -1.0, 0.5]],
            dtype=torch.float64,
        ),
    )
    expected_quaternion = torch.tensor(
        [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
        dtype=torch.float64,
    ).expand(4, -1)
    torch.testing.assert_close(transformed.quaternions_wxyz, expected_quaternion)
    torch.testing.assert_close(
        transformed.log_scales,
        source.log_scales + math.log(2.0),
    )
    rotation = torch.tensor(transform.rotation, dtype=torch.float64).reshape(3, 3)
    expected_covariances = 4.0 * rotation @ gaussian_covariances(source) @ rotation.T
    torch.testing.assert_close(
        gaussian_covariances(transformed),
        expected_covariances,
    )
    assert transformed.coordinates == GaussianCoordinates.scene()
    assert transformed.means.dtype == source.means.dtype
    torch.testing.assert_close(transformed.features, source.features)
    torch.testing.assert_close(transformed.instance_ids, source.instance_ids)


def test_asset_tensor_validation_checks_shape_dtype_and_coordinates() -> None:
    asset = _asset(asset_id="avatar", role=GaussianAssetRole.MOVABLE)
    tensors = _gaussians(coordinates=GaussianCoordinates.asset_local_metres())

    validate_asset_tensors(asset, tensors)

    with pytest.raises(ValueError, match="gaussian_count"):
        validate_asset_tensors(replace(asset, gaussian_count=5), tensors)
    with pytest.raises(ValueError, match="coordinate convention"):
        validate_asset_tensors(
            _asset(asset_id="background", role=GaussianAssetRole.BACKGROUND),
            tensors,
        )


def test_compose_frame_uses_stable_unique_identity_and_exact_objects() -> None:
    composition = _composition(GaussianDeformationKind.ARTICULATED)
    background = _gaussians(coordinates=GaussianCoordinates.scene())
    local = _gaussians(coordinates=GaussianCoordinates.asset_local_metres())

    composed = compose_frame_gaussians(
        composition,
        frame_index=0,
        background_tensors=background,
        object_tensors={"player-1": local},
    )

    assert composed.gaussian_count == 8
    assert composed.instance_ids.tolist() == [0, 0, 0, 0, 9, 9, 9, 9]
    with pytest.raises(ValueError, match="tensor objects differ"):
        compose_frame_gaussians(
            composition,
            frame_index=0,
            background_tensors=background,
            object_tensors={},
        )


def test_compose_supports_background_only_and_rejects_duplicate_ids() -> None:
    background = _gaussians(coordinates=GaussianCoordinates.scene())
    local = _gaussians(coordinates=GaussianCoordinates.asset_local_metres())
    first = transform_gaussians(assign_instance_id(local, 1), GaussianTransform.identity())

    background_only = compose_gaussians(background, ())
    assert background_only.gaussian_count == background.gaussian_count
    with pytest.raises(ValueError, match="Duplicate movable instance ids"):
        compose_gaussians(background, (first, first))


def test_foreground_frame_composes_multiple_positive_identity_assets() -> None:
    asset = _asset(asset_id="avatar", role=GaussianAssetRole.MOVABLE)
    transform = GaussianTransform.identity()
    composition = GaussianForegroundComposition(
        scene_id="b00",
        composition_id="foreground-test",
        assets=(asset,),
        objects=(
            GaussianSceneObject("player-1", 9, asset.asset_id, GaussianDeformationKind.ARTICULATED),
            GaussianSceneObject("player-2", 10, asset.asset_id, GaussianDeformationKind.ARTICULATED),
        ),
        frames=(
            GaussianFrame(
                0,
                (
                    GaussianInstance("player-1", 0, transform),
                    GaussianInstance("player-2", 4, transform),
                ),
            ),
            GaussianFrame(
                1,
                (
                    GaussianInstance("player-1", 1, transform),
                    GaussianInstance("player-2", 5, transform),
                ),
            ),
        ),
    )
    local = _gaussians(coordinates=GaussianCoordinates.asset_local_metres())

    foreground = compose_foreground_frame_gaussians(
        composition,
        frame_index=0,
        object_tensors={
            "player-1": assign_instance_id(local, 9),
            "player-2": assign_instance_id(local, 10),
        },
    )

    assert foreground.gaussian_count == 8
    assert set(foreground.instance_ids.tolist()) == {9, 10}
    assert 0 not in foreground.instance_ids.tolist()
    with pytest.raises(ValueError, match="must carry only instance_id 10"):
        compose_foreground_frame_gaussians(
            composition,
            frame_index=0,
            object_tensors={
                "player-1": assign_instance_id(local, 9),
                "player-2": assign_instance_id(local, 11),
            },
        )


def test_tensor_set_rejects_nonfinite_unnormalized_and_overflowing_scale() -> None:
    source = _gaussians(coordinates=GaussianCoordinates.asset_local_metres())
    non_finite = source.means.clone()
    non_finite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="only finite"):
        replace(source, means=non_finite)

    bad_quaternion = source.quaternions_wxyz.clone()
    bad_quaternion[0] *= 2.0
    with pytest.raises(ValueError, match="normalized in wxyz order"):
        replace(source, quaternions_wxyz=bad_quaternion)

    overflowing = source.log_scales.clone()
    overflowing[0, 0] = 1000.0
    with pytest.raises(ValueError, match="finite and strictly positive"):
        replace(source, log_scales=overflowing)

    with pytest.raises(TypeError, match="same dtype"):
        replace(source, features=source.features.float())


def test_articulated_validation_rejects_rigid_only_frames() -> None:
    composition = _composition(GaussianDeformationKind.ARTICULATED)
    first = _gaussians(coordinates=GaussianCoordinates.asset_local_metres())
    rigid_scene = transform_gaussians(
        first,
        _rotation_z_90_transform(scale=1.0),
    )
    rigid_local = replace(
        rigid_scene,
        coordinates=GaussianCoordinates.asset_local_metres(),
    )

    with pytest.raises(ValueError, match="rigid-only across all frames"):
        validate_articulated_deformation(
            composition,
            object_id="player-1",
            frame_tensors={0: first, 1: rigid_local},
        )


def test_articulated_validation_detects_local_geometry_change() -> None:
    composition = _composition(GaussianDeformationKind.ARTICULATED)
    first = _gaussians(coordinates=GaussianCoordinates.asset_local_metres())
    changed_means = first.means.clone()
    changed_means[0, 0] += 0.25
    articulated = replace(first, means=changed_means)

    report = validate_articulated_deformation(
        composition,
        object_id="player-1",
        frame_tensors={0: first, 1: articulated},
    )

    assert report.object_id == "player-1"
    assert report.frame_count == 2
    assert report.deformed_frame_indices == (1,)
    assert report.max_mean_residual > 1.0e-5
