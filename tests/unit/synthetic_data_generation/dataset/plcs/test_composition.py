"""Tests for explicit renderer-ready PLCS avatar appearance."""

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
    GaussianSceneObject,
    GaussianTransform,
)
from src.synthetic_data_generation.composition.gaussians import (
    GaussianTensorSet,
    assign_instance_id,
    compose_foreground_frame_gaussians,
)
from src.synthetic_data_generation.dataset.plcs.composition import (
    AvatarAppearance,
    compose_prevalidated_frame_gaussians,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def test_avatar_appearance_requires_explicit_linear_rgb() -> None:
    appearance = AvatarAppearance(
        features=torch.tensor(((0.1, 0.2, 0.3),), dtype=torch.float32),
        appearance_model="rgb",
        appearance_space="linear_rgb",
    )

    assert appearance.features.shape == (1, 3)
    with pytest.raises(ValueError, match=r"shape \[N,3\]"):
        AvatarAppearance(
            features=torch.zeros((1, 4), dtype=torch.float32),
            appearance_model="rgb",
            appearance_space="linear_rgb",
        )
    with pytest.raises(ValueError, match="appearance_model"):
        AvatarAppearance(
            features=torch.zeros((1, 3), dtype=torch.float32),
            appearance_model="spherical-harmonics",
            appearance_space="linear_rgb",
        )
    with pytest.raises(ValueError, match="unit range"):
        AvatarAppearance(
            features=torch.tensor(((1.1, 0.0, 0.0),), dtype=torch.float32),
            appearance_model="rgb",
            appearance_space="linear_rgb",
        )


def test_fused_frame_composition_is_exactly_generic_composition_equivalent() -> None:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    matrix[:3, 3] = (2.0, -1.0, 0.5)
    transform = GaussianTransform(
        scale=1.25,
        rigid=RigidTransform.from_matrix(matrix),
    )
    asset = GaussianAsset(
        asset_id="avatar-001",
        asset_class="smplh-player",
        role=GaussianAssetRole.MOVABLE,
        coordinates=GaussianCoordinates.asset_local_metres(),
        gaussian_count=2,
        feature_dim=3,
        floating_dtype="float32",
        appearance_model="rgb",
        appearance_space="linear_rgb",
    )
    composition = GaussianForegroundComposition(
        scene_id="B00",
        composition_id="B00-plcs",
        assets=(asset,),
        objects=(
            GaussianSceneObject(
                object_id="player-001",
                instance_id=4,
                asset_id=asset.asset_id,
                deformation_kind=GaussianDeformationKind.ARTICULATED,
            ),
        ),
        frames=tuple(
            GaussianFrame(
                frame_index=frame_index,
                instances=(
                    GaussianInstance(
                        object_id="player-001",
                        source_frame_index=frame_index,
                        scene_from_asset=transform,
                    ),
                ),
            )
            for frame_index in range(2)
        ),
    )
    local = GaussianTensorSet(
        means=torch.tensor(((0.1, 0.2, 0.3), (-0.2, 0.4, 0.1))),
        quaternions_wxyz=torch.tensor(((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))),
        log_scales=torch.log(torch.full((2, 3), 0.05)),
        opacity_logits=torch.tensor((2.0, 3.0)),
        features=torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        instance_ids=torch.zeros(2, dtype=torch.int64),
        coordinates=GaussianCoordinates.asset_local_metres(),
        appearance_model="rgb",
        appearance_space="linear_rgb",
    )

    expected = compose_foreground_frame_gaussians(
        composition,
        frame_index=0,
        object_tensors={"player-001": assign_instance_id(local, 4)},
    )
    actual = compose_prevalidated_frame_gaussians(
        composition,
        frame_index=0,
        object_tensors={"player-001": local},
    )

    for name in (
        "means",
        "quaternions_wxyz",
        "log_scales",
        "opacity_logits",
        "features",
        "instance_ids",
    ):
        torch.testing.assert_close(
            getattr(actual, name),
            getattr(expected, name),
            rtol=0.0,
            atol=0.0,
        )
    assert actual.coordinates == expected.coordinates
