"""Tests for explicit renderer-ready PLCS avatar appearance."""

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

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
from src.synthetic_data_generation.dataset.camera_profiles import (
    SampledCamera,
    SampledCameraRig,
)
from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.plcs.composition import (
    AvatarAppearance,
    PLCSAvatarFrameTensors,
    compose_prevalidated_frame_gaussians,
    transform_asset_points,
)
from src.synthetic_data_generation.dataset.plcs.coordinates import (
    PLCSSourceSupportPlane,
)
from src.synthetic_data_generation.dataset.plcs.handler import (
    _empty_supervision,
    _write_frame_supervision,
)
from src.synthetic_data_generation.dataset.plcs.production import PLCSProductionMode
from src.synthetic_data_generation.dataset.plcs.timeline import (
    PLCSObjectTrack,
    build_global_timeline,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.tasks.plcs.generate_dataset.sampling.motion_source import PLCSMotionClip


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


def test_rgb_gaussian_and_coco17_supervision_share_independently_verified_geometry(
    tmp_path: Path,
) -> None:
    yaw = np.pi / 2.0
    clip = PLCSMotionClip.from_amass_arrays(
        source_path=tmp_path / "coincident-point.npz",
        category="general",
        gender="neutral",
        fps=30.0,
        poses=np.zeros((2, 156), dtype=np.float64),
        trans=np.zeros((2, 3), dtype=np.float64),
        betas=np.zeros(16, dtype=np.float64),
    )
    track = PLCSObjectTrack(
        object_id="player-001",
        instance_id=4,
        asset_id="avatar-001",
        clip=clip,
        support_plane=PLCSSourceSupportPlane.from_surface_minimum(
            initial_root_translation_z_m=0.0,
            support_local_z_m=-4.0,
        ),
        start_frame=0,
        anchor_position_court_m=(0.0, 0.0, 0.0),
        yaw_radians=yaw,
    )
    timeline = build_global_timeline(
        scene_id="B00",
        production_mode=PLCSProductionMode.SINGLE_OBJECT,
        target_court=TargetCourtBinding(
            court_instance_id="court-001",
            candidate_id="candidate-001",
            scene_from_court=RigidTransform.identity(),
            selection_seed=7,
        ),
        tracks=(track,),
    )
    transform = timeline.frames[0].entries[0].scene_from_asset
    assert transform is not None
    local_point = torch.tensor(((0.1, 0.2, 1.0),), dtype=torch.float32)
    local_gaussians = GaussianTensorSet(
        means=local_point,
        quaternions_wxyz=torch.tensor(((1.0, 0.0, 0.0, 0.0),)),
        log_scales=torch.log(torch.full((1, 3), 0.05)),
        opacity_logits=torch.tensor((2.0,)),
        features=torch.tensor(((1.0, 0.0, 0.0),)),
        instance_ids=torch.zeros(1, dtype=torch.int64),
        coordinates=GaussianCoordinates.asset_local_metres(),
        appearance_model="rgb",
        appearance_space="linear_rgb",
    )
    asset = GaussianAsset(
        asset_id=track.asset_id,
        asset_class="smplh-player",
        role=GaussianAssetRole.MOVABLE,
        coordinates=GaussianCoordinates.asset_local_metres(),
        gaussian_count=1,
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
                object_id=track.object_id,
                instance_id=track.instance_id,
                asset_id=track.asset_id,
                deformation_kind=GaussianDeformationKind.ARTICULATED,
            ),
        ),
        frames=tuple(
            GaussianFrame(
                frame_index=frame_index,
                instances=(
                    GaussianInstance(
                        object_id=track.object_id,
                        source_frame_index=frame_index,
                        scene_from_asset=transform,
                    ),
                ),
            )
            for frame_index in range(2)
        ),
    )
    composed = compose_prevalidated_frame_gaussians(
        composition,
        frame_index=0,
        object_tensors={track.object_id: local_gaussians},
    )

    joints = torch.zeros((52, 3), dtype=torch.float32)
    joints[:] = local_point[0]
    frame_tensors = {
        track.object_id: {
            0: PLCSAvatarFrameTensors(gaussians=local_gaussians, joints_m=joints)
        }
    }
    camera = SceneCamera(
        camera_id="camera-0",
        source_frame_index=0,
        width=200,
        height=200,
        intrinsics=(100.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path="request-only",
    )
    rig = SampledCameraRig(
        profile="contract-test",
        seed=7,
        court_instance_id="court-001",
        cameras=(
            SampledCamera(
                slot_id="camera-0",
                court_local_center_m=(0.0, 0.0, 0.0),
                court_local_look_at_m=(0.0, 0.0, 1.0),
                hfov_degrees=90.0,
                scene_camera=camera,
            ),
        ),
    )
    output = _empty_supervision(frame_count=1, camera_count=1, object_count=1)
    court_points: NDArray[np.float64] = np.zeros((20, 3), dtype=np.float64)
    court_points[:, 2] = 10.0
    _write_frame_supervision(
        output,
        timeline=timeline,
        rig=rig,
        frame_index=0,
        frame_tensors=frame_tensors,
        court_points_court_m=court_points,
    )

    expected_scene = np.asarray((-0.2, 0.1, 5.0), dtype=np.float32)
    np.testing.assert_allclose(
        transform_asset_points(local_point, transform=transform).numpy()[0],
        expected_scene,
        atol=1.0e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        composed.means.numpy()[0], expected_scene, atol=1.0e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        output.human_kp_3d[0, 0, 5],
        expected_scene,
        atol=1.0e-6,
        rtol=0.0,
    )
    expected_normalized_uv = np.asarray((0.48, 0.51), dtype=np.float32)
    assert output.human_vis[0, 0, 0, 5]
    np.testing.assert_allclose(
        output.human_kp[0, 0, 0, 5],
        expected_normalized_uv,
        atol=1.0e-6,
        rtol=0.0,
    )
