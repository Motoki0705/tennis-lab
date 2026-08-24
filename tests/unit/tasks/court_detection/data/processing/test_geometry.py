"""Calibration-aware pose-safe Court processing geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.tasks.court_detection.configuration import CourtAugmentationConfig
from src.tasks.court_detection.data.contracts import (
    CourtKeypointChannels,
    CourtPoseAuthority,
    CourtRawSample,
    CourtSampleMetadata,
)
from src.tasks.court_detection.data.processing.geometry import CourtProcessingGeometry
from src.tasks.court_detection.geometry.pose import (
    build_pose_target,
    canonical_semantic_court_points,
    project_canonical_points,
    validate_projection_round_trip,
)

_CONFIG_DIR = Path(__file__).resolve().parents[6] / "src/tasks/court_detection/configs"


def _authority() -> CourtPoseAuthority:
    center = np.asarray((0.0, -30.0, 12.0), dtype=np.float64)
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((right, down, forward))
    transform[:3, 3] = center
    return CourtPoseAuthority(
        source_schema="canonical_court_dataset_v3",
        camera=SceneCamera(
            camera_id="camera",
            source_frame_index=0,
            width=640,
            height=480,
            intrinsics=(500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0),
            camera_to_scene=RigidTransform.from_matrix(transform),
            image_path="generated/camera.png",
        ),
        target_court=TargetCourtBinding(
            court_instance_id="court",
            candidate_id="candidate",
            scene_from_court=RigidTransform.identity(),
            selection_seed=779,
        ),
    )


def _pose_safe_config():
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "data/source=synthetic_court",
                "data.source.keypoint_court_scope=target_court",
                "data/processing=kp",
                "data/augmentation=pose_safe",
                "loss=query_pose",
                "model=query_encoder",
            ],
        )
    return CourtAugmentationConfig.from_mapping(config.data.augmentation)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transform_points_preserves_floating_input_dtype(dtype: torch.dtype) -> None:
    points = torch.tensor([[[1.25, 2.5], [3.75, 4.0]]], dtype=dtype)
    matrix = torch.tensor(
        [[2.0, 0.0, 3.0], [0.0, 0.5, -1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    transformed = CourtProcessingGeometry._transform_points(points, matrix)

    assert transformed.dtype == dtype
    torch.testing.assert_close(
        transformed,
        torch.tensor([[[5.5, 0.25], [10.5, 1.0]]], dtype=dtype),
    )


def test_transform_points_preserves_001588_large_coordinate_precision() -> None:
    points = torch.tensor(
        [[[387122.0463825724, 48177.58336823048]]],
        dtype=torch.float64,
    )
    scale = 256.0 / 959.0
    offset_y = (256.0 - 539.0 * scale) * 0.5
    matrix = torch.tensor(
        [[scale, 0.0, 0.0], [0.0, scale, offset_y], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    transformed = CourtProcessingGeometry._transform_points(points, matrix)

    torch.testing.assert_close(
        transformed[0, 0],
        torch.tensor(
            [103340.19173507667, 12916.810575878],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1.0e-9,
    )
    rounded_source = CourtProcessingGeometry._transform_points(points.float(), matrix)
    assert float(torch.max(torch.abs(transformed - rounded_source.double()))) > 4.0e-3


@pytest.mark.parametrize(
    ("points", "matrix", "error", "message"),
    [
        (
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.eye(3),
            TypeError,
            "floating dtype",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.eye(3),
            ValueError,
            "shape",
        ),
        (
            torch.tensor([[1.0, 2.0]]),
            torch.eye(2),
            ValueError,
            "matrix",
        ),
        (
            torch.tensor([[1.0, 2.0]]),
            torch.eye(3, device="meta"),
            ValueError,
            "device",
        ),
    ],
)
def test_transform_points_rejects_invalid_contracts(
    points: torch.Tensor,
    matrix: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        CourtProcessingGeometry._transform_points(points, matrix)


def test_pose_safe_geometry_letterboxes_kp_and_k_with_one_isotropic_plan() -> None:
    authority = _authority()
    source_target = build_pose_target(authority)
    points = project_canonical_points(
        source_target,
        canonical_semantic_court_points(source_target),
    ).float()
    raw = CourtRawSample(
        sample_id="sample",
        image=Image.new("RGB", (640, 480)),
        keypoint_channels=CourtKeypointChannels(
            channel_names=tuple(f"kp_{index}" for index in range(14)),
            points_xy=points.unsqueeze(1),
            point_visible=torch.ones(14, 1, dtype=torch.bool),
            physical_indices=torch.arange(14).view(14, 1),
            horizontal_flip_permutation=tuple(range(14)),
        ),
        court_instances=(),
        dense_target_refs={},
        metadata=CourtSampleMetadata(
            source_kind="synthetic_court",
            source_schema="canonical_court_dataset_v3",
            source_sample_id="sample",
            scene_id="scene",
            provenance={},
        ),
        pose_authority=authority,
    )
    geometry = CourtProcessingGeometry(
        _pose_safe_config(),
        is_train=True,
        require_pose=True,
    )

    plan = geometry.sample(raw)
    transformed = geometry.apply(raw, dense_targets={}, plan=plan)

    assert plan.output_size_hw == (256, 256)
    torch.testing.assert_close(plan.matrix[0, 0], plan.matrix[1, 1])
    assert transformed.image_tensor.shape == (3, 256, 256)
    assert transformed.pose_target is not None
    assert transformed.keypoint_channels is not None
    target = transformed.pose_target
    assert float(target.intrinsics[0, 0]) == 200.0
    assert float(target.intrinsics[1, 1]) == 200.0
    assert float(target.intrinsics[1, 2]) == pytest.approx(127.8, abs=1.0e-5)
    validate_projection_round_trip(
        target,
        transformed.keypoint_channels.points_xy[:, 0],
    )
