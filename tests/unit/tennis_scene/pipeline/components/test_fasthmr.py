"""FastHMR camera and native SMPL contracts checked without model assets."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.submodules.models.fasthmr.assets import FastHMRConfig
from src.submodules.models.fasthmr.inference import (
    FastHMR,
    FastHMRPrediction,
    camera_translation,
    fuse_flip_pose,
)
from src.tennis_scene.pipeline.body_types import BodyParameters
from src.tennis_scene.pipeline.components.fasthmr import native_smpl_parameters


def test_native_smpl_adapter_preserves_camera_root_and_hand_joints() -> None:
    expected: NDArray[np.float32] = np.zeros((2, 24, 3), np.float32)
    expected[0, 0] = [0.15, -0.42, 0.63]
    expected[1, 22:] = [[0.3, 0.1, -0.2], [-0.14, 0.32, 0.6]]
    matrices = Rotation.from_rotvec(expected.reshape(-1, 3)).as_matrix().reshape(2, 24, 3, 3).astype(np.float32)
    translation = np.array([[1, -2, 7], [-2, 1, 5]], np.float32)
    betas: NDArray[np.float32] = np.full((2, 10), 0.31, np.float32)
    prediction = FastHMRPrediction(matrices, betas, translation)
    body = native_smpl_parameters(prediction, 2)
    np.testing.assert_allclose(body.global_orient, expected[:, 0], atol=1e-6)
    np.testing.assert_allclose(body.body_pose.reshape(2, 23, 3), expected[:, 1:], atol=1e-6)
    np.testing.assert_array_equal(body.transl, translation)
    with pytest.raises(ValueError, match="body_pose"):
        BodyParameters(body.body_pose, body.global_orient, body.betas, body.transl)
    matrices[0, 0, 0] *= -1
    with pytest.raises(ValueError, match="proper rotations"):
        native_smpl_parameters(prediction, 2)


def test_crop_camera_and_horizontal_flip_keep_physical_axes() -> None:
    translation = camera_translation(
        torch.tensor([[2., -0.5, 0.5]]), torch.tensor([[500., 900., 200.]]), (1200, 1600),
    )
    torch.testing.assert_close(translation, torch.tensor([[-1., 1., 10.]]))
    predicted, flipped = torch.zeros((1, 133)), torch.zeros((1, 133))
    predicted[0, 87] = 0.75
    flipped[0, 3 * 1:3 * 2] = torch.tensor([2., 4., 6.])
    flipped[0, 87] = 1.5
    result = fuse_flip_pose(predicted, flipped)
    torch.testing.assert_close(result[0, 3 * 2:3 * 3], torch.tensor([-1., 2., 3.]))
    assert result[0, 87] == 0.75


def test_temporal_decoder_keeps_short_tail_and_flip_evaluations(tmp_path: Path) -> None:
    cfg = FastHMRConfig(tmp_path, tmp_path, tmp_path / "coco", "cpu", "pt32", 4, 10)
    fast = FastHMR(cfg)
    seen: list[tuple[int, float]] = []

    def diffusion(features: torch.Tensor) -> torch.Tensor:
        seen.append((features.shape[1], float(features[0, 0, 0])))
        return torch.zeros((1, features.shape[1], 133))

    class Body:
        def rotations(self, pose: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
            assert len(shape) == len(pose) and pose.dtype is torch.float32
            return torch.eye(3).expand(len(pose), 24, 3, 3)

    fast.models = (None, None, diffusion, None, None)
    # The released temporal decoder accepts both a full 243-frame window
    # and a variable-length tail. Each orientation must see the same samples.
    features = torch.arange(244, dtype=torch.float32).reshape(-1, 1)
    camera = torch.tensor([[2., -0.5, 0.5]]).repeat(244, 1)
    boxes = np.tile(np.array([500., 900., 200.], np.float32), (244, 1))
    body: Any = Body()
    prediction = fast._decode(features, features + 10, torch.zeros((1, 10)), camera, boxes, (1200, 1600), body)
    assert seen == [(243, 0.), (243, 10.), (1, 243.), (1, 253.)]
    assert prediction.rotations.shape == (244, 24, 3, 3)
    np.testing.assert_allclose(prediction.translation[[0, -1]], [[-1., 1., 10.], [-1., 1., 10.]])
