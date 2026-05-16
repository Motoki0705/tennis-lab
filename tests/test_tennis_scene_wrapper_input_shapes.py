from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from src.tennis_scene.pipeline.components.blcs import BLCSConfig, BLCSModule
from src.tennis_scene.pipeline.components.input_shapes import (
    normalize_court_keypoint_sequence,
)
from src.tennis_scene.pipeline.components.plcs import PLCSModule


def _make_float_array(shape: tuple[int, ...]) -> np.ndarray:
    values = np.linspace(0.05, 0.95, num=int(np.prod(shape)), dtype=np.float32)
    return values.reshape(shape)


def _assert_tensor_shape(value: object, shape: tuple[int, ...]) -> torch.Tensor:
    assert isinstance(value, torch.Tensor)
    assert tuple(value.shape) == shape
    return value


class _BLCSStubPredictor:
    def __init__(self, position: torch.Tensor, *, num_court_tokens: int = 20) -> None:
        self.model = SimpleNamespace(num_court_tokens=num_court_tokens)
        self.position = position
        self.last_call: dict[str, object] | None = None

    def predict(self, **kwargs: object) -> dict[str, torch.Tensor]:
        self.last_call = kwargs
        return {"position": self.position.clone()}


class _PLCSStubPredictor:
    def __init__(
        self,
        position_meters: torch.Tensor,
        yaw_radians: torch.Tensor,
        *,
        num_court_tokens: int = 20,
    ) -> None:
        self.model = SimpleNamespace(num_court_tokens=num_court_tokens)
        self.position_meters = position_meters
        self.yaw_radians = yaw_radians
        self.last_call: dict[str, object] | None = None

    def predict(self, **kwargs: object) -> dict[str, torch.Tensor]:
        self.last_call = kwargs
        return {
            "position_meters": self.position_meters.clone(),
            "yaw_radians": self.yaw_radians.clone(),
            "canonical_pose": torch.ones(1, dtype=torch.float32),
        }


def test_normalize_court_keypoint_sequence_pads_k14_visibility_to_zero() -> None:
    court_kp = _make_float_array((3, 14, 2))
    court_vis = np.ones((3, 14), dtype=np.float32)

    normalized_kp, normalized_vis = normalize_court_keypoint_sequence(
        court_kp=court_kp,
        court_vis=court_vis,
        target_batch_size=1,
        target_num_cameras=1,
        target_num_frames=3,
        target_num_keypoints=20,
    )

    assert normalized_kp.shape == (1, 1, 3, 20, 2)
    assert normalized_vis.shape == (1, 1, 3, 20)
    np.testing.assert_allclose(normalized_kp[0, 0, :, :14], court_kp)
    np.testing.assert_allclose(normalized_vis[0, 0, :, :14], court_vis)
    np.testing.assert_array_equal(normalized_kp[0, 0, :, 14:], 0.0)
    np.testing.assert_array_equal(normalized_vis[0, 0, :, 14:], 0.0)


def test_blcs_process_normalizes_single_camera_per_frame_inputs() -> None:
    num_frames = 3
    predictor = _BLCSStubPredictor(
        torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
            dtype=torch.float32,
        )
    )
    module = BLCSModule(BLCSConfig(checkpoint_path="dummy.ckpt", device="cpu"))
    module._predictor = predictor

    ball_uv = _make_float_array((num_frames, 2))
    court_kp = _make_float_array((num_frames, 14, 2))
    ball_vis = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    result = module.process(ball_uv=ball_uv, court_kp=court_kp, ball_vis=ball_vis)

    is_valid, errors = result.validate()
    assert is_valid, errors
    np.testing.assert_array_equal(result.visibility, np.array([True, False, True]))
    np.testing.assert_array_equal(result.ball_3d[1], np.zeros(3, dtype=np.float32))

    assert predictor.last_call is not None
    ball_uv_t = _assert_tensor_shape(predictor.last_call["ball_uv"], (1, 1, 3, 2))
    court_kp_t = _assert_tensor_shape(
        predictor.last_call["court_kp"],
        (1, 1, 3, 20, 2),
    )
    ball_vis_t = _assert_tensor_shape(predictor.last_call["ball_vis"], (1, 1, 3))
    ball_mask_t = _assert_tensor_shape(predictor.last_call["ball_mask"], (1, 1, 3))
    court_vis_t = _assert_tensor_shape(
        predictor.last_call["court_vis"],
        (1, 1, 3, 20),
    )
    assert predictor.last_call["denormalize"] is True
    np.testing.assert_allclose(ball_uv_t.numpy()[0, 0], ball_uv)
    np.testing.assert_allclose(court_kp_t.numpy()[0, 0, :, :14], court_kp)
    np.testing.assert_allclose(ball_vis_t.numpy()[0, 0], ball_vis)
    np.testing.assert_array_equal(ball_mask_t.numpy(), np.ones((1, 1, 3), dtype=np.float32))
    np.testing.assert_array_equal(court_vis_t.numpy()[0, 0, :, 14:], 0.0)


def test_blcs_process_normalizes_multicamera_fixed_court_inputs() -> None:
    num_cameras = 2
    num_frames = 3
    predictor = _BLCSStubPredictor(
        torch.ones((1, num_frames, 3), dtype=torch.float32)
    )
    module = BLCSModule(BLCSConfig(checkpoint_path="dummy.ckpt", device="cpu"))
    module._predictor = predictor

    ball_uv = _make_float_array((num_cameras, num_frames, 2))
    court_kp = _make_float_array((14, 2))

    result = module.process(ball_uv=ball_uv, court_kp=court_kp)

    is_valid, errors = result.validate()
    assert is_valid, errors
    assert result.ball_3d.shape == (num_frames, 3)

    assert predictor.last_call is not None
    ball_uv_t = _assert_tensor_shape(
        predictor.last_call["ball_uv"],
        (1, num_cameras, num_frames, 2),
    )
    court_kp_t = _assert_tensor_shape(
        predictor.last_call["court_kp"],
        (1, num_cameras, num_frames, 20, 2),
    )
    court_vis_t = _assert_tensor_shape(
        predictor.last_call["court_vis"],
        (1, num_cameras, num_frames, 20),
    )
    np.testing.assert_allclose(ball_uv_t.numpy()[0], ball_uv)
    np.testing.assert_allclose(court_kp_t.numpy()[0, 0, 0, :14], court_kp)
    np.testing.assert_allclose(court_kp_t.numpy()[0, 1, 2, :14], court_kp)
    np.testing.assert_array_equal(court_vis_t.numpy()[0, :, :, 14:], 0.0)


def test_plcs_process_normalizes_single_camera_per_frame_inputs() -> None:
    num_players = 2
    num_frames = 3
    predictor = _PLCSStubPredictor(
        position_meters=torch.ones((num_players, num_frames, 3), dtype=torch.float32),
        yaw_radians=torch.zeros((num_players, num_frames), dtype=torch.float32),
    )
    module = PLCSModule(checkpoint_path="dummy.ckpt", device="cpu")
    module._predictor = predictor

    human_kp = _make_float_array((num_players, num_frames, 17, 2))
    court_kp = _make_float_array((num_frames, 14, 2))
    track_ids = np.array([10, 11], dtype=np.int32)

    result = module.process(
        human_kp_2d=human_kp,
        court_kp=court_kp,
        track_ids=track_ids,
    )

    is_valid, errors = result.validate()
    assert is_valid, errors
    np.testing.assert_array_equal(result.track_ids, track_ids)

    assert predictor.last_call is not None
    human_kp_t = _assert_tensor_shape(
        predictor.last_call["human_kp"],
        (num_players, 1, num_frames, 17, 2),
    )
    human_vis_t = _assert_tensor_shape(
        predictor.last_call["human_vis"],
        (num_players, 1, num_frames, 17),
    )
    human_mask_t = _assert_tensor_shape(
        predictor.last_call["human_mask"],
        (num_players, 1, num_frames),
    )
    court_kp_t = _assert_tensor_shape(
        predictor.last_call["court_kp"],
        (num_players, 1, num_frames, 20, 2),
    )
    court_vis_t = _assert_tensor_shape(
        predictor.last_call["court_vis"],
        (num_players, 1, num_frames, 20),
    )
    assert predictor.last_call["denormalize"] is True
    np.testing.assert_allclose(human_kp_t.numpy()[:, 0], human_kp)
    np.testing.assert_array_equal(
        human_vis_t.numpy(),
        np.ones((num_players, 1, num_frames, 17), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        human_mask_t.numpy(),
        np.ones((num_players, 1, num_frames), dtype=np.float32),
    )
    np.testing.assert_allclose(
        court_kp_t.numpy()[:, 0, :, :14],
        np.broadcast_to(court_kp, (num_players, num_frames, 14, 2)),
    )
    np.testing.assert_array_equal(court_vis_t.numpy()[:, 0, :, 14:], 0.0)


def test_plcs_process_normalizes_multicamera_fixed_court_inputs() -> None:
    num_players = 2
    num_cameras = 2
    num_frames = 3
    predictor = _PLCSStubPredictor(
        position_meters=torch.full(
            (num_players, num_frames, 3),
            fill_value=2.0,
            dtype=torch.float32,
        ),
        yaw_radians=torch.full(
            (num_players, num_frames),
            fill_value=0.5,
            dtype=torch.float32,
        ),
    )
    module = PLCSModule(checkpoint_path="dummy.ckpt", device="cpu")
    module._predictor = predictor

    human_kp = _make_float_array((num_players, num_cameras, num_frames, 17, 2))
    court_kp = _make_float_array((num_cameras, 14, 2))
    court_vis = np.array(
        [
            [1.0] * 14,
            [0.0, 1.0] * 7,
        ],
        dtype=np.float32,
    )

    result = module.process(
        human_kp_2d=human_kp,
        court_kp=court_kp,
        court_vis=court_vis,
    )

    is_valid, errors = result.validate()
    assert is_valid, errors
    assert result.position.shape == (num_players, num_frames, 3)
    assert result.yaw.shape == (num_players, num_frames)

    assert predictor.last_call is not None
    human_kp_t = _assert_tensor_shape(
        predictor.last_call["human_kp"],
        (num_players, num_cameras, num_frames, 17, 2),
    )
    court_kp_t = _assert_tensor_shape(
        predictor.last_call["court_kp"],
        (num_players, num_cameras, num_frames, 20, 2),
    )
    court_vis_t = _assert_tensor_shape(
        predictor.last_call["court_vis"],
        (num_players, num_cameras, num_frames, 20),
    )
    np.testing.assert_allclose(human_kp_t.numpy(), human_kp)
    np.testing.assert_allclose(court_kp_t.numpy()[0, :, 0, :14], court_kp)
    np.testing.assert_allclose(court_kp_t.numpy()[1, :, 2, :14], court_kp)
    np.testing.assert_allclose(court_vis_t.numpy()[0, :, 1, :14], court_vis)
    np.testing.assert_array_equal(court_vis_t.numpy()[:, :, :, 14:], 0.0)