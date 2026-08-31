"""Contract tests for top-down ViTPose over one completed local track."""

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import src.submodules.models.vitpose.pose2d as pose2d_module
from src.submodules.configuration import ViTPoseHeadConfig
from src.submodules.models.tracker.common import TrackResult
from src.submodules.models.vitpose.pose2d import (
    Pose2DRequest,
    ViTPosePose2D,
)


class _FakePose(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_shapes: list[tuple[int, ...]] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.batch_shapes.append(tuple(images.shape))
        return torch.zeros(
            (images.shape[0], 17, 64, 48),
            dtype=images.dtype,
            device=images.device,
        )


def _head_config() -> ViTPoseHeadConfig:
    return ViTPoseHeadConfig(
        in_channels=8,
        out_channels=17,
        num_deconv_layers=0,
        num_deconv_filters=(),
        num_deconv_kernels=(),
        final_conv_kernel=1,
        num_conv_layers=0,
        num_conv_kernels=(),
    )


def test_vitpose_consumes_one_completed_track_and_returns_unidentified_coco17(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed_track = TrackResult(
        tracks={
            7: torch.tensor(
                [
                    [40.0, 20.0, 80.0, 100.0],
                    [42.0, 22.0, 82.0, 102.0],
                    [44.0, 24.0, 84.0, 104.0],
                ]
            )
        },
        num_frames=3,
    )
    completed_boxes = completed_track.bbx_xys(7, base_enlarge=1.0)
    video_path = tmp_path / "camera-near.mp4"
    decoder_calls: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []

    def _fake_get_batch(
        path: str,
        boxes: torch.Tensor,
        *,
        img_ds: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert path == str(video_path)
        assert img_ds == 0.5
        torch.testing.assert_close(boxes, completed_boxes)
        return torch.zeros((3, 3, 256, 256)), boxes.clone()

    def _fake_keypoints_from_heatmaps(
        *,
        heatmaps: NDArray[np.float32],
        center: NDArray[np.float32],
        scale: NDArray[np.float32],
        use_udp: bool,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        assert heatmaps.shape[1:] == (17, 64, 48)
        assert use_udp
        decoder_calls.append((center.copy(), scale.copy()))
        predictions = np.repeat(center[:, None, :], 17, axis=1)
        predictions[..., 0] += np.arange(17, dtype=np.float32)
        confidence = np.full(
            (center.shape[0], 17, 1),
            0.75,
            dtype=np.float32,
        )
        return predictions.astype(np.float32), confidence

    monkeypatch.setattr(pose2d_module, "get_batch", _fake_get_batch)
    monkeypatch.setattr(
        pose2d_module,
        "keypoints_from_heatmaps",
        _fake_keypoints_from_heatmaps,
    )
    fake_pose = _FakePose()
    model = ViTPosePose2D(
        checkpoint=tmp_path / "unused.ckpt",
        device="cpu",
        flip_test=False,
        batch_size=2,
        head_config=_head_config(),
    )
    model._pose = fake_pose
    model._loaded = True
    request = Pose2DRequest(video_path=video_path, bbx_xys=completed_boxes)

    result = model.predict(request)

    assert fake_pose.batch_shapes == [(2, 3, 256, 192), (1, 3, 256, 192)]
    assert result.keypoints.shape == (3, 17, 3)
    assert result.keypoints.dtype == torch.float32
    torch.testing.assert_close(result.keypoints[:, 0, :2], completed_boxes[:, :2])
    torch.testing.assert_close(
        result.keypoints[..., 2], torch.full((3, 17), 0.75)
    )
    assert [call[0].shape for call in decoder_calls] == [(2, 2), (1, 2)]
    assert set(vars(request)) == {"video_path", "bbx_xys"}
    assert set(vars(result)) == {"keypoints"}
