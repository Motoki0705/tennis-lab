"""The RTMPose-X array contract is checked with asymmetric synthetic logits."""

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.submodules.models import RTMPoseBoxSequenceRequest, RTMPoseXPose2D
from src.submodules.models.rtmpose import pose2d as rt


def test_native_box_rows_and_flip_preserve_raw_coco17_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = RTMPoseXPose2D(tmp_path / "model.pth", device="cpu", batch_size=2, show_progress=False)
    monkeypatch.setattr(model, "load", lambda: None)
    sizes: list[int] = []

    def forward(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flip = len(sizes) % 2 == 1
        sizes.append(len(pixels))
        np.testing.assert_allclose(pixels[0, :, 192, 144].numpy(),
                                   [2.1632845, -1.8606442, -1.71730], atol=1e-5)
        x = torch.full((len(pixels), 26, 576), -.25)
        y = torch.full((len(pixels), 26, 768), -.4)
        joint = 2 if flip else 1
        x[:, joint, 575 - 216 if flip else 216] = 2.4 if flip else 2.0
        y[:, joint, 384] = 1.6 if flip else 1.2
        x[:, 17, 0] = 10  # a Halpe-only joint
        y[:, 17, 0] = 10
        return x, y

    monkeypatch.setattr(model, "_forward", forward)
    frame: NDArray[np.uint8] = np.full((100, 100, 3), [5, 10, 250], dtype=np.uint8)
    request = RTMPoseBoxSequenceRequest(
        [(4, frame), (5, frame), (6, frame)],
        torch.tensor([[10., 10., 40., 50.], [40., 20., 70., 60.], [20., 20., 50., 60.]]),
        torch.tensor([4, 4, 6]),
    )
    keypoints = model.predict(request).keypoints.numpy()
    assert sizes == [2, 2, 1, 1]
    assert keypoints.shape == (3, 17, 3) and keypoints.dtype == np.float32
    np.testing.assert_allclose(keypoints[:, 1], [[20.3125, 30, 1.4], [50.3125, 40, 1.4], [30.3125, 40, 1.4]], atol=1e-6)
    np.testing.assert_allclose(keypoints[:, 2], np.tile([-1, -1, -.4], (3, 1)), atol=1e-6)


def test_rectangular_crop_rejects_missing_frame_instead_of_reordering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = RTMPoseXPose2D(tmp_path / "model.pth", device="cpu", batch_size=1, show_progress=False)
    monkeypatch.setattr(model, "load", lambda: None)
    frame: NDArray[np.uint8] = np.zeros((8, 8, 3), np.uint8)
    request = RTMPoseBoxSequenceRequest([(2, frame)], torch.tensor([[1., 1., 3., 5.]]), torch.tensor([1]))
    with pytest.raises(ValueError, match="Missing requested source frame 1"):
        model.predict(request)


def test_halpe_order_and_flip_are_checked_against_checkpoint_metadata() -> None:
    names = dict(enumerate(rt.HALPE_NAMES))
    rt._validate_halpe({"keypoint_id2name": names, "flip_indices": rt.HALPE_FLIP})
    names[1], names[2] = names[2], names[1]
    with pytest.raises(ValueError, match="Halpe26 names or flip indices"):
        rt._validate_halpe({"keypoint_id2name": names, "flip_indices": rt.HALPE_FLIP})


def test_missing_rtmpose_weights_are_reported_before_initializing_graph(tmp_path: Path) -> None:
    model = RTMPoseXPose2D(tmp_path / "model.pth", device="cpu", batch_size=2)
    with pytest.raises(FileNotFoundError, match="Required checkpoint"):
        model.load()
