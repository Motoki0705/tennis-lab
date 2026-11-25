"""CPU fixtures for the official EdgeCrafter image and output contracts."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import src.submodules.models.ecpose.inference as inference
from src.submodules.models.ecpose.inference import decode_predictions, preprocess_images
from src.submodules.models.ecpose.releases import IMAGE_SIZE, RELEASES
from src.utils.checksum import dual_sha256


def test_load_network_restores_on_cpu_under_a_different_default_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Deployable(torch.nn.Linear):
        def deploy(self) -> Self:
            return self

    def config(_path: str) -> SimpleNamespace:
        return SimpleNamespace(
            yaml_cfg={"eval_spatial_size": IMAGE_SIZE, "ViTAdapter": {}},
            model=Deployable(2, 1),
        )

    checkpoint = tmp_path / "ecpose.pth"
    weight, bias = torch.tensor([[0.25, -0.5]]), torch.tensor([0.75])
    torch.save({"ema": {"module": {"weight": weight, "bias": bias}}}, checkpoint)
    release = replace(
        RELEASES["ecpose-l-o365"],
        size=checkpoint.stat().st_size,
        sha256=dual_sha256(checkpoint),
    )
    monkeypatch.setattr(inference, "require_source", lambda _: None)
    monkeypatch.setattr(inference, "_yaml_config", lambda _: config)

    with torch.device("meta"):
        network = inference.load_network(tmp_path, checkpoint, release)

    restored = network.model.state_dict()
    torch.testing.assert_close(restored["weight"], weight)
    torch.testing.assert_close(restored["bias"], bias)


def test_preprocess_converts_bgr_and_preserves_native_width_and_height() -> None:
    frames: list[NDArray[np.uint8]] = [
        np.full((3, 7, 3), (0, 128, 255), dtype=np.uint8),
        np.zeros((8, 4, 3), np.uint8),
    ]
    pixels, sizes = preprocess_images(frames)

    assert pixels.shape == (2, 3, 640, 640)
    assert pixels[0, :, 100, 200].tolist() == pytest.approx(
        [2.2489, 0.2052, -1.8044], abs=1e-4
    )
    assert sizes.tolist() == [[7, 3], [4, 8]]


def test_ecdet_uses_person_label_zero_and_clips_boxes_in_native_pixels() -> None:
    logits = torch.full((2, 300, 80), -10.0)
    boxes = torch.zeros((2, 300, 4))
    logits[0, 3, 0] = 4.0
    logits[0, 2, 0] = 3.0
    logits[0, 4, 5] = 5.0  # Highest score belongs to another class.
    logits[0, 1, 0] = 5.0  # Zero-area box is rejected.
    boxes[0, 3] = torch.tensor([0.50, 0.25, 0.20, 0.25])
    boxes[0, 2] = torch.tensor([0.98, 0.84, 0.10, 0.20])

    prediction, empty = decode_predictions(
        RELEASES["ecdet-l-coco"],
        (logits, boxes),
        torch.tensor([[200, 80], [40, 90]]),
        confidence=0.25,
    )

    assert prediction.boxes_xyxy[0].tolist() == pytest.approx([80, 10, 120, 30])
    assert prediction.boxes_xyxy[1].tolist() == pytest.approx([186, 59.2, 200, 75.2])
    assert prediction.scores.tolist() == pytest.approx([0.982014, 0.952574], abs=1e-6)
    assert prediction.keypoints_xyc is None
    assert empty.boxes_xyxy.shape == (0, 4)
    assert empty.scores.shape == (0,)
    assert empty.keypoints_xyc is None


def test_ecpose_keeps_query_rows_joint_order_and_explicit_availability() -> None:
    logits = torch.full((2, 60, 2), -10.0)
    logits[0, 5, 1], logits[0, 2, 1] = 3.0, 6.0
    logits[0, 4, 0], logits[0, 1, 1] = 8.0, 5.0
    xy = torch.full((2, 60, 17, 2), torch.nan)
    xy[0, 2, 0] = torch.tensor([0.40, 0.20])  # Nose
    xy[0, 2, 9] = torch.tensor([0.20, 0.30])  # Left wrist
    xy[0, 2, 10] = torch.tensor([0.70, 0.60])  # Right wrist
    xy[0, 2, 16] = torch.tensor([1.20, 0.60])  # Out of image
    xy[0, 5, 0] = torch.tensor([0.02, 0.10])
    xy[0, 5, 1] = torch.tensor([0.30, 0.70])
    xy[0, 1, 0] = torch.tensor([0.45, 0.60])  # Cannot define a box

    prediction, empty = decode_predictions(
        RELEASES["ecpose-x-o365"],
        (logits, xy),
        torch.tensor([[320, 100], [120, 240]]),
        confidence=0.25,
        pose_box_margin=0.10,
    )

    assert prediction.boxes_xyxy[0].tolist() == pytest.approx([48, 16, 240, 64])
    assert prediction.boxes_xyxy[1].tolist() == pytest.approx([0, 4, 104.96, 76])
    assert prediction.scores.tolist() == pytest.approx([0.997527, 0.952574], abs=1e-6)
    keypoints = prediction.keypoints_xyc
    assert keypoints is not None
    assert keypoints[0, 0].tolist() == pytest.approx([128, 20, 1])
    assert keypoints[0, 9].tolist() == pytest.approx([64, 30, 1])
    assert keypoints[0, 10].tolist() == pytest.approx([224, 60, 1])
    assert keypoints[0, 16].tolist() == keypoints[0, 1].tolist() == [0, 0, 0]
    assert keypoints[1, 0].tolist() == pytest.approx([6.4, 10, 1])
    assert empty.boxes_xyxy.shape == (0, 4)
    assert empty.scores.shape == (0,)
    assert empty.keypoints_xyc is not None and empty.keypoints_xyc.shape == (0, 17, 3)


def test_ecpose_requires_the_caller_to_select_the_box_policy() -> None:
    with pytest.raises(ValueError, match="explicit nonnegative joint-envelope margin"):
        decode_predictions(
            RELEASES["ecpose-l-o365"],
            (torch.empty(0), torch.empty(0)),
            torch.empty(0),
            confidence=0.0,
        )
