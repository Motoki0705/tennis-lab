"""CPU fixtures for the RT-DETRv4 comparison adapter's image and box contracts."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn
from torchvision.ops import box_convert

RUN = (
    Path(__file__).resolve().parents[5]
    / "knowledge/runs/run-meiji-player-model-comparison"
)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(RUN))
    return import_module("rtdetrv4")


def test_prepare_preserves_wh_and_uses_rgb_unit_range(adapter):
    frame: NDArray[np.uint8] = np.empty((2, 5, 3), np.uint8)
    frame[:] = (9, 55, 241)

    pixels, sizes = adapter.preprocess([frame, np.zeros((4, 1, 3), np.uint8)])

    assert pixels.shape == (2, 3, 640, 640)
    assert sizes.tolist() == [[5, 2], [1, 4]]
    torch.testing.assert_close(pixels[0, :, 80, 420], torch.tensor([241, 55, 9]) / 255)
    assert torch.count_nonzero(pixels[1]) == 0


def test_only_person_rows_survive_with_clipped_native_xyxy_and_matching_scores(adapter):
    labels = np.array([[4, 0, 0, 0], [1, 2, 3, 4]])
    boxes = np.array(
        [
            [[0, 0, 100, 40], [-3, 8, 120, 45], [20, 3, 30, 15], [0, 43, 90, 58]],
            [[1, 1, 4, 6]] * 4,
        ],
        np.float32,
    )
    scores = np.array([[0.98, 0.81, 0.20, 0.19], [0.99, 0.97, 0.91, 0.87]], np.float32)

    first, second = adapter.person_predictions(
        labels, boxes, scores, np.array([[100, 40], [11, 50]])
    )

    np.testing.assert_array_equal(first.boxes_xyxy, [[0, 8, 100, 40], [20, 3, 30, 15]])
    np.testing.assert_array_equal(first.scores, np.array([0.81, 0.20], np.float32))
    assert first.keypoints_xyc is second.keypoints_xyc is None
    assert second.boxes_xyxy.shape == (0, 4)
    assert second.scores.shape == (0,)


def test_nonfinite_coordinates_are_rejected_before_clipping(adapter):
    with pytest.raises(ValueError, match="Non-finite RT-DETRv4"):
        adapter.person_predictions(
            np.array([[0]]),
            np.array([[[0, 0, np.inf, 5]]]),
            np.array([[0.99]]),
            np.array([[10, 10]]),
        )


def test_detector_wires_native_sizes_to_postprocessing_and_declares_forward(adapter):
    calls = []

    class Stages:
        def stage(self, stage, operation):
            calls.append(stage.value)
            return operation()

        def forward(self, backend, operation):
            calls.append(backend.value)
            return operation()

    class Postprocess(nn.Module):
        def forward(self, output, sizes):
            boxes = box_convert(output["pred_boxes"], in_fmt="cxcywh", out_fmt="xyxy")
            return (
                torch.zeros((2, 1), dtype=torch.long),
                boxes * sizes.repeat(1, 2)[:, None],
                output["pred_logits"].sigmoid(),
            )

    def model(pixels):
        assert pixels.shape == (2, 3, 640, 640)
        return torch.tensor([[0.0], [2.0]]), torch.tensor([[[0.5, 0.5, 0.5, 0.5]]] * 2)

    detector = adapter.Detector(
        adapter.Backend.PT32, torch.device("cpu"), model, Postprocess()
    )
    first, second = detector(
        [np.zeros((40, 20, 3), np.uint8), np.zeros((12, 100, 3), np.uint8)], Stages()
    )

    np.testing.assert_array_equal(first.boxes_xyxy, [[5, 10, 15, 30]])
    np.testing.assert_array_equal(second.boxes_xyxy, [[25, 3, 75, 9]])
    np.testing.assert_allclose([first.scores[0], second.scores[0]], [0.5, 0.880797])
    assert calls == ["preprocess", "h2d", "PT32", "postprocess", "d2h", "postprocess"]
