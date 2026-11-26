"""RF-DETR-L shape, preprocessing and score semantics without vendor weights."""

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.submodules.models import PersonDetectionRequest, RFDetrPersonDetector
from src.submodules.models.rfdetr import person_detector as rf


def test_batch_preserves_per_image_size_rgb_order_and_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    detector = RFDetrPersonDetector(tmp_path / "model.pth", device="cpu")
    monkeypatch.setattr(detector, "load", lambda: None)
    batches = []

    def forward(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batches.append(len(pixels))
        np.testing.assert_allclose(pixels[0, :, 350, 350].numpy(), [2.2489083, -2.0357141, -1.8044444], atol=1e-5)
        boxes = torch.tensor([.25, .5, .75, .5]).expand(len(pixels), 300, 4).clone()
        logits = torch.full((len(pixels), 300, 91), -10.)
        logits[:, 0, 1] = 0
        logits[:, 0, 0] = 10  # class zero does not denote person
        return boxes, logits

    monkeypatch.setattr(detector, "_forward", forward)
    requests = []
    for shape in ((8, 16), (16, 8), (32, 24)):
        frame: NDArray[np.uint8] = np.zeros((*shape, 3), np.uint8)
        frame[..., 2] = 255
        requests.append(PersonDetectionRequest(frame))
    results = detector.predict_batch(requests, batch_size=2)
    assert batches == [2, 1]
    np.testing.assert_array_equal(
        [result.boxes_xyxy[0] for result in results],
        [[0, 2, 10, 6], [0, 4, 5, 12], [0, 8, 15, 24]],
    )
    assert [result.scores.tolist() for result in results] == [[.5], [.5], [.5]]


def test_decode_ranks_query_class_globally_before_selection() -> None:
    ordinal = torch.arange(300, dtype=torch.float32)
    boxes = torch.zeros((2, 300, 4))
    boxes[..., 0] = .5 + ordinal * .001
    boxes[..., 1] = .5
    boxes[..., 2:] = .2
    logits = torch.full((2, 300, 91), -10.)
    logits[:, :, 90] = 5
    logits[0, 0, 1] = 5  # earlier flattened tie than any class-90 entry
    logits[1, 0, 1] = 4  # all 300 class-90 entries outrank it
    logits[0, 1, 1] = 6
    sizes = torch.tensor([[100., 200.], [200., 100.]])
    primary, empty = rf.decode_persons(boxes, logits, sizes, confidence=.30)
    np.testing.assert_allclose(primary.boxes_xyxy, [[80.2, 40, 120.2, 60], [80, 40, 120, 60]], atol=1e-5)
    assert primary.scores[0] > primary.scores[1] > .99
    assert empty.boxes_xyxy.shape == (0, 4)
    assert empty.scores.shape == (0,)


def test_empty_batch_and_missing_checkpoint_do_not_import_rfdetr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_import(name: str) -> None:
        pytest.fail(f"Unexpected import of {name}")

    monkeypatch.setattr(rf, "import_module", unexpected_import)
    detector = RFDetrPersonDetector(tmp_path / "missing.pth", device="cpu")
    assert detector.predict_batch([], batch_size=2) == []
    assert not detector.is_loaded
    with pytest.raises(FileNotFoundError, match="Required checkpoint"):
        detector.load()


def test_unavailable_rfdetr_version_points_to_project_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_package(name: str) -> str:
        raise rf.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(rf.metadata, "version", no_package)
    with pytest.raises(ImportError, match="rfdetr==1.12.1.*project extra"):
        rf.require_rf_detr_version()
