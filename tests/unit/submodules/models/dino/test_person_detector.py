"""Unit tests for DINO person-only decoding and checkpoint validation."""

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.submodules.models import DinoPersonDetector, PersonDetectionRequest
from src.submodules.models.dino.architecture import (
    preprocess_frame,
    use_fp32_deformable_attention,
    validate_checkpoint_args,
    validate_dino_repository,
)
from src.submodules.models.dino.person_detector import decode_person_detections
from src.tasks.person_tracking.duplicate_boxes import merge_person_boxes
from src.utils.inference_recording import InferenceRecorder


def test_mixed_dino_wraps_only_deformable_attention_in_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    class DeformableAttention(torch.nn.Module):
        def forward(self, query, reference_points, value, shapes, starts, mask):
            assert not torch.is_autocast_enabled("cpu")
            assert query.dtype == reference_points.dtype == value.dtype == torch.float32
            assert shapes.dtype == starts.dtype == torch.int64
            assert mask.dtype == torch.bool
            return query + reference_points + value

    upstream = ModuleType("fake_dino_attention")
    upstream.__dict__["MSDeformAttn"] = DeformableAttention
    monkeypatch.setitem(sys.modules, "_tennis_lab_third_party_dino_models.dino.ops.modules", upstream)
    model = torch.nn.Sequential(DeformableAttention(), torch.nn.Linear(2, 2)).eval()
    use_fp32_deformable_attention(model)
    args = [torch.ones(1, 2, dtype=torch.bfloat16)] * 3 + [
        torch.ones(1, 2, dtype=torch.int64), torch.zeros(1, dtype=torch.int64), torch.ones(1, dtype=torch.bool),
    ]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        value = model[0](*args)
        assert value.dtype == torch.float32
        assert model[1](value).dtype == torch.bfloat16
    torch.testing.assert_close(value, torch.full((1, 2), 3.0))


def test_decode_person_detections_filters_sorts_and_scales() -> None:
    logits = torch.full((1, 3, 91), -10.0)
    logits[0, :, 1] = torch.tensor([0.0, 2.0, 1.0])
    boxes = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.5, 0.2, 0.2], [0.8, 0.5, 0.1, 0.2]]]
    )

    result = decode_person_detections(
        {"pred_logits": logits, "pred_boxes": boxes},
        image_sizes=[(100, 200)],
        confidence=0.6,
    )[0]

    np.testing.assert_allclose(
        result.boxes_xyxy,
        np.array([[30, 40, 70, 60], [150, 40, 170, 60]], dtype=np.float32),
    )
    assert result.boxes_xyxy.shape == (2, 4)
    assert result.boxes_xyxy.dtype == np.float32
    assert result.scores.shape == (2,)
    assert result.scores.dtype == np.float32
    assert result.scores[0] > result.scores[1] > 0.6
    assert set(vars(result)) == {"boxes_xyxy", "scores"}


def test_decode_person_detections_returns_shaped_empty_arrays() -> None:
    result = decode_person_detections(
        {
            "pred_logits": torch.full((1, 2, 91), -10.0),
            "pred_boxes": torch.zeros((1, 2, 4)),
        },
        image_sizes=[(100, 200)],
        confidence=0.3,
    )[0]
    assert result.boxes_xyxy.shape == (0, 4)
    assert result.scores.shape == (0,)
    assert result.boxes_xyxy.dtype == result.scores.dtype == np.float32


def test_decode_batch_uses_each_image_size_threshold_and_row_order() -> None:
    logits = torch.full((3, 4, 91), -10.0)
    logits[0, :, 1] = torch.tensor([0.0, 2.0, -1e-5, 1.0])
    logits[1, :, 1] = torch.tensor([3.0, -1.0, 0.0, 3.0])
    # A different class's high score must not select a person.
    logits[2, :, 2] = 10.0
    boxes = torch.tensor(
        [
            [[0.5, 0.5, 0.2, 0.4], [0.25, 0.5, 0.2, 0.2],
             [0.8, 0.5, 0.1, 0.2], [0.1, 0.1, 0.4, 0.4]],
            [[0.5, 0.5, 0.5, 0.2], [0.25, 0.5, 0.2, 0.2],
             [0.8, 0.8, 0.1, 0.2], [0.9, 0.9, 0.4, 0.6]],
            [[0.5, 0.5, 0.2, 0.4]] * 4,
        ]
    )

    results = decode_person_detections(
        {"pred_logits": logits, "pred_boxes": boxes},
        image_sizes=[(100, 200), (300, 80), (13, 47)],
        confidence=0.5,
    )

    assert len(results) == 3
    np.testing.assert_allclose(
        results[0].boxes_xyxy,
        [[30, 40, 70, 60], [0, 0, 60, 30], [80, 30, 120, 70]],
        rtol=0, atol=1e-5,
    )
    np.testing.assert_allclose(
        results[1].boxes_xyxy,
        [[20, 120, 60, 180], [56, 180, 80, 300], [60, 210, 68, 270]],
        rtol=0, atol=2e-5,
    )
    np.testing.assert_allclose(results[0].scores, [0.8807971, 0.7310586, 0.5])
    np.testing.assert_allclose(results[1].scores, [0.95257413, 0.95257413, 0.5])
    assert results[2].boxes_xyxy.shape == (0, 4)
    assert results[2].scores.shape == (0,)


@pytest.mark.parametrize(
    ("logit_shape", "box_shape", "sizes"),
    [
        ((2, 3, 91), (2, 3, 4), [(100, 200)]),
        ((2, 3, 91), (1, 3, 4), [(100, 200), (300, 80)]),
        ((2, 3, 91), (2, 4, 4), [(100, 200), (300, 80)]),
        ((2, 3, 91), (2, 3, 5), [(100, 200), (300, 80)]),
        ((2, 3, 1), (2, 3, 4), [(100, 200), (300, 80)]),
        ((3, 91), (3, 4), [(100, 200)]),
    ],
)
def test_decode_rejects_mismatched_batch_axes(
    logit_shape: tuple[int, ...], box_shape: tuple[int, ...],
    sizes: list[tuple[int, int]],
) -> None:
    with pytest.raises(ValueError, match="Expected DINO logits"):
        decode_person_detections(
            {"pred_logits": torch.zeros(logit_shape), "pred_boxes": torch.zeros(box_shape)},
            image_sizes=sizes, confidence=0.5,
        )


class _TinyDino(torch.nn.Module):
    """Input-dependent CPU outputs; no upstream weights or CUDA op required."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        assert not torch.is_grad_enabled()
        self.inputs.append(images.clone())
        logits = images.new_full((len(images), 4, 91), -10.0)
        logits[:, :, 1] = images.new_tensor([-1.0, 0.0, 2.0, 2.0])
        boxes = images.new_tensor(
            [[0.8, 0.5, 0.1, 0.2], [0.8, 0.5, 0.1, 0.2],
             [0.25, 0.5, 0.2, 0.2], [0.26, 0.5, 0.2, 0.2]]
        ).repeat(len(images), 1, 1)
        # Distinguish images even when resizing makes their shapes identical.
        boxes[:, :, 0] += images[:, 0, 0, 0, None] * 0.02
        return {"pred_logits": logits, "pred_boxes": boxes}


@pytest.fixture
def tiny_detector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DinoPersonDetector, _TinyDino, list[bool]]:
    detector = DinoPersonDetector(
        checkpoint=tmp_path / "dino.pth", repository=tmp_path / "DINO",
        device="cpu", confidence=0.5, short_side=8, max_long_side=16,
    )
    model = _TinyDino()
    loads: list[bool] = []

    def load() -> None:
        loads.append(True)
        detector._model = model

    monkeypatch.setattr(detector, "_load_impl", load)
    return detector, model, loads


def test_recording_reports_actual_compatible_and_partial_batches(
    tiny_detector: tuple[DinoPersonDetector, _TinyDino, list[bool]], tmp_path: Path,
) -> None:
    detector, model, loads = tiny_detector
    detector.checkpoint.write_bytes(b"fake CPU weights")
    requests = [PersonDetectionRequest(np.full((6, 12, 3), i * 40, dtype=np.uint8)) for i in range(3)]
    expected = detector.predict_batch(requests, batch_size=2)
    detector.unload()
    model.inputs.clear()
    with InferenceRecorder(tmp_path / "dino.json"):
        actual = detector.predict_batch(requests, batch_size=2)
        detector.unload()
    record = json.loads((tmp_path / "dino.json").read_text())
    batches = [e["attributes"]["batch_shape"] for e in record["events"] if e["stage"] == "inference"]
    assert batches == [[2, 3, 8, 16], [1, 3, 8, 16]]
    assert loads == [True, True]
    for plain, recorded in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(recorded.boxes_xyxy, plain.boxes_xyxy)
        np.testing.assert_array_equal(recorded.scores, plain.scores)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("merge_duplicates", [False, True])
def test_predict_batch_matches_single_and_duplicate_rows(
    tiny_detector: tuple[DinoPersonDetector, _TinyDino, list[bool]],
    batch_size: int, merge_duplicates: bool,
) -> None:
    detector, model, loads = tiny_detector
    requests = [PersonDetectionRequest(np.full((h, w, 3), i * 40, dtype=np.uint8))
                for i, (h, w) in enumerate([(6, 12), (9, 18), (6, 12), (12, 24), (6, 12)])]
    expected = [detector.predict(request) for request in requests]
    model.inputs.clear()

    actual = detector.predict_batch(requests, batch_size=batch_size)

    assert len(actual) == len(requests)
    assert loads == [True] and detector.is_loaded
    assert [len(images) for images in model.inputs] == {
        1: [1, 1, 1, 1, 1], 2: [2, 2, 1], 4: [4, 1],
    }[batch_size]
    for index, (single, batched) in enumerate(zip(expected, actual, strict=True)):
        np.testing.assert_array_equal(single.boxes_xyxy, batched.boxes_xyxy)
        np.testing.assert_array_equal(single.scores, batched.scores)
        rows: NDArray[np.int64] = np.arange(index * 3, (index + 1) * 3, dtype=np.int64)
        keep, dropped = merge_person_boxes(
            index, rows, batched.boxes_xyxy, batched.scores, enabled=merge_duplicates,
        )
        single_keep, single_dropped = merge_person_boxes(
            index, rows, single.boxes_xyxy, single.scores, enabled=merge_duplicates,
        )
        np.testing.assert_array_equal(keep, [0, 2] if merge_duplicates else [0, 1, 2])
        np.testing.assert_array_equal(keep, single_keep)
        assert dropped == single_dropped
        if merge_duplicates:
            assert [(item.kept_row, item.dropped_row) for item in dropped] == [(index * 3, index * 3 + 1)]


def test_predict_batch_splits_shapes_without_padding_or_reordering(
    tiny_detector: tuple[DinoPersonDetector, _TinyDino, list[bool]],
) -> None:
    detector, model, _ = tiny_detector
    requests = [PersonDetectionRequest(np.full((h, w, 3), i * 40, dtype=np.uint8))
                for i, (h, w) in enumerate([(6, 12), (9, 18), (12, 6), (6, 12)])]
    expected = [detector.predict(request) for request in requests]
    expected_inputs = torch.cat([images.flatten() for images in model.inputs])
    model.inputs.clear()

    actual = detector.predict_batch(requests, batch_size=4)

    assert [tuple(images.shape) for images in model.inputs] == [
        (2, 3, 8, 16), (1, 3, 16, 8), (1, 3, 8, 16),
    ]
    # Compare actual normalized tensors, not only the model's reported shapes.
    torch.testing.assert_close(
        torch.cat([images.flatten() for images in model.inputs]),
        expected_inputs, rtol=0, atol=0,
    )
    for single, batched in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(single.boxes_xyxy, batched.boxes_xyxy)


def test_predict_empty_batch_does_not_load_model(
    tiny_detector: tuple[DinoPersonDetector, _TinyDino, list[bool]],
) -> None:
    detector, model, loads = tiny_detector
    assert detector.predict_batch([], batch_size=2) == []
    assert not detector.is_loaded and not loads and not model.inputs


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_predict_batch_rejects_invalid_limits_before_loading(
    tiny_detector: tuple[DinoPersonDetector, _TinyDino, list[bool]],
    batch_size: int,
) -> None:
    detector, _, loads = tiny_detector
    error = ValueError if type(batch_size) is int else TypeError
    with pytest.raises(error, match="batch_size"):
        detector.predict_batch([], batch_size=batch_size)
    assert not loads


@pytest.mark.parametrize(
    ("shape", "dtype", "error"),
    [((3, 4), np.uint8, ValueError), ((3, 4, 4), np.uint8, ValueError),
     ((0, 4, 3), np.uint8, ValueError), ((3, 4, 3), np.float32, TypeError)],
)
def test_predict_batch_validates_each_frame(
    tiny_detector: tuple[DinoPersonDetector, _TinyDino, list[bool]],
    shape: tuple[int, ...], dtype: type[np.uint8], error: type[Exception],
) -> None:
    detector, _, _ = tiny_detector
    requests = [PersonDetectionRequest(np.zeros((3, 4, 3), np.uint8)),
                PersonDetectionRequest(np.zeros(shape, dtype=dtype))]
    with pytest.raises(error):
        detector.predict_batch(requests, batch_size=2)


def test_preprocess_frame_matches_official_resize_constraint() -> None:
    frame: NDArray[np.uint8] = np.zeros((1080, 1920, 3), dtype=np.uint8)
    tensor = preprocess_frame(frame, short_side=800, max_long_side=1333)
    assert tensor.shape == (3, 750, 1333)
    assert tensor.dtype == torch.float32


def test_checkpoint_architecture_mismatch_is_explicit() -> None:
    args = Namespace(backbone="swin_T_224_1k")
    with pytest.raises(ValueError, match="Unsupported DINO checkpoint architecture"):
        validate_checkpoint_args(args)


def test_uninitialized_dino_submodule_is_explicit(tmp_path: Path) -> None:
    repository = tmp_path / "DINO"

    with pytest.raises(FileNotFoundError, match="git submodule update --init"):
        validate_dino_repository(repository)
