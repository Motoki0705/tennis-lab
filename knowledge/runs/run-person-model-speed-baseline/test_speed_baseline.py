from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from parity import compare
from speed_inputs import ClipRows, InputSet, publish, require_sha, write_crops
from speed_models import dino_decode_gpu

from src.submodules.models.dino.architecture import COCO_PERSON_CLASS_ID
from src.submodules.models.dino.person_detector import (
    PersonDetectionResult,
    decode_person_detections,
)


def test_detector_postprocess_retains_production_order_and_coordinates() -> None:
    logits = torch.zeros((2, 3, COCO_PERSON_CLASS_ID + 1))
    logits[0, :, COCO_PERSON_CLASS_ID] = torch.tensor([0.7, 2.0, -2.0])
    boxes = torch.tensor(
        [[[0.2, 0.4, 0.3, 0.6], [0.85, 0.7, 0.4, 0.2], [0.45, 0.55, 0.1, 0.1]]] * 2
    )
    raw = {"pred_logits": logits, "pred_boxes": boxes}
    sizes = [(80, 120), (160, 40)]
    expected = decode_person_detections(raw, image_sizes=sizes, confidence=0.6)
    actual = dino_decode_gpu(raw, sizes, confidence=0.6)
    for (xyxy, score), result in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(xyxy.numpy(), result.boxes_xyxy)
        np.testing.assert_array_equal(score.numpy(), result.scores)
    assert actual[0][0].shape == (2, 4)
    assert actual[1][0].shape == (0, 4)


def test_fixed_crops_prioritize_observations_and_pin_exact_bytes(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(2):
        path = tmp_path / f"frame{index}.npy"
        image = np.full((80, 120, 3), 22 + index, dtype=np.uint8)
        np.save(path, image)
        rows.append(
            {
                "id": f"frame{index}",
                "frame": index,
                "clip_id": "clip",
                "camera": "cam0",
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape": [80, 120, 3],
            }
        )
    tracked = ClipRows(
        "clip",
        {},
        {"cam0": np.ones((2, 1), bool)},
        {"cam0": np.array([[[44, 48, 38.4]]] * 2)},
        (0, 1),
        [],
    )
    boxes = np.array(
        [
            [8, 8, 24, 24],
            [34, 8, 50, 24],
            [58, 8, 74, 24],
            [78, 8, 94, 24],
            [30, 32, 58, 64],
        ],
        np.float32,
    )
    scores = np.array([0.91, 0.82, 0.7, 0.61, 0.33], np.float32)
    crops = write_crops(
        tmp_path, rows, [tracked], lambda _: PersonDetectionResult(boxes, scores)
    )
    manifest = publish(tmp_path, rows, [], [], crops, {"fixture": True})
    data = InputSet(tmp_path)
    assert manifest.is_file() and data.manifest["pose_sweep_crop_count"] == 8
    assert [row["id"] for row in data.rows("crops")[:4]] == [
        "frame0/dino_000",
        "frame0/dino_001",
        "frame0/dino_002",
        "frame0/dino_004",
    ]
    crop = data.rows("crops")[3]
    assert crop["scene_observation_iou"] == pytest.approx(1.0)
    path = require_sha(tmp_path / crop["path"], crop["sha256"])
    assert np.load(path).shape == (3, 256, 256)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        require_sha(path, crop["sha256"])


def test_pose_parity_preserves_pixel_units_and_fixed_ids() -> None:
    reference = np.tile([800.0, 500.0, 0.8], (17, 1))
    candidate = reference.copy()
    candidate[:, 1] += 1.2
    result = compare("vitpose-flip", {"crop": reference}, {"crop": candidate})
    assert not result.passed
    assert result.metrics["pose_xy_px_p95"] == pytest.approx(1.2)
    assert (
        compare("vitpose-flip", {"crop": reference}, {"other": reference}).reason
        == "Input ID coverage differs"
    )
