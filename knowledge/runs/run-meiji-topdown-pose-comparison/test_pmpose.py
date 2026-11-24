from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from evaluate import AccuracyStages
from pmpose import (
    HEATMAP_HW,
    JOINT_ORDER,
    NATIVE_JOINTS,
    _bbox_contour,
    _infer,
    _verify_joint_schema,
    make_adapter,
)
from pose_contract import PoseInput

from src.utils.inference_benchmark import (
    Backend,
    BatchStages,
    ConditionUnavailable,
    Stage,
    Status,
)


@pytest.mark.parametrize("box", [[1.25, 2.75, 5.3, 9], [-1.5, 4.25, 11.6, 12.4]])
def test_bbox_prompt_matches_demo_raster_contour(box):
    frame = np.zeros((11, 10, 3), dtype=np.uint8)
    bounds = np.array(box, dtype=np.float32)
    height, width = frame.shape[:2]
    raster = np.zeros((height, width), dtype=np.uint8)
    x1, y1 = np.clip(np.floor(bounds[:2]), 0, [width, height]).astype(int)
    x2, y2 = np.clip(np.ceil(bounds[2:]), 0, [width, height]).astype(int)
    raster[y1:y2, x1:x2] = 255
    contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert _bbox_contour(PoseInput(frame, bounds)) == [contours[0].ravel().tolist()]


def test_bbox_prompt_rejects_a_one_pixel_intersection():
    row = PoseInput(
        np.zeros((11, 10, 3), dtype=np.uint8),
        np.array([9.25, 1, 12, 6], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="does not yield a polygon"):
        _bbox_contour(row)


def test_right_eye_metadata_cannot_be_replaced_by_left_eye():
    meta = {
        "keypoint_id2name": dict(enumerate(JOINT_ORDER)),
        "flip_indices": [
            0,
            2,
            1,
            4,
            3,
            6,
            5,
            8,
            7,
            10,
            9,
            12,
            11,
            14,
            13,
            16,
            15,
            17,
            18,
            19,
            20,
            21,
            22,
        ],
    }
    _verify_joint_schema(meta)
    meta["keypoint_id2name"][2] = "left_eye"
    with pytest.raises(ValueError, match="pinned COCO17"):
        _verify_joint_schema(meta)


@pytest.mark.parametrize("timed", [False, True], ids=["accuracy", "shared-speed"])
def test_flip_decode_keeps_original_row_order_and_auxiliary_joints(timed):
    frame = np.zeros((11, 20, 3), dtype=np.uint8)
    rows = tuple(
        PoseInput(frame, np.array(box, dtype=np.float32))
        for box in ([1, 2, 6, 7], [14, 1, 18, 8])
    )
    factory_pixels = torch.zeros(2, 3, 2, 4)
    factory_pixels[:, :, 0, 0] = 1
    samples = [
        SimpleNamespace(
            metainfo={
                "input_center": [30, 50],
                "input_scale": np.array([24, 32]),
                "input_size": [192, 256],
            }
        ),
        SimpleNamespace(
            metainfo={
                "input_center": [12, 30],
                "input_scale": np.array([48, 64]),
                "input_size": [192, 256],
            }
        ),
    ]

    def preprocess(inputs):
        assert inputs == rows
        return factory_pixels, samples

    def decode(maps):
        # A deterministic stand-in for the upstream decoder. The adapter must
        # undo the crop coordinates and place results along the right row axis.
        indices = maps.reshape(len(maps), -1).argmax(axis=1)
        y, x = np.divmod(indices, HEATMAP_HW[1])
        return np.column_stack((x * 4, y * 4))[None], None

    original = torch.zeros(2, NATIVE_JOINTS, *HEATMAP_HW)
    mirror = torch.zeros_like(original)
    for row, joint, other, x, y in [
        (0, 1, 2, 18, 20),
        (0, 2, 1, 4, 20),
        (0, 10, 9, 36, 42),
        (1, 1, 2, 24, 17),
    ]:
        original[row, joint, y, x] = 0.25
        mirror[row, other, y, 47 - x] = 0.75
    zeros = torch.zeros(2, NATIVE_JOINTS, 1, 1)
    original_oks, mirror_oks = zeros.clone(), zeros.clone()
    original_oks[0, 1], mirror_oks[0, 2] = 0.25, 0.75
    original_presence, mirror_presence = zeros.clone(), zeros.clone()
    original_presence[1, 21], mirror_presence[1, 21] = 0.5, 1.0
    original_outputs = (original, original_presence, zeros, original_oks, zeros)
    mirror_outputs = (mirror, mirror_presence, zeros, mirror_oks, zeros)

    def forward(pixels):
        return original_outputs if pixels[0, 0, 0, 0] == 1 else mirror_outputs

    infer = _infer(
        SimpleNamespace(preprocess=preprocess, decode=decode),
        forward,
        Backend.PT32,
        torch.device("cpu"),
        torch.float32,
    )
    event = SimpleNamespace(record=lambda: None, elapsed_time=lambda other: 1.0)
    cuda = SimpleNamespace(events=(event, event), synchronize=lambda: None)
    stages = (
        BatchStages(cuda, Backend.PT32, timed=True)
        if timed
        else AccuracyStages(Backend.PT32)
    )
    output = infer(rows, stages)
    if isinstance(stages, BatchStages):
        names = [
            sample.name for sample in stages._finish() if sample.stage is Stage.FORWARD
        ]
        assert names == ["original", "flipped"]
    else:
        assert stages.forward_count == 2
    assert output.keypoints_xyc.shape == (2, 17, 3)
    assert output.keypoints_xyc.dtype == np.float32
    np.testing.assert_array_equal(output.keypoints_xyc[0, 1], [27, 44, 0.5])
    np.testing.assert_array_equal(output.keypoints_xyc[0, 2], [20, 44, 0])
    np.testing.assert_array_equal(output.keypoints_xyc[0, 10], [36, 55, 0])
    np.testing.assert_array_equal(output.keypoints_xyc[1, 1], [12, 15, 0])
    assert np.isnan(output.keypoints_xyc[0, 9, :2]).all()
    assert output.keypoints_xyc[0, 9, 2] == 0
    assert output.extras["presence"].shape == (2, 23)
    assert output.extras["presence"][1, 21] == 0.75
    assert output.extras["heatmap_peak"][0, 10] == 0.5


@pytest.mark.parametrize(
    ("backend", "source", "reason"),
    [
        (Backend.PT32, None, "BBoxMaskPose checkout"),
        (Backend.PT16, Path("unused"), "differs from PMPose-l"),
        (Backend.TRT16, None, "TensorRT engine bundle"),
    ],
)
def test_unavailable_assets_are_reported_on_load(tmp_path, backend, source, reason):
    path = tmp_path / "wrong.pth"
    path.write_bytes(b"different weights")
    adapter = make_adapter(
        model_id="pmpose-l",
        backend=backend,
        checkpoint=path,
        device="cpu",
        source_root=source,
    )
    with pytest.raises(ConditionUnavailable, match=reason) as error:
        adapter.load()
    assert error.value.status is Status.UNAVAILABLE
