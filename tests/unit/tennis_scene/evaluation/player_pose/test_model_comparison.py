"""Contracts for the archived baseline and candidate prediction interchange."""

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, Predictions
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesOutput,
)
from src.tennis_scene.pipeline.contracts import SourceVideo

RUN = (
    Path(__file__).resolve().parents[5]
    / "knowledge/runs/run-meiji-player-model-comparison"
)
FRAMES = (
    FrameKey("video_000/clip_003", "cam3", 120),
    FrameKey("video_000/clip_003", "cam0", 122),
)


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(RUN))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    return importlib.import_module("prediction_rows"), importlib.import_module(
        "baseline"
    )


def test_pose_rows_keep_both_people_and_zero_detection_frames(comparison, tmp_path):
    rows, _ = comparison
    header = rows.PredictionHeader(
        "meiji_4cam/pin", "1" * 64, "ecpose-l-o365", "pt16", True
    )
    keypoints: NDArray[np.float32] = np.zeros((2, 17, 3), np.float32)
    keypoints[1, 9] = (121, 56, 0.75)
    predictions = {
        FRAMES[0]: Predictions(
            np.array([[2, 4, 17, 32], [82, 37, 136, 119]], np.float32),
            np.array([0.5, 0.875], np.float32),
            keypoints,
        ),
        FRAMES[1]: Predictions(
            np.empty((0, 4), np.float32),
            np.empty(0, np.float32),
            np.empty((0, 17, 3), np.float32),
        ),
    }
    path = tmp_path / "predictions.jsonl"
    rows.write_predictions(path, header, predictions, expected_frames=FRAMES)

    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert saved[0] == {
        "schema": "player_model_predictions_v1",
        "benchmark_id": "meiji_4cam/pin",
        "manifest_sha256": "1" * 64,
        "model_id": "ecpose-l-o365",
        "backend": "pt16",
        "has_pose": True,
    }
    assert saved[1]["camera_id"] == "cam0"
    assert saved[1]["boxes_xyxy"] == saved[1]["keypoints_xyc"] == []
    assert saved[2]["keypoints_xyc"][1][9] == [121, 56, 0.75]
    restored = rows.read_predictions(path, header, expected_frames=FRAMES)
    assert restored[FRAMES[1]].keypoints_xyc.shape == (0, 17, 3)
    np.testing.assert_array_equal(restored[FRAMES[0]].boxes_xyxy[1], [82, 37, 136, 119])
    np.testing.assert_array_equal(
        restored[FRAMES[0]].keypoints_xyc[1, 9], [121, 56, 0.75]
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("duplicate", "Duplicate prediction frame"),
        ("missing", "1 missing frames, 0 unexpected"),
        ("extra", "0 missing frames, 1 unexpected"),
        ("manifest", "Prediction header differs"),
    ],
)
def test_prediction_reader_rejects_unpinned_rows(comparison, tmp_path, change, message):
    rows, _ = comparison
    header = rows.PredictionHeader(
        "meiji_4cam/pin", "1" * 64, "dino-swin-l", "trt16", False
    )
    saved = [
        header.as_json(),
        *(
            {**key._asdict(), "boxes_xyxy": [], "scores": [], "keypoints_xyc": None}
            for key in FRAMES
        ),
    ]
    match change:
        case "duplicate":
            saved.append(saved[1])
        case "missing":
            saved.pop()
        case "extra":
            saved.append({**saved[1], "source_frame": 121})
        case "manifest":
            saved[0]["manifest_sha256"] = "2" * 64
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in saved))

    with pytest.raises(ValueError, match=message):
        rows.read_predictions(path, header, expected_frames=FRAMES)


def test_archived_predictions_align_poses_by_source_row_and_frame(comparison):
    _, baseline = comparison
    video = SourceVideo("cam3", Path("video.mp4"), "1" * 64, 3, 60, 1920, 1080)
    offsets = np.array([0, 1, 1, 2], np.int64)
    source_rows = np.array([7, 11], np.int64)
    detections = PersonDetectionOutput(
        "cam3",
        offsets,
        np.array([[2, 4, 17, 32], [82, 37, 136, 119]], np.float32),
        np.array([0.5, 0.875], np.float32),
        source_rows,
    )
    keypoints: NDArray[np.float32] = np.zeros((2, 17, 3), np.float32)
    keypoints[1, 2] = (121, 56, 0.75)
    keypoints[1, 9] = (102, 57, -0.2)
    poses = PersonPoseFeaturesOutput(
        "cam3", offsets, source_rows, keypoints, "2" * 64, "3" * 64
    )

    result = baseline.camera_predictions(video, detections, poses, (1, 2))
    assert list(result) == [1, 2]
    assert result[1].boxes_xyxy.shape == (0, 4)
    np.testing.assert_array_equal(result[2].boxes_xyxy[0], [82, 37, 136, 119])
    np.testing.assert_array_equal(result[2].keypoints_xyc[0, 2], [121, 56, 0.75])
    np.testing.assert_array_equal(result[2].keypoints_xyc[0, 9], [0, 0, 0])

    for mismatch in (
        replace(poses, source_rows=np.array([8, 11], np.int64)),
        replace(poses, frame_offsets=np.array([0, 1, 2, 2], np.int64)),
    ):
        with pytest.raises(ValueError, match="source row mismatch"):
            baseline.camera_predictions(video, detections, mismatch, (1, 2))
