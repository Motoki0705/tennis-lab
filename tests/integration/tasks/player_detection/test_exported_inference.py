"""Opt-in real-frame check of the exported fine-tuned DINO checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.player_detection.configuration import FrameSelectionConfig
from src.tasks.player_detection.data.detection_dataset import select_detection_frames
from src.tasks.player_detection.data.store import PlayerFrameStore
from src.tasks.player_detection.inference import (
    DinoPlayerDetector,
    PlayerDetectionRequest,
)


def test_exported_player_checkpoint_predicts_a_held_out_frame() -> None:
    if os.environ.get("PLAYER_DETECTION_GPU_TEST") != "1":
        pytest.skip("Set PLAYER_DETECTION_GPU_TEST=1 for the GPU/artifact check")
    if not torch.cuda.is_available():
        pytest.skip("DINO deformable attention requires CUDA")
    root = Path(os.environ["PLAYER_DETECTION_ARTIFACT_ROOT"])
    checkpoint = root / "ckpt/player_detection/chat-player-v1-e8-best-pr937.pth"
    store_dir = root / "data/player_detection/chat-player-v1"
    prediction_file = (
        root
        / "outputs/player_detection/train/dino_swinl_ft/chat-player-v1-e8-20260926"
        / "predictions/pred_test.npz"
    )
    required = (checkpoint, store_dir / "index.npz", prediction_file)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.fail(f"Required player inference artifacts are missing: {missing}")

    torch.set_float32_matmul_precision("high")
    store = PlayerFrameStore(store_dir)
    selection = select_detection_frames(
        store, "test", FrameSelectionConfig(True, True, 4.0), frame_stride=1
    )
    row = int(selection.frames[0])
    image = store.read_bgr(row)
    detector = DinoPlayerDetector(
        checkpoint,
        root / "third_party/DINO",
        device="cuda",
        confidence=0.3,
        short_side=800,
        max_long_side=1333,
    )
    try:
        detector.load()
        assert detector.is_loaded
        assert detector.checkpoint_info is not None
        assert detector.checkpoint_info.epoch == 3
        result = detector.predict(PlayerDetectionRequest(image))
    finally:
        detector.unload()
    assert not detector.is_loaded
    assert result.boxes_xyxy.ndim == 2 and result.boxes_xyxy.shape[1] == 4
    assert result.scores.shape == (len(result.boxes_xyxy),)
    assert np.isfinite(result.boxes_xyxy).all()
    assert np.isfinite(result.scores).all()
    assert (result.scores >= 0.3).all()

    with np.load(prediction_file) as archived:
        assert archived["scene_ids"][0] == store.frame_key(row)
        keep = archived["scores"][0] >= 0.3
        expected_boxes = archived["boxes_xyxy"][0, keep]
        expected_scores = archived["scores"][0, keep]
    np.testing.assert_allclose(result.boxes_xyxy, expected_boxes, atol=1.0, rtol=0)
    np.testing.assert_allclose(result.scores, expected_scores, atol=0.001, rtol=0)
