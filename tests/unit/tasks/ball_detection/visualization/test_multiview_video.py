"""Tests for CPU multiview ball-prediction video generation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.model_io.contracts import BallPrediction
from src.tasks.ball_detection.visualization.multiview_video import (
    decode_model_images,
    load_multiview_clip,
    run_cpu_visualization,
)


class _PeakPredictor:
    device = torch.device("cpu")

    def predict(self, images: torch.Tensor) -> BallPrediction:
        batch_size, frame_count, _, height, width = images.shape
        heatmaps = torch.zeros(batch_size, frame_count, height, width)
        heatmaps[:, :, height // 2, width // 2] = 0.9
        return BallPrediction(
            coords=torch.zeros(batch_size, frame_count, 2),
            confidence=torch.full((batch_size, frame_count), 0.9),
            heatmaps=heatmaps,
        )


def _write_clip(root: Path, *, frame_count: int = 8) -> Path:
    clip_dir = root / "videos" / "video_002" / "clips" / "clip_023"
    media_dir = clip_dir / "media"
    media_dir.mkdir(parents=True)
    cameras = []
    for camera_index in range(3):
        camera_id = f"cam{camera_index}"
        relative_video = f"media/{camera_id}.mp4"
        video_path = clip_dir / relative_video
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (80, 48)
        )
        assert writer.isOpened()
        for frame_index in range(frame_count):
            frame: NDArray[np.uint8] = np.full(
                (48, 80, 3),
                (20 * camera_index, 10 * frame_index, 100),
                dtype=np.uint8,
            )
            writer.write(frame)
        writer.release()
        cameras.append({"camera_id": camera_id, "video": relative_video})
    manifest = {
        "version": 2,
        "clip_id": "video_002/clip_023",
        "camera_ids": [f"cam{index}" for index in range(3)],
        "cameras": cameras,
        "fps": 10.0,
        "num_frames": frame_count,
        "width": 80,
        "height": 48,
    }
    (clip_dir / "clip.json").write_text(json.dumps(manifest), encoding="utf-8")
    return clip_dir


def test_load_and_decode_multiview_clip(tmp_path: Path) -> None:
    clip = load_multiview_clip(_write_clip(tmp_path))

    images = decode_model_images(
        clip.cameras[0],
        frame_count=clip.frame_count,
        original_size_wh=clip.original_size_wh,
        image_size_hw=(24, 40),
    )

    assert clip.clip_id == "video_002/clip_023"
    assert [camera.camera_id for camera in clip.cameras] == ["cam0", "cam1", "cam2"]
    assert images.shape == (8, 3, 24, 40)
    assert images.dtype == torch.float32
    assert 0.0 <= float(images.min()) <= float(images.max()) <= 1.0


def test_cpu_visualization_writes_synced_video_preview_and_metadata(
    tmp_path: Path,
) -> None:
    clip_dir = _write_clip(tmp_path)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted-test-checkpoint")
    output = tmp_path / "visualization.mp4"

    result = run_cpu_visualization(
        predictor=_PeakPredictor(),  # type: ignore[arg-type]
        checkpoint_path=checkpoint,
        clip_dir=clip_dir,
        output_path=output,
        image_size_hw=(24, 40),
        sequence_length=8,
        window_stride=8,
        inference_batch_size=1,
        peak_threshold=0.5,
        cpu_threads=1,
    )

    capture = cv2.VideoCapture(str(result.video_path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1936
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 432
    capture.release()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["device"] == "cpu"
    assert metadata["clip_id"] == "video_002/clip_023"
    assert metadata["predictions_at_or_above_threshold"] == {
        "cam0": 8,
        "cam1": 8,
        "cam2": 8,
    }
    assert len(metadata["checkpoint_sha256"]) == 64
    assert len(metadata["video_sha256"]) == 64
    assert result.preview_path.is_file()
