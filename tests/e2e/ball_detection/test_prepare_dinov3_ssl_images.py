from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.tasks.ball_detection.scripts.youtube.prepare_dinov3_ssl_images import (
    _parse_gate_output,
    run_pipeline,
)


def test_prepare_dinov3_ssl_images_promotes_mock_accepted_frames(
    tmp_path: Path,
) -> None:
    source_video = tmp_path / "tennis_source.mp4"
    _write_tiny_video(source_video, frame_count=8)
    cfg = OmegaConf.create(
        {
            "workflow": {
                "root": str(tmp_path / "dino_ssl"),
                "sources": [
                    {
                        "video_id": "local_tennis",
                        "url": "https://www.youtube.com/watch?v=local_tennis",
                        "local_video": str(source_video),
                        "title": "tennis rally test clip",
                    }
                ],
                "discovery": {
                    "enabled": False,
                    "queries": [],
                    "max_results_per_query": 0,
                },
                "paths": {
                    "videos_dir": "videos",
                    "source_dir": "source",
                    "h264_dir": "h264",
                    "sampled_dir": "sampled",
                    "images_dir": "images",
                    "manifests_dir": "manifests",
                },
                "processing": {"max_new_videos": 1, "reprocess_existing": False},
                "download": {
                    "enabled": False,
                    "format": "best",
                    "merge_output_format": "mkv",
                    "js_runtimes": None,
                    "remote_components": None,
                    "download_archive": None,
                    "overwrite": False,
                    "extra_args": [],
                },
                "transcode": {
                    "enabled": True,
                    "ffmpeg_binary": "ffmpeg",
                    "encoder": "libx264",
                    "hwaccel": None,
                    "hwaccel_output_format": None,
                    "preset": "ultrafast",
                    "tune": None,
                    "rate_control": None,
                    "cq": None,
                    "bitrate": None,
                    "maxrate": None,
                    "bufsize": None,
                    "profile": None,
                    "pix_fmt": "yuv420p",
                    "crf": 30,
                    "overwrite": False,
                },
                "frames": {
                    "frames_per_video": 4,
                    "output_ext": "jpg",
                    "jpeg_quality": 90,
                    "overwrite": False,
                },
                "gate": {
                    "backend": "mock",
                    "mock": {"accept_all": False},
                    "contact_sheet": {
                        "max_images": 4,
                        "columns": 2,
                        "thumb_width": 64,
                        "thumb_height": 36,
                    },
                    "command_backend": {
                        "command": ["echo", "tennis"],
                        "prompt": "",
                        "accept_labels": ["tennis"],
                    },
                    "openai_compatible": {
                        "base_url": "http://127.0.0.1:8000/v1",
                        "model": "unused",
                        "timeout_sec": 1,
                        "accept_labels": ["tennis"],
                        "prompt": "",
                    },
                },
            }
        }
    )

    result = run_pipeline(cfg)

    images = sorted((tmp_path / "dino_ssl" / "images").glob("*.jpg"))
    assert result["image_count"] == 4
    assert len(images) == 4
    assert (tmp_path / "dino_ssl" / "manifests" / "images.jsonl").exists()


def test_prepare_dinov3_ssl_images_parses_vllm_json_gate_output() -> None:
    parsed = _parse_gate_output(
        '```json\n{"label": "non_tennis", "confidence": 0.91, '
        '"reason": "basketball court"}\n```'
    )

    assert parsed == {
        "label": "non_tennis",
        "confidence": 0.91,
        "reason": "basketball court",
    }


def _write_tiny_video(path: Path, *, frame_count: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 4.0, (64, 36))
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame: NDArray[np.uint8] = np.full(
                (36, 64, 3),
                index * 20,
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()
