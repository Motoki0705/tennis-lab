"""Export helpers for generating dataset-compatible outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from src.tools.annotation.backend.models import CourtFrameAnnotation
from src.tools.annotation.backend.video import VideoFrameProvider
from src.wasb.tennis_format import row_from_visibility, save_label_csv


@dataclass(frozen=True)
class WasbExportConfig:
    """Export configuration for WASB-style ball labels."""

    output_dir: Path
    game_name: str = "game_tmp"
    clip_name: str = "Clip1"


def export_wasb_clip(
    *,
    provider: VideoFrameProvider,
    out_cfg: WasbExportConfig,
    start_frame: int,
    clip_length: int,
    annotations_by_local: dict[int, tuple[float, float, int, float]],
    jpeg_quality: int = 95,
) -> Path:
    """Export a contiguous clip to WASB-compatible `Label.csv` format.

    Args:
        provider: Frame provider.
        out_cfg: Output directory configuration.
        start_frame: Global start frame index.
        clip_length: Number of frames to export.
        annotations_by_local: Mapping local_idx -> (x_px, y_px, visibility, score).
        jpeg_quality: JPEG quality for exported frames.

    Returns:
        Path to the exported clip directory.
    """
    clip_dir = out_cfg.output_dir / "wasb" / out_cfg.game_name / out_cfg.clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for local_idx in range(clip_length):
        global_idx = start_frame + local_idx
        filename = f"{local_idx:04d}.jpg"
        frame_bgr = provider.read_frame_bgr(global_idx)
        cv2.imwrite(
            str(clip_dir / filename),
            frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        )

        x_px, y_px, vis, score = annotations_by_local.get(
            local_idx, (0.0, 0.0, 0, 0.0)
        )
        if int(vis) == 0:
            x_px, y_px, score = 0.0, 0.0, 0.0
        rows.append(
            row_from_visibility(
                file_name=filename,
                x=float(x_px),
                y=float(y_px),
                visibility=int(vis),
                score=float(score),
            )
        )

    save_label_csv(clip_dir / "Label.csv", rows)
    return clip_dir


@dataclass(frozen=True)
class CourtExportConfig:
    """Export configuration for court keypoint JSON annotations."""

    output_dir: Path


def export_court_keypoints(
    *,
    provider: VideoFrameProvider,
    out_cfg: CourtExportConfig,
    annotations: list[CourtFrameAnnotation],
    jpeg_quality: int = 95,
) -> Path:
    """Export annotated frames as `*_keypoints.json` plus their images."""
    import json

    out_dir = out_cfg.output_dir / "court_keypoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        stem = f"frame_{ann.frame_idx:06d}"
        image_name = f"{stem}.jpg"
        json_name = f"{stem}_keypoints.json"

        frame_bgr = provider.read_frame_bgr(ann.frame_idx)
        cv2.imwrite(
            str(out_dir / image_name),
            frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        )

        payload = {
            "image_path": image_name,
            "frame_idx": int(ann.frame_idx),
            "keypoints": [
                {
                    "x": float(kp.x_px),
                    "y": float(kp.y_px),
                    "visibility": int(kp.visibility),
                }
                for kp in ann.keypoints
            ],
        }
        (out_dir / json_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return out_dir

