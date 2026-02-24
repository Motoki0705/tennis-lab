"""Ball assist inference helpers for the annotation tool."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.tools.annotation.backend.models import (
    BallAssistMeta,
    BallClipConfig,
    BallFrameAnnotation,
)
from src.tools.annotation.backend.video import VideoFrameProvider


@dataclass(frozen=True)
class BallAssistConfig:
    """Configuration for running WASB ball assist."""

    checkpoint_path: Path | None
    model_type: Literal["wasb", "hrcnet"]
    device: Literal["cpu", "cuda"]
    batch_size: int
    score_threshold: float
    max_disp: int


def build_assist_meta(cfg: BallAssistConfig) -> BallAssistMeta:
    """Build metadata for a ball assist run."""
    return BallAssistMeta(
        checkpoint_path=str(cfg.checkpoint_path) if cfg.checkpoint_path else None,
        model_type=cfg.model_type,
        device=cfg.device,
        batch_size=cfg.batch_size,
        score_threshold=cfg.score_threshold,
        max_disp=cfg.max_disp,
        created_at=datetime.now().isoformat(),
    )


@lru_cache(maxsize=2)
def _load_predictor(
    checkpoint_path: str,
    model_type: str,
    device: str,
    score_threshold: float,
    max_disp: int,
):
    """Load and cache a WASB-compatible predictor."""
    from src.tasks.wasb.inference import HRCNetWASBPredictor, WASBPredictor

    if model_type == "hrcnet":
        predictor_cls = HRCNetWASBPredictor
    else:
        predictor_cls = WASBPredictor

    return predictor_cls.load_from_checkpoint(
        checkpoint_path,
        device=device,
        score_threshold=score_threshold,
        max_disp=max_disp,
    )


def run_ball_assist_for_clip(
    *,
    provider: VideoFrameProvider,
    clip_cfg: BallClipConfig,
    assist_cfg: BallAssistConfig,
) -> dict[int, BallFrameAnnotation]:
    """Run batched WASB inference for the current clip.

    Args:
        provider: Video frame provider.
        clip_cfg: Clip configuration.
        assist_cfg: Assist inference configuration.

    Returns:
        Mapping of local_idx -> BallFrameAnnotation.

    """
    if assist_cfg.checkpoint_path is None:
        raise FileNotFoundError("WASB checkpoint is not configured")

    checkpoint = Path(assist_cfg.checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"WASB checkpoint not found: {checkpoint}")

    predictor = _load_predictor(
        str(checkpoint),
        assist_cfg.model_type,
        assist_cfg.device,
        assist_cfg.score_threshold,
        assist_cfg.max_disp,
    )
    predictor.reset_tracker()

    clip_length = int(clip_cfg.clip_length)
    start_frame = int(clip_cfg.start_frame)
    batch_size = max(1, int(assist_cfg.batch_size))

    results: dict[int, BallFrameAnnotation] = {}

    for offset in range(0, clip_length, batch_size):
        end = min(clip_length, offset + batch_size)
        frames: list[np.ndarray] = []
        for local_idx in range(offset, end):
            frame_bgr = provider.read_frame_bgr(start_frame + local_idx)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        if not frames:
            continue

        frames_arr = np.stack(frames, axis=0)
        preds = predictor.predict(frames_arr)

        frame_indices = preds["frame_indices"]
        ball_xy_px = preds["ball_xy_px"]
        visibility = preds["visibility"]
        scores = preds["score"]

        for i, local_idx in enumerate(frame_indices):
            idx = int(local_idx)
            if idx < 0 or idx >= clip_length:
                continue
            visible = bool(visibility[i])
            x_px, y_px = float(ball_xy_px[i][0]), float(ball_xy_px[i][1])
            score = float(scores[i])
            if not visible:
                results[idx] = BallFrameAnnotation(
                    visibility=0,
                    x_px=0.0,
                    y_px=0.0,
                    score=0.0,
                    source="assist",
                )
                continue
            results[idx] = BallFrameAnnotation(
                visibility=1,
                x_px=x_px,
                y_px=y_px,
                score=score,
                source="assist",
            )

    return results
