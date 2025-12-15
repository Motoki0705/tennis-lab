"""Infer ball positions and render an overlay video (Hydra-based).

This script runs batched WASB inference on a single input video, optionally
applies trajectory completion, then saves a new video with the ball position
plotted on each frame.

Example commands:
    `uv run python -m src.wasb.scripts.plot_ball_video video_path=data/tennis/raw/videos/match.mp4`
    `uv run python -m src.wasb.scripts.plot_ball_video video_path=... checkpoint=... model=hrcnet device=cuda`

Config entry point: `src/wasb/configs/plot_ball_video.yaml`
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import hydra
import numpy as np
from hydra.utils import to_absolute_path
from numpy.typing import NDArray
from omegaconf import DictConfig

from src.wasb.inference import (
    HRCNetWASBPredictor,
    SingleVideoBallLocalizationPipeline,
    WASBPredictor,
    build_completer,
)


def _resolve_output_path(video_path: Path, output_path: str | None) -> Path:
    if output_path is None:
        return video_path.with_name(f"{video_path.stem}_ball.mp4")
    return Path(output_path)


def _render_overlay_video(
    *,
    video_path: Path,
    output_path: Path,
    xy_px: NDArray[np.floating[Any]] | Sequence[tuple[float, float]],
    visibility_code: NDArray[np.integer[Any]] | Sequence[int],
    fps: float,
    radius: int,
    thickness: int,
    color_detected_bgr: tuple[int, int, int],
    color_completed_bgr: tuple[int, int, int],
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        try:
            frame_idx = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if frame_idx >= len(visibility_code):
                    break

                vis = int(visibility_code[frame_idx])
                if vis > 0:
                    x, y = xy_px[frame_idx]
                    xf, yf = float(x), float(y)
                    if not (np.isfinite(xf) and np.isfinite(yf)):
                        writer.write(frame_bgr)
                        frame_idx += 1
                        continue

                    xi, yi = int(round(xf)), int(round(yf))
                    if 0 <= xi < width and 0 <= yi < height:
                        color = color_detected_bgr if vis == 1 else color_completed_bgr
                        cv2.circle(frame_bgr, (xi, yi), radius, color, thickness)

                writer.write(frame_bgr)
                frame_idx += 1
        finally:
            writer.release()
    finally:
        cap.release()


@hydra.main(config_path="../configs", config_name="plot_ball_video", version_base="1.3")  # type: ignore[misc]
def main(cfg: DictConfig) -> int:
    """Run ball localization and render an overlay video using Hydra config."""
    video_path = Path(to_absolute_path(str(cfg.video_path)))
    checkpoint = Path(to_absolute_path(str(cfg.checkpoint)))
    model_name = str(getattr(cfg, "model", "wasb")).lower()
    device = str(getattr(cfg, "device", "cpu"))
    batch_size = int(getattr(cfg, "batch_size", 64))
    max_frames = getattr(cfg, "max_frames", None)

    output_path = _resolve_output_path(
        video_path, getattr(cfg, "output_path", None)
    )
    output_path = Path(to_absolute_path(str(output_path)))

    score_threshold = float(getattr(cfg, "score_threshold", 0.5))

    if model_name == "wasb":
        predictor = WASBPredictor.load_from_checkpoint(
            checkpoint, device=device, score_threshold=score_threshold
        )
    elif model_name == "hrcnet":
        predictor = HRCNetWASBPredictor.load_from_checkpoint(
            checkpoint, device=device, score_threshold=score_threshold
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    completer = None
    completion_cfg = getattr(cfg, "completion", None)
    if completion_cfg is not None and bool(getattr(completion_cfg, "enabled", True)):
        completer = build_completer(
            method=str(getattr(completion_cfg, "method", "hybrid")),
            checkpoint_path=getattr(completion_cfg, "checkpoint_path", None),
            device=str(getattr(completion_cfg, "device", device)),
            score_threshold=score_threshold,
            max_gap=int(getattr(completion_cfg, "max_gap", 15)),
            physics_gap_threshold=int(getattr(completion_cfg, "physics_gap_threshold", 5)),
        )

    pipeline = SingleVideoBallLocalizationPipeline(
        predictor, completer=completer, batch_size=batch_size
    )
    result = pipeline.run(video_path, max_frames=max_frames)

    render_cfg = getattr(cfg, "render", None)
    radius = int(getattr(render_cfg, "radius", 6)) if render_cfg is not None else 6
    thickness = (
        int(getattr(render_cfg, "thickness", -1)) if render_cfg is not None else -1
    )

    detected_bgr: tuple[int, int, int] = cast(
        tuple[int, int, int],
        tuple(int(c) for c in getattr(render_cfg, "color_detected_bgr", [0, 255, 0])),
    )
    completed_bgr: tuple[int, int, int] = cast(
        tuple[int, int, int],
        tuple(int(c) for c in getattr(render_cfg, "color_completed_bgr", [0, 255, 255])),
    )

    use_completion = bool(getattr(render_cfg, "use_completion", True)) if render_cfg is not None else True
    if use_completion and result.completion is not None:
        xy_px = result.completion.xy
        visibility_code = result.completion.visibility
    else:
        xy_px = result.ball_xy_px
        visibility_code = result.visibility.astype("int32")

    _render_overlay_video(
        video_path=video_path,
        output_path=output_path,
        xy_px=xy_px,
        visibility_code=visibility_code,
        fps=result.fps,
        radius=radius,
        thickness=thickness,
        color_detected_bgr=detected_bgr,
        color_completed_bgr=completed_bgr,
    )

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
