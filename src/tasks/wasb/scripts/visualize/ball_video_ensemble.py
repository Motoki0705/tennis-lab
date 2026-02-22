"""Infer ball positions via heatmap ensemble and render an overlay video (Hydra-based).

This script loads multiple trained WASB Lightning checkpoints (e.g. HRNet/HRCNet
with different `frames_in`, plus a DinoV3 heatmap model), ensembles their per-frame
heatmaps, then writes an output video with the predicted ball location plotted.

Example commands:
    `uv run python -m src.wasb.scripts.visualize.ball_video_ensemble video_path=data/samples/clip.mp4`
    `uv run python -m src.wasb.scripts.visualize.ball_video_ensemble video_path=... ensemble.checkpoints='[a.ckpt,b.ckpt,c.ckpt,d.ckpt,e.ckpt]'`

Config entry point: `src/wasb/configs/plot_ball_video_ensemble.yaml`
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

from src.wasb.inference.ball_detection import HeatmapEnsemblePredictor
from src.wasb.pipeline.video_ball_localization_pipeline import (
    VideoBallLocalizationPipeline,
)


def _resolve_output_path(video_path: Path, output_path: str | None) -> Path:
    if output_path is None:
        return video_path.with_name(f"{video_path.stem}_ball_ensemble.mp4")
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

                vis = bool(visibility_code[frame_idx])
                if vis:
                    x, y = xy_px[frame_idx]
                    xf, yf = float(x), float(y)
                    if not (np.isfinite(xf) and np.isfinite(yf)):
                        writer.write(frame_bgr)
                        frame_idx += 1
                        continue

                    xi, yi = int(round(xf)), int(round(yf))
                    if 0 <= xi < width and 0 <= yi < height:
                        cv2.circle(frame_bgr, (xi, yi), radius, color_detected_bgr, thickness)

                writer.write(frame_bgr)
                frame_idx += 1
        finally:
            writer.release()
    finally:
        cap.release()


@hydra.main(
    config_path="../../configs",
    config_name="plot_ball_video_ensemble",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:
    """Run ensemble ball localization and render an overlay video using Hydra config."""
    video_path = Path(to_absolute_path(str(cfg.video_path)))
    device = str(getattr(cfg, "device", "cpu"))
    batch_size = int(getattr(cfg, "batch_size", 64))
    max_frames = getattr(cfg, "max_frames", None)

    output_path = _resolve_output_path(video_path, getattr(cfg, "output_path", None))
    output_path = Path(to_absolute_path(str(output_path)))

    ensemble_cfg = getattr(cfg, "ensemble", None)
    if ensemble_cfg is None:
        raise ValueError("Missing `ensemble` config block")

    checkpoints = [Path(to_absolute_path(str(p))) for p in list(getattr(ensemble_cfg, "checkpoints", []))]
    if len(checkpoints) == 0:
        raise ValueError("ensemble.checkpoints must be a non-empty list")

    output_heatmap_hw = getattr(ensemble_cfg, "output_heatmap_hw", None)
    output_hw_tuple: tuple[int, int] | None = None
    if output_heatmap_hw is not None:
        output_hw_tuple = (int(output_heatmap_hw[0]), int(output_heatmap_hw[1]))

    predictor = HeatmapEnsemblePredictor.load_from_checkpoint(
        tuple(checkpoints),
        device=device,
        ensemble_cfg=ensemble_cfg,
        output_heatmap_hw=output_hw_tuple,
    )

    pipeline = VideoBallLocalizationPipeline(predictor, batch_size=batch_size)
    result = pipeline.run(video_path, max_frames=max_frames)

    render_cfg = getattr(cfg, "render", None)
    radius = int(getattr(render_cfg, "radius", 6)) if render_cfg is not None else 6
    thickness = int(getattr(render_cfg, "thickness", -1)) if render_cfg is not None else -1

    detected_bgr: tuple[int, int, int] = cast(
        tuple[int, int, int],
        tuple(int(c) for c in getattr(render_cfg, "color_detected_bgr", [0, 255, 0])),
    )
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
    )

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
