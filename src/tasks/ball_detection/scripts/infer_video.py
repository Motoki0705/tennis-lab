"""Run ball-detection inference on one video and save an annotated video.

Example commands:
    `uv run python -m src.tasks.ball_detection.scripts.infer_video`
    `uv run python -m src.tasks.ball_detection.scripts.infer_video video_path=path/to/video.mp4`
    `uv run python -m src.tasks.ball_detection.scripts.infer_video checkpoint_path=checkpoints/ball_detection/best.pt`
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import nn
from tqdm.auto import tqdm

from src.tasks.ball_detection.models import build_ball_detection_model


@dataclass(slots=True, frozen=True)
class FramePrediction:
    """One video-frame prediction produced by sliding-window inference."""

    frame_index: int
    frame_name: str
    confidence: float
    visible: bool
    x_original: float
    y_original: float
    support_count: int


def _resolve_path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(to_absolute_path(str(value))).resolve()


def _resolve_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(
            "Expected checkpoint payload to be a dict, "
            f"got {type(payload).__name__}."
        )
    return payload


def _load_model(cfg: DictConfig, *, checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = build_ball_detection_model(cfg).to(device)
    payload = _resolve_checkpoint_payload(checkpoint_path)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid model_state_dict.")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames_bgr: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_bgr.append(frame)
    finally:
        cap.release()

    if not frames_bgr:
        raise RuntimeError(f"Video contains no frames: {video_path}")
    if fps <= 0.0:
        fps = 30.0
    return frames_bgr, fps


def _preprocess_frame(frame_bgr: np.ndarray, *, image_size: tuple[int, int]) -> np.ndarray:
    image_h, image_w = image_size
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (image_w, image_h))
    return frame_rgb.astype(np.float32) / 255.0


def _to_model_input(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError(
            f"Expected input with shape (B, T, C, H, W), got {tuple(images.shape)}."
        )
    return images.permute(0, 2, 1, 3, 4).contiguous()


def _predict_frames(
    *,
    model: nn.Module,
    device: torch.device,
    frames_bgr: list[np.ndarray],
    cfg: DictConfig,
) -> list[FramePrediction]:
    num_frames = int(cfg.model.num_frames)
    image_h = int(cfg.data.image_size[0])
    image_w = int(cfg.data.image_size[1])
    heatmap_h = int(cfg.data.heatmap_size[0])
    heatmap_w = int(cfg.data.heatmap_size[1])
    threshold = float(cfg.inference.confidence_threshold)
    batch_size = int(cfg.inference.batch_size)

    if len(frames_bgr) < num_frames:
        raise ValueError(
            f"Video has {len(frames_bgr)} frames but model.num_frames={num_frames}."
        )
    if batch_size <= 0:
        raise ValueError("inference.batch_size must be positive.")

    original_height, original_width = frames_bgr[0].shape[:2]
    preprocessed = [
        _preprocess_frame(frame, image_size=(image_h, image_w))
        for frame in frames_bgr
    ]

    frame_count = len(preprocessed)
    starts = list(range(0, frame_count - num_frames + 1))
    heatmap_sums = np.zeros((frame_count, heatmap_h, heatmap_w), dtype=np.float32)
    heatmap_counts = np.zeros(frame_count, dtype=np.int32)

    progress = tqdm(
        range(0, len(starts), batch_size),
        desc="Infer video",
        leave=False,
        dynamic_ncols=True,
    )
    for batch_start_index in progress:
        batch_starts = starts[batch_start_index : batch_start_index + batch_size]
        batch = []
        for start in batch_starts:
            window = np.stack(
                [preprocessed[start + offset].transpose(2, 0, 1) for offset in range(num_frames)]
            )
            batch.append(window)
        inputs = torch.from_numpy(np.stack(batch)).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            probs = torch.sigmoid(model(_to_model_input(inputs))).squeeze(1).cpu().numpy()

        for sample_index, start in enumerate(batch_starts):
            for offset in range(num_frames):
                frame_index = start + offset
                heatmap_sums[frame_index] += probs[sample_index, offset]
                heatmap_counts[frame_index] += 1
    progress.close()

    predictions: list[FramePrediction] = []
    for frame_index in range(frame_count):
        support_count = int(heatmap_counts[frame_index])
        if support_count <= 0:
            predictions.append(
                FramePrediction(
                    frame_index=frame_index,
                    frame_name=f"{frame_index:06d}.jpg",
                    confidence=0.0,
                    visible=False,
                    x_original=0.0,
                    y_original=0.0,
                    support_count=0,
                )
            )
            continue
        avg_heatmap = heatmap_sums[frame_index] / float(support_count)
        avg_heatmap = np.nan_to_num(avg_heatmap, nan=0.0, posinf=0.0, neginf=0.0)
        confidence = float(avg_heatmap.max())
        peak_y, peak_x = np.unravel_index(int(avg_heatmap.argmax()), avg_heatmap.shape)
        visible = confidence >= threshold
        x_original = float(peak_x * original_width / max(heatmap_w, 1)) if visible else 0.0
        y_original = float(peak_y * original_height / max(heatmap_h, 1)) if visible else 0.0
        predictions.append(
            FramePrediction(
                frame_index=frame_index,
                frame_name=f"{frame_index:06d}.jpg",
                confidence=confidence,
                visible=visible,
                x_original=x_original,
                y_original=y_original,
                support_count=support_count,
            )
        )
    return predictions


def _annotate_and_write_video(
    *,
    frames_bgr: list[np.ndarray],
    predictions: list[FramePrediction],
    output_video_path: Path,
    fps: float,
    cfg: DictConfig,
) -> None:
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*str(cfg.inference.video_codec))
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {output_video_path}")

    radius = int(cfg.inference.draw.radius)
    thickness = int(cfg.inference.draw.thickness)
    font_scale = float(cfg.inference.draw.font_scale)
    try:
        for frame_bgr, prediction in zip(frames_bgr, predictions, strict=True):
            annotated = frame_bgr.copy()
            if prediction.visible:
                cv2.circle(
                    annotated,
                    center=(int(round(prediction.x_original)), int(round(prediction.y_original))),
                    radius=radius,
                    color=(0, 0, 255),
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )
            label = (
                f"frame={prediction.frame_index:04d} "
                f"conf={prediction.confidence:.3f} "
                f"support={prediction.support_count}"
            )
            cv2.putText(
                annotated,
                label,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)
    finally:
        writer.release()


def _write_predictions_json(
    *,
    checkpoint_path: Path,
    video_path: Path,
    output_json_path: Path,
    predictions: list[FramePrediction],
    fps: float,
) -> None:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": len(predictions),
        "predictions": [asdict(prediction) for prediction in predictions],
    }
    output_json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="infer_video",
)
def main(cfg: DictConfig) -> None:
    """Load a checkpoint, infer one video, and save the overlay video."""
    checkpoint_path = _resolve_path(str(cfg.checkpoint_path))
    video_path = _resolve_path(str(cfg.video_path))
    output_video_path = _resolve_path(str(cfg.output_video_path))
    output_json_path = _resolve_path(str(cfg.output_json_path))
    if checkpoint_path is None or video_path is None or output_video_path is None:
        raise ValueError("checkpoint_path, video_path, and output_video_path are required.")

    device_name = str(cfg.inference.device).lower()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = _load_model(cfg, checkpoint_path=checkpoint_path, device=device)
    frames_bgr, fps = _read_video_frames(video_path)
    predictions = _predict_frames(model=model, device=device, frames_bgr=frames_bgr, cfg=cfg)
    _annotate_and_write_video(
        frames_bgr=frames_bgr,
        predictions=predictions,
        output_video_path=output_video_path,
        fps=fps,
        cfg=cfg,
    )
    if output_json_path is not None:
        _write_predictions_json(
            checkpoint_path=checkpoint_path,
            video_path=video_path,
            output_json_path=output_json_path,
            predictions=predictions,
            fps=fps,
        )


if __name__ == "__main__":
    main()
