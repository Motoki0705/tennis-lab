"""CPU inference and H.264 rendering for one synchronized multiview clip."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.data.components.multiview import child_path, read_object
from src.tasks.ball_detection.inference import BallDetectionPredictor
from src.tasks.ball_detection.visualization.api.predict import (
    PredictionSequence,
    predict_frame_tensor,
)
from src.utils.video.writer import VideoWriter


@dataclass(frozen=True, slots=True)
class MultiviewCamera:
    """One synchronized camera stream declared by ``clip.json``."""

    camera_id: str
    video_path: Path


@dataclass(frozen=True, slots=True)
class MultiviewClip:
    """Validated metadata required for prediction and rendering."""

    clip_id: str
    clip_dir: Path
    fps: float
    frame_count: int
    original_size_wh: tuple[int, int]
    cameras: tuple[MultiviewCamera, ...]


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    """Paths and measured runtime for a completed visualization."""

    video_path: Path
    preview_path: Path
    metadata_path: Path
    inference_seconds: float
    render_seconds: float


def load_multiview_clip(clip_dir: Path) -> MultiviewClip:
    """Load one v2 clip manifest and validate its three declared videos."""
    clip_dir = clip_dir.resolve()
    manifest_path = clip_dir / "clip.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"clip.json not found: {manifest_path}")
    manifest = read_object(manifest_path)
    if manifest.get("version") != 2:
        raise ValueError(f"Unsupported clip manifest version: {manifest_path}")

    camera_ids = manifest.get("camera_ids")
    cameras_raw = manifest.get("cameras")
    if not isinstance(camera_ids, list) or not isinstance(cameras_raw, list):
        raise ValueError(f"Invalid camera inventory: {manifest_path}")
    if len(camera_ids) != 3 or len(cameras_raw) != 3:
        raise ValueError(f"Expected exactly three cameras: {manifest_path}")
    if len(set(camera_ids)) != 3:
        raise ValueError(f"Camera IDs must be unique: {manifest_path}")

    cameras: list[MultiviewCamera] = []
    for expected_id, camera in zip(camera_ids, cameras_raw, strict=True):
        if not isinstance(camera, dict) or camera.get("camera_id") != expected_id:
            raise ValueError(f"Camera ordering disagrees with camera_ids: {manifest_path}")
        camera_id = str(expected_id)
        if Path(camera_id).name != camera_id or camera_id in {".", ".."}:
            raise ValueError(f"Invalid camera ID {camera_id!r}: {manifest_path}")
        video_path = child_path(clip_dir, str(camera.get("video")))
        if not video_path.is_file():
            raise FileNotFoundError(f"Camera video not found: {video_path}")
        cameras.append(MultiviewCamera(camera_id=camera_id, video_path=video_path))

    fps = float(manifest.get("fps", 0.0))
    frame_count = int(manifest.get("num_frames", 0))
    width = int(manifest.get("width", 0))
    height = int(manifest.get("height", 0))
    if fps <= 0 or frame_count < 8 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid clip timing or dimensions: {manifest_path}")
    return MultiviewClip(
        clip_id=str(manifest["clip_id"]),
        clip_dir=clip_dir,
        fps=fps,
        frame_count=frame_count,
        original_size_wh=(width, height),
        cameras=tuple(cameras),
    )


def decode_model_images(
    camera: MultiviewCamera,
    *,
    frame_count: int,
    original_size_wh: tuple[int, int],
    image_size_hw: tuple[int, int],
) -> torch.Tensor:
    """Decode a complete stream once into model-ready CPU RGB floats."""
    model_height, model_width = image_size_hw
    capture = cv2.VideoCapture(
        str(camera.video_path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open camera video: {camera.video_path}")
    frames: list[torch.Tensor] = []
    try:
        for frame_index in range(frame_count):
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Video ended before frame {frame_index}: {camera.video_path}"
                )
            height, width = frame_bgr.shape[:2]
            if (width, height) != original_size_wh:
                raise ValueError(
                    f"Unexpected frame size {(width, height)} at "
                    f"{camera.video_path}:{frame_index}; expected {original_size_wh}."
                )
            resized_rgb = cv2.cvtColor(
                cv2.resize(frame_bgr, (model_width, model_height)),
                cv2.COLOR_BGR2RGB,
            )
            frames.append(
                torch.from_numpy(np.ascontiguousarray(resized_rgb))
                .permute(2, 0, 1)
                .to(torch.float32)
                .div_(255.0)
            )
        if capture.read()[0]:
            raise ValueError(
                f"Video contains more than {frame_count} frames: {camera.video_path}"
            )
    finally:
        capture.release()
    return torch.stack(frames)


def predict_multiview_clip(
    *,
    predictor: BallDetectionPredictor,
    clip: MultiviewClip,
    image_size_hw: tuple[int, int],
    sequence_length: int,
    window_stride: int,
    inference_batch_size: int,
    peak_threshold: float,
) -> tuple[dict[str, PredictionSequence], float]:
    """Predict all cameras sequentially so only one decoded stream occupies RAM."""
    started = time.monotonic()
    predictions: dict[str, PredictionSequence] = {}
    for camera in clip.cameras:
        print(f"[cpu-inference] decode {clip.clip_id}/{camera.camera_id}", flush=True)
        model_images = decode_model_images(
            camera,
            frame_count=clip.frame_count,
            original_size_wh=clip.original_size_wh,
            image_size_hw=image_size_hw,
        )
        print(
            f"[cpu-inference] predict {clip.clip_id}/{camera.camera_id} "
            f"frames={clip.frame_count}",
            flush=True,
        )
        predictions[camera.camera_id] = predict_frame_tensor(
            predictor=predictor,
            model_images=model_images,
            sequence_length=sequence_length,
            window_stride=window_stride,
            inference_batch_size=inference_batch_size,
            image_size_hw=image_size_hw,
            peak_threshold=peak_threshold,
        )
        del model_images
    return predictions, time.monotonic() - started


def render_multiview_video(
    *,
    clip: MultiviewClip,
    predictions: dict[str, PredictionSequence],
    output_path: Path,
    preview_path: Path,
    image_size_hw: tuple[int, int],
    peak_threshold: float,
    tile_size_wh: tuple[int, int] = (640, 360),
    trajectory_length: int = 10,
    crf: int = 17,
) -> float:
    """Render synchronized cameras with peak predictions and short trajectories."""
    if set(predictions) != {camera.camera_id for camera in clip.cameras}:
        raise ValueError("Predictions must contain exactly the clip camera IDs.")
    if trajectory_length <= 0:
        raise ValueError("trajectory_length must be positive.")
    tile_width, tile_height = tile_size_wh
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("Tile dimensions must be positive.")

    started = time.monotonic()
    captures = [cv2.VideoCapture(str(camera.video_path)) for camera in clip.cameras]
    if any(not capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise RuntimeError("Failed to open one or more multiview camera videos.")

    gap = 8
    header_height = 72
    canvas_width = len(captures) * tile_width + (len(captures) - 1) * gap
    canvas_height = header_height + tile_height
    if canvas_width % 2 or canvas_height % 2:
        raise ValueError(
            f"Rendered H.264 frame must have even dimensions, got "
            f"{canvas_width}x{canvas_height}."
        )
    candidate_frame = _preview_frame_index(predictions)
    model_height, model_width = image_size_hw

    try:
        with VideoWriter(output_path, fps=clip.fps, crf=crf) as writer:
            for frame_index in range(clip.frame_count):
                canvas: NDArray[np.uint8] = np.full(
                    (canvas_height, canvas_width, 3),
                    (18, 18, 18),
                    dtype=np.uint8,
                )
                _draw_header(
                    canvas,
                    clip_id=clip.clip_id,
                    frame_index=frame_index,
                    frame_count=clip.frame_count,
                    fps=clip.fps,
                    peak_threshold=peak_threshold,
                )
                for camera_index, (camera, capture) in enumerate(
                    zip(clip.cameras, captures, strict=True)
                ):
                    ok, frame_bgr = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"Video ended before frame {frame_index}: {camera.video_path}"
                        )
                    tile = cast(
                        NDArray[np.uint8],
                        np.asarray(
                            cv2.resize(frame_bgr, (tile_width, tile_height)),
                            dtype=np.uint8,
                        ),
                    )
                    prediction = predictions[camera.camera_id]
                    _draw_prediction(
                        tile,
                        prediction=prediction,
                        frame_index=frame_index,
                        camera_id=camera.camera_id,
                        peak_threshold=peak_threshold,
                        model_size_wh=(model_width, model_height),
                        trajectory_length=trajectory_length,
                    )
                    x0 = camera_index * (tile_width + gap)
                    canvas[
                        header_height : header_height + tile_height,
                        x0 : x0 + tile_width,
                    ] = tile

                frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                writer.write_frame(cast(NDArray[np.uint8], frame_rgb))
                if frame_index == candidate_frame:
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(preview_path), canvas):
                        raise OSError(f"Failed to write preview image: {preview_path}")
    finally:
        for capture in captures:
            capture.release()
    return time.monotonic() - started


def run_cpu_visualization(
    *,
    predictor: BallDetectionPredictor,
    checkpoint_path: Path,
    clip_dir: Path,
    output_path: Path,
    image_size_hw: tuple[int, int],
    sequence_length: int,
    window_stride: int,
    inference_batch_size: int,
    peak_threshold: float,
    cpu_threads: int,
) -> VisualizationResult:
    """Run the complete CPU-only prediction and rendering pipeline."""
    if predictor.device != torch.device("cpu"):
        raise ValueError(f"CPU visualization requires a CPU predictor, got {predictor.device}.")
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive.")
    torch.set_num_threads(cpu_threads)
    cv2.setNumThreads(1)

    clip = load_multiview_clip(clip_dir)
    predictions, inference_seconds = predict_multiview_clip(
        predictor=predictor,
        clip=clip,
        image_size_hw=image_size_hw,
        sequence_length=sequence_length,
        window_stride=window_stride,
        inference_batch_size=inference_batch_size,
        peak_threshold=peak_threshold,
    )
    output_path = output_path.resolve()
    preview_path = output_path.with_name(f"{output_path.stem}_preview.jpg")
    metadata_path = output_path.with_suffix(".json")
    render_seconds = render_multiview_video(
        clip=clip,
        predictions=predictions,
        output_path=output_path,
        preview_path=preview_path,
        image_size_hw=image_size_hw,
        peak_threshold=peak_threshold,
    )
    metadata = {
        "schema_version": "ball_multiview_visualization.v1",
        "clip_id": clip.clip_id,
        "camera_ids": [camera.camera_id for camera in clip.cameras],
        "frame_count": clip.frame_count,
        "fps": clip.fps,
        "device": "cpu",
        "cpu": platform.processor(),
        "cpu_threads": cpu_threads,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_image_size_hw": list(image_size_hw),
        "sequence_length": sequence_length,
        "window_stride": window_stride,
        "inference_batch_size": inference_batch_size,
        "peak_threshold": peak_threshold,
        "inference_seconds": inference_seconds,
        "render_seconds": render_seconds,
        "predictions_at_or_above_threshold": {
            camera_id: int(prediction.visibility.sum().item())
            for camera_id, prediction in predictions.items()
        },
        "maximum_confidence": {
            camera_id: float(prediction.confidences.max().item())
            for camera_id, prediction in predictions.items()
        },
        "video": str(output_path),
        "video_sha256": _sha256(output_path),
        "preview": str(preview_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return VisualizationResult(
        video_path=output_path,
        preview_path=preview_path,
        metadata_path=metadata_path,
        inference_seconds=inference_seconds,
        render_seconds=render_seconds,
    )


def default_cpu_threads() -> int:
    """Use a bounded CPU thread count to avoid oversubscribing inference."""
    return min(os.cpu_count() or 1, 16)


def _preview_frame_index(predictions: dict[str, PredictionSequence]) -> int:
    scores = torch.stack(
        [prediction.confidences for prediction in predictions.values()]
    ).amax(dim=0)
    return int(scores.argmax().item())


def _draw_header(
    canvas: NDArray[np.uint8],
    *,
    clip_id: str,
    frame_index: int,
    frame_count: int,
    fps: float,
    peak_threshold: float,
) -> None:
    cv2.putText(
        canvas,
        f"ConvNeXt-UNet CPU inference | {clip_id}",
        (16, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    seconds = frame_index / fps
    text = (
        f"frame {frame_index + 1}/{frame_count}  t={seconds:.2f}s  "
        f"green: confidence >= {peak_threshold:.2f}  yellow cross: peak below threshold"
    )
    cv2.putText(
        canvas,
        text,
        (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )


def _draw_prediction(
    tile: NDArray[np.uint8],
    *,
    prediction: PredictionSequence,
    frame_index: int,
    camera_id: str,
    peak_threshold: float,
    model_size_wh: tuple[int, int],
    trajectory_length: int,
) -> None:
    tile_height, tile_width = tile.shape[:2]
    model_width, model_height = model_size_wh
    visible = bool(prediction.visibility[frame_index].item())
    confidence = float(prediction.confidences[frame_index].item())
    point = _scaled_point(
        prediction.coords_px[frame_index],
        source_size_wh=(model_width, model_height),
        target_size_wh=(tile_width, tile_height),
    )

    start = max(0, frame_index - trajectory_length + 1)
    tail: list[tuple[int, int]] = []
    for index in range(start, frame_index + 1):
        if float(prediction.confidences[index].item()) >= peak_threshold:
            tail.append(
                _scaled_point(
                    prediction.coords_px[index],
                    source_size_wh=(model_width, model_height),
                    target_size_wh=(tile_width, tile_height),
                )
            )
    for first, second in zip(tail, tail[1:], strict=False):
        cv2.line(tile, first, second, (70, 220, 90), 2, cv2.LINE_AA)

    if visible:
        cv2.circle(tile, point, 9, (70, 245, 90), 2, cv2.LINE_AA)
        cv2.circle(tile, point, 2, (70, 245, 90), -1, cv2.LINE_AA)
    else:
        cv2.drawMarker(
            tile,
            point,
            (0, 205, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=11,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

    cv2.rectangle(tile, (8, 8), (216, 38), (18, 18, 18), -1)
    cv2.putText(
        tile,
        f"{camera_id}  conf={confidence:.3f}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def _scaled_point(
    coord: torch.Tensor,
    *,
    source_size_wh: tuple[int, int],
    target_size_wh: tuple[int, int],
) -> tuple[int, int]:
    source_width, source_height = source_size_wh
    target_width, target_height = target_size_wh
    x = round(float(coord[0].item()) * (target_width - 1) / (source_width - 1))
    y = round(float(coord[1].item()) * (target_height - 1) / (source_height - 1))
    return min(max(x, 0), target_width - 1), min(max(y, 0), target_height - 1)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


__all__ = [
    "MultiviewCamera",
    "MultiviewClip",
    "VisualizationResult",
    "decode_model_images",
    "default_cpu_threads",
    "load_multiview_clip",
    "predict_multiview_clip",
    "render_multiview_video",
    "run_cpu_visualization",
]
