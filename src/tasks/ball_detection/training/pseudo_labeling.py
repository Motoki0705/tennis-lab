"""Pseudo-label generation utilities for semi-supervised ball detection."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from src.tasks.ball_detection.input_adapter import to_model_input

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True)
class VideoCacheMetadata:
    """Metadata describing one raw video and its shared frame cache.

    Attributes:
        source_video_path: Absolute or repo-relative raw video path.
        fps: Source video FPS.
        width: Original frame width in pixels.
        height: Original frame height in pixels.
        total_frames: Total frame count in the source video.
        cached_frames: Number of contiguous frames materialized in the cache.
    """

    source_video_path: str
    fps: float
    width: int
    height: int
    total_frames: int
    cached_frames: int


@dataclass(frozen=True)
class FramePrediction:
    """Frame-level pseudo prediction aggregated from overlapping windows.

    Attributes:
        frame_name: Cache image file name such as ``000123.jpg``.
        confidence: Peak heatmap confidence.
        visible: Whether the frame passed the pseudo confidence threshold.
        x_half: Ball x coordinate in heatmap pixel space.
        y_half: Ball y coordinate in heatmap pixel space.
        x_original: Ball x coordinate in original-image pixel space.
        y_original: Ball y coordinate in original-image pixel space.
        support_count: Number of overlapping windows contributing to the frame.
    """

    frame_name: str
    confidence: float
    visible: bool
    x_half: float
    y_half: float
    x_original: float
    y_original: float
    support_count: int


def generate_phase_pseudo_labels(
    *,
    model: nn.Module,
    device: torch.device,
    config: DictConfig,
    label_root: Path,
    phase_index: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate pseudo labels from raw videos for one semi-supervised phase."""
    semi_cfg = config.training.semi_supervised
    phase_name = f"phase_{phase_index:02d}"
    phase_dir = label_root / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(str(semi_cfg.pseudo_cache_root)).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = phase_dir / "manifest.jsonl"

    raw_videos = _list_raw_videos(
        raw_video_dir=Path(str(semi_cfg.raw_video_dir)).expanduser(),
        raw_video_glob=str(semi_cfg.get("raw_video_glob", "video_*.mp4")),
    )
    if dry_run:
        raw_videos = raw_videos[: int(semi_cfg.get("dry_run_pseudo_max_videos", 2))]

    videos_summary: list[dict[str, Any]] = []
    accepted_total = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for video_path in tqdm(raw_videos, desc=f"Pseudo {phase_name}", dynamic_ncols=True):
            video_key = _resolve_video_key(video_path)
            cache_video_dir = cache_root / video_key
            phase_video_dir = phase_dir / video_key
            phase_video_dir.mkdir(parents=True, exist_ok=True)

            accepted_starts, predictions, meta, scanned_frame_count = _generate_video_pseudo_labels(
                model=model,
                device=device,
                config=config,
                video_path=video_path,
                cache_video_dir=cache_video_dir,
            )

            label_csv_path = phase_video_dir / "Label.csv"
            _write_label_csv(
                label_csv_path,
                predictions,
                accepted_counts=_count_window_support(
                    starts=accepted_starts,
                    frame_count=scanned_frame_count,
                    window_size=int(config.model.num_frames),
                ),
            )

            video_summary = {
                "video": video_key,
                "source_video_path": str(video_path),
                "cache_dir": str(cache_video_dir),
                "frame_count": scanned_frame_count,
                "accepted_windows": len(accepted_starts),
                "accepted_starts": accepted_starts,
                "clip_meta": asdict(meta),
            }
            (phase_video_dir / "summary.json").write_text(
                json.dumps(video_summary, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            videos_summary.append(video_summary)
            accepted_total += len(accepted_starts)

            if accepted_starts:
                manifest_handle.write(
                    json.dumps(
                        {
                            "video": video_key,
                            "source_video_path": str(video_path),
                            "image_dir": str(cache_video_dir / "frames"),
                            "label_csv": str(label_csv_path),
                            "frame_count": scanned_frame_count,
                            "original_width": meta.width,
                            "original_height": meta.height,
                            "accepted_starts": accepted_starts,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

    summary = {
        "phase": phase_name,
        "label_root": str(label_root),
        "cache_root": str(cache_root),
        "manifest_path": str(manifest_path),
        "videos": len(raw_videos),
        "accepted_windows": accepted_total,
        "video_summaries": videos_summary,
    }
    (phase_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary


def select_pseudo_windows(
    *,
    predictions: Sequence[FramePrediction],
    config: DictConfig,
) -> list[int]:
    """Filter pseudo windows by visibility coverage and motion consistency."""
    semi_cfg = config.training.semi_supervised
    num_frames = int(config.model.num_frames)
    accepted: list[int] = []
    for start in range(0, len(predictions) - num_frames + 1):
        window = predictions[start : start + num_frames]
        visible_indices = [idx for idx, pred in enumerate(window) if pred.visible]
        if len(visible_indices) < int(semi_cfg.get("pseudo_min_visible_frames", 5)):
            continue
        visible_ids = np.asarray(visible_indices, dtype=np.int32)
        if visible_ids.size > 1 and int(np.diff(visible_ids).max()) > int(
            semi_cfg.get("pseudo_max_frame_gap", 2)
        ):
            continue
        points = np.asarray(
            [[window[idx].x_original, window[idx].y_original] for idx in visible_indices],
            dtype=np.float32,
        )
        if _has_motion_outlier(frame_ids=visible_ids, points=points, config=config):
            continue
        accepted.append(start)
    return accepted


def _generate_video_pseudo_labels(
    *,
    model: nn.Module,
    device: torch.device,
    config: DictConfig,
    video_path: Path,
    cache_video_dir: Path,
) -> tuple[list[int], list[FramePrediction], VideoCacheMetadata, int]:
    """Generate accepted pseudo windows for one raw video."""
    semi_cfg = config.training.semi_supervised
    num_frames = int(config.model.num_frames)
    chunk_frames = int(semi_cfg.get("pseudo_chunk_frames", num_frames * 16))
    target_windows = int(semi_cfg.get("pseudo_windows_per_video", 8))
    sample_stride = int(config.data.get("sample_stride", 1))
    if chunk_frames < num_frames:
        raise ValueError(
            "training.semi_supervised.pseudo_chunk_frames must be >= model.num_frames."
        )
    if target_windows <= 0:
        raise ValueError(
            "training.semi_supervised.pseudo_windows_per_video must be positive."
        )

    meta = _load_or_init_cache_metadata(video_path=video_path, cache_video_dir=cache_video_dir)
    if meta.total_frames < num_frames:
        return [], [], meta, 0

    frames_dir = cache_video_dir / "frames"
    chunk_start_step = max(chunk_frames - num_frames + 1, 1)
    accepted_starts: list[int] = []
    frame_predictions: dict[int, FramePrediction] = {}
    scanned_frame_count = 0

    for chunk_start in range(0, meta.total_frames - num_frames + 1, chunk_start_step):
        if len(accepted_starts) >= target_windows:
            break
        chunk_end = min(meta.total_frames, chunk_start + chunk_frames)
        meta = _ensure_cached_frames(
            video_path=video_path,
            cache_video_dir=cache_video_dir,
            meta=meta,
            target_frame_exclusive=chunk_end,
        )
        frame_paths = [
            frames_dir / f"{frame_index:06d}.jpg"
            for frame_index in range(chunk_start, chunk_end)
        ]
        local_predictions = _infer_chunk_predictions(
            model=model,
            device=device,
            config=config,
            frame_paths=frame_paths,
            original_width=meta.width,
            original_height=meta.height,
        )
        for local_index, prediction in enumerate(local_predictions):
            frame_predictions[chunk_start + local_index] = prediction

        local_starts = select_pseudo_windows(predictions=local_predictions, config=config)
        for local_start in local_starts:
            global_start = chunk_start + local_start
            if global_start % sample_stride != 0:
                continue
            accepted_starts.append(global_start)
            if len(accepted_starts) >= target_windows:
                break
        scanned_frame_count = max(scanned_frame_count, chunk_end)

    if scanned_frame_count <= 0:
        return accepted_starts, [], meta, 0

    finalized_predictions: list[FramePrediction] = []
    for frame_index in range(scanned_frame_count):
        prediction = frame_predictions.get(frame_index)
        if prediction is None:
            finalized_predictions.append(
                FramePrediction(
                    frame_name=f"{frame_index:06d}.jpg",
                    confidence=0.0,
                    visible=False,
                    x_half=0.0,
                    y_half=0.0,
                    x_original=0.0,
                    y_original=0.0,
                    support_count=0,
                )
            )
        else:
            finalized_predictions.append(prediction)
    return accepted_starts, finalized_predictions, meta, scanned_frame_count


def _list_raw_videos(*, raw_video_dir: Path, raw_video_glob: str) -> list[Path]:
    """Return raw videos sorted by file name."""
    if not raw_video_dir.exists():
        raise FileNotFoundError(f"Raw video directory not found: {raw_video_dir}")
    videos = sorted(path for path in raw_video_dir.glob(raw_video_glob) if path.is_file())
    if not videos:
        raise FileNotFoundError(
            f"No raw videos found in {raw_video_dir} matching pattern {raw_video_glob!r}."
        )
    return videos


def _resolve_video_key(video_path: Path) -> str:
    """Resolve a stable cache key from a raw video path."""
    return video_path.stem


def _load_or_init_cache_metadata(
    *,
    video_path: Path,
    cache_video_dir: Path,
) -> VideoCacheMetadata:
    """Load shared frame-cache metadata or initialize it from the raw video."""
    cache_video_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_video_dir / "meta.json"
    if meta_path.exists():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        cached_frames = int(payload.get("cached_frames", 0))
        return VideoCacheMetadata(
            source_video_path=str(payload["source_video_path"]),
            fps=float(payload["fps"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            total_frames=int(payload["total_frames"]),
            cached_frames=cached_frames,
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open raw video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    if fps <= 0.0:
        raise ValueError(f"Invalid FPS in raw video: {video_path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame size in raw video: {video_path}")
    if total_frames <= 0:
        raise ValueError(f"Raw video contains no frames: {video_path}")

    metadata = VideoCacheMetadata(
        source_video_path=str(video_path),
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        cached_frames=_count_contiguous_cached_frames(cache_video_dir / "frames"),
    )
    _write_cache_metadata(cache_video_dir, metadata)
    return metadata


def _ensure_cached_frames(
    *,
    video_path: Path,
    cache_video_dir: Path,
    meta: VideoCacheMetadata,
    target_frame_exclusive: int,
) -> VideoCacheMetadata:
    """Materialize raw video frames into the shared cache up to the target frame."""
    frames_dir = cache_video_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    target_frame_exclusive = min(target_frame_exclusive, meta.total_frames)
    if meta.cached_frames >= target_frame_exclusive:
        return meta

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open raw video for caching: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, meta.cached_frames)
        frame_index = meta.cached_frames
        while frame_index < target_frame_exclusive:
            ok, frame = cap.read()
            if not ok:
                break
            frame_path = frames_dir / f"{frame_index:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to cache frame image: {frame_path}")
            frame_index += 1
    finally:
        cap.release()

    updated = VideoCacheMetadata(
        source_video_path=meta.source_video_path,
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        total_frames=meta.total_frames,
        cached_frames=max(meta.cached_frames, frame_index),
    )
    _write_cache_metadata(cache_video_dir, updated)
    return updated


def _count_contiguous_cached_frames(frames_dir: Path) -> int:
    """Count the contiguous ``000000.jpg`` cache prefix already on disk."""
    if not frames_dir.exists():
        return 0
    frame_index = 0
    while (frames_dir / f"{frame_index:06d}.jpg").exists():
        frame_index += 1
    return frame_index


def _write_cache_metadata(cache_video_dir: Path, meta: VideoCacheMetadata) -> None:
    """Persist cache metadata for reuse across phases."""
    (cache_video_dir / "meta.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _infer_chunk_predictions(
    *,
    model: nn.Module,
    device: torch.device,
    config: DictConfig,
    frame_paths: Sequence[Path],
    original_width: int,
    original_height: int,
) -> list[FramePrediction]:
    """Infer frame predictions for one contiguous cached video chunk."""
    data_cfg = config.data
    semi_cfg = config.training.semi_supervised
    num_frames = int(config.model.num_frames)
    image_h = int(data_cfg.image_size[0])
    image_w = int(data_cfg.image_size[1])
    heatmap_h = int(data_cfg.heatmap_size[0])
    heatmap_w = int(data_cfg.heatmap_size[1])
    if len(frame_paths) < num_frames:
        return []

    heatmap_sums = np.zeros((len(frame_paths), heatmap_h, heatmap_w), dtype=np.float32)
    heatmap_counts = np.zeros(len(frame_paths), dtype=np.int32)
    starts = list(range(0, len(frame_paths) - num_frames + 1))

    model.eval()
    for batch_starts in _batched(
        starts,
        int(semi_cfg.get("pseudo_inference_batch_size", 8)),
    ):
        batch = []
        for start in batch_starts:
            frames = []
            for frame_path in frame_paths[start : start + num_frames]:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise RuntimeError(f"Failed to read cached pseudo frame: {frame_path}")
                frame = (
                    cv2.cvtColor(
                        cv2.resize(frame, (image_w, image_h)),
                        cv2.COLOR_BGR2RGB,
                    ).astype(np.float32)
                    / 255.0
                )
                frames.append(frame.transpose(2, 0, 1))
            batch.append(np.stack(frames))
        inputs = torch.from_numpy(np.stack(batch)).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            probs = torch.sigmoid(
                model(_to_model_input(inputs, config))
            ).squeeze(1).cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        for batch_index, start in enumerate(batch_starts):
            for offset in range(num_frames):
                frame_index = start + offset
                heatmap_sums[frame_index] += probs[batch_index, offset]
                heatmap_counts[frame_index] += 1

    predictions: list[FramePrediction] = []
    threshold = float(semi_cfg.get("pseudo_confidence_threshold", 0.55))
    for frame_index, frame_path in enumerate(frame_paths):
        support_count = int(heatmap_counts[frame_index])
        if support_count <= 0:
            predictions.append(
                FramePrediction(
                    frame_name=frame_path.name,
                    confidence=0.0,
                    visible=False,
                    x_half=0.0,
                    y_half=0.0,
                    x_original=0.0,
                    y_original=0.0,
                    support_count=0,
                )
            )
            continue
        avg = heatmap_sums[frame_index] / float(support_count)
        avg = np.nan_to_num(avg, nan=0.0, posinf=1.0, neginf=0.0)
        peak = float(avg.max())
        peak_y, peak_x = np.unravel_index(int(avg.argmax()), avg.shape)
        visible = peak >= threshold
        predictions.append(
            FramePrediction(
                frame_name=frame_path.name,
                confidence=peak,
                visible=visible,
                x_half=float(peak_x if visible else 0.0),
                y_half=float(peak_y if visible else 0.0),
                x_original=float(peak_x * original_width / max(heatmap_w, 1))
                if visible
                else 0.0,
                y_original=float(peak_y * original_height / max(heatmap_h, 1))
                if visible
                else 0.0,
                support_count=support_count,
            )
        )
    return predictions


def _has_motion_outlier(
    *,
    frame_ids: np.ndarray,
    points: np.ndarray,
    config: DictConfig,
) -> bool:
    """Return whether the visible points contain implausible motion."""
    semi_cfg = config.training.semi_supervised
    if len(points) < 2:
        return True
    velocities = []
    speeds = []
    for idx in range(1, len(points)):
        dt = max(int(frame_ids[idx] - frame_ids[idx - 1]), 1)
        velocity = (points[idx] - points[idx - 1]) / float(dt)
        velocities.append(velocity)
        speeds.append(float(np.linalg.norm(velocity)))
    if _contains_outlier(
        np.asarray(speeds, dtype=np.float32),
        hard_cap=float(semi_cfg.get("pseudo_max_speed_px_per_frame", 20.0)),
        mad_scale=float(semi_cfg.get("pseudo_mad_scale", 4.0)),
        min_mad=float(semi_cfg.get("pseudo_min_mad_px", 1.0)),
    ):
        return True
    if len(velocities) < 2:
        return False
    accelerations = [
        float(np.linalg.norm(velocities[idx] - velocities[idx - 1]))
        for idx in range(1, len(velocities))
    ]
    return _contains_outlier(
        np.asarray(accelerations, dtype=np.float32),
        hard_cap=float(semi_cfg.get("pseudo_max_accel_px_per_frame2", 14.0)),
        mad_scale=float(semi_cfg.get("pseudo_mad_scale", 4.0)),
        min_mad=float(semi_cfg.get("pseudo_min_mad_px", 1.0)),
    )


def _contains_outlier(
    values: np.ndarray,
    *,
    hard_cap: float,
    mad_scale: float,
    min_mad: float,
) -> bool:
    """Return whether any value violates the configured hard or MAD thresholds."""
    if values.size == 0:
        return False
    if float(values.max()) > hard_cap:
        return True
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return bool(np.any(values > median + mad_scale * max(mad, min_mad)))


def _write_label_csv(
    path: Path,
    predictions: Sequence[FramePrediction],
    *,
    accepted_counts: Sequence[int],
) -> None:
    """Write phase-specific pseudo labels for all cached frames in one video."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file name",
                "visibility",
                "x-coordinate",
                "y-coordinate",
                "status",
                "confidence",
                "support-count",
                "accepted-window-count",
            ]
        )
        for index, prediction in enumerate(predictions):
            accepted_count = int(accepted_counts[index]) if index < len(accepted_counts) else 0
            writer.writerow(
                [
                    prediction.frame_name,
                    int(prediction.visible),
                    str(int(round(prediction.x_original))) if prediction.visible else "",
                    str(int(round(prediction.y_original))) if prediction.visible else "",
                    0,
                    f"{prediction.confidence:.6f}",
                    prediction.support_count,
                    accepted_count,
                ]
            )


def _count_window_support(
    *,
    starts: Sequence[int],
    frame_count: int,
    window_size: int,
) -> list[int]:
    """Count how many accepted windows include each cached frame."""
    counts = [0] * frame_count
    for start in starts:
        for offset in range(window_size):
            frame_index = start + offset
            if 0 <= frame_index < frame_count:
                counts[frame_index] += 1
    return counts


def _batched(values: Sequence[int], batch_size: int) -> Iterator[list[int]]:
    """Yield fixed-size integer batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    batch: list[int] = []
    for value in values:
        batch.append(int(value))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _to_model_input(images: torch.Tensor, config: DictConfig) -> torch.Tensor:
    """Convert cached RGB frames into the configured model input layout."""
    return to_model_input(images, config.get("model", {}) or {})


__all__ = [
    "FramePrediction",
    "VideoCacheMetadata",
    "generate_phase_pseudo_labels",
    "select_pseudo_windows",
]
