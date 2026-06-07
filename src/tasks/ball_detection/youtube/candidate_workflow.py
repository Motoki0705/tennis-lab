"""Candidate clip selection and prediction for YouTube ball data."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.tasks.ball_detection.data.argumentation import normalize_tensor_images_imagenet
from src.tasks.ball_detection.inference import BallDetectionPredictor
from src.utils.data.heatmaps import heatmaps_to_peaks

JSONDict = dict[str, Any]
LEFT_KEYS = {81, 65361, 2424832}
RIGHT_KEYS = {83, 65363, 2555904}


@dataclass(frozen=True)
class CandidateSelectionConfig:
    """Settings for candidate selection over one video's raw frames."""

    resume: bool
    start_index: int | None
    window_name: str
    max_display_width: int
    max_display_height: int
    min_frames: int
    copy_mode: str
    overwrite: bool
    skip_small: int
    skip_medium: int
    skip_large: int


@dataclass(frozen=True)
class CandidatePredictionConfig:
    """Settings for prediction over selected candidate clips."""

    checkpoint: Path
    device: str
    sequence_length: int
    window_stride: int
    batch_size: int
    image_size: tuple[int, int]
    normalize_imagenet: bool
    imagenet_mean: tuple[float, float, float]
    imagenet_std: tuple[float, float, float]
    peak_threshold: float
    nms_kernel: int
    max_candidates_per_frame: int
    aggregation: str
    overwrite: bool


@dataclass
class SelectionState:
    """Mutable candidate selection UI state."""

    records: list[JSONDict]
    current_index: int
    clip_start_index: int | None = None
    display_scale: float = 1.0
    notification: str | None = None
    notification_until: float = 0.0


def run_candidate_selection(
    *,
    root: Path,
    video_id: str,
    raw_dir: Path,
    staging_dir: Path,
    config: CandidateSelectionConfig,
) -> int:
    """Select meaningful raw-frame ranges for one video."""
    _validate_selection_config(config)
    records = _read_jsonl(raw_dir / "frames.jsonl")
    if not records:
        raise FileNotFoundError(
            f"No raw frames found for {video_id}: {raw_dir / 'frames.jsonl'}"
        )
    if any(str(record.get("video_id")) != video_id for record in records):
        raise ValueError(f"Raw frame manifest contains a video other than {video_id}.")

    candidates = _candidate_documents(staging_dir)
    resume_index = _resume_index(candidates, len(records)) if config.resume else 0
    start_index = resume_index if config.start_index is None else int(config.start_index)
    if start_index >= len(records):
        print(
            f"[clip_and_predict] selection complete for {video_id}: "
            "last candidate reaches the final raw frame"
        )
        return 0
    if start_index < 0:
        raise ValueError("select.start_index must be non-negative.")
    state = SelectionState(
        records=records,
        current_index=start_index,
    )

    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    print(
        f"[clip_and_predict] mode=select video={video_id} "
        f"resume_index={state.current_index} candidates={len(candidates)}"
    )
    print(
        "[clip_and_predict] keys: arrows=1, a/d=10, z/x=50, "
        "k=start/end candidate, Esc=clear marker, q=quit"
    )
    try:
        while True:
            record = records[state.current_index]
            image_path = _resolve_path(root, record["image_path"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read raw frame: {image_path}")
            state.display_scale = _display_scale(image, config)
            cv2.imshow(config.window_name, _render_selection(image, state, config))
            key = cv2.waitKeyEx(20)
            if key < 0:
                continue
            if key in RIGHT_KEYS:
                _move_selection(state, config.skip_small)
                continue
            if key in LEFT_KEYS:
                _move_selection(state, -config.skip_small)
                continue
            ascii_key = key & 0xFF
            character = chr(ascii_key).lower()
            if character == "q":
                return 0
            if ascii_key == 27:
                state.clip_start_index = None
                print("[clip_and_predict] cleared candidate marker")
                continue
            if character == "d":
                _move_selection(state, config.skip_medium)
                continue
            if character == "a":
                _move_selection(state, -config.skip_medium)
                continue
            if character == "x":
                _move_selection(state, config.skip_large)
                continue
            if character == "z":
                _move_selection(state, -config.skip_large)
                continue
            if character == "k":
                if state.clip_start_index is None:
                    state.clip_start_index = state.current_index
                    print(f"[clip_and_predict] candidate start={state.current_index}")
                    continue
                try:
                    candidate = create_candidate(
                        root=root,
                        video_id=video_id,
                        staging_dir=staging_dir,
                        records=records,
                        start_index=state.clip_start_index,
                        end_index=state.current_index,
                        config=config,
                    )
                except (FileExistsError, ValueError) as error:
                    print(f"[clip_and_predict] candidate rejected: {error}")
                    continue
                print(f"[clip_and_predict] selected {candidate['clip_id']}")
                state.notification = f"SELECTED: {candidate['clip_id']}"
                state.notification_until = time.monotonic() + 2.0
                state.clip_start_index = None
                _move_selection(state, config.skip_small)
    finally:
        cv2.destroyWindow(config.window_name)


def create_candidate(
    *,
    root: Path,
    video_id: str,
    staging_dir: Path,
    records: list[JSONDict],
    start_index: int,
    end_index: int,
    config: CandidateSelectionConfig,
) -> JSONDict:
    """Create one selected staging candidate without running inference."""
    if end_index < start_index:
        raise ValueError(f"candidate end {end_index} is before start {start_index}")
    selected = records[start_index : end_index + 1]
    if len(selected) < config.min_frames:
        raise ValueError(
            f"candidate requires at least {config.min_frames} frames, got {len(selected)}"
        )
    new_range = set(range(start_index, end_index + 1))
    for existing in _candidate_documents(staging_dir):
        existing_range = set(
            range(int(existing["raw_start_index"]), int(existing["raw_end_index"]) + 1)
        )
        if new_range & existing_range:
            raise ValueError(
                f"candidate overlaps existing clip={existing['clip_id']} "
                f"range={existing['raw_start_index']}..{existing['raw_end_index']}"
            )

    number = _next_candidate_number(staging_dir)
    clip_name = f"clip_{number:06d}"
    clip_id = f"{video_id}_{clip_name}"
    candidate_dir = staging_dir / clip_name
    if candidate_dir.exists() and not config.overwrite:
        raise FileExistsError(f"candidate directory exists: {candidate_dir}")
    temporary = staging_dir / f".{clip_name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    candidate_frames: list[JSONDict] = []
    try:
        for offset, record in enumerate(selected):
            source_path = _resolve_path(root, record["image_path"])
            extension = source_path.suffix or ".jpg"
            file_name = f"{offset:06d}{extension}"
            _copy_frame(source_path, temporary / file_name, config.copy_mode)
            candidate_frames.append({
                "frame_id": record["frame_id"],
                "file_name": file_name,
                "source_image_path": record["image_path"],
                "source_frame_index": record["source_frame_index"],
                "timestamp_sec": record["timestamp_sec"],
            })
        first = selected[0]
        candidate = {
            "schema_name": "ball_youtube_candidate_clip_v2",
            "clip_id": clip_id,
            "video_id": video_id,
            "split": first["split"],
            "status": "selected",
            "raw_start_index": start_index,
            "raw_end_index": end_index,
            "start_frame_index": first["source_frame_index"],
            "end_frame_index": selected[-1]["source_frame_index"],
            "frame_count": len(selected),
            "fps": first["fps"],
            "width": first["width"],
            "height": first["height"],
            "frames": candidate_frames,
            "source": {
                "type": "youtube",
                "source_url": first.get("source_url"),
                "source_title": first.get("source_title"),
            },
            "selection": {
                "selected_at": _utc_now(),
            },
            "prediction": None,
        }
        _write_json_atomic(temporary / "candidate.json", candidate)
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        temporary.replace(candidate_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return candidate


def predict_candidates(
    *,
    root: Path,
    video_id: str,
    staging_dir: Path,
    config: CandidatePredictionConfig,
) -> int:
    """Run prediction only for selected candidates belonging to one video."""
    _validate_prediction_config(config)
    candidate_paths = sorted(staging_dir.glob("clip_*/candidate.json"))
    if not candidate_paths:
        raise FileNotFoundError(f"No selected candidates found for {video_id}: {staging_dir}")
    candidate_docs = [_read_json(path) for path in candidate_paths]
    if any(str(document.get("video_id")) != video_id for document in candidate_docs):
        raise ValueError(f"Staging directory contains a video other than {video_id}.")

    pending = [
        (path, document)
        for path, document in zip(candidate_paths, candidate_docs, strict=True)
        if config.overwrite or document.get("status") == "selected"
    ]
    if not pending:
        print(f"[clip_and_predict] no selected candidates require prediction for {video_id}")
        return 0

    predictor = BallDetectionPredictor.load_from_checkpoint(
        config.checkpoint,
        device=config.device,
    )
    checkpoint_frames = predictor.model_config.get("num_frames")
    if checkpoint_frames is not None and int(checkpoint_frames) != config.sequence_length:
        raise ValueError(
            "prediction.sequence_length does not match checkpoint model.num_frames: "
            f"{config.sequence_length} != {checkpoint_frames}."
        )

    print(
        f"[clip_and_predict] mode=predict video={video_id} candidates={len(pending)}"
    )
    for candidate_path, document in pending:
        predictions = _predict_candidate(
            candidate_dir=candidate_path.parent,
            document=document,
            predictor=predictor,
            config=config,
        )
        document["status"] = "pseudo_labeled"
        document["prediction"] = {
            "checkpoint": str(config.checkpoint),
            "sequence_length": config.sequence_length,
            "window_stride": config.window_stride,
            "aggregation": config.aggregation,
            "peak_threshold": config.peak_threshold,
            "nms_kernel": config.nms_kernel,
            "max_candidates_per_frame": config.max_candidates_per_frame,
            "predicted_at": _utc_now(),
        }
        for frame, prediction in zip(document["frames"], predictions, strict=True):
            frame["review_status"] = "pending"
            frame["predictions"] = prediction
            candidates = prediction["candidates"]
            frame["prediction"] = candidates[0] if candidates else None
            frame["balls"] = [
                {
                    "ball_id": f"b{index:03d}",
                    "prediction_id": candidate["prediction_id"],
                    "x": candidate["x"],
                    "y": candidate["y"],
                    "state": "visible",
                    "role": "target",
                    "confidence": candidate["confidence"],
                    "label_source": "pseudo",
                }
                for index, candidate in enumerate(candidates, start=1)
            ]
            frame.pop("ball", None)
        _write_jsonl(candidate_path.parent / "predictions.jsonl", predictions)
        _write_json_atomic(candidate_path, document)
        print(f"  predicted {document['clip_id']}: {len(predictions)} frames")
    return 0


def _predict_candidate(
    *,
    candidate_dir: Path,
    document: JSONDict,
    predictor: BallDetectionPredictor,
    config: CandidatePredictionConfig,
) -> list[JSONDict]:
    frame_count = len(document["frames"])
    if frame_count < config.sequence_length:
        raise ValueError(
            f"{document['clip_id']} has {frame_count} frames, "
            f"shorter than sequence_length={config.sequence_length}."
        )
    model_frames = [
        _load_model_frame(candidate_dir / str(frame["file_name"]), config)
        for frame in document["frames"]
    ]
    starts = build_window_starts(
        frame_count=frame_count,
        sequence_length=config.sequence_length,
        stride=config.window_stride,
    )
    prediction_counts = np.zeros(frame_count, dtype=np.int32)
    heatmap_sums: torch.Tensor | None = None
    heatmap_maxima: torch.Tensor | None = None
    for start_chunk in _chunked(starts, config.batch_size):
        batch = torch.stack(
            [
                torch.stack(model_frames[start : start + config.sequence_length])
                for start in start_chunk
            ]
        )
        outputs = predictor.predict(batch, return_heatmaps=True)
        heatmaps = outputs["heatmaps"].to(torch.float32)
        if heatmap_sums is None:
            heatmap_sums = torch.zeros(
                (frame_count, *heatmaps.shape[-2:]),
                dtype=torch.float32,
            )
            heatmap_maxima = torch.zeros_like(heatmap_sums)
        for window_index, start in enumerate(start_chunk):
            for offset in range(config.sequence_length):
                frame_index = start + offset
                prediction_counts[frame_index] += 1
                frame_heatmap = heatmaps[window_index, offset]
                heatmap_sums[frame_index] += frame_heatmap
                heatmap_maxima[frame_index] = torch.maximum(
                    heatmap_maxima[frame_index],
                    frame_heatmap,
                )

    if heatmap_sums is None or heatmap_maxima is None:
        raise RuntimeError(f"No prediction windows were produced for {document['clip_id']}.")
    if config.aggregation == "mean_heatmap":
        aggregated = heatmap_sums / torch.from_numpy(prediction_counts).clamp_min(1).view(-1, 1, 1)
    else:
        aggregated = heatmap_maxima
    peak_coords, peak_values, peak_valid = heatmaps_to_peaks(
        aggregated,
        threshold=config.peak_threshold,
        nms_kernel=config.nms_kernel,
        max_peaks=config.max_candidates_per_frame,
    )

    width = int(document["width"])
    height = int(document["height"])
    predictions: list[JSONDict] = []
    for index, frame in enumerate(document["frames"]):
        candidates = []
        valid_indices = torch.nonzero(peak_valid[index], as_tuple=False).flatten()
        for rank, peak_index in enumerate(valid_indices.tolist(), start=1):
            candidates.append({
                "prediction_id": f"p{rank:03d}",
                "rank": rank,
                "x": float(peak_coords[index, peak_index, 0] * max(width - 1, 0)),
                "y": float(peak_coords[index, peak_index, 1] * max(height - 1, 0)),
                "confidence": float(peak_values[index, peak_index]),
            })
        predictions.append({
            "frame_id": frame["frame_id"],
            "method": "local_peak_nms",
            "candidates": candidates,
            "prediction_count": int(prediction_counts[index]),
        })
    return predictions


def build_window_starts(*, frame_count: int, sequence_length: int, stride: int) -> list[int]:
    """Build sliding-window starts, always covering the final frame."""
    if frame_count < sequence_length:
        raise ValueError(
            f"frame_count={frame_count} is shorter than "
            f"sequence_length={sequence_length}."
        )
    starts = list(range(0, frame_count - sequence_length + 1, stride))
    last_start = frame_count - sequence_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _load_model_frame(path: Path, config: CandidatePredictionConfig) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read candidate frame: {path}")
    image_h, image_w = config.image_size
    image = cv2.resize(image, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).to(torch.float32) / 255.0
    if config.normalize_imagenet:
        tensor = normalize_tensor_images_imagenet(
            tensor.unsqueeze(0),
            mean=config.imagenet_mean,
            std=config.imagenet_std,
        ).squeeze(0)
    return tensor


def _resume_index(candidates: list[JSONDict], frame_count: int) -> int:
    if not candidates:
        return 0
    last_end = max(int(candidate["raw_end_index"]) for candidate in candidates)
    return min(last_end + 1, frame_count)


def _candidate_documents(staging_dir: Path) -> list[JSONDict]:
    return [
        _read_json(path)
        for path in sorted(staging_dir.glob("clip_*/candidate.json"))
    ]


def _next_candidate_number(staging_dir: Path) -> int:
    numbers = []
    for path in staging_dir.glob("clip_*"):
        try:
            numbers.append(int(path.name.removeprefix("clip_")))
        except ValueError:
            continue
    return max(numbers, default=0) + 1


def _validate_selection_config(config: CandidateSelectionConfig) -> None:
    skips = (config.skip_small, config.skip_medium, config.skip_large)
    if any(skip <= 0 for skip in skips):
        raise ValueError("All selection skip sizes must be positive.")
    if config.skip_large > 50:
        raise ValueError("select.skip_large must be at most 50.")
    if not (config.skip_small <= config.skip_medium <= config.skip_large):
        raise ValueError("Selection skip sizes must be ordered small <= medium <= large.")
    if config.min_frames <= 0:
        raise ValueError("select.min_frames must be positive.")
    if config.copy_mode not in {"hardlink", "copy"}:
        raise ValueError("select.copy_mode must be hardlink or copy.")


def _validate_prediction_config(config: CandidatePredictionConfig) -> None:
    if config.aggregation not in {"mean_heatmap", "max_heatmap"}:
        raise ValueError(
            f"Unsupported prediction.aggregation={config.aggregation!r}; "
            "expected mean_heatmap or max_heatmap."
        )
    if config.sequence_length <= 0:
        raise ValueError("prediction.sequence_length must be positive.")
    if config.window_stride <= 0 or config.window_stride > config.sequence_length:
        raise ValueError(
            f"prediction.window_stride must be in [1, {config.sequence_length}]."
        )
    if config.batch_size <= 0:
        raise ValueError("prediction.batch_size must be positive.")
    if config.nms_kernel <= 0 or config.nms_kernel % 2 == 0:
        raise ValueError("prediction.nms_kernel must be a positive odd integer.")
    if config.max_candidates_per_frame <= 0:
        raise ValueError("prediction.max_candidates_per_frame must be positive.")


def _render_selection(
    image: np.ndarray,
    state: SelectionState,
    config: CandidateSelectionConfig,
) -> np.ndarray:
    scale = state.display_scale
    canvas = cv2.resize(
        image,
        (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    record = state.records[state.current_index]
    marker = "none" if state.clip_start_index is None else str(state.clip_start_index)
    lines = [
        (
            f"{record['video_id']} raw {state.current_index + 1}/{len(state.records)} "
            f"source={record['source_frame_index']} time={float(record['timestamp_sec']):.2f}s"
        ),
        f"candidate_start={marker}",
        (
            f"arrows=+/-{config.skip_small} a/d=+/-{config.skip_medium} "
            f"z/x=+/-{config.skip_large} k=start/end Esc=clear q=quit"
        ),
    ]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 72), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0.0)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (12, 20 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    if state.notification is not None:
        if time.monotonic() < state.notification_until:
            _draw_selection_notification(canvas, state.notification)
        else:
            state.notification = None
    return canvas


def _draw_selection_notification(canvas: np.ndarray, message: str) -> None:
    text_size, baseline = cv2.getTextSize(
        message,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        2,
    )
    box_width = min(canvas.shape[1], text_size[0] + 32)
    box_height = text_size[1] + baseline + 24
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (30, 130, 30), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0.0, dst=canvas)
    cv2.putText(
        canvas,
        message,
        (16, text_size[1] + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _move_selection(state: SelectionState, step: int) -> None:
    state.current_index = min(
        max(state.current_index + step, 0),
        len(state.records) - 1,
    )


def _display_scale(image: np.ndarray, config: CandidateSelectionConfig) -> float:
    height, width = image.shape[:2]
    return min(
        1.0,
        config.max_display_width / max(width, 1),
        config.max_display_height / max(height, 1),
    )


def _copy_frame(source: Path, target: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _chunked(values: Sequence[int], chunk_size: int) -> Iterator[list[int]]:
    for index in range(0, len(values), chunk_size):
        yield list(values[index : index + chunk_size])


def _read_json(path: Path) -> JSONDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JSONDict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[JSONDict]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
