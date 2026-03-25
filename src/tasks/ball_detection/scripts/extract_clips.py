"""Extract manually specified clips from raw tennis videos.

Example commands:
    `uv run python -m src.tasks.ball_detection.scripts.extract_clips`
    `uv run python -m src.tasks.ball_detection.scripts.extract_clips clips.video_1='[[2,10],[13,20]]'`

Config entry point: `src/tasks/ball_detection/configs/clip_extract.yaml`
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class RequestedClipRange:
    """A user-specified start/end range in seconds."""

    start_seconds: float
    end_seconds: float


@dataclass(slots=True)
class ClipSummary:
    """Metadata for one extracted clip."""

    clip_index: int
    file_name: str
    relative_path: str
    absolute_path: str
    requested_start_seconds: float
    requested_end_seconds: float
    actual_start_seconds: float
    actual_end_seconds: float
    start_frame: int
    end_frame_exclusive: int
    num_frames: int


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw_video_summary(input_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = input_dir / "summary.json"
    if not summary_path.exists():
        return {}

    summary = _read_json(summary_path)
    entries = summary.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid raw video summary format: {summary_path}")

    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        file_name = str(entry.get("file_name", "")).strip()
        if file_name:
            result[file_name] = entry
    return result


def _validate_codec(cfg: DictConfig) -> None:
    codec = str(cfg.extraction.codec)
    if len(codec) != 4:
        raise ValueError("`extraction.codec` must be a four-character codec tag.")


def _parse_requested_clips(cfg: DictConfig) -> dict[str, list[RequestedClipRange]]:
    clips_cfg = cfg.get("clips")
    if clips_cfg is None:
        raise ValueError(
            "No clips configured. Set `clips` in configs/clip_extract.yaml."
        )

    clip_mapping = OmegaConf.to_container(clips_cfg, resolve=True)
    if not isinstance(clip_mapping, dict):
        raise ValueError(
            "`clips` must be a mapping from video name to [start, end] ranges."
        )
    if not clip_mapping:
        raise ValueError(
            "No clips configured. Set `clips` in configs/clip_extract.yaml."
        )

    parsed: dict[str, list[RequestedClipRange]] = {}
    for video_name, ranges in clip_mapping.items():
        video_key = str(video_name).strip()
        if not video_key:
            raise ValueError("Clip config contains an empty video key.")
        if not isinstance(ranges, list) or not ranges:
            raise ValueError(
                f"`clips.{video_key}` must be a non-empty list of [start, end]."
            )

        parsed_ranges: list[RequestedClipRange] = []
        for index, item in enumerate(ranges, start=1):
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(
                    f"`clips.{video_key}[{index}]` must be a two-value list: [start, end]."
                )
            start_seconds = float(item[0])
            end_seconds = float(item[1])
            if start_seconds < 0:
                raise ValueError(
                    f"`clips.{video_key}[{index}]` start must be >= 0 seconds."
                )
            if end_seconds <= start_seconds:
                raise ValueError(
                    f"`clips.{video_key}[{index}]` end must be greater than start."
                )
            parsed_ranges.append(
                RequestedClipRange(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
            )
        parsed[video_key] = parsed_ranges
    return parsed


def _clear_existing_clips(video_output_dir: Path) -> None:
    if not video_output_dir.exists():
        return
    for clip_path in video_output_dir.glob("clip_*.mp4"):
        clip_path.unlink()


def _open_video(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return capture


def _read_video_info(video_path: Path) -> dict[str, Any]:
    capture = _open_video(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    if fps <= 0:
        raise ValueError(f"Invalid FPS for video: {video_path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame size for video: {video_path}")
    if num_frames <= 0:
        raise ValueError(f"Video contains no frames: {video_path}")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "duration_seconds": num_frames / fps,
    }


def _build_writer(
    output_path: Path,
    codec: str,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create clip writer: {output_path}")
    return writer


def _seconds_to_frame_index(seconds: float, fps: float) -> int:
    return max(int(round(seconds * fps)), 0)


def _extract_requested_clips(
    *,
    video_path: Path,
    video_output_dir: Path,
    requested_ranges: list[RequestedClipRange],
    codec: str,
) -> tuple[list[ClipSummary], dict[str, Any]]:
    video_info = _read_video_info(video_path)
    fps = float(video_info["fps"])
    width = int(video_info["width"])
    height = int(video_info["height"])
    num_frames = int(video_info["num_frames"])

    capture = _open_video(video_path)
    clips: list[ClipSummary] = []
    try:
        for clip_index, requested_range in enumerate(requested_ranges, start=1):
            start_frame = _seconds_to_frame_index(requested_range.start_seconds, fps)
            end_frame_exclusive = _seconds_to_frame_index(
                requested_range.end_seconds,
                fps,
            )
            if start_frame >= num_frames:
                raise ValueError(
                    f"Requested clip starts beyond video length: {video_path} "
                    f"start={requested_range.start_seconds}s duration={video_info['duration_seconds']:.3f}s"
                )
            if end_frame_exclusive > num_frames:
                raise ValueError(
                    f"Requested clip ends beyond video length: {video_path} "
                    f"end={requested_range.end_seconds}s duration={video_info['duration_seconds']:.3f}s"
                )
            if end_frame_exclusive <= start_frame:
                raise ValueError(
                    f"Requested clip collapses to zero frames after FPS rounding: {video_path} "
                    f"range=({requested_range.start_seconds}, {requested_range.end_seconds})"
                )

            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            clip_path = video_output_dir / f"clip_{clip_index}.mp4"
            writer = _build_writer(
                clip_path,
                codec=codec,
                fps=fps,
                width=width,
                height=height,
            )
            frames_written = 0
            try:
                while start_frame + frames_written < end_frame_exclusive:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    writer.write(frame)
                    frames_written += 1
            finally:
                writer.release()

            if frames_written != end_frame_exclusive - start_frame:
                raise RuntimeError(
                    f"Failed to extract the requested number of frames from {video_path}. "
                    f"Expected {end_frame_exclusive - start_frame}, wrote {frames_written}."
                )

            clips.append(
                ClipSummary(
                    clip_index=clip_index,
                    file_name=clip_path.name,
                    relative_path=str(
                        clip_path.relative_to(video_output_dir.parent.parent)
                    ),
                    absolute_path=str(clip_path.resolve()),
                    requested_start_seconds=requested_range.start_seconds,
                    requested_end_seconds=requested_range.end_seconds,
                    actual_start_seconds=start_frame / fps,
                    actual_end_seconds=end_frame_exclusive / fps,
                    start_frame=start_frame,
                    end_frame_exclusive=end_frame_exclusive,
                    num_frames=frames_written,
                )
            )
    finally:
        capture.release()

    return clips, video_info


def extract_clips(cfg: DictConfig) -> dict[str, Any]:
    """Extract manually specified clips under `data/tennis/extracted/clips`."""
    _validate_codec(cfg)
    requested_clips = _parse_requested_clips(cfg)

    input_dir = Path(to_absolute_path(str(cfg.input_dir)))
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    summary_path = Path(to_absolute_path(str(cfg.summary_path)))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    raw_video_summary = _load_raw_video_summary(input_dir)
    codec = str(cfg.extraction.codec)
    clear_existing_video_dir = bool(cfg.extraction.clear_existing_video_dir)

    videos_summary: list[dict[str, Any]] = []
    total_clips = 0

    for video_name, video_ranges in requested_clips.items():
        video_path = input_dir / f"{video_name}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Configured video does not exist: {video_path}")

        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        if clear_existing_video_dir:
            _clear_existing_clips(video_output_dir)

        clips, video_info = _extract_requested_clips(
            video_path=video_path,
            video_output_dir=video_output_dir,
            requested_ranges=video_ranges,
            codec=codec,
        )
        total_clips += len(clips)

        raw_entry = raw_video_summary.get(video_path.name, {})
        videos_summary.append(
            {
                "video_name": video_path.name,
                "relative_video_path": str(
                    video_path.relative_to(input_dir.parent.parent.parent)
                ),
                "absolute_video_path": str(video_path.resolve()),
                "video_output_dir": str(video_output_dir.resolve()),
                "source_url": raw_entry.get("url"),
                "source_title": raw_entry.get("title"),
                "fps": video_info["fps"],
                "width": video_info["width"],
                "height": video_info["height"],
                "num_frames": video_info["num_frames"],
                "duration_seconds": video_info["duration_seconds"],
                "clip_count": len(clips),
                "requested_ranges": [asdict(item) for item in video_ranges],
                "clips": [asdict(clip) for clip in clips],
            }
        )

    summary = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "updated_at": _timestamp(),
        "video_count": len(videos_summary),
        "clip_count": total_clips,
        "videos": videos_summary,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


@hydra.main(config_path="../configs", config_name="clip_extract", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for extracting manually specified clips from raw videos."""
    summary = extract_clips(cfg)
    print(
        json.dumps(
            {
                "video_count": summary["video_count"],
                "clip_count": summary["clip_count"],
                "output_dir": summary["output_dir"],
                "summary_path": summary["summary_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
