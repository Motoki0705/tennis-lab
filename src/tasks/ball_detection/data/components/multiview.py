"""Strict video_ball_annotation.v2 parsing and atomic CPU frame preparation."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from src.tasks.ball_detection.data.types import FrameLabel

COORDINATE_STATUSES = frozenset({"observed", "interpolated", "occlusion_estimated"})


def read_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, rejecting scalar/list documents."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def child_path(root: Path, relative: str) -> Path:
    """Resolve a manifest child without escaping the declared root."""
    fragment = Path(relative)
    if fragment.is_absolute() or ".." in fragment.parts or fragment == Path("."):
        raise ValueError(f"Expected a relative child path: {relative!r}")
    result = (root / fragment).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes dataset: {relative!r}")
    return result


@dataclass(frozen=True)
class AnnotatedVideo:
    """One camera stream, with unknown positions kept distinct from negatives."""

    clip_id: str
    camera_id: str
    video_path: Path
    annotation_path: Path
    source_sha256: str
    original_size: tuple[int, int]
    labels: tuple[FrameLabel | None, ...]
    breaks: tuple[bool, ...]
    status_counts: dict[str, int]

    def valid_starts(self, num_frames: int, stride: int) -> list[int]:
        """Keep consecutive supervised windows without crossing track breaks."""
        return [
            start
            for start in range(0, len(self.labels) - num_frames + 1, stride)
            if all(
                label is not None for label in self.labels[start : start + num_frames]
            )
            and not any(self.breaks[start + 1 : start + num_frames])
        ]


def read_annotated_video(
    *,
    clip_id: str,
    camera_id: str,
    video_path: Path,
    annotation_path: Path,
    original_size: tuple[int, int],
    num_frames: int,
    accepted_statuses: frozenset[str],
) -> AnnotatedVideo:
    """Validate frame alignment and use original decoded-pixel coordinates."""
    if not accepted_statuses or not accepted_statuses <= COORDINATE_STATUSES:
        raise ValueError("Invalid accepted_statuses")
    document = read_object(annotation_path)
    if document["schema_version"] != "video_ball_annotation.v2":
        raise ValueError(f"Unsupported annotation schema: {annotation_path}")
    source = document["source"]
    if (source["width"], source["height"]) != original_size or source[
        "frame_count"
    ] != num_frames:
        raise ValueError(
            f"Annotation/clip dimensions or frame count disagree: {annotation_path}"
        )
    if source["file_name"] != video_path.name or not video_path.is_file():
        raise ValueError(f"Annotation video is missing or mismatched: {video_path}")
    coordinates = document["coordinate_system"]
    if any(
        coordinates[key] != expected
        for key, expected in {
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "frame_index": "zero_based",
        }.items()
    ):
        raise ValueError(f"Unsupported coordinate system: {annotation_path}")
    digest = source["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError(f"Invalid source SHA-256: {annotation_path}")
    frames = document["frames"]
    if len(frames) != num_frames:
        raise ValueError(f"Incomplete annotation frames: {annotation_path}")
    labels: list[FrameLabel | None] = []
    breaks: list[bool] = []
    counts: Counter[str] = Counter()
    for index, frame in enumerate(frames):
        if type(frame["frame_index"]) is not int or frame["frame_index"] != index:
            raise ValueError(
                f"Non-consecutive frame_index in {annotation_path}: expected {index}"
            )
        if (
            frame["track_id"] != document["target"]["track_id"]
            or frame["label"] != "tennis_ball"
        ):
            raise ValueError(f"Unexpected target at {annotation_path}:{index}")
        status = frame["status"]
        if status not in COORDINATE_STATUSES | {"unresolved"}:
            raise ValueError(
                f"Unknown annotation status {status!r}: {annotation_path}:{index}"
            )
        center = frame["center_px"]
        label = None
        if status == "unresolved":
            if center is not None:
                raise ValueError(
                    f"Unresolved frame has coordinates: {annotation_path}:{index}"
                )
        else:
            if not isinstance(center, dict):
                raise ValueError(f"Missing center_px: {annotation_path}:{index}")
            x, y = float(center["x"]), float(center["y"])
            width, height = original_size
            if not (
                math.isfinite(x)
                and math.isfinite(y)
                and 0 <= x < width
                and 0 <= y < height
            ):
                raise ValueError(f"Invalid center_px: {annotation_path}:{index}")
            if status in accepted_statuses:
                # visibility means supervised position in the common Dataset;
                # an occlusion estimate remains a positive positional target.
                label = FrameLabel(
                    visibility=1.0,
                    x=x,
                    y=y,
                    instance_id=str(frame["track_id"]),
                    state=status,
                )
        if type(frame["break_before"]) is not bool:
            raise ValueError(f"break_before must be boolean: {annotation_path}:{index}")
        labels.append(label)
        breaks.append(frame["break_before"])
        counts[status] += 1
    return AnnotatedVideo(
        clip_id,
        camera_id,
        video_path,
        annotation_path,
        digest,
        original_size,
        tuple(labels),
        tuple(breaks),
        dict(counts),
    )


def frame_cache_path(
    root: Path, stream: AnnotatedVideo, size_hw: tuple[int, int]
) -> Path:
    """Include recording and camera identity in the unchanged Dataset window ID."""
    height, width = size_hw
    recording, clip = stream.clip_id.split("/")
    return root / f"{height}x{width}_bmp" / f"{recording}__{clip}" / stream.camera_id


def cache_metadata(stream: AnnotatedVideo, size_hw: tuple[int, int]) -> dict[str, Any]:
    """Record the source identity and decode settings for deterministic reuse."""
    stat = stream.video_path.stat()
    return {
        "version": 2,
        "sha256": stream.source_sha256,
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "size_hw": list(size_hw),
        "num_frames": len(stream.labels),
        "format": "bmp",
    }


def validate_frame_cache(
    path: Path, stream: AnnotatedVideo, size_hw: tuple[int, int]
) -> None:
    """Reject stale or partial caches instead of treating missing frames as negatives."""
    if read_object(path / "frames.json") != cache_metadata(stream, size_hw):
        raise ValueError(f"Stale frame cache; remove and prepare again: {path}")
    expected = {f"{index:06d}.bmp" for index in range(len(stream.labels))}
    if {p.name for p in path.iterdir()} != expected | {"frames.json"}:
        raise ValueError(f"Incomplete frame cache; remove and prepare again: {path}")


def prepare_frame_cache(
    path: Path, stream: AnnotatedVideo, size_hw: tuple[int, int]
) -> None:
    """Decode every frame on CPU, verify source hash/count/size, then publish atomically."""
    if path.exists():
        validate_frame_cache(path, stream, size_hw)
        return
    with stream.video_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != stream.source_sha256:
        raise ValueError(f"Video SHA-256 differs from annotation: {stream.video_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = cache_metadata(stream, size_hw)
    height, width = size_hw
    with tempfile.TemporaryDirectory(
        prefix=f".{path.name}-", dir=path.parent
    ) as temporary:
        staging = Path(temporary) / "frames"
        staging.mkdir()
        # Each video has its own decoding thread; parallel streams must not
        # each allocate an FFmpeg thread pool spanning the entire host.
        capture = cv2.VideoCapture(
            str(stream.video_path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        try:
            for index in range(len(stream.labels)):
                ok, frame = capture.read()
                if not ok or (frame.shape[1], frame.shape[0]) != stream.original_size:
                    raise ValueError(
                        f"Video decode/size failure at {stream.video_path}:{index}"
                    )
                frame = cv2.resize(frame, (width, height))
                if not cv2.imwrite(
                    str(staging / f"{index:06d}.bmp"),
                    frame,
                ):
                    raise OSError(f"Failed to write decoded frame: {staging}")
            if capture.read()[0]:
                raise ValueError(
                    f"Video has more frames than annotations: {stream.video_path}"
                )
        finally:
            capture.release()
        (staging / "frames.json").write_text(json.dumps(metadata), encoding="utf-8")
        staging.rename(path)
