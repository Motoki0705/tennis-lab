"""Per-attempt context, summaries and strict frame selection."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.tennis_scene.chat_annotation.runtime.contracts import (
    BallAnnotation,
    ClipManifest,
    PlayerAnnotation,
    SupportedAnnotation,
)
from src.tennis_scene.chat_annotation.runtime.validation import (
    validate_annotation,
)

from .common import (
    atomic_write_text,
    compact_ranges,
    load_annotation,
    load_manifest,
    load_task,
)

BALL_ROW_KEYS = {"reviewed", "balls", "interpolation_break", "notes"}

PLAYER_ROW_KEYS = {"reviewed", "players", "notes"}


class Ctx:
    def __init__(self, attempt_dir: Path) -> None:
        self.dir = attempt_dir.resolve()
        self.task = load_task(self.dir)
        self.target: str = self.task["target"]
        self.manifest_path = Path(self.task["manifest"])
        self.manifest: ClipManifest = load_manifest(self.manifest_path)
        self.video = Path(self.task["video"])
        self.annotation_path = Path(self.task["annotation"])
        self.work = self.dir / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.n = len(self.manifest.frames)
        self.w = self.manifest.width
        self.h = self.manifest.height

    def annotation(self) -> SupportedAnnotation:
        annotation = load_annotation(self.annotation_path)
        expected = BallAnnotation if self.target == "ball" else PlayerAnnotation
        if not isinstance(annotation, expected):
            raise ValueError(f"annotation schema does not match target {self.target}")
        return annotation

    def check_range(self, start: int, stop: int | None) -> tuple[int, int]:
        stop = self.n if stop is None else stop
        if not 0 <= start < stop <= self.n:
            raise ValueError(f"range must satisfy 0 <= start < stop <= {self.n}")
        return start, stop


def summarize(
    annotation: SupportedAnnotation, manifest: ClipManifest
) -> dict[str, Any]:
    report = validate_annotation(annotation, manifest)
    unreviewed = [f.frame_index for f in annotation.frames if not f.reviewed]
    noted = [f.frame_index for f in annotation.frames if f.notes]
    counts: dict[str, Any] = {}
    if isinstance(annotation, BallAnnotation):
        statuses: Counter[str] = Counter()
        tracks: Counter[str] = Counter()
        with_ball = 0
        for frame in annotation.frames:
            with_ball += bool(frame.balls)
            for ball in frame.balls:
                statuses[ball.status] += 1
                tracks[ball.track_id] += 1
        breaks = [f.frame_index for f in annotation.frames if f.interpolation_break]
        counts = {
            "ball_status": dict(statuses),
            "tracks": dict(tracks),
            "frames_with_ball_entry": with_ball,
            "interpolation_break_frames": len(breaks),
        }
    elif isinstance(annotation, PlayerAnnotation):
        sources: Counter[str] = Counter()
        tracks = Counter()
        per_frame: Counter[int] = Counter()
        flags: Counter[str] = Counter()
        for frame in annotation.frames:
            per_frame[len(frame.players)] += 1
            for player in frame.players:
                sources[player.bbox_source] += 1
                tracks[player.track_id] += 1
                flags["occluded"] += player.occluded
                flags["truncated"] += player.truncated
        counts = {
            "bbox_source": dict(sources),
            "tracks": dict(tracks),
            "players_per_frame": {str(k): v for k, v in sorted(per_frame.items())},
            "flags": dict(flags),
        }
    return {
        "annotation_status_field": annotation.status,
        "validation_status": report.status,
        "error_count": len(report.errors),
        "errors": report.errors[:15],
        "issue_count": len(report.issues),
        "frames": report.target_frames,
        "reviewed": report.reviewed_frames,
        "unreviewed_ranges": compact_ranges(unreviewed)[:60],
        "frames_with_notes": len(noted),
        "clip_issues": list(annotation.issues),
        "counts": counts,
    }


def brief(indices: list[int]) -> dict[str, Any]:
    ranges = compact_ranges(indices)
    return {
        "count": len(set(indices)),
        "ranges": ranges[:12] + ([["..."]] if len(ranges) > 12 else []),
    }


def dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def write_annotation(ctx: Ctx, annotation: SupportedAnnotation) -> None:
    text = json.dumps(annotation.model_dump(mode="json"), ensure_ascii=False, indent=1)
    atomic_write_text(ctx.annotation_path, text + "\n")


def parse_frames(spec: str, n: int) -> list[int]:
    """'S:E' half-open ranges and single indices, comma separated."""
    frames: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = (int(v) for v in part.split(":"))
            frames.extend(range(a, b))
        else:
            frames.append(int(part))
    bad = [f for f in frames if not 0 <= f < n]
    if bad:
        raise ValueError(f"frames outside clip: {bad[:5]}")
    return sorted(set(frames))
