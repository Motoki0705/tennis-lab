"""Shared paths and helpers for the Codex CLI annotation campaign.

The shared chat_annotation runtime owns schemas, video decoding and validation.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.chat_annotation.layout import published_video_path
from src.tennis_scene.chat_annotation.runtime.contracts import (
    ClipManifest,
    SupportedAnnotation,
    annotation_clip_id,
    parse_annotation,
    read_json,
)
from src.tennis_scene.chat_annotation.runtime.media import (
    Timeline,
    check_clip,
    decode_range,
)

from .configuration import paths

TARGETS = ("ball",)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, *, indent: int | None = 1) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=False)
    atomic_write_text(path, text + "\n")


def manifest_index() -> dict[str, Path]:
    """clip_id (MP4 stem) -> clip_manifest.json, rejecting duplicate identities."""
    index: dict[str, Path] = {}
    for path in sorted(
        paths().annotation_root.glob("_preparation/*/*/clips/*/clip_manifest.json")
    ):
        if path.parent.name.startswith(".building-"):
            continue
        manifest = ClipManifest.model_validate(read_json(path))
        clip_id = annotation_clip_id(manifest)
        if clip_id in index:
            raise ValueError(f"duplicate clip identity: {clip_id}")
        index[clip_id] = path
    return index


def load_manifest(path: Path) -> ClipManifest:
    return ClipManifest.model_validate(read_json(path))


def locate_video(manifest: ClipManifest) -> Path:
    path: Path = published_video_path(paths().annotation_root, manifest)
    if not path.is_file():
        raise FileNotFoundError(f"clip video not found in videos/ or done/: {path}")
    return path


def load_task(attempt_dir: Path) -> dict[str, Any]:
    path = attempt_dir.resolve() / "task.json"
    if not path.is_file():
        raise FileNotFoundError(f"task.json not found in {attempt_dir}")
    task: dict[str, Any] = read_json(path)
    if Path(task["attempt_dir"]).resolve() != attempt_dir.resolve():
        raise ValueError("task.json attempt_dir does not match the given directory")
    directory = attempt_dir.resolve()
    if not directory.is_relative_to(paths().tasks.resolve()):
        raise ValueError("attempt is outside this campaign")
    if task["target"] != "ball":
        raise ValueError("local-agent campaigns annotate balls only")
    output = Path(task["annotation"])
    if (
        output.resolve().parent != directory
        or output.name != f"annotation_{task['clip_id']}.json"
    ):
        raise ValueError(
            "annotation must be the clip JSON directly inside this attempt"
        )
    manifest_path = Path(task["manifest"]).resolve()
    if not manifest_path.is_relative_to(
        (paths().annotation_root / "_preparation").resolve()
    ):
        raise ValueError("manifest must belong to this annotation root")
    manifest = load_manifest(manifest_path)
    if annotation_clip_id(manifest) != task["clip_id"]:
        raise ValueError("task clip ID does not match the manifest")
    if Path(task["video"]).resolve() != locate_video(manifest).resolve():
        raise ValueError("task video does not match the published clip")
    previous = task.get("previous_annotation")
    if previous and not Path(previous).resolve().is_relative_to(
        paths().tasks.resolve()
    ):
        raise ValueError("previous annotation must belong to this campaign")
    return task


def load_annotation(path: Path) -> SupportedAnnotation:
    return parse_annotation(read_json(path))


def drop_page_cache(path: Path) -> None:
    """Advise the kernel to drop this file's cached pages (WSL2 does not hand page cache back
    to Windows, and hundreds of clips are read repeatedly)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


TIMELINE_SLOTS = 6  # concurrent check_clip decodes (~0.8 GB each) across every worker and the prefetcher


def cached_timeline(
    candidates: list[Path], identity: dict[str, Any]
) -> Timeline | None:
    for path in candidates:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if cached.get("identity") == identity:
            t = cached["timeline"]
            return Timeline(
                t["width"],
                t["height"],
                Fraction(t["time_base"]),
                Fraction(t["rate"]),
                tuple(t["pts"]),
                tuple(t["durations"]),
            )
    return None


@contextlib.contextmanager
def timeline_slot(max_wait: float = 900.0) -> Iterator[None]:
    """flock one of the pre-created cache/locks/timeline_slot_N files (opened read-only, so it
    works inside the worker sandbox). Raises after max_wait so a worker sees a clear retry message."""
    import fcntl
    import time

    deadline = time.monotonic() + max_wait
    while True:
        for i in range(TIMELINE_SLOTS):
            handle = (paths().locks / f"timeline_slot_{i}").open("rb")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield
            finally:
                handle.close()
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                "host busy: every clip-verification slot stayed in use; rerun this command in a minute"
            )
        time.sleep(1.0)


def verified_timeline(
    video: Path, manifest: ClipManifest, work_dir: Path | None = None
) -> Timeline:
    """Full check_clip (hash + every frame's PTS/duration vs manifest) once per video file.

    check_clip decodes the whole clip (~0.8 GB peak, seconds of CPU); later calls reuse the
    verified timeline while the file's path/size/mtime are unchanged. Cache: the shared
    cache/timeline (written outside the worker sandbox) or <work_dir>/timeline.json.
    """
    stat = video.stat()
    identity = {
        "video": str(video.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "video_sha256": manifest.sha256,
    }
    candidates = [
        (paths().campaign_dir / "cache" / "timeline") / f"{manifest.sha256}.json"
    ]
    if work_dir is not None:
        candidates.append(work_dir / "timeline.json")
    hit = cached_timeline(candidates, identity)
    if hit is not None:
        return hit
    with (
        timeline_slot()
    ):  # at most TIMELINE_SLOTS whole-clip decodes at once across all workers
        hit = cached_timeline(
            candidates, identity
        )  # another process may have verified it meanwhile
        if hit is not None:
            return hit
        timeline = check_clip(video, manifest)
    payload = {
        "identity": identity,
        "verified_at": utc_now(),
        "timeline": {
            "width": timeline.width,
            "height": timeline.height,
            "time_base": str(timeline.time_base),
            "rate": str(timeline.rate),
            "pts": list(timeline.pts),
            "durations": list(timeline.durations),
        },
    }
    for path in candidates:
        try:
            atomic_write_json(path, payload, indent=None)
            break
        except OSError:  # the shared cache is read-only inside the worker sandbox
            continue
    drop_page_cache(video)
    return timeline


def iter_frames(
    video: Path,
    manifest: ClipManifest,
    start: int,
    stop: int,
    work_dir: Path | None = None,
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """Presentation-order BGR frames [start, stop) matched to manifest PTS."""
    timeline = verified_timeline(video, manifest, work_dir)
    try:
        for index, frame in enumerate(
            decode_range(video, timeline, start, stop), start=start
        ):
            image: NDArray[np.uint8] = frame.to_ndarray(format="bgr24").astype(
                np.uint8, copy=False
            )
            yield index, image
    finally:
        drop_page_cache(video)


def compact_ranges(indices: list[int]) -> list[list[int]]:
    """Sorted unique ints -> half-open [start, stop) ranges."""
    result: list[list[int]] = []
    for index in sorted(set(indices)):
        if result and result[-1][1] == index:
            result[-1][1] = index + 1
        else:
            result.append([index, index + 1])
    return result
