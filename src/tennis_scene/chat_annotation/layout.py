"""Published clips grouped by their source video filename."""

from __future__ import annotations

from pathlib import Path

from .runtime.contracts import ClipManifest


def video_path(output_root: Path, manifest: ClipManifest) -> Path:
    return (
        output_root
        / "videos"
        / Path(manifest.source.filename).stem
        / str(manifest.filename)
    )


def done_video_path(output_root: Path, manifest: ClipManifest) -> Path:
    return (
        output_root / "done" / Path(manifest.source.filename).stem / manifest.filename
    )


def published_video_path(output_root: Path, manifest: ClipManifest) -> Path:
    """Locate a clip after preparation or completion, rejecting ambiguous copies."""
    pending = video_path(output_root, manifest)
    done = done_video_path(output_root, manifest)
    for path in (pending, done):
        if any(
            parent.is_symlink() for parent in (path, path.parent, path.parent.parent)
        ):
            raise ValueError("published video paths must not contain symlinks")
    if pending.exists() and done.exists():
        raise ValueError("clip exists in both videos and done")
    return done if done.exists() else pending
