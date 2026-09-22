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
