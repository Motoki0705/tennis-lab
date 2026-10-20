"""Provenance for presentation GIFs made from measured ball predictions."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.store import BallFrameStore
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


def build_overlay_manifest(
    *,
    store_dir: Path,
    clip_id: str,
    frame_names: Sequence[str],
    start_frame: int,
    checkpoint: Path,
    resolved_config: dict[str, Any],
) -> dict[str, Any]:
    """Hash the store tables, clip payload and checkpoint used for one GIF."""
    store = BallFrameStore(store_dir)
    clip = store.clip_by_id(clip_id)
    inputs = [store_dir / "metadata.json", store_dir / "index.npz", store.image_path(clip)]
    commit = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"], text=True,
    ).strip()
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "config": resolved_config,
        "inputs": [{"path": str(path), "sha256": dual_sha256(path)} for path in inputs],
        "checkpoint": {"path": str(checkpoint), "sha256": dual_sha256(checkpoint)},
        "clip_id": clip_id,
        "frame_names": list(frame_names),
        "frame_range": {"start": start_frame, "stop": start_frame + len(frame_names)},
        "elements": {"video": "measured", "predictions": "measured"},
    }
