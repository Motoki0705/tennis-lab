"""NPZ scene loading and lightweight metadata helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _decode_npz_meta(raw: Any) -> dict[str, Any]:
    """Decode a meta payload stored as JSON bytes/string in NPZ."""
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw if isinstance(raw, dict) else {}


@dataclass(frozen=True)
class SceneMeta:
    """Lightweight metadata for scene indexing."""

    scene_idx: int
    scene_path: Path
    num_frames: int
    num_cameras: int


def load_npz_scene(scene_path: Path) -> dict[str, Any]:
    """Load a full NPZ scene into memory."""
    payload: dict[str, Any] = {}
    with np.load(scene_path, allow_pickle=True) as data:
        for key in data.files:
            if key == "meta":
                payload[key] = _decode_npz_meta(data[key])
            else:
                payload[key] = data[key].copy()
    return payload


def extract_scene_meta(scene_path: Path, scene_idx: int) -> SceneMeta:
    """Extract metadata from a scene file without full loading."""
    with np.load(scene_path, allow_pickle=True, mmap_mode="r") as data:
        meta = {}
        if "meta" in data:
            meta = _decode_npz_meta(data["meta"])
        num_cameras = int(data["num_cameras"]) if "num_cameras" in data else 0
    num_frames = int(meta.get("num_frames", 0)) if isinstance(meta, dict) else 0
    return SceneMeta(
        scene_idx=scene_idx,
        scene_path=scene_path,
        num_frames=num_frames,
        num_cameras=num_cameras,
    )


def extract_scene_meta_parallel(
    scene_files: list[Path],
    max_workers: int = 8,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[SceneMeta]:
    """Extract scene metadata in parallel."""
    results: dict[int, SceneMeta] = {}
    failed: list[int] = []
    total = len(scene_files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_scene_meta, path, idx): idx
            for idx, path in enumerate(scene_files)
        }

        for i, future in enumerate(as_completed(futures)):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # pragma: no cover - best-effort
                print(f"Warning: Failed to load scene {scene_files[idx]}: {exc}")
                failed.append(idx)

            if progress_callback:
                progress_callback(i + 1, total)

    _ = failed
    return [results[i] for i in range(total) if i in results]
