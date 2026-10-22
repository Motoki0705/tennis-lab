"""Explicit, verified video or dataset-store RGB sources for presentation."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.annotation.migration.legacy_format.contracts import sha256_file
from src.tasks.annotation.migration.legacy_format.media import check_clip, decode_range
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.base.annotation.media import Timeline

from .annotation_correction_inputs import AnnotationComparison

RGBDecoder = Callable[[int, int], Generator[NDArray[np.uint8], None, None]]


@dataclass(frozen=True)
class FrameInput:
    timeline: Timeline
    metadata: dict[str, Any]
    decode: RGBDecoder


def open_frame_input(
    clip: AnnotationComparison,
    *,
    source: Literal["video", "store"],
    store_dir: Path | None,
) -> FrameInput:
    """Select one reader explicitly; never substitute a source after failure."""
    manifest = clip.prepared.manifest
    if source == "video":
        if store_dir is not None:
            raise ValueError("video input cannot specify store_dir")
        timeline = check_clip(clip.video, manifest)

        def video_frames(
            start: int, stop: int
        ) -> Generator[NDArray[np.uint8], None, None]:
            decoder = decode_range(clip.video, timeline, start, stop)
            try:
                for frame in decoder:
                    yield frame.to_ndarray(format="rgb24")
            finally:
                decoder.close()

        return FrameInput(
            timeline,
            {
                "source": "video",
                "path": str(clip.video),
                "sha256": manifest.sha256,
                "saved_resolution": [manifest.width, manifest.height],
                "jpeg_quality": None,
            },
            video_frames,
        )
    if source != "store" or store_dir is None:
        raise ValueError(
            "frame source must be video or store; store requires store_dir"
        )
    store = BallFrameStore(store_dir)
    stored_clip = store.clip_by_id(f"chat_annotation/{clip.annotation.clip_id}")
    if not (
        stored_clip.media_sha256 == clip.cache.identity.video_sha256 == manifest.sha256
    ):
        raise ValueError("store/candidates/clip_manifest media SHA-256 mismatch")
    if stored_clip.frame_count != len(manifest.frames):
        raise ValueError("store frame_count does not match clip_manifest frames")
    if (stored_clip.source_width, stored_clip.source_height) != (
        manifest.width,
        manifest.height,
    ):
        raise ValueError("store source dimensions do not match clip_manifest")
    if Fraction(stored_clip.fps) != Fraction(manifest.nominal_fps):
        raise ValueError("store FPS does not match clip_manifest")
    rows = store.clip_rows(stored_clip)
    actual_pts = [
        int(pts) * Fraction(stored_clip.time_base) for pts in store.frames["pts"][rows]
    ]
    expected_pts = [
        frame.clip_pts * Fraction(manifest.time_base) for frame in manifest.frames
    ]
    if actual_pts != expected_pts:
        raise ValueError("store presentation timestamps do not match clip_manifest")
    quality = store.metadata["jpeg_quality"]
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("store jpeg_quality must be an integer in [1, 100]")
    timeline = Timeline(
        manifest.width,
        manifest.height,
        Fraction(manifest.time_base),
        Fraction(manifest.nominal_fps),
        tuple(frame.clip_pts for frame in manifest.frames),
        tuple(frame.duration_pts for frame in manifest.frames),
    )

    def store_frames(start: int, stop: int) -> Generator[NDArray[np.uint8], None, None]:
        if not 0 <= start < stop <= stored_clip.frame_count:
            raise ValueError("invalid store frame interval")
        for index in range(start, stop):
            yield cv2.cvtColor(store.read_bgr(int(rows[index])), cv2.COLOR_BGR2RGB)

    payload = store.image_path(stored_clip)
    return FrameInput(
        timeline,
        {
            "source": "store",
            "store_dir": str(store_dir),
            "clip_id": stored_clip.clip_id,
            "media_sha256": stored_clip.media_sha256,
            "saved_resolution": [stored_clip.width, stored_clip.height],
            "source_resolution": [stored_clip.source_width, stored_clip.source_height],
            "coordinate_scale": [
                stored_clip.width / stored_clip.source_width,
                stored_clip.height / stored_clip.source_height,
            ],
            "jpeg_quality": quality,
            "storage": "video_q90"
            if stored_clip.clip_id in store.video_frames
            else "jpeg_shard",
            "metadata": {
                "path": str(store_dir / "metadata.json"),
                "sha256": sha256_file(store_dir / "metadata.json"),
            },
            "index": {
                "path": str(store_dir / "index.npz"),
                "sha256": sha256_file(store_dir / "index.npz"),
            },
            "image_payload": {
                "path": str(payload),
                "sha256": sha256_file(payload),
                "bytes": payload.stat().st_size,
            },
        },
        store_frames,
    )
