"""DINOv3 patch-token precomputation over an issue #634 dataset.

Backbone-agnostic core: the encoder is injected as a callable so the logic is
testable without DINOv3 weights. The CLI wiring (real backbone, Hydra config)
lives in ``src/tasks/slcs/scripts/precompute_dino_tokens.py``.

Per clip camera, frames at the contract's explicit sample indices
(:func:`src.tasks.slcs.data.dino_tokens.sample_frame_indices`) are decoded,
resized to the spec's fixed input size and encoded; results are written via
:func:`src.tasks.slcs.data.dino_tokens.write_dino_tokens` (completion marker
last). Completed clips are skipped unless ``overwrite=True``; per-clip
failures are recorded and reported — the run result never hides them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.slcs.data.contract import (
    ClipManifest,
    DatasetContractError,
    DatasetIndex,
)
from src.tasks.slcs.data.dino_tokens import (
    DinoTokenSpec,
    has_dino_tokens,
    sample_frame_indices,
    write_dino_tokens,
)
from src.utils.video.reader import OpenCVVideoFrameReader

# Maps a uint8 RGB frame batch (B, H, W, 3) at the spec input size to patch
# tokens (B, S, C) float16.
FrameEncoder = Callable[[NDArray[np.uint8]], NDArray[np.float16]]


@dataclass
class PrecomputeReport:
    """Outcome of a precompute run (failures are first-class, never hidden)."""

    processed: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed


def read_frames_at(
    video_path: Path,
    frame_indices: NDArray[np.int64],
    *,
    image_height: int,
    image_width: int,
) -> NDArray[np.uint8]:
    """Decode exactly the requested frames, resized to the spec input size.

    Uses sequential decoding (indices are sorted by contract), so no
    codec-dependent random-access seeking is involved.
    """
    wanted = {int(i) for i in frame_indices}
    if len(wanted) != len(frame_indices):
        raise DatasetContractError(
            f"frame_indices contain duplicates: {frame_indices}."
        )
    frames: dict[int, NDArray[np.uint8]] = {}
    last_wanted = int(frame_indices[-1])
    reader: Iterator = iter(OpenCVVideoFrameReader(video_path))
    for packet in reader:
        if packet.index in wanted:
            resized = cv2.resize(
                packet.frame,
                (image_width, image_height),
                interpolation=cv2.INTER_AREA,
            )
            frames[packet.index] = np.ascontiguousarray(resized[..., ::-1])
        if packet.index >= last_wanted:
            break
    missing = sorted(wanted - set(frames))
    if missing:
        raise DatasetContractError(
            f"{video_path}: could not decode frames {missing} "
            f"(video shorter than the manifest claims?)."
        )
    return np.stack([frames[int(i)] for i in frame_indices], axis=0)


def precompute_clip_tokens(
    manifest: ClipManifest,
    encoder: FrameEncoder,
    spec: DinoTokenSpec,
    *,
    batch_size: int,
    overwrite: bool,
    generator: dict[str, object] | None = None,
) -> Path:
    """Encode all cameras of one clip and write the token annotation."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    frame_idx = sample_frame_indices(manifest.num_frames, spec.frame_stride)
    tokens_by_camera: dict[str, tuple[NDArray[np.float16], NDArray[np.int64]]] = {}
    for camera_id in manifest.camera_ids:
        video_path = manifest.media_path(camera_id)
        frames = read_frames_at(
            video_path,
            frame_idx,
            image_height=spec.image_height,
            image_width=spec.image_width,
        )
        chunks: list[NDArray[np.float16]] = []
        for start in range(0, frames.shape[0], batch_size):
            chunk = encoder(frames[start : start + batch_size])
            chunk = np.asarray(chunk, dtype=np.float16)
            if chunk.ndim != 3 or chunk.shape[1:] != (spec.num_tokens, spec.embed_dim):
                raise DatasetContractError(
                    f"encoder returned shape {chunk.shape}; expected "
                    f"(B, {spec.num_tokens}, {spec.embed_dim})."
                )
            chunks.append(chunk)
        tokens_by_camera[camera_id] = (np.concatenate(chunks, axis=0), frame_idx)
    return Path(
        write_dino_tokens(
            manifest,
            tokens_by_camera,
            spec,
            generator=generator,
            overwrite=overwrite,
        )
    )


def run_precompute(
    dataset_root: str | Path,
    encoder: FrameEncoder,
    spec: DinoTokenSpec,
    *,
    batch_size: int,
    overwrite: bool,
    generator: dict[str, object] | None = None,
) -> PrecomputeReport:
    """Precompute tokens for every clip in the dataset index."""
    index = DatasetIndex.load(dataset_root)
    report = PrecomputeReport()
    for ref in index.clips:
        clip_dir = index.clip_dir(ref)
        if not overwrite and has_dino_tokens(clip_dir):
            report.skipped_existing.append(ref.clip_id)
            continue
        try:
            manifest = ClipManifest.load(clip_dir)
            precompute_clip_tokens(
                manifest,
                encoder,
                spec,
                batch_size=batch_size,
                overwrite=overwrite,
                generator=generator,
            )
            report.processed.append(ref.clip_id)
        # Continue across clips so the report contains every per-clip failure.
        except Exception as exc:
            report.failed[ref.clip_id] = f"{type(exc).__name__}: {exc}"
    return report


__all__ = [
    "FrameEncoder",
    "PrecomputeReport",
    "precompute_clip_tokens",
    "read_frames_at",
    "run_precompute",
]
