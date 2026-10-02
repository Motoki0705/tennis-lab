"""CPU CLIP crop cache shared by source thresholds, with exact input identity.

All raw tracks participate in the occlusion test. Only core-entering tracks
can ever link/qualify, so the other tracks need no embeddings. No reviewed
label, detector score or preliminary selection influences crop eligibility.
Keys include video hash, encoder weight hash, frame and exact resized RGB
pixels. The cache never approximates a new box with a neighbouring old box.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.player_association.appearance.encoders import AppearanceEncoder
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    TrackAppearance,
    crop,
    sample_tracks,
)
from src.tasks.player_association.association.associate import CameraTracks
from src.utils.video import OpenCVVideoFrameReader


def cached_appearance(tracks: CameraTracks, core: np.ndarray, video: dict[str, Any], encoder: AppearanceEncoder,
                      weight_sha: str, cache: Path) -> tuple[tuple[TrackAppearance, ...], dict[str, Any]]:
    config = CropSamplingConfig()
    samples = sample_tracks(tracks.boxes_xyxy, tracks.observed, tracks.image_size, config)
    wanted: dict[int, list[int]] = {}
    core_rows: np.ndarray = np.asarray(core.any(1))
    for row in np.flatnonzero(core_rows):
        for frame in samples[row].frames:
            wanted.setdefault(int(frame), []).append(int(row))
    vectors: dict[tuple[int, int], np.ndarray] = {}
    pending: dict[str, tuple[np.ndarray, list[tuple[int, int]]]] = {}
    counts = {'cached_crops': 0, 'new_crops': 0, 'sampled_crops': sum(len(v) for v in wanted.values()),
              'core_tracks': int(core_rows.sum()), 'all_tracks': len(core_rows)}
    cache.mkdir(parents=True, exist_ok=True)
    prefix = f"{video['sha256']}:{encoder.name}:{weight_sha}:".encode()
    def flush() -> None:
        if not pending:
            return
        keys = list(pending)
        matrix = encoder.embed(torch.from_numpy(np.stack([pending[k][0] for k in keys]))).numpy()
        if len(matrix) != len(keys) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1., atol=1e-4):
            raise ValueError('Invalid cached CLIP embeddings')
        for key, vector in zip(keys, matrix, strict=True):
            path = cache / f'{key}.npz'
            if path.exists():
                raise FileExistsError(path)
            with path.open('xb') as out:
                np.savez_compressed(out, embedding=vector, sha256=np.asarray(hashlib.sha256(vector.tobytes()).hexdigest()))
            for unit in pending[key][1]:
                vectors[unit] = vector
        counts['new_crops'] += len(keys)
        pending.clear()
    if wanted:
        for packet in OpenCVVideoFrameReader(Path(video['path']), max_frames=max(wanted) + 1):
            for row in wanted.get(packet.index, ()):
                patch = crop(packet.frame, tracks.boxes_xyxy[row, packet.index], encoder.input_size)
                key = hashlib.sha256(prefix + str(packet.index).encode() + patch.tobytes()).hexdigest()
                path = cache / f'{key}.npz'
                unit = row, packet.index
                if path.exists():
                    with np.load(path, allow_pickle=False) as a:
                        vector = a['embedding']
                        if hashlib.sha256(vector.tobytes()).hexdigest() != str(a['sha256']):
                            raise ValueError('Appearance cache hash mismatch')
                    vectors[unit] = vector
                    counts['cached_crops'] += 1
                elif key in pending:
                    pending[key][1].append(unit)
                else:
                    pending[key] = patch, [unit]
                if len(pending) >= 16:
                    flush()
        flush()
    if len(vectors) != counts['sampled_crops']:
        raise ValueError('Video ended before all CLIP crops were read')
    result = []
    for row, sample in enumerate(samples):
        frames = sample.frames if core_rows[row] else np.empty(0, np.int64)
        embeddings = np.stack([vectors[row, int(f)] for f in frames]) if len(frames) else np.empty((0, 0), np.float32)
        result.append(TrackAppearance(frames, embeddings))
    return tuple(result), {**counts, 'tracks': [{'frames': a.frames.tolist(), 'rejected': s.rejected} for a, s in zip(result, samples, strict=True)]}
