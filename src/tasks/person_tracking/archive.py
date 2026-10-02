"""Portable shared features, with explicit provenance and no object/pickle arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.person_tracking.contracts import DetectionFeatures

_FIELDS = ('rows', 'boxes', 'scores', 'poses', 'embeddings', 'appearance_valid')
# v2 explicitly preserves unbounded ViTPose heatmap peaks. v1 required [0,1].
_SCHEMA = 'person_detection_features_v2'


def save_features(path: Path, frames: list[DetectionFeatures], provenance: dict[str, Any]) -> None:
    if any(f.parts is not None for f in frames):
        raise ValueError('Use the native part archive; v2 cannot discard part descriptors')
    if not frames or [f.frame for f in frames] != list(range(len(frames))):
        raise ValueError('Feature sequence must cover consecutive frames starting at zero')
    if len({f.embeddings.shape[1] for f in frames}) != 1:
        raise ValueError('Feature sequence changed embedding dimension')
    arrays = {field: np.concatenate([getattr(f, field) for f in frames]) for field in _FIELDS}
    rows = arrays['rows']
    if len(np.unique(rows)) != len(rows):
        raise ValueError('Detection row reused across frames')
    arrays['offsets'] = np.cumsum([0, *[len(f.rows) for f in frames]], dtype=np.int64)
    arrays['metadata'] = np.asarray(json.dumps({'schema': _SCHEMA, 'provenance': provenance}, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def load_features(path: Path) -> tuple[list[DetectionFeatures], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {*_FIELDS, 'offsets', 'metadata'}:
            raise ValueError('Feature archive fields differ from the contract')
        metadata = json.loads(str(archive['metadata'].item()))
        if metadata['schema'] != _SCHEMA:
            raise ValueError('Unsupported feature archive schema')
        arrays = {field: archive[field] for field in _FIELDS}
        offsets = archive['offsets']
    count = len(arrays['rows'])
    if offsets.ndim != 1 or offsets.dtype != np.int64 or len(offsets) < 2 or offsets[0] != 0 \
            or offsets[-1] != count or (np.diff(offsets) < 0).any() \
            or any(len(array) != count for array in arrays.values()) or len(np.unique(arrays['rows'])) != count:
        raise ValueError('Feature archive timeline or detection row axis is inconsistent')
    frames = [DetectionFeatures(frame, **{field: arrays[field][start:end] for field in _FIELDS})
              for frame, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))]
    return frames, dict(metadata['provenance'])
