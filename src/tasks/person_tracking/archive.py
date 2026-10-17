"""Portable shared features, with explicit provenance and no object/pickle arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.person_tracking.contracts import DetectionFeatures

_FIELDS = ('rows', 'boxes', 'scores', 'poses', 'embeddings', 'appearance_valid')
# v2 requires explicit migration; v3 additionally records missing score origins.
_LEGACY_SCHEMA = 'person_detection_features_v2'
_SCHEMA = 'person_detection_features_v3'


def save_features(path: Path, frames: list[DetectionFeatures], provenance: dict[str, Any]) -> None:
    if any(f.poses is None for f in frames):
        raise ValueError('The pose feature archive requires inferred poses')
    if any(f.parts is not None for f in frames):
        raise ValueError('Use the native part archive; pose archives cannot discard part descriptors')
    if not frames or [f.frame for f in frames] != list(range(len(frames))):
        raise ValueError('Feature sequence must cover consecutive frames starting at zero')
    if len({f.embeddings.shape[1] for f in frames}) != 1:
        raise ValueError('Feature sequence changed embedding dimension')
    arrays = {field: np.concatenate([getattr(f, field) for f in frames]) for field in _FIELDS}
    arrays['score_origin'] = np.concatenate([f.score_origins() for f in frames])
    rows = arrays['rows']
    if len(np.unique(rows)) != len(rows):
        raise ValueError('Detection row reused across frames')
    arrays['offsets'] = np.cumsum([0, *[len(f.rows) for f in frames]], dtype=np.int64)
    arrays['metadata'] = np.asarray(json.dumps({'schema': _SCHEMA, 'provenance': provenance}, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def load_features(path: Path) -> tuple[list[DetectionFeatures], dict[str, Any]]:
    """Read only v3 archives; historical inputs require explicit migration."""
    with np.load(path, allow_pickle=False) as archive:
        if 'metadata' not in archive.files:
            raise ValueError('Feature archive fields differ from the contract')
        metadata = json.loads(str(archive['metadata'].item()))
        if metadata['schema'] == _LEGACY_SCHEMA:
            raise ValueError(
                f'{path}: {_LEGACY_SCHEMA} requires explicit migration to {_SCHEMA}; '
                'run python -m src.tasks.person_tracking.scripts.migrate_features '
                '<absolute-source.npz> <absolute-new-destination.npz>'
            )
        if metadata['schema'] != _SCHEMA:
            raise ValueError('Unsupported feature archive schema')
        fields = (*_FIELDS, 'score_origin')
        if set(archive.files) != {*fields, 'offsets', 'metadata'}:
            raise ValueError('Feature archive fields differ from the contract')
        arrays = {field: archive[field] for field in fields}
        offsets = archive['offsets']
    return _restore_frames(arrays, offsets), dict(metadata['provenance'])


def _restore_frames(arrays: dict[str, np.ndarray], offsets: np.ndarray) -> list[DetectionFeatures]:
    """Validate the shared timeline and observation contract after schema decoding."""
    count = len(arrays['rows'])
    if offsets.ndim != 1 or offsets.dtype != np.int64 or len(offsets) < 2 or offsets[0] != 0 \
            or offsets[-1] != count or (np.diff(offsets) < 0).any() \
            or any(len(array) != count for array in arrays.values()) or len(np.unique(arrays['rows'])) != count:
        raise ValueError('Feature archive timeline or detection row axis is inconsistent')
    return [DetectionFeatures(frame, **{field: array[start:end] for field, array in arrays.items()})
            for frame, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))]
