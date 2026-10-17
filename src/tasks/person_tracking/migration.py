"""Explicit, non-destructive migration of v2 feature archives to v3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.tasks.person_tracking.archive import (
    _FIELDS,
    _LEGACY_SCHEMA,
    _restore_frames,
    save_features,
)
from src.tasks.person_tracking.contracts import ScoreOrigin


def migrate_v2_features(source: Path, destination: Path) -> Path:
    """Preserve all v2 observations and provenance, explicitly recording LEGACY.

    Both paths must be absolute and the destination must be new. Validate the
    complete source before creating output; never modify the source archive.
    """
    if not source.is_absolute() or not destination.is_absolute():
        raise ValueError("Migration source and destination must be absolute paths")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Migration destination already exists: {destination}")
    source, destination = source.resolve(), destination.resolve()
    with np.load(source, allow_pickle=False) as archive:
        if 'metadata' not in archive.files:
            raise ValueError('Feature archive fields differ from the contract')
        metadata = json.loads(str(archive['metadata'].item()))
        if metadata['schema'] != _LEGACY_SCHEMA:
            raise ValueError(f'Migration requires a {_LEGACY_SCHEMA} source')
        if set(archive.files) != {*_FIELDS, 'offsets', 'metadata'}:
            raise ValueError('Feature archive fields differ from the contract')
        arrays = {field: archive[field] for field in _FIELDS}
        offsets = archive['offsets']
    arrays['score_origin'] = np.full(len(arrays['rows']), ScoreOrigin.LEGACY, np.uint8)
    frames = _restore_frames(arrays, offsets)
    save_features(destination, frames, dict(metadata['provenance']))
    return destination
