"""Post-association agreement of predicted identities on the z=0 court plane.

This measures the ground-plane 3D hypothesis, not a reconstructed body. It
does not add a tuned acceptance threshold or repair an undecided identity.
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np

from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_footpoints,
)


def court_consistency(cameras: Sequence[CameraTracks], player_ids: Sequence[np.ndarray],
                      config: FootpointConfig) -> list[dict[str, Any]]:
    if len(cameras) != len(player_ids) or len({c.camera.camera_id for c in cameras}) != len(cameras):
        raise ValueError('Require one identity array per unique camera')
    points = []
    for camera, identities in zip(cameras, player_ids, strict=True):
        if identities.shape != camera.observed.shape:
            raise ValueError('Identity timeline differs from observations')
        xy, valid = ground_footpoints(camera.boxes_xyxy, camera.observed, camera.camera, camera.image_size[1], config)
        points.append((xy, valid))
    rows = []
    all_ids = sorted({int(i) for ids in player_ids for i in np.unique(ids) if i >= 0})
    for identity in all_ids:
        per_camera = []
        for ids, (xy, valid) in zip(player_ids, points, strict=True):
            seen = valid & (ids == identity)
            if (seen.sum(axis=0) > 1).any():
                raise ValueError('Unresolved same-camera identity overlap')
            per_camera.append(((xy * seen[..., None]).sum(axis=0), seen.any(axis=0)))
        for a, b in combinations(range(len(cameras)), 2):
            first, av = per_camera[a]
            second, bv = per_camera[b]
            distances = np.linalg.norm(first[av & bv] - second[av & bv], axis=-1)
            rows.append({'identity': identity, 'camera_a': cameras[a].camera.camera_id, 'camera_b': cameras[b].camera.camera_id,
                'shared_frames': len(distances), 'median_m': float(np.median(distances)) if len(distances) else None,
                'p95_m': float(np.percentile(distances, 95)) if len(distances) else None,
                'max_m': float(distances.max()) if len(distances) else None})
    return rows
