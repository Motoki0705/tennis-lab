"""CPU-only dataset statistics; no model execution or label mutation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.play_intervals import (
    PlayIntervalConfig,
    mask_intervals,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tennis_scene.chat_annotation.player_pose.dataset import PlayerPoseStore
from src.utils.checksum import dual_sha256

from .aggregation import aggregate
from .configuration import StatisticsConfig
from .contracts import ClipInput, Measurements
from .inputs import read_clip
from .metrics import annotations as annotation_metrics
from .metrics import gaps, interpolation, motion, pose, spatial, windows
from .scopes import scope_masks, selections
from .summaries import serialize


def compute_clip_statistics(data: ClipInput, config: StatisticsConfig) -> tuple[dict[str, Any], dict[str, Measurements]]:
    choices = selections(data, config)
    values = pose.signals(data, config)
    measurements = {}
    scopes = scope_masks(data, choices[config.scope_stride])
    for name, mask in scopes.items():
        metrics = annotation_metrics.compute(data, mask)
        for block in (gaps.compute(data, mask), interpolation.compute(data, mask), spatial.compute(data, mask, config),
                      motion.compute(data, mask, config), pose.compute(data, mask, config, values)):
            metrics.merge(block)
        measurements[name] = metrics
    measurements['windows'] = windows.compute(data, choices, config, values)
    xy = spatial.normalized_xy(data)
    report = dict(
        clip=asdict(data.clip), scopes={name: serialize(m) for name, m in measurements.items()},
        intervals={name: mask_intervals(mask) for name, mask in scopes.items()},
        times=data.states.times.tolist(),
        trajectory=[[i, float(p[0]), float(p[1])] for i, p in enumerate(xy) if data.states.visible_observation[i]],
        original_annotation=asdict(data.details), pose_availability=data.pose_reason or 'available',
        boundary_frames=np.flatnonzero(data.breaks).tolist(),
    )
    return report, measurements


def compute_dataset_statistics(
    store: BallFrameStore, clip_ids: Sequence[str], config: StatisticsConfig, *, project_root: Path,
    pose_directory: Path | None = None, progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    config.validate()
    if not clip_ids or len(set(clip_ids)) != len(clip_ids):
        raise ValueError('Select a nonempty, unique list of catalogued clips')
    identity = {name: dual_sha256(store.directory / name) for name in ('metadata.json', 'index.npz')}
    # A UI catalog may have been opened earlier. Freeze fresh arrays under the
    # hashes captured for this run, then recheck those hashes before publication.
    store = BallFrameStore(store.directory)
    metadata = json.loads((store.directory / 'metadata.json').read_text())
    records = {c['clip_id']: c for c in metadata['clips']}
    pose_identity = dual_sha256(pose_directory / 'manifest.json') if pose_directory else None
    pose_reader = PlayerPoseStore(pose_directory) if pose_directory else None
    if pose_reader is not None and pose_reader.manifest['coordinate_system'] != 'stored_jpeg_pixels':
        raise ValueError('Pose statistics require stored-JPEG pixel coordinates')
    if pose_reader is not None and pose_reader.ball_store.directory.resolve() != store.directory.resolve():
        raise ValueError('Statistics pose dataset does not match the selected ball snapshot')
    reports: dict[str, Any] = {}
    raw: dict[str, dict[str, Measurements]] = {}
    for index, clip_id in enumerate(clip_ids):
        clip = store.clip_by_id(clip_id)
        data = read_clip(store, clip, records[clip_id], project_root, pose_reader)
        reports[clip_id], raw[clip_id] = compute_clip_statistics(data, config)
        if progress is not None:
            progress(index + 1, len(clip_ids))
    groups: dict[str, list[str]] = {'all': list(clip_ids)}
    group_memberships: dict[str, set[str]] = {}
    for clip_id in clip_ids:
        clip = store.clip_by_id(clip_id)
        for key in (f'source/{clip.source}', f'split/{clip.split}', f'source_split/{clip.source}/{clip.split}'):
            groups.setdefault(key, []).append(clip_id)
        group_memberships.setdefault(f'{clip.source}/{clip.group_id}', set()).add(clip.split)
    aggregates = {key: {scope: aggregate([raw[c][scope] for c in members]) for scope in next(iter(raw.values()))}
                  for key, members in groups.items()}
    for name, digest in identity.items():
        if dual_sha256(store.directory / name) != digest:
            raise ValueError('Dataset snapshot changed during statistics computation')
    if pose_directory is not None and dual_sha256(pose_directory / 'manifest.json') != pose_identity:
        raise ValueError('Pose approval manifest changed during statistics computation')
    payload: dict[str, Any] = dict(
        schema='ball_dataset_statistics.v1', config=config.to_dict(), selection=asdict(PlayIntervalConfig(window_stride=config.scope_stride)),
        identity=dict(store=str(store.directory), hashes=identity, pose_manifest_sha256=pose_identity,
                      clip_ids_sha256=hashlib.sha256(json.dumps(sorted(clip_ids)).encode()).hexdigest()),
        clip_count=len(clip_ids), independent_videos=len(group_memberships),
        split_overlap={k: sorted(v) for k, v in group_memberships.items() if len(v) > 1},
        raw_annotation_availability=dict(Counter(report['original_annotation']['availability'] for report in reports.values())),
        pose_available_clips=sum(report['pose_availability'] == 'available' for report in reports.values()),
        groups=aggregates, clips=reports,
        limitations=['Annotation boundaries mix hits, bounces and cuts; undetected cuts remain possible.',
                     'Durations at the final frame and timestamp jumps use nominal FPS.',
                     'Pose flags are heuristics; no annotations, poses or training masks are changed.',
                     'Model accuracy and camera position are not inferred as ground truth.'],
    )
    # Catch nonfinite values at this one outward boundary, never emit NaN JSON.
    json.dumps(payload, allow_nan=False)
    return payload
