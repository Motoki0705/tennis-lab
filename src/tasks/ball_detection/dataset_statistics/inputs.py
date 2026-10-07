"""Verified readers. Missing optional provenance stays explicitly unavailable."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.play_intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tennis_scene.chat_annotation.player_pose.dataset import PlayerPoseStore
from src.utils.checksum import dual_sha256

from .contracts import AnnotationDetails, ClipInput, PoseInput


def annotation_details(meta: dict[str, Any], clip: ClipRecord, trusted_root: Path) -> AnnotationDetails:
    provenance = meta.get('provenance', {})
    issues = tuple(str(x) for x in provenance.get('annotation_issues', []))
    if clip.source == 'tracknet':
        return AnnotationDetails('unsupported_source', 'TrackNet has no notes/interpolation endpoints in its schema', issues)
    if 'annotation_path' not in meta:
        return AnnotationDetails('missing_path', 'Snapshot has no original annotation path', issues)
    path = Path(meta['annotation_path'])
    if not path.is_absolute() or not path.resolve().is_relative_to(trusted_root.resolve()):
        return AnnotationDetails('outside_root', 'Original annotation is outside the configured project root', issues)
    if not path.is_file():
        return AnnotationDetails('missing_file', 'Original annotation file is not present', issues)
    if dual_sha256(path) != clip.annotation_sha256:
        return AnnotationDetails('hash_mismatch', 'Original annotation differs from the dataset snapshot', issues)
    raw = json.loads(path.read_text())
    expected = 'tennis_chat_ball_annotation.v1' if clip.source == 'chat_annotation' else 'video_ball_annotation.v2'
    if raw['schema_version'] != expected or len(raw['frames']) != clip.frame_count:
        raise ValueError(f'{clip.clip_id}: original annotation schema/timeline mismatch')
    notes: dict[int, str] = {}
    endpoints: dict[int, tuple[int, int]] = {}
    for index, row in enumerate(raw['frames']):
        if row['frame_index'] != index:
            raise ValueError('Original annotation frames are not dense')
        text = row.get('notes', '')
        if text:
            notes[index] = str(text)
        if clip.source == 'chat_annotation':
            balls = row['balls']
            pair = balls[0]['interpolation_frames'] if len(balls) == 1 else None
        else:
            pair = row.get('support_frames') if row['status'] == 'interpolated' else None
        if pair is not None:
            if len(pair) != 2 or not 0 <= pair[0] < index < pair[1] < clip.frame_count:
                raise ValueError('Interpolation endpoints do not enclose the labelled frame')
            endpoints[index] = (int(pair[0]), int(pair[1]))
    source_notes = raw.get('annotation_notes', [])
    if isinstance(source_notes, str):
        source_notes = [source_notes]
    if not isinstance(source_notes, list):
        source_notes = [json.dumps(source_notes, ensure_ascii=False)]
    return AnnotationDetails('available', None, issues, notes, endpoints, clip.annotation_sha256,
                             all('notes' in row for row in raw['frames']), tuple(str(note) for note in source_notes))


def read_pose(reader: PlayerPoseStore | None, clip_id: str) -> tuple[PoseInput | None, str | None]:
    if reader is None:
        return None, 'pose_not_selected: select the pose-approved dataset to measure poses'
    arrays = reader.read_clip(clip_id)
    if arrays is None:
        return None, 'pose_skipped'
    ids = tuple(str(x) for x in arrays['player_ids'])
    observed = arrays['observed']
    if observed.dtype != np.bool_ or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError('Pose observed mask/player IDs are invalid')
    if (arrays['raw_track_ids'][observed] < 0).any() or (arrays['detection_rows'][observed] < 0).any():
        raise ValueError('Observed pose lacks its tracking/detection identity')
    boxes = arrays['boxes_xyxy'].astype(np.float64)
    if ((boxes[..., 2:] - boxes[..., :2])[observed] <= 0).any():
        raise ValueError('Observed pose has an invalid box')
    return PoseInput(arrays['keypoints'][..., :2].astype(np.float64), arrays['keypoints'][..., 2].astype(np.float64),
                     observed, boxes, ids, arrays['raw_track_ids'].astype(np.int64)), None


def read_clip(store: BallFrameStore, clip: ClipRecord, meta: dict[str, Any], trusted_root: Path,
              pose_reader: PlayerPoseStore | None) -> ClipInput:
    states = annotation_states(store, clip)
    delta = np.diff(states.times)
    nominal = 1.0 / float(Fraction(clip.fps))
    typical = float(np.median(delta)) if len(delta) else nominal
    jumps = delta > PlayIntervalConfig().timestamp_gap_factor * typical
    # There is no per-frame duration in the store. At timestamp jumps and the
    # terminal frame use nominal FPS rather than assigning the intervening gap.
    durations = np.r_[np.where(jumps, nominal, delta), nominal].astype(np.float64)
    breaks = states.boundaries.copy() | np.isin(states.events, [1, 2])
    breaks[0] = True
    breaks[1:] |= jumps
    # A target/reference transition is also a continuity boundary.
    breaks[1:] |= states.target[1:] != states.target[:-1]
    pose, reason = read_pose(pose_reader, clip.clip_id)
    return ClipInput(clip, states, durations, breaks, annotation_details(meta, clip, trusted_root), pose, reason)
