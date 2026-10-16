"""Pose discontinuity measurements and candidate flags, never training filters."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..configuration import StatisticsConfig
from ..contracts import ClipInput, Measurements
from ..scopes import runs

JOINTS = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
          'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
          'right_knee', 'left_ankle', 'right_ankle')
BONES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


@dataclass(frozen=True)
class PoseSignals:
    speed: NDArray[np.float64]
    relative_speed: NDArray[np.float64]
    spike: NDArray[np.float64]
    bone_change: NDArray[np.float64]
    swapped: NDArray[np.float64]
    translation: NDArray[np.float64]
    flags: NDArray[np.bool_]
    valid: NDArray[np.bool_]
    recovery: tuple[tuple[int, int, int, float], ...]


def signals(data: ClipInput, config: StatisticsConfig) -> PoseSignals | None:
    p = data.pose
    if p is None:
        return None
    n, people, joints, _ = p.points.shape
    valid = p.observed[..., None] & (p.scores >= config.pose_min_score)
    speed = np.full((n, people, joints), np.nan)
    relative = speed.copy()
    spike = speed.copy()
    bone = np.full((n, people, len(BONES)), np.nan)
    swapped = np.full((n, people, len(PAIRS)), np.nan)
    translation = np.full((n, people), np.nan)
    scale = np.array([np.median((p.boxes[:, i, 3] - p.boxes[:, i, 1])[p.observed[:, i]])
                      if p.observed[:, i].any() else np.nan for i in range(people)])
    pair_valid = valid[1:] & valid[:-1] & ~data.breaks[1:, None, None]
    delta = np.diff(p.points, axis=0)
    dt = np.diff(data.states.times)
    speed[1:] = np.where(pair_valid, np.linalg.norm(delta, axis=-1) / scale[None, :, None] / dt[:, None, None], np.nan)
    for player in range(people):
        for index in np.flatnonzero(pair_valid[:, player].sum(axis=1) >= 3):
            center_delta = np.median(delta[index, player, pair_valid[index, player]], axis=0)
            translation[index + 1, player] = np.linalg.norm(center_delta) / scale[player] / dt[index]
            relative[index + 1, player] = np.where(pair_valid[index, player],
                np.linalg.norm(delta[index, player] - center_delta, axis=-1) / scale[player] / dt[index], np.nan)
    if n > 2:
        weight = dt[:-1] / (dt[:-1] + dt[1:])
        linear = p.points[:-2] + (p.points[2:] - p.points[:-2]) * weight[:, None, None, None]
        triple = pair_valid[:-1] & pair_valid[1:]
        spike[1:-1] = np.where(triple, np.linalg.norm(p.points[1:-1] - linear, axis=-1) / scale[None, :, None], np.nan)
    for index, (a, b) in enumerate(BONES):
        length = np.linalg.norm(p.points[:, :, a] - p.points[:, :, b], axis=-1)
        bone[1:, :, index] = np.where(pair_valid[:, :, a] & pair_valid[:, :, b],
                                      np.abs(np.diff(length, axis=0)) / scale, np.nan)
    for index, (a, b) in enumerate(PAIRS):
        normal = np.linalg.norm(delta[:, :, a], axis=-1) + np.linalg.norm(delta[:, :, b], axis=-1)
        swap = np.linalg.norm(p.points[1:, :, a] - p.points[:-1, :, b], axis=-1) + np.linalg.norm(p.points[1:, :, b] - p.points[:-1, :, a], axis=-1)
        swapped[1:, :, index] = np.where(pair_valid[:, :, a] & pair_valid[:, :, b], (normal - swap) / scale, np.nan)
    recovery = []
    for player in range(people):
        ids = np.flatnonzero(p.observed[:, player])
        for a, b in zip(ids[:-1], ids[1:], strict=True):
            common = valid[a, player] & valid[b, player]
            if b > a + 1 and common.any() and not data.breaks[a + 1:b + 1].any():
                step = np.median(np.linalg.norm(p.points[b, player, common] - p.points[a, player, common], axis=1)) / scale[player]
                recovery.append((int(a), int(b), player, float(step)))
    flags = (speed > config.pose_jump_primary) | (relative > config.pose_jump_primary) | (spike > config.pose_spike_threshold)
    return PoseSignals(speed, relative, spike, bone, swapped, translation, flags, valid, tuple(recovery))


def compute(data: ClipInput, scope: NDArray[np.bool_], config: StatisticsConfig, values: PoseSignals | None) -> Measurements:
    result = Measurements()
    p = data.pose
    result.rates['pose/available_frames'] = (int(scope.sum()) if p is not None else 0, int(scope.sum()))
    if values is None or p is None:
        result.views['pose/availability'] = data.pose_reason
        return result
    result.views['pose/availability'] = 'available'
    n, people = int(scope.sum()), len(p.player_ids)
    # Derivative samples require all contributing frames inside this scope.
    pair = scope.copy()
    pair[0] = False
    pair[1:] &= scope[:-1]
    triple = pair.copy()
    triple[-1] = False
    triple[:-1] &= scope[1:]
    frame_flags = values.flags.copy()
    frame_flags &= ((pair[:, None, None] & ((values.speed > config.pose_jump_primary) | (values.relative_speed > config.pose_jump_primary))) |
                    (triple[:, None, None] & (values.spike > config.pose_spike_threshold)))
    for j, name in enumerate(JOINTS):
        prefix = f'pose/{name}'
        result.sample(prefix + '/score', p.scores[:, :, j][scope[:, None] & p.observed], 'heatmap_peak', n * people)
        result.sample(prefix + '/speed', values.speed[pair, :, j], 'body_heights/s', int(pair.sum()) * people)
        result.sample(prefix + '/relative_speed', values.relative_speed[pair, :, j], 'body_heights/s', int(pair.sum()) * people)
        result.sample(prefix + '/spike', values.spike[triple, :, j], 'body_heights', int(triple.sum()) * people)
        result.rates[prefix + '/flagged'] = (int(frame_flags[scope, :, j].sum()), int(values.valid[scope, :, j].sum()))
    result.rates['pose/person_observed'] = (int(p.observed[scope].sum()), n * people)
    result.rates['pose/all_people_missing'] = (int((~p.observed.any(axis=1) & scope).sum()), n)
    result.sample('pose/translation_speed', values.translation[pair], 'body_heights/s', int(pair.sum()) * people)
    result.sample('pose/bone_change', values.bone_change[pair], 'body_heights', int(pair.sum()) * people * len(BONES))
    result.sample('pose/swap_advantage', values.swapped[pair], 'body_heights', int(pair.sum()) * people * len(PAIRS))
    result.rates['pose/bone_change_candidate'] = (int((values.bone_change[pair] > config.pose_bone_change_threshold).sum()), int(np.isfinite(values.bone_change[pair]).sum()))
    result.rates['pose/left_right_swap_candidate'] = (int((values.swapped[pair] > config.pose_spike_threshold).sum()), int(np.isfinite(values.swapped[pair]).sum()))
    result.rates['pose/whole_person_jump'] = (int(((values.speed[pair] > config.pose_jump_primary).sum(axis=2) >= 14).sum()), int(np.isfinite(values.translation[pair]).sum()))
    result.sample('pose/recovery_step', [value for a, b, _, value in values.recovery if scope[a:b + 1].all()], 'body_heights')
    tracking_pairs = pair[1:, None] & p.observed[1:] & p.observed[:-1] & ~data.breaks[1:, None]
    result.rates['pose/raw_track_transition'] = (int(((p.raw_tracks[1:] != p.raw_tracks[:-1]) & tracking_pairs).sum()), int(tracking_pairs.sum()))
    for threshold in config.pose_jump_speeds:
        flagged = (values.speed > threshold) & pair[:, None, None]
        prefix = f'pose/threshold_{threshold:g}'
        result.rates[prefix + '/joints'] = (int(flagged.sum()), int(np.isfinite(values.speed[pair]).sum()))
        result.rates[prefix + '/frames'] = (int(flagged.any(axis=(1, 2)).sum()), n)
        result.rates[prefix + '/wrists'] = (int(flagged[:, :, [9, 10]].sum()), int(np.isfinite(values.speed[pair][:, :, [9, 10]]).sum()))
        remaining = values.valid & ~flagged
        before = values.valid.any(axis=2)
        lost = before & ~remaining.any(axis=2) & scope[:, None]
        result.rates[prefix + '/people_lost'] = (int(lost.sum()), int(before[scope].sum()))
        result.rates[prefix + '/all_people_lost'] = (int((before.any(axis=1) & ~remaining.any(axis=(1, 2)) & scope).sum()), int((before.any(axis=1) & scope).sum()))
    result.views['pose/people'] = []
    for player, name in enumerate(p.player_ids):
        intervals = runs(scope & ~p.observed[:, player], data.breaks)
        result.sample(f'pose/player_slot_{player}/missing_frames', [b - a for a, b in intervals], 'frames')
        result.views['pose/people'].append(dict(player_id=name, observed=int(p.observed[scope, player].sum()), frames=n,
                                               flagged=int(frame_flags[:, player].any(axis=1).sum()), missing_intervals=intervals))
    candidates = np.argwhere(frame_flags)
    result.counts['pose/flagged_joint_frames'] = len(candidates)
    # Full flags are exposed as compact records for frame navigation, not auto-removal.
    result.views['pose/flags'] = [[int(t), int(p), int(j)] for t, p, j in candidates]
    if len(candidates):
        strength = np.nan_to_num(values.speed[tuple(candidates.T)], nan=0)
        for index in np.argsort(strength)[-20:][::-1]:
            t, player, joint = candidates[index]
            result.findings.append(dict(frame=int(t), kind='pose/jump_candidate', player=p.player_ids[player],
                                        joint=JOINTS[joint], value=float(strength[index])))
    return result
