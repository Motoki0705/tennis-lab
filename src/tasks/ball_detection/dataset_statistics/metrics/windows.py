from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.play_intervals import PlaySelection, mask_intervals

from ..configuration import StatisticsConfig
from ..contracts import ClipInput, Measurements
from .pose import PoseSignals
from .spatial import normalized_xy


def compute(data: ClipInput, choices: dict[int, PlaySelection], config: StatisticsConfig, pose: PoseSignals | None) -> Measurements:
    result = Measurements()
    s, length = data.states, 32
    uv = normalized_xy(data)
    velocity = np.linalg.norm(np.diff(uv, axis=0), axis=1) / np.diff(s.times)
    good_edge = s.visible_observation[1:] & s.visible_observation[:-1] & ~data.breaks[1:] & (s.track[1:] == s.track[:-1])
    for stride, selection in choices.items():
        prefix = f'windows/stride_{stride}'
        coverage = np.zeros(data.clip.frame_count, np.int64)
        records = []
        position_supervision: NDArray[np.int64] = np.zeros(length, np.int64)
        position_missing: NDArray[np.int64] = np.zeros(length, np.int64)
        for start in selection.window_starts:
            end = start + length
            coverage[start:end] += 1
            position_supervision += s.supervision[start:end]
            position_missing += s.located[start:end] == 0
            gaps = mask_intervals(s.located[start:end] == 0)
            speeds = velocity[start:end - 1][good_edge[start:end - 1]]
            record = dict(start=start, stop=end, observed=int(s.supervision[start:end].sum()),
                          interpolated=int((s.kinds[start:end, 1] > 0).sum()),
                          coordinate_gap_max=max((b - a for a, b in gaps), default=0),
                          seconds=float(s.times[end - 1] - s.times[start]),
                          boundary_count=int(data.breaks[start + 1:end].sum()),
                          mean_speed=float(speeds.mean()) if len(speeds) else None,
                          pose_missing=None, pose_flagged=None)
            if data.pose is not None and pose is not None:
                record['pose_missing'] = int((~data.pose.observed[start:end].any(axis=1)).sum())
                flags = ((pose.speed[start + 1:end] > config.pose_jump_primary) | (pose.relative_speed[start + 1:end] > config.pose_jump_primary)).any(axis=(1, 2))
                flags[:-1] |= (pose.spike[start + 1:end - 1] > config.pose_spike_threshold).any(axis=(1, 2))
                record['pose_flagged'] = int(flags.sum())
            records.append(record)
        result.counts[prefix + '/count'] = len(records)
        result.counts[prefix + '/unique_frames'] = int((coverage > 0).sum())
        result.counts[prefix + '/frame_occurrences'] = int(coverage.sum())
        result.counts[prefix + '/observed_occurrences'] = int((coverage * s.supervision).sum())
        result.rates[prefix + '/coverage'] = (int((coverage > 0).sum()), data.clip.frame_count)
        result.rates[prefix + '/play_coverage'] = (int((coverage > 0).sum()), int(selection.play.sum()))
        result.rates[prefix + '/boundary_windows'] = (sum(r['boundary_count'] > 0 for r in records), len(records))
        for threshold in config.pose_jump_speeds:
            affected = 0
            if pose is not None:
                flags = (pose.speed > threshold).any(axis=(1, 2))
                affected = sum(bool(flags[start + 1:start + length].any()) for start in selection.window_starts)
            result.rates[prefix + f'/pose_threshold_{threshold:g}/affected_windows'] = (affected, len(records) if pose is not None else 0)
        for field, unit in [('observed', 'frames/window'), ('interpolated', 'frames/window'),
                            ('coordinate_gap_max', 'frames'), ('seconds', 'seconds'), ('mean_speed', 'normalized_xy/s'),
                            ('pose_missing', 'frames/window'), ('pose_flagged', 'frames/window')]:
            result.sample(prefix + '/' + field, [np.nan if r[field] is None else r[field] for r in records], unit)
        result.sample(prefix + '/multiplicity', coverage[selection.play], 'occurrences/frame')
        result.views[prefix + '/position_supervision'] = position_supervision.tolist()
        result.views[prefix + '/position_missing'] = position_missing.tolist()
        result.views[prefix + '/records'] = records
        result.views[prefix + '/mdd_zero_position'] = 0
    return result
