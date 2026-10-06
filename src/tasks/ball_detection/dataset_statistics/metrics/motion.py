from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..configuration import StatisticsConfig
from ..contracts import ClipInput, Measurements
from ..scopes import edges
from .spatial import normalized_xy


def compute(data: ClipInput, scope: NDArray[np.bool_], config: StatisticsConfig) -> Measurements:
    result = Measurements()
    s = data.states
    uv = normalized_xy(data)
    dt = np.diff(s.times)
    same_track = s.track[1:] == s.track[:-1]
    for name, valid in [('observed', s.visible_observation),
                        ('located_single', s.reviewed & (s.count == 1) & (s.located == 1))]:
        accepted = edges(data, scope, valid) & same_track
        indices = np.flatnonzero(accepted)
        delta = np.diff(uv, axis=0)
        velocity = delta / dt[:, None]
        pixels = np.diff(s.xy / data.clip.scale, axis=0)
        prefix = f'motion/{name}'
        eligible = int((scope[1:] & scope[:-1]).sum())
        result.sample(prefix + '/step', np.linalg.norm(pixels[accepted], axis=1), 'source_px/frame_pair', eligible)
        result.sample(prefix + '/speed_px', np.linalg.norm(pixels[accepted], axis=1) / dt[accepted], 'source_px/s', eligible)
        result.sample(prefix + '/speed_normalized', np.linalg.norm(velocity[accepted], axis=1), 'normalized_xy/s', eligible)
        result.sample(prefix + '/abs_vx', np.abs(velocity[accepted, 0]), 'image_width/s', eligible)
        result.sample(prefix + '/abs_vy', np.abs(velocity[accepted, 1]), 'image_height/s', eligible)
        horizontal, vertical = np.abs(delta[accepted]).sum(axis=0)
        result.sample(prefix + '/vertical_share', [vertical / (horizontal + vertical)] if horizontal + vertical > 0 else [], 'fraction', 1)
        consecutive = accepted[1:] & accepted[:-1]
        acceleration = np.linalg.norm(np.diff(velocity, axis=0), axis=1) / ((dt[1:] + dt[:-1]) / 2)
        result.sample(prefix + '/acceleration', acceleration[consecutive], 'normalized_xy/s2', int((scope[2:] & scope[1:-1] & scope[:-2]).sum()))
        angle = np.arctan2(delta[:, 1], delta[:, 0])
        change = np.abs(np.angle(np.exp(1j * np.diff(angle))))
        moving = np.linalg.norm(delta, axis=1) > 0
        result.sample(prefix + '/turn', change[consecutive & moving[1:] & moving[:-1]], 'radians')
        cells = np.clip((uv[indices] * [config.grid_x, config.grid_y]).astype(int), [0, 0], [config.grid_x - 1, config.grid_y - 1])
        ends = np.clip((uv[indices + 1] * [config.grid_x, config.grid_y]).astype(int), [0, 0], [config.grid_x - 1, config.grid_y - 1])
        region = cells[:, 1] * config.grid_x + cells[:, 0]
        destination = ends[:, 1] * config.grid_x + ends[:, 0]
        grid_size = config.grid_x * config.grid_y
        transitions: NDArray[np.int64] = np.zeros((grid_size, grid_size), np.int64)
        np.add.at(transitions, (region, destination), 1)
        movement = np.bincount(region, weights=np.linalg.norm(delta[accepted], axis=1), minlength=grid_size)
        speed = np.bincount(region, weights=np.linalg.norm(velocity[accepted], axis=1), minlength=grid_size)
        result.views[prefix + '/transitions'] = [[int(a), int(b), int(transitions[a, b])]
                                                 for a, b in np.argwhere(transitions > 0)]
        result.views[prefix + '/movement'] = movement.reshape(config.grid_y, config.grid_x).tolist()
        result.views[prefix + '/speed_sum'] = speed.reshape(config.grid_y, config.grid_x).tolist()
        result.views[prefix + '/edge_count'] = np.bincount(region, minlength=grid_size).reshape(config.grid_y, config.grid_x).tolist()
        for axis, label in enumerate(('x', 'y')):
            result.views[prefix + f'/direction_{label}_sum'] = np.bincount(region, weights=delta[accepted, axis], minlength=grid_size).reshape(config.grid_y, config.grid_x).tolist()
        if len(indices):
            order = np.argsort(np.linalg.norm(velocity[accepted], axis=1))[-5:][::-1]
            result.findings.extend(dict(frame=int(indices[k] + 1), kind=prefix + '/fast_motion', value=float(np.linalg.norm(velocity[indices[k]]))) for k in order)
    return result
