"""Synthetic trajectory metrics with explicit truth, masks, units and support."""
from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[Any]


def metric_values(prediction: Array, arrays: dict[str, Array]) -> dict[str, Array]:
    """Return unreduced values so dataset metrics weight frames, not rallies.

    Prediction is S,T,3. Derivatives use actual seconds; a free-flight stencil
    requires every participating frame to be free of event/fence exclusions.
    Reprojection compares against synthetic truth, on in-image true-camera
    frames. Behind-camera predictions are counted and never treated as valid.
    """
    truth = arrays['positions_3d_m'].astype(np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape[1:] != truth.shape or not np.isfinite(prediction).all():
        raise ValueError('Expected finite S,T,3 predictions matching truth')
    times = arrays['timestamps_seconds'].astype(np.float64)
    dt = np.diff(times)
    if len(times) < 4 or not (dt > 0).all() or not np.allclose(dt, dt[0], rtol=1e-8, atol=1e-12):
        raise ValueError('Metrics require >=4 uniformly timed frames')
    error2 = np.square(prediction - truth).sum(-1)
    gap = arrays['occlusion_mask'].all(0)
    no_evidence = (arrays['occlusion_mask'] | arrays['out_of_frame_mask']).all(0)
    values = {f'error2_{name}': error2[:, mask].ravel() for name, mask in (
        ('overall', np.ones(len(times), dtype=bool)), ('gap', gap),
        ('no_evidence', no_evidence), ('event_pm5', arrays['event_region_mask']))}
    free = arrays['free_flight_mask']
    for name, order in (('acceleration', 2), ('jerk', 3)):
        magnitude = np.linalg.norm(np.diff(prediction, n=order, axis=1) / dt[0]**order, axis=-1)
        support = np.logical_and.reduce([free[offset:len(free) - order + offset] for offset in range(order + 1)])
        values[name + '_all'] = magnitude.ravel()
        values[name + '_free_flight'] = magnitude[:, support].ravel()
    rotation, translation, intrinsic = (arrays['camera_true_' + key] for key in ('R', 't', 'K'))
    camera = np.einsum('vij,stj->svti', rotation, prediction) + translation[None, :, None]
    target_camera = np.einsum('vij,tj->vti', rotation, truth) + translation[:, None]
    projected = np.einsum('vij,svtj->svti', intrinsic, camera)
    target_projected = np.einsum('vij,vtj->vti', intrinsic, target_camera)
    support2d = np.broadcast_to(~arrays['out_of_frame_mask'], camera.shape[:-1])
    front = camera[..., 2] > 1e-6
    target_uv = target_projected[..., :2] / target_projected[..., 2, None]
    # No fictitious finite pixel error is assigned to an undefined projection.
    uv = projected[..., :2] / np.where(front, projected[..., 2], 1)[..., None]
    error_px = np.linalg.norm(uv - target_uv[None], axis=-1)
    for name, mask in (('all', np.ones_like(support2d)), ('observed', ~arrays['occlusion_mask'][None]), ('gap', arrays['occlusion_mask'][None])):
        selected = support2d & mask
        values['reprojection_px_' + name] = error_px[selected & front]
        values['behind_' + name] = (~front[selected]).astype(np.float64)
    return values


def summarize_metrics(values: dict[str, list[Array]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, chunks in values.items():
        combined = np.concatenate(chunks)
        if name.startswith('error2_'):
            result[name.replace('error2_', 'rmse_m_')] = {'count': len(combined), 'value': float(np.sqrt(combined.mean())) if len(combined) else None}
        elif name.startswith('behind_'):
            result[name] = {'count': len(combined), 'invalid_count': int(combined.sum()), 'fraction': float(combined.mean()) if len(combined) else None}
        else:
            result[name] = {'count': len(combined), 'mean': float(combined.mean()) if len(combined) else None,
                            'p50': float(np.quantile(combined, .5)) if len(combined) else None,
                            'p95': float(np.quantile(combined, .95)) if len(combined) else None}
    # A finite front-only reprojection summary is explicitly conditional when
    # any prediction is behind the camera; it cannot count as successful fidelity.
    result['reprojection_all_defined'] = result['behind_all']['invalid_count'] == 0
    return result
