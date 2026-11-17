"""Diagnostic frame motion; a local fit never replaces the shared court."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..homography import court_template_xy
from ..hybrid_homography import project_candidate
from ..line_evidence import FloatArray
from ..paint_observations import PAINT_NAMES, image_lines
from ..quality import projection_jacobian
from .fit import HomographyChart, PaintGroup, SharedPaintConfig, fit_common_court


def inspect_drift(shared: FloatArray, groups: tuple[PaintGroup, ...], size_wh: tuple[int, int],
                  config: SharedPaintConfig) -> dict[str, Any]:
    """3px coherent warning, 10px mean/25px maximum failure at 1920x1080.

    One noisy line is not enough to claim camera motion. Unidentifiable or
    imprecise local geometry is explicitly unobservable, never a zero shift.
    """
    if len({(g.clip_id, g.frame_index) for g in groups}) > 1:
        raise ValueError("Drift inspection requires one source frame")
    local = fit_common_court(shared, groups, size_wh, replace(config, min_clips_per_line=1))
    diagnostic: dict[str, Any] = {"schema": "video_court_drift_v1", "status": "failed", "reason": "drift_unobservable",
                                  "local_geometry": local.diagnostic, "geometry_reasons": list(local.reasons),
                                  "mean_displacement_px": None, "max_displacement_px": None}
    used = {row['line'] for row in local.diagnostic.get('group_residuals', []) if row['used']}
    shared_lines = image_lines(shared)
    diagnostic['common_line_residuals'] = [
        {'line': PAINT_NAMES[group.line_index], 'used_by_local_fit': PAINT_NAMES[group.line_index] in used,
         'signed_median_px': float(np.median(np.c_[group.points, np.ones(len(group.points))] @ shared_lines[group.line_index])),
         'p95_px': float(np.percentile(np.abs(np.c_[group.points, np.ones(len(group.points))] @ shared_lines[group.line_index]), 95)),
         'points': len(group.points)} for group in groups]
    if local.matrix is None:
        return diagnostic
    world = court_template_xy().astype(float)
    reference = project_candidate(shared, world)
    actual = project_candidate(local.matrix, world)
    if reference is None or actual is None:
        return diagnostic
    visible = (reference[:, 0] >= 0) & (reference[:, 0] <= size_wh[0] - 1) & (reference[:, 1] >= 0) & (reference[:, 1] <= size_wh[1] - 1)
    if not visible.any():
        return diagnostic
    chart = HomographyChart(size_wh)
    parameters = chart.pack(local.matrix)
    locations, labels = [], []
    for group in groups:
        if PAINT_NAMES[group.line_index] not in used:
            continue
        axis = int(np.argmax(np.ptp(group.points, axis=0)))
        ordered = group.points[np.argsort(group.points[:, axis], kind='stable')]
        locations.extend(ordered[np.rint(np.array([.1, .9]) * (len(ordered) - 1)).astype(int)])
        labels.extend([group.line_index, group.line_index])

    def residual(p: FloatArray) -> FloatArray:
        return np.asarray(np.einsum('ij,ij->i', np.c_[locations, np.ones(len(locations))], image_lines(chart.unpack(p))[labels]), dtype=np.float64)

    jacobian = np.column_stack([(residual(parameters + np.eye(8)[i] * 1e-6)
                                - residual(parameters - np.eye(8)[i] * 1e-6)) / 2e-6 for i in range(8)])
    if np.linalg.matrix_rank(jacobian) != 8:
        return diagnostic
    sigma = max(1., float(np.median(np.abs(residual(parameters)))))
    covariance = np.linalg.inv(jacobian.T @ jacobian) * sigma ** 2
    normalized = (np.c_[world, np.ones(len(world))] @ chart.world.T)[:, :2]
    derivative = projection_jacobian(normalized, np.r_[parameters, 1.].reshape(3, 3)) * float(np.hypot(*size_wh))
    projected_covariance = np.einsum('nij,jk,nlk->nil', derivative, covariance, derivative)
    std = np.sqrt(np.maximum(np.linalg.eigvalsh(projected_covariance)[:, -1], 0))
    scale = float(np.hypot(*size_wh) / np.hypot(1920, 1080))
    diagnostic['max_visible_projection_std_px'] = float(std[visible].max())
    if float(std[visible].max()) > config.paint.max_projection_std_px * scale:
        diagnostic['geometry_reasons'].append('uncertain_local_projection')
        return diagnostic
    difference = actual[visible] - reference[visible]
    distance = np.linalg.norm(difference, axis=1)
    mean, maximum = float(distance.mean()), float(distance.max())
    # The gradient of mean displacement preserves correlation across keypoints;
    # it is not std/sqrt(number_of_pixels).
    unit = np.divide(difference, distance[:, None], out=np.zeros_like(difference), where=distance[:, None] > 1e-10)
    mean_derivative = np.einsum('ni,nij->j', unit, derivative[visible]) / int(visible.sum())
    mean_std = float(np.sqrt(max(0., mean_derivative @ covariance @ mean_derivative)))
    diagnostic.update(local_homography=local.matrix.tolist(), mean_displacement_px=mean, max_displacement_px=maximum,
                      mean_displacement_std_px=mean_std, thresholds_px={'warning': 3 * scale, 'mean_failure': 10 * scale, 'max_failure': 25 * scale})
    if mean > 10 * scale or maximum > 25 * scale:
        diagnostic.update(status='failed', reason='camera_drift_exceeds_tolerance')
    elif mean - 2 * mean_std > 3 * scale:
        diagnostic.update(status='warning', reason='camera_moved_possible')
    else:
        diagnostic.update(status='ok', reason=None)
    return diagnostic
