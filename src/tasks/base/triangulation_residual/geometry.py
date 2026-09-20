"""Observation-only triangulation, explicit missing-data seeds, and features."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from src.tasks.base.triangulation_residual.contracts import (
    CameraRig,
    GeometryInput,
    feature_dimension,
)
from src.utils.geometry.triangulation import project_multiview, triangulate_multiview
from src.utils.schema.court import HALF_LENGTH


class InsufficientGeometryError(ValueError):
    """No observed root anchor exists anywhere in the supplied time window."""


def fill_missing_seed(
    raw: np.ndarray, valid: np.ndarray, root_indices: tuple[int, ...]
) -> np.ndarray:
    """Linearly interpolate observed 3D tracks (edge-hold outside support).

    A joint with no 3D evidence anywhere is placed at the interpolated root
    anchor. The original validity mask is NEVER changed to indicate measured
    evidence. No GT, bone template or model output is used for this seed.
    """
    frames, joints, _ = raw.shape
    filled: np.ndarray = np.full_like(raw, np.nan)
    time = np.arange(frames)
    for joint in range(joints):
        support = np.flatnonzero(valid[:, joint])
        if len(support):
            for xyz in range(3):
                filled[:, joint, xyz] = np.interp(
                    time, support, raw[support, joint, xyz]
                )
    available = [j for j in root_indices if np.isfinite(filled[:, j]).all()]
    if not available:
        raise InsufficientGeometryError(
            "No triangulated root evidence; cannot initialize from observations"
        )
    anchor = filled[:, available].mean(axis=1)
    never_observed = ~valid.any(axis=0)
    filled[:, never_observed] = anchor[:, None]
    if not np.isfinite(filled).all():
        raise ValueError("Initializer interpolation produced nonfinite coordinates")
    return filled


def prepare_geometry(
    observations_px: np.ndarray,
    scores: np.ndarray,
    court_px: np.ndarray,
    court_scores: np.ndarray,
    rig: CameraRig,
    *,
    root_indices: tuple[int, ...],
    fps: float,
    min_score: float = 0.3,
    refinement_steps: int = 5,
) -> GeometryInput:
    """Inputs use (V,T,J,2), (V,T,J), physical CourtKP14 and fixed cameras.

    2D features are normalized by each view's W,H, global 3D by court scale;
    relative pose remains in metres. Raw observations/geometry are retained
    separately from finite neural features. All models share this entrypoint.
    """
    obs = np.asarray(observations_px, dtype=np.float64)
    conf = np.asarray(scores, dtype=np.float64)
    if obs.ndim != 4 or obs.shape[-1] != 2 or conf.shape != obs.shape[:-1]:
        raise ValueError("Expected observations (V,T,J,2) and scores (V,T,J)")
    views, frames, joints, _ = obs.shape
    if (
        views != len(rig.K)
        or frames < 2
        or not root_indices
        or any(j < 0 or j >= joints for j in root_indices)
    ):
        raise ValueError("Invalid view/time/joint/root contract")
    if (
        not np.isfinite(conf).all()
        or (conf < 0).any()
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("Scores must be finite and nonnegative; fps must be positive")
    if court_px.shape != (views, 14, 2) or court_scores.shape != (views, 14):
        raise ValueError("Expected static physical CourtKP14 observations")
    if (
        not np.isfinite(court_scores).all()
        or (court_scores < 0).any()
        or (court_scores > 1).any()
    ):
        raise ValueError("Court scores must be finite in [0,1]")
    finite = np.isfinite(obs).all(axis=-1)
    observed = finite & (conf > 0)
    if np.any((conf > 0) & ~finite):
        raise ValueError("Positive-confidence observations must have finite UV")
    conf = np.clip(conf, 0, 1)
    result = triangulate_multiview(
        obs.transpose(1, 2, 0, 3),
        conf.transpose(1, 2, 0),
        rig.matrices,
        min_score=min_score,
        refinement_steps=refinement_steps,
    )
    # Explicit numerical guard, not an anatomical filter: points >100m are
    # unusable for this court-sized task and keep invalid provenance.
    valid = result.valid & (np.abs(result.points) < 100).all(-1)
    raw = np.where(valid[..., None], result.points, np.nan)
    seed = fill_missing_seed(raw, valid, root_indices)
    root = seed[:, root_indices].mean(axis=1)
    relative = seed - root[:, None]
    projected, depth = project_multiview(seed, rig.matrices)
    projected = projected.transpose(2, 0, 1, 3)
    project_valid = (depth.transpose(2, 0, 1) > 1e-4) & np.isfinite(projected).all(-1)
    uv = np.where(observed[..., None], obs, 0) / rig.image_size[:, None, None]
    reprojected = (
        np.where(project_valid[..., None], projected, 0) / rig.image_size[:, None, None]
    )
    residual = np.where((observed & project_valid)[..., None], uv - reprojected, 0)
    court_valid = (court_scores > 0) & np.isfinite(court_px).all(-1)
    if np.any((court_scores > 0) & ~court_valid):
        raise ValueError("Visible court points must be finite")
    court_uv = np.where(court_valid[..., None], court_px, 0) / rig.image_size[:, None]
    cam = np.concatenate(
        (
            np.stack(
                (
                    rig.K[:, 0, 0] / rig.image_size[:, 0],
                    rig.K[:, 1, 1] / rig.image_size[:, 1],
                    rig.K[:, 0, 2] / rig.image_size[:, 0],
                    rig.K[:, 1, 2] / rig.image_size[:, 1],
                ),
                axis=-1,
            ),
            rig.R.reshape(views, 9),
            rig.centers / HALF_LENGTH,
        ),
        axis=-1,
    )
    rays = seed[:, :, None] - rig.centers
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-8)
    angles = np.zeros((frames, joints))
    for a, b in combinations(range(views), 2):
        angle = np.rad2deg(
            np.arccos(np.clip(np.abs((rays[:, :, a] * rays[:, :, b]).sum(-1)), 0, 1))
        )
        usable = result.used_views[..., a] & result.used_views[..., b] & valid
        angles = np.maximum(angles, np.where(usable, angle, 0))
    used = result.used_views.transpose(2, 0, 1)

    def repeat(value: np.ndarray) -> np.ndarray:
        repeated: np.ndarray = np.broadcast_to(value, (views, *value.shape))
        return repeated

    # Every block is ordered explicitly; the feature count is a checkpoint contract.
    blocks = [
        np.where(observed[..., None], uv - 0.5, 0).reshape(views, frames, -1),
        np.where(project_valid[..., None], reprojected - 0.5, 0).reshape(
            views, frames, -1
        ),
        residual.reshape(views, frames, -1),
        np.broadcast_to(
            np.where(court_valid[..., None], court_uv - 0.5, 0).reshape(views, 1, 28),
            (views, frames, 28),
        ),
        repeat(seed / HALF_LENGTH).reshape(views, frames, -1),
        np.broadcast_to(cam[:, None], (views, frames, 16)),
        conf,
        observed,
        repeat(valid),
        used,
        project_valid,
        np.broadcast_to(court_scores[:, None], (views, frames, 14)),
        repeat(relative).reshape(views, frames, -1),
        repeat(root / HALF_LENGTH),
        repeat(angles / 90),
    ]
    features = np.concatenate(blocks, axis=-1).astype(np.float32)
    if (
        features.shape != (views, frames, feature_dimension(joints))
        or not np.isfinite(features).all()
    ):
        raise ValueError("Residual feature contract violated")
    return GeometryInput(
        features,
        seed.astype(np.float32),
        raw,
        valid,
        root.astype(np.float32),
        relative.astype(np.float32),
        (np.arange(frames) * 30 / fps).astype(np.float32),
        np.ones((views, frames), dtype=bool),
        uv.astype(np.float32),
        conf.astype(np.float32),
        reprojected.astype(np.float32),
        residual.astype(np.float32),
        used,
        project_valid,
        angles.astype(np.float32),
    )
