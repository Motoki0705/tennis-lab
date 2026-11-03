"""Fit joint log curvature in native grid units, without copying dense videos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from src.tennis_scene.ball_gmm.config import CandidateGMMConfig
from src.tennis_scene.pipeline.components.ball_evidence import BallHeatmapEvidence
from src.utils.geometry.probabilistic_triangulation import CameraGMM


@dataclass(frozen=True)
class PeakGaussian:
    mean_cells: NDArray[np.float64]
    covariance_cells2: NDArray[np.float64]
    bounded: bool


def fit_peak(
    patch: NDArray[np.floating], valid: NDArray[np.bool_], *,
    config: CandidateGMMConfig,
) -> PeakGaussian | None:
    """3x3 least-squares log quadratic; reject non-concave/insufficient support.

    Eigenvalue bounds are an explicit regularization policy, recorded per peak.
    Offsets are restricted to the peak's own cell. No isotropic fallback exists.
    """
    if (patch.ndim != 2 or patch.shape[0] != patch.shape[1]
            or patch.shape[0] < 3 or patch.shape[0] % 2 != 1 or valid.shape != patch.shape
            or valid.dtype != bool or not np.isfinite(patch).all()
            or (patch < 0).any() or (patch > 1).any()):
        raise ValueError("Log-curvature fit requires a finite odd square probability patch and mask")
    mid = len(patch) // 2
    patch = patch[mid-1:mid+2, mid-1:mid+2]
    valid = valid[mid-1:mid+2, mid-1:mid+2]
    y, x = np.mgrid[-1:2, -1:2].astype(float)
    design = np.stack((np.ones_like(x), x, y, -x*x/2, -x*y, -y*y/2), -1)
    support = valid & (patch > config.log_floor)
    if not support[1, 1] or int(support.sum()) < 6:
        return None
    coef, _, rank, _ = np.linalg.lstsq(design[support], np.log(patch[support]), rcond=None)
    if rank < 6:
        return None
    precision = np.array([[coef[3], coef[4]], [coef[4], coef[5]]])
    eigen, basis = np.linalg.eigh(precision)
    if eigen[0] <= 1e-8:
        return None
    unbounded_cov = np.linalg.inv(precision)
    offset = unbounded_cov @ coef[1:3]
    variance = 1 / eigen
    bounded_variance = np.clip(variance, config.min_variance_cells2, config.max_variance_cells2)
    clipped_offset = np.clip(offset, -0.5, 0.5)
    bounded = bool(np.any(variance != bounded_variance) or np.any(offset != clipped_offset))
    return PeakGaussian(clipped_offset, (basis * bounded_variance) @ basis.T, bounded)


@dataclass(frozen=True)
class CandidateGaussians:
    means_px: NDArray[np.float64]  # T,K,2
    covariance_px2: NDArray[np.float64]  # T,K,2,2
    weights: NDArray[np.float64]
    presence: NDArray[np.float64]
    indices: NDArray[np.int64]  # original top-8 slots, -1 for padding
    bounded: NDArray[np.bool_]
    rejected: NDArray[np.int64]


def candidate_gaussians(evidence: BallHeatmapEvidence, config: CandidateGMMConfig) -> CandidateGaussians:
    t, h, w = evidence.heatmaps.shape
    if min(h, w) < 2:
        raise ValueError("Candidate GMM requires a two-dimensional native grid")
    k = config.top_k
    means = np.zeros((t, k, 2))
    covariances = np.tile(np.eye(2), (t, k, 1, 1))
    weights = np.zeros((t, k))
    presence = np.zeros(t)
    indices = np.full((t, k), -1, np.int64)
    bounded = np.zeros((t, k), bool)
    rejected = np.zeros(t, np.int64)
    scale = (np.asarray(evidence.source_size_wh) - 1) / [w - 1, h - 1]
    for frame in range(t):
        selected = 0
        for source in np.flatnonzero(evidence.candidate_valid[frame]):
            if selected == k:
                break
            peak = fit_peak(evidence.patches[frame, source], evidence.patch_valid[frame, source], config=config)
            if peak is None:
                rejected[frame] += 1
                continue
            means[frame, selected] = (evidence.candidate_cells[frame, source] + peak.mean_cells) * scale
            covariances[frame, selected] = (peak.covariance_cells2 * scale[:, None] * scale[None, :]
                                                  * config.covariance_scale)
            weights[frame, selected] = evidence.candidate_scores[frame, source]
            indices[frame, selected] = source
            bounded[frame, selected] = peak.bounded
            selected += 1
        if selected:
            score = float(weights[frame].max())
            presence[frame] = expit(config.presence_slope * logit(np.clip(score, 1e-7, 1-1e-7)) + config.presence_bias)
            weights[frame] /= weights[frame].sum()
        else:
            # Required conditional distribution for a camera with presence=0;
            # these padding components have zero mass in all active subsets.
            weights[frame, 0] = 1.0
    return CandidateGaussians(means, covariances, weights, presence, indices, bounded, rejected)


def frame_gmm(views: tuple[CandidateGaussians, ...], frame: int) -> CameraGMM:
    return CameraGMM(
        np.stack([v.means_px[frame] for v in views]),
        np.stack([v.covariance_px2[frame] for v in views]),
        np.stack([v.weights[frame] for v in views]),
        np.array([v.presence[frame] for v in views]),
    )
