"""Tuning-only calibration; physical existence cannot be inferred from these labels."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from src.tennis_scene.ball_gmm.config import CandidateGMMConfig
from src.tennis_scene.ball_gmm.heatmap import CandidateGaussians


def fit_calibration(
    candidates: tuple[CandidateGaussians, ...], labels: np.ndarray, scoreable: np.ndarray,
    *, stop: int, base: CandidateGMMConfig,
) -> tuple[CandidateGMMConfig, dict[str, Any]]:
    if (stop < 1 or stop > labels.shape[1] or len(candidates) != len(labels)
            or labels.shape != (*scoreable.shape, 2)):
        raise ValueError("Calibration requires selected-view label axes and an explicit tuning stop")
    features, targets, distances = [], [], []
    for i, candidate in enumerate(candidates):
        for frame in range(stop):
            available = candidate.indices[frame] >= 0
            if not scoreable[i, frame] or not available.any():
                continue
            delta = candidate.means_px[frame, available] - labels[i, frame]
            nearest = int(np.argmin(np.linalg.norm(delta, axis=-1)))
            recovered = bool(np.linalg.norm(delta[nearest]) <= 20)
            # Recover the pre-calibration score from presence of the base config.
            score = expit((logit(np.clip(candidate.presence[frame], 1e-7, 1-1e-7))-base.presence_bias)/base.presence_slope)
            features.append(logit(np.clip(score, 1e-7, 1-1e-7)))
            targets.append(float(recovered))
            if recovered:
                cov = candidate.covariance_px2[frame, available][nearest] / base.covariance_scale
                distances.append(float(delta[nearest] @ np.linalg.solve(cov, delta[nearest])))
    if len(features) < 10 or not distances:
        raise ValueError("Insufficient tuning support for candidate calibration")
    x, y = np.asarray(features), np.asarray(targets)

    def loss(parameters):
        logits = parameters[0]*x + parameters[1]
        return float(np.mean(np.logaddexp(0, logits)-y*logits)+.001*np.square(parameters-[1, 0]).sum())

    fit = minimize(loss, [1., 0.], bounds=[(.05, 20.), (-20., 20.)], method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(f"Tuning calibration failed: {fit.message}")
    # Gaussian scale MLE on the explicitly reported <=20 px localization cohort.
    # It models localized-peak noise; gross errors are handled by the mixture.
    scale = max(float(np.mean(distances)/2), 1e-4)
    settings = replace(base, covariance_scale=scale, presence_slope=float(fit.x[0]), presence_bias=float(fit.x[1]))
    return settings, {"frames": [0, stop], "scoreable_candidate_points": len(y),
        "recoverable_points": int(y.sum()), "covariance_fit_points": len(distances),
        "target": "any selected candidate within 20 native pixels on scoreable tuning labels",
        "presence_identifiability": "usable top-K support proxy; no physical existence labels",
        "covariance_fit_cohort": "nearest selected candidate with error <=20 px",
        "raw_brier": float(np.mean((expit(x)-y)**2)),
        "calibrated_brier": float(np.mean((expit(fit.x[0]*x+fit.x[1])-y)**2))}
