"""Confidence-ranked court candidates and explicitly trimmed KP + LINE fitting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from .confidence_homography import (
    ConfidenceHomographyResult,
    _has_support,
    estimate_confidence_homography,
)
from .line_evidence import FloatArray, LineDiagnostics, LineEvidence, line_residuals


@dataclass(frozen=True)
class HybridHomographyConfig:
    """One image-independent configuration; distances scale with its diagonal."""

    min_score: float = 0.05
    threshold_diagonal_ratio: float = 0.005
    line_probability_threshold: float = 0.5
    max_kp: int = 8
    max_candidates: int = 1001
    refine_candidates: int = 6
    samples_per_line: int = 40
    max_line_observations: int = 768
    max_selection_rounds: int = 4
    max_nfev: int = 70
    kp_weight: float = 0.25
    min_line_support: float = 0.55
    ambiguity_gap: float = 0.02
    ambiguity_displacement_ratio: float = 0.03

    def __post_init__(self) -> None:
        if not 0 <= self.min_score <= 1 or not 0 < self.line_probability_threshold < 1:
            raise ValueError("Invalid probability thresholds")
        if not 0 < self.threshold_diagonal_ratio < 0.1:
            raise ValueError("Invalid geometric threshold")
        if not 0 < self.kp_weight <= 1 or not 0 < self.min_line_support <= 1:
            raise ValueError("Invalid hybrid weights/support")
        if (
            not 0 <= self.ambiguity_gap < 1
            or not 0 < self.ambiguity_displacement_ratio < 1
        ):
            raise ValueError("Invalid ambiguity thresholds")
        if (
            self.max_kp < 4
            or self.max_line_observations < 16
            or self.samples_per_line < 4
        ):
            raise ValueError("Insufficient KP/LINE capacity")
        if (
            min(
                self.max_candidates,
                self.refine_candidates,
                self.max_selection_rounds,
                self.max_nfev,
            )
            < 1
        ):
            raise ValueError("Solver budgets must be positive")


DEFAULT_HYBRID_CONFIG = HybridHomographyConfig()


@dataclass(frozen=True)
class HybridHypothesis:
    matrix: FloatArray
    projected: FloatArray
    selected: NDArray[np.bool_]
    score: float
    line: LineDiagnostics
    selection_history: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class HybridHomographyResult:
    """No substitute H on failure; best rejected hypothesis stays diagnostic.

    `selected` is exactly the last optimization's KP set. Rejected coordinates
    never occur in that optimization's residual vector, including robust losses.
    The selection history records every actual nonlinear solve, not gate probes.
    """

    matrix: FloatArray | None
    projected: FloatArray
    selected: NDArray[np.bool_]
    residuals_px: FloatArray
    status: str
    kp_only: ConfidenceHomographyResult
    line_selection: HybridHypothesis | None
    best: HybridHypothesis | None
    candidate_count: int
    refined_count: int
    alternative_score_gap: float | None


def project_candidate(matrix: FloatArray, template: FloatArray) -> FloatArray | None:
    homogeneous = np.c_[template, np.ones(len(template))] @ matrix.T
    depth = homogeneous[:, 2]
    if not np.isfinite(homogeneous).all() or np.linalg.matrix_rank(matrix) < 3:
        return None
    epsilon = 1e-9 * max(float(np.max(np.abs(depth))), 1e-300)
    if not (np.all(depth > epsilon) or np.all(depth < -epsilon)):
        return None
    return np.asarray(homogeneous[:, :2] / depth[:, None], dtype=np.float64)


def select_keypoints(
    template: FloatArray,
    observed: FloatArray,
    scores: FloatArray,
    projected: FloatArray,
    line_distance_px: FloatArray,
    threshold_px: float,
    config: HybridHomographyConfig,
) -> NDArray[np.bool_]:
    """Hard residual/LINE gates, then confidence/residual ranking and strict cap.

    All observations can be *tested* against a candidate. Only the returned
    subset enters its KP objective. The cap never permits the complete input
    set, even if every point passes; no all-point soft-weighted loss exists.
    """
    residual = np.linalg.norm(projected - observed, axis=1)
    eligible = (
        np.isfinite(observed).all(axis=1)
        & (scores >= config.min_score)
        & (residual <= threshold_px)
        & (line_distance_px <= threshold_px)
    )
    indices = np.flatnonzero(eligible)
    merit = scores[indices] / (1 + (residual[indices] / threshold_px) ** 2)
    order = indices[np.argsort(-merit, kind="stable")]
    selected: NDArray[np.bool_] = np.zeros(len(template), dtype=bool)
    selected[order[: min(config.max_kp, len(template) - 1)]] = True
    if selected.sum() < 4 or not _has_support(template[selected], observed[selected]):
        selected[:] = False
    return selected


def selected_kp_residuals(
    projected: FloatArray,
    observed: FloatArray,
    scores: FloatArray,
    selected: NDArray[np.bool_],
    threshold_px: float,
) -> FloatArray:
    """Only selected rows are touched; NaNs/extreme rejected KP have zero effect."""
    weights = 0.5 + 0.5 * scores[selected]
    delta = (projected[selected] - observed[selected]) / threshold_px
    return np.asarray(
        (delta * np.sqrt(weights / weights.sum())[:, None]).ravel(), dtype=np.float64
    )


def estimate_hybrid_homography(
    template_xy: NDArray[np.floating],
    image_xy: NDArray[np.floating],
    scores: NDArray[np.floating],
    line_probability: NDArray[np.floating],
    *,
    edges: NDArray[np.integer],
    image_size_hw: tuple[int, int],
    config: HybridHomographyConfig = DEFAULT_HYBRID_CONFIG,
) -> HybridHomographyResult:
    """KP-generated multiple hypotheses -> LINE selection -> trimmed joint fit.

    PROSAC is retained as an explicit seed/baseline. The bounded additional pool
    enumerates four-point combinations in increasing worst confidence rank;
    this is deterministic enumeration, not a second call to PROSAC. No semantic
    LINE labels, camera intrinsics, LINE-only initializer or GT are assumed.
    """
    template = np.asarray(template_xy, dtype=np.float64)
    observed = np.asarray(image_xy, dtype=np.float64)
    quality = np.asarray(scores, dtype=np.float64)
    segments = np.asarray(edges)
    if (
        segments.ndim != 2
        or segments.shape[1] != 2
        or not np.issubdtype(segments.dtype, np.integer)
    ):
        raise ValueError("edges must be integer pairs of template indices")
    segments = segments.astype(np.int64)
    if len(segments) < 4 or np.any(segments < 0) or np.any(segments >= len(template)):
        raise ValueError("Invalid court segments")
    if not 5 <= len(template) <= 14 or np.any(segments[:, 0] == segments[:, 1]):
        raise ValueError("Hybrid trimming needs 5--14 ordered KP and nonzero segments")
    height, width = image_size_hw
    if min(height, width) < 2:
        raise ValueError("Invalid image size")
    diagonal = float(np.hypot(width, height))
    threshold = config.threshold_diagonal_ratio * diagonal
    # Reuse its strict coordinate/score validation and the published baseline.
    baseline = estimate_confidence_homography(
        template,
        observed,
        quality,
        reprojection_threshold_px=threshold,
        min_score=config.min_score,
    )
    empty: NDArray[np.bool_] = np.zeros(len(template), dtype=bool)

    def result(
        status: str,
        *,
        best: HybridHypothesis | None = None,
        initial: HybridHypothesis | None = None,
        count: int = 0,
        refined: int = 0,
        gap: float | None = None,
    ) -> HybridHomographyResult:
        accepted = best is not None and status == "ok"
        projected = (
            best.projected
            if accepted and best is not None
            else np.full_like(template, np.nan)
        )
        return HybridHomographyResult(
            best.matrix if accepted and best is not None else None,
            projected,
            best.selected if accepted and best is not None else empty.copy(),
            np.linalg.norm(projected - observed, axis=1),
            status,
            baseline,
            initial,
            best,
            count,
            refined,
            gap,
        )

    # Invalid probability arrays are caller errors; lack of meaningful evidence
    # is an explicit estimation failure, never a switch back to KP-only output.
    try:
        evidence = LineEvidence.from_probability(
            line_probability,
            image_size_hw,
            threshold=config.line_probability_threshold,
            max_observations=config.max_line_observations,
        )
    except ValueError as error:
        if str(error) == "insufficient_or_diffuse_line_evidence":
            return result(str(error))
        raise
    observed_line_distance = evidence.distances(np.nan_to_num(observed, nan=-diagonal))

    def evaluate(
        matrix: FloatArray, history: tuple[tuple[int, ...], ...] = ()
    ) -> HybridHypothesis | None:
        projected = project_candidate(matrix, template)
        if projected is None or np.max(np.abs(projected)) > 10 * diagonal:
            return None
        hull = cv2.convexHull(projected.astype(np.float32))
        if not 0.015 * width * height <= cv2.contourArea(hull) <= 9 * width * height:
            return None
        selected = select_keypoints(
            template,
            observed,
            quality,
            projected,
            observed_line_distance,
            threshold,
            config,
        )
        if not selected.any():
            return None
        line, diagnostic = line_residuals(
            projected,
            segments,
            evidence,
            threshold,
            samples_per_line=config.samples_per_line,
        )
        kp = selected_kp_residuals(projected, observed, quality, selected, threshold)
        score = float(line @ line + config.kp_weight * (kp @ kp))
        return HybridHypothesis(matrix, projected, selected, score, diagnostic, history)

    hypotheses: list[HybridHypothesis] = []
    if baseline.matrix is not None:
        initial = evaluate(baseline.matrix)
        if initial is not None:
            hypotheses.append(initial)
    order = baseline.ranked_indices
    combinations_ranked = sorted(
        combinations(range(len(order)), 4), key=lambda v: (max(v), sum(v), v)
    )
    for indices in combinations_ranked[: config.max_candidates]:
        subset = order[list(indices)]
        if not _has_support(template[subset], observed[subset]):
            continue
        matrix = cast(
            FloatArray | None,
            cv2.findHomography(template[subset], observed[subset], method=0)[0],
        )
        if matrix is None:
            continue
        candidate = evaluate(np.asarray(matrix, dtype=np.float64))
        if candidate is not None:
            hypotheses.append(candidate)
    if not hypotheses:
        return result("no_jointly_supported_candidate")
    hypotheses.sort(key=lambda h: h.score)
    initial = hypotheses[0]
    seeds: list[HybridHypothesis] = []
    for candidate in hypotheses:
        if all(
            np.sqrt(np.mean((candidate.projected - old.projected) ** 2))
            > 0.002 * diagonal
            for old in seeds
        ):
            seeds.append(candidate)
        if len(seeds) == config.refine_candidates:
            break

    center = template.mean(axis=0)
    scale = float(np.sqrt(np.mean((template - center) ** 2)))
    normalization = np.array(
        [
            [1 / scale, 0, -center[0] / scale],
            [0, 1 / scale, -center[1] / scale],
            [0, 0, 1],
        ]
    )
    image_scale = np.diag([diagonal, diagonal, 1.0])
    source_inverse = np.linalg.inv(normalization)

    def pack(matrix: FloatArray) -> FloatArray:
        normalized = np.linalg.solve(image_scale, matrix @ source_inverse)
        return np.asarray((normalized / normalized[2, 2]).ravel()[:8], dtype=np.float64)

    def unpack(parameters: FloatArray) -> FloatArray:
        return np.asarray(
            image_scale @ np.append(parameters, 1).reshape(3, 3) @ normalization,
            dtype=np.float64,
        )

    solutions: list[HybridHypothesis] = []
    # Smooth distance fields widen the convergence basin; final fit/acceptance
    # always uses the original ridges, in the same original-image pixel units.
    fields = (gaussian_filter(evidence.distance, threshold / 2), evidence.distance)
    for seed in seeds:
        current = seed
        history: list[tuple[int, ...]] = []
        valid = True
        for field in fields:
            stable = False
            for _ in range(config.max_selection_rounds):
                fixed = current.selected.copy()
                history.append(tuple(np.flatnonzero(fixed).tolist()))

                def residual(
                    parameters: FloatArray,
                    fixed: NDArray[np.bool_] = fixed,
                    field: FloatArray = field,
                ) -> FloatArray:
                    projected = project_candidate(unpack(parameters), template)
                    size = (
                        2 * int(fixed.sum())
                        + 2 * len(segments) * config.samples_per_line
                        + len(evidence.points)
                    )
                    if projected is None:
                        return np.full(size, 100.0)
                    line, _ = line_residuals(
                        projected,
                        segments,
                        evidence,
                        threshold,
                        samples_per_line=config.samples_per_line,
                        distance_field=field,
                    )
                    kp = selected_kp_residuals(
                        projected, observed, quality, fixed, threshold
                    )
                    return np.asarray(
                        np.r_[np.sqrt(config.kp_weight) * kp, line], dtype=np.float64
                    )

                optimized = least_squares(
                    residual,
                    pack(current.matrix),
                    method="trf",
                    max_nfev=config.max_nfev,
                    ftol=1e-6,
                    xtol=1e-7,
                    gtol=1e-6,
                )
                updated = evaluate(unpack(optimized.x), tuple(history))
                if not optimized.success or updated is None:
                    valid = False
                    break
                current = updated
                if np.array_equal(fixed, current.selected):
                    stable = True
                    break
            if not valid or not stable:
                valid = False
                break
        if valid:
            solutions.append(current)
    if not solutions:
        return result(
            "joint_optimization_failed", initial=initial, count=len(hypotheses)
        )
    solutions.sort(key=lambda h: h.score)
    best = solutions[0]
    alternatives = [
        h
        for h in solutions[1:]
        if np.sqrt(np.mean((h.projected - best.projected) ** 2))
        > config.ambiguity_displacement_ratio * diagonal
    ]
    gap = alternatives[0].score - best.score if alternatives else None
    status = "ok"
    template_delta = template[segments[:, 1]] - template[segments[:, 0]]
    transverse = np.abs(template_delta[:, 0]) > np.abs(template_delta[:, 1])
    supported = np.asarray(best.line["segment_support"]) >= 0.5
    if (
        min(float(best.line["forward_support"]), float(best.line["reverse_support"]))
        < config.min_line_support
        or float(best.line["supported_segments"]) < 4
        or np.sum(supported & transverse) < 2
        or np.sum(supported & ~transverse) < 2
    ):
        status = "insufficient_line_support"
    elif gap is not None and gap < config.ambiguity_gap:
        status = "ambiguous_candidates"
    return result(
        status,
        best=best,
        initial=initial,
        count=len(hypotheses),
        refined=len(solutions),
        gap=gap,
    )
