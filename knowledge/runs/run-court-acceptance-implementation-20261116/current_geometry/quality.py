"""Image-only court quality: visible markings and sensitivity of their fit.

The score is a ranking cost, not a calibrated probability or a reference error.
Each physical marking has total weight one: a thousand pixels along one line
must not masquerade as a thousand independent constraints on a homography.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .confidence_homography import _has_support
from .homography import COURT_HOMOGRAPHY_EDGES, court_template_xy
from .line_evidence import FloatArray, LineEvidence, clip_segments, line_residuals

QUALITY_SCHEMA = "court_quality_v1"


@dataclass(frozen=True, slots=True)
class CourtQualityConfig:
    line_tolerance_ratio: float = 0.005
    min_line_support: float = 0.55
    min_visible_segment_support: float = 0.25
    min_line_coverage: float = 0.5
    max_normalized_condition: float = 10000.0
    max_projection_std_ratio: float = 0.005
    max_leave_line_out_ratio: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "line_tolerance_ratio", "min_line_support", "min_visible_segment_support",
            "min_line_coverage", "max_projection_std_ratio", "max_leave_line_out_ratio",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"Court quality {name} must be finite in (0,1]")
        if not np.isfinite(self.max_normalized_condition) or self.max_normalized_condition <= 1:
            raise ValueError("Court quality condition limit must be finite and > 1")


DEFAULT_QUALITY_CONFIG = CourtQualityConfig()


@dataclass(frozen=True)
class CourtQuality:
    accepted: bool
    score: float | None
    reasons: tuple[str, ...]
    metrics: dict[str, object]

    def diagnostics(self) -> dict[str, object]:
        return {
            "schema": QUALITY_SCHEMA,
            "accepted": self.accepted,
            "score": self.score,
            "score_direction": "lower_is_better",
            "reasons": list(self.reasons),
            **self.metrics,
        }


def projection_jacobian(points: FloatArray, matrix: FloatArray) -> FloatArray:
    """Jacobian for eight H parameters with h33=1, in caller-normalized units."""
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    depth = homogeneous[:, 2]
    if np.any(np.abs(depth) < 1e-12):
        raise ValueError("Projection Jacobian crosses a pole")
    uv = homogeneous[:, :2] / depth[:, None]
    base = np.c_[points, np.ones(len(points))] / depth[:, None]
    jacobian = np.zeros((len(points), 2, 8))
    jacobian[:, 0, :3] = base
    jacobian[:, 1, 3:6] = base
    jacobian[:, :, 6:] = -uv[:, :, None] * base[:, None, :2]
    return jacobian


def _projection(points: FloatArray, matrix: FloatArray) -> FloatArray:
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(homogeneous[:, :2] / homogeneous[:, 2, None], dtype=np.float64)


def _sensitivity(
    matrix: FloatArray,
    template: FloatArray,
    evidence: LineEvidence,
    segments: FloatArray,
    visible: NDArray[np.bool_],
    tolerance: float,
    cell_sigma: float,
    diagonal: float,
    condition_limit: float,
    observed: FloatArray,
    supported_kp: NDArray[np.bool_],
) -> dict[str, object]:
    """Line-group normalized covariance and linear leave-one-marking influence.

    These are local diagnostics around a hypothesis; they cannot prove that a
    neural LINE prediction or its semantic correspondence is correct.
    """
    center = template.mean(axis=0)
    scale = float(np.sqrt(np.mean((template - center) ** 2)))
    source = np.array([[1 / scale, 0, -center[0] / scale], [0, 1 / scale, -center[1] / scale], [0, 0, 1]])
    # A centered image chart avoids position-dependent conditioning.
    image = np.array([[1 / diagonal, 0, -evidence.width / (2 * diagonal)],
                      [0, 1 / diagonal, -evidence.height / (2 * diagonal)], [0, 0, 1]])
    normalized = image @ matrix @ np.linalg.inv(source)
    if abs(normalized[2, 2]) < 1e-12:
        return {"stability_status": "singular_parameter_chart"}
    normalized /= normalized[2, 2]
    inverse = np.linalg.inv(matrix)
    groups: list[tuple[FloatArray, FloatArray]] = []
    coverage = np.zeros(len(segments))
    positions = np.linspace(0.05, 0.95, 24)
    for index, (segment, is_visible) in enumerate(zip(segments, visible, strict=True)):
        if not is_visible:
            continue
        delta = segment[1] - segment[0]
        length = float(np.linalg.norm(delta))
        direction = delta / length
        samples = segment[0] + positions[:, None] * delta
        distance, nearest = evidence.tree.query(samples)
        tangent = evidence.tangents[nearest]
        supported = (distance <= tolerance) & (np.abs(tangent @ direction) >= np.sqrt(0.5))
        if supported.sum() < 3:
            continue
        coverage[index] = float(np.ptp(positions[supported]) / 0.9)
        court_samples = _projection(samples[supported], inverse)
        normalized_samples = _projection(court_samples, source)
        jacobian = projection_jacobian(normalized_samples, normalized)
        normal = np.array([-direction[1], direction[0]])
        line_jacobian = np.einsum("nci,c->ni", jacobian, normal)
        matched_points = np.asarray(evidence.tree.data[nearest[supported]])
        residual = (samples[supported] - matched_points) @ normal / diagonal
        weight = np.sqrt(1.0 / len(residual))
        groups.append((line_jacobian * weight, residual * weight))
    metrics: dict[str, object] = {"line_coverage": coverage.tolist(), "independent_line_groups": len(groups)}
    if not groups:
        return {**metrics, "stability_status": "insufficient_line_constraints"}
    line_residual = np.concatenate([g[1] for g in groups])
    line_sigma = max(cell_sigma, float(np.sqrt(line_residual @ line_residual / len(groups))) * diagonal)
    groups = [(j * diagonal / line_sigma, r * diagonal / line_sigma) for j, r in groups]
    # The estimated H is constrained jointly by KP and LINE. Assessing LINE
    # alone would call a well-anchored fit unobservable when one marking is
    # removed. Retain the actual independent KP evidence in every line probe.
    kp_jacobian = projection_jacobian(_projection(template[supported_kp], source), normalized).reshape(-1, 8)
    kp_residual_px = (_projection(template[supported_kp], matrix) - observed[supported_kp]).ravel()
    kp_sigma = max(cell_sigma, float(np.sqrt(np.mean(kp_residual_px**2)))) if len(kp_residual_px) else cell_sigma
    fixed_jacobian = kp_jacobian * diagonal / kp_sigma
    fixed_residual = kp_residual_px / kp_sigma
    jacobian = np.vstack([fixed_jacobian, *[g[0] for g in groups]])
    residual = np.concatenate([fixed_residual, *[g[1] for g in groups]])
    _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    if len(singular) < 8 or singular[-1] <= np.finfo(float).eps * singular[0]:
        return {**metrics, "stability_status": "rank_deficient_line_constraints"}
    condition = float(singular[0] / singular[-1])
    metrics["normalized_condition"] = condition
    if condition > condition_limit:
        return {**metrics, "stability_status": "ill_conditioned_line_constraints"}
    covariance = (right.T / singular**2) @ right
    query = projection_jacobian(_projection(template, source), normalized)
    query_covariance = np.einsum("nci,ij,ndj->ncd", query, covariance, query) * diagonal**2
    std = np.sqrt(np.maximum(np.linalg.eigvalsh(query_covariance)[:, -1], 0))
    full_delta = -covariance @ jacobian.T @ residual
    maximum_influence = 0.0
    for dropped in range(len(groups)):
        kept = [group for i, group in enumerate(groups) if i != dropped]
        if not kept:
            return {**metrics, "stability_status": "leave_line_out_unconstrained"}
        reduced = np.vstack([fixed_jacobian, *[g[0] for g in kept]])
        reduced_residual = np.concatenate([fixed_residual, *[g[1] for g in kept]])
        delta, _, rank, _ = np.linalg.lstsq(reduced, -reduced_residual, rcond=1 / condition_limit)
        if rank < 8:
            return {**metrics, "stability_status": "leave_line_out_unconstrained"}
        displacement = np.einsum("nci,i->nc", query, delta - full_delta) * diagonal
        maximum_influence = max(maximum_influence, float(np.linalg.norm(displacement, axis=1).max()))
    return {
        **metrics, "stability_status": "ok", "line_localization_sigma_px": line_sigma,
        "kp_localization_sigma_px": kp_sigma,
        "max_projection_std_px": float(std.max()), "max_leave_line_out_px": maximum_influence,
    }


def assess_court_quality(
    matrix: NDArray[np.floating] | None,
    observed: NDArray[np.floating],
    valid: NDArray[np.bool_],
    line_probability: NDArray[np.floating],
    *,
    image_size_hw: tuple[int, int],
    config: CourtQualityConfig,
    evaluation_size_hw: tuple[int, int] | None = None,
    template_xy: NDArray[np.floating] | None = None,
) -> CourtQuality:
    """Assess one image-only H without using camera identity or a reference.

    All distances are source pixels. ``evaluation_size_hw`` supplies a common
    original-image diagonal for region comparisons, never a GT-based threshold.
    The covariance uses complete template queries, including off-image corners;
    visibility is not a reason to fail, insufficient constraints are.
    """
    template = court_template_xy().astype(np.float64) if template_xy is None else np.asarray(template_xy, np.float64)
    raw = np.asarray(observed, dtype=np.float64)
    raw_valid = np.asarray(valid, dtype=bool)
    if template.shape != (14, 2) or raw.shape != template.shape or raw_valid.shape != (14,):
        raise ValueError("Court quality requires ordered KP14")
    if min(*image_size_hw, *(evaluation_size_hw or image_size_hw)) < 2:
        raise ValueError("Court quality requires image dimensions >= 2")
    if matrix is None:
        return CourtQuality(False, None, ("no_homography",), {})
    homography = np.asarray(matrix, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
        return CourtQuality(False, None, ("invalid_homography",), {})
    homogeneous = np.c_[template, np.ones(14)] @ homography.T
    depth = homogeneous[:, 2]
    epsilon = 1e-9 * max(float(np.max(np.abs(depth))), 1e-300)
    if not (np.all(depth > epsilon) or np.all(depth < -epsilon)):
        return CourtQuality(False, None, ("projection_pole",), {})
    projected = homogeneous[:, :2] / depth[:, None]
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected[[0, 1, 3, 2]].astype(np.float32)):
        return CourtQuality(False, None, ("invalid_court_polygon",), {})
    diagonal = float(np.hypot(*(evaluation_size_hw or image_size_hw)))
    tolerance = config.line_tolerance_ratio * diagonal
    raw_residual = np.linalg.norm(projected - raw, axis=1)
    supported_kp = raw_valid & np.isfinite(raw).all(axis=1) & (raw_residual <= tolerance)
    reasons: list[str] = []
    if supported_kp.sum() < 4 or not _has_support(template[supported_kp], raw[supported_kp]):
        reasons.append("insufficient_independent_keypoints")
    try:
        evidence = LineEvidence.from_probability(line_probability, image_size_hw)
    except ValueError as error:
        if str(error) != "insufficient_or_diffuse_line_evidence":
            raise
        return CourtQuality(False, None, (str(error),), {"supported_keypoints": int(supported_kp.sum())})
    edges = np.asarray(COURT_HOMOGRAPHY_EDGES)
    _, line = line_residuals(projected, edges, evidence, tolerance)
    segments, visible = clip_segments(projected[edges], evidence.width, evidence.height)
    visible &= np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) >= 0.02 * evidence.diagonal
    supports = np.asarray(line["segment_support"])
    if min(line["forward_support"], line["reverse_support"]) < config.min_line_support:
        reasons.append("insufficient_line_support")
    if not visible.any() or np.any(supports[visible] < config.min_visible_segment_support):
        reasons.append("visible_line_mismatch")
    cell_sigma = 0.5 * max((image_size_hw[0] - 1) / (line_probability.shape[0] - 1),
                           (image_size_hw[1] - 1) / (line_probability.shape[1] - 1))
    stability = _sensitivity(homography, template, evidence, segments, visible, tolerance,
                             cell_sigma, diagonal, config.max_normalized_condition, raw, supported_kp)
    coverage = np.asarray(stability.get("line_coverage", np.zeros(len(edges))))
    supported_lines = visible & (supports >= 0.5) & (coverage >= config.min_line_coverage)
    transverse = np.abs(template[edges[:, 1], 0] - template[edges[:, 0], 0]) > 0
    if min(int((supported_lines & transverse).sum()), int((supported_lines & ~transverse).sum())) < 2:
        reasons.append("insufficient_line_distribution")
    std_ratio: float | None = None
    influence_ratio: float | None = None
    if stability["stability_status"] != "ok":
        reasons.append(str(stability["stability_status"]))
    else:
        std_ratio = float(stability["max_projection_std_px"]) / diagonal  # type: ignore[arg-type]
        influence_ratio = float(stability["max_leave_line_out_px"]) / diagonal  # type: ignore[arg-type]
        if std_ratio > config.max_projection_std_ratio:
            reasons.append("unstable_projection")
        if influence_ratio > config.max_leave_line_out_ratio:
            reasons.append("unstable_line_influence")
    score = None if std_ratio is None or influence_ratio is None else float(
        (line["forward_rms_px"] + line["reverse_rms_px"]) / tolerance
        + std_ratio / config.max_projection_std_ratio + influence_ratio / config.max_leave_line_out_ratio
        + 2 - line["forward_support"] - line["reverse_support"]
    )
    return CourtQuality(not reasons, score, tuple(reasons), {
        "supported_keypoints": int(supported_kp.sum()), "support_indices": np.flatnonzero(supported_kp).tolist(),
        "evaluation_diagonal_px": diagonal, "line_tolerance_px": tolerance, "line": dict(line),
        "max_projection_std_ratio": std_ratio, "max_leave_line_out_ratio": influence_ratio, **stability,
    })
