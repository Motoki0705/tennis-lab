"""RGB-only refinement of an explicit model H, with no substitution on failure."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .homography import court_template_xy
from .hybrid_homography import project_candidate
from .line_evidence import FloatArray
from .paint_correspondence import court_consensus, line_errors
from .paint_observations import (
    PAINT_NAMES,
    PaintConfig,
    PaintLine,
    extract_paint_lines,
    image_lines,
)
from .quality import CourtQuality, projection_jacobian

DEFAULT_PAINT_CONFIG = PaintConfig()

PAINT_QUALITY_SCHEMA = "court_rgb_quality_v1"


@dataclass(frozen=True)
class PaintResult:
    matrix: FloatArray | None
    projected: FloatArray
    quality: CourtQuality
    fitted_matrix: FloatArray | None
    initial_matrix: FloatArray | None


class _Chart:
    def __init__(self, size: tuple[int, int]) -> None:
        world = court_template_xy().astype(np.float64)
        centre = world.mean(axis=0)
        scale = float(np.sqrt(np.mean((world - centre) ** 2)))
        self.source = np.array([[1 / scale, 0, -centre[0] / scale], [0, 1 / scale, -centre[1] / scale], [0, 0, 1.]])
        self.diagonal = float(np.hypot(*size))
        self.image = np.array([[1 / self.diagonal, 0, -size[1] / (2 * self.diagonal)],
                               [0, 1 / self.diagonal, -size[0] / (2 * self.diagonal)], [0, 0, 1.]])

    def pack(self, matrix: FloatArray) -> FloatArray:
        normalized = self.image @ matrix @ np.linalg.inv(self.source)
        return np.asarray((normalized / normalized[2, 2]).ravel()[:8], dtype=np.float64)

    def unpack(self, parameters: FloatArray) -> FloatArray:
        return np.asarray(np.linalg.solve(self.image, np.r_[parameters, 1.].reshape(3, 3)) @ self.source, dtype=np.float64)


def _fit(initial: FloatArray, lines: tuple[PaintLine, ...], chart: _Chart,
          config: PaintConfig) -> tuple[FloatArray | None, dict[str, object]]:
    usable = [line for line in lines if line.status == "ok"]
    if len(usable) < 4:
        return None, {"stop_reason": "insufficient_lines"}
    points = np.concatenate([line.points[np.linspace(0, len(line.points) - 1, min(24, len(line.points))).astype(int)] for line in usable])
    labels = np.concatenate([np.full(min(24, len(line.points)), line.index, dtype=int) for line in usable])
    weights = np.concatenate([np.full(min(24, len(line.points)), np.sqrt(24 / min(24, len(line.points)))) for line in usable])

    def residual(parameters: FloatArray) -> FloatArray:
        matrix = chart.unpack(parameters)
        if abs(np.linalg.det(matrix)) < 1e-12:
            return np.full(len(points), 1e6)
        projected_lines = image_lines(matrix)[labels]
        return np.asarray(np.einsum("ij,ij->i", np.c_[points, np.ones(len(points))], projected_lines) * weights, dtype=np.float64)

    optimized = least_squares(residual, chart.pack(initial), loss="soft_l1", f_scale=config.line_inlier_distance_px,
                              max_nfev=config.max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    matrix = chart.unpack(optimized.x)
    rank = int(np.linalg.matrix_rank(optimized.jac))
    okay = optimized.success and rank == 8 and project_candidate(matrix, court_template_xy().astype(np.float64)) is not None
    return (matrix if okay else None), {"stop_reason": "converged" if okay else "optimizer_failed",
        "optimizer_status": int(optimized.status), "nfev": int(optimized.nfev), "rank": rank,
        "fitted_line_names": [PAINT_NAMES[line.index] for line in usable]}


def _uncertainty(matrix: FloatArray, lines: tuple[PaintLine, ...], chart: _Chart,
                 visible: NDArray[np.bool_], config: PaintConfig) -> dict[str, object]:
    usable = [line for line in lines if line.status == "ok"]
    # Two separated locations per physical line model coherent intercept/angle
    # error. Thousands of ridge pixels cannot drive uncertainty to zero.
    points = np.concatenate([line.points[np.rint(np.array([.1, .9]) * (len(line.points) - 1)).astype(int)] for line in usable])
    labels: NDArray[np.int64] = np.repeat([line.index for line in usable], 2)
    parameters = chart.pack(matrix)

    def residual(p: FloatArray) -> FloatArray:
        return np.asarray(np.einsum("ij,ij->i", np.c_[points, np.ones(len(points))], image_lines(chart.unpack(p))[labels]), dtype=np.float64)

    jacobian = np.column_stack([(residual(parameters + np.eye(8)[i] * 1e-6) - residual(parameters - np.eye(8)[i] * 1e-6)) / 2e-6 for i in range(8)])
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-10 else None
    if condition is None or condition > config.max_normalized_condition:
        return {"stability_status": "ill_conditioned_paint", "normalized_condition": condition}
    sigma = max(1., float(np.median(np.abs(residual(parameters)))))
    covariance = np.linalg.inv(jacobian.T @ jacobian) * sigma ** 2
    world = court_template_xy().astype(np.float64)
    normalized_world = (np.c_[world, np.ones(len(world))] @ chart.source.T)[:, :2]
    derivative = projection_jacobian(normalized_world, np.r_[parameters, 1.].reshape(3, 3)) * chart.diagonal
    projected_cov = np.einsum("nij,jk,nlk->nil", derivative, covariance, derivative)
    std = np.sqrt(np.maximum(np.linalg.eigvalsh(projected_cov)[:, -1], 0))
    base = project_candidate(matrix, world)
    assert base is not None
    changes: list[float] = []
    change_names: list[str] = []
    essential: list[str] = []
    for removed in usable:
        reduced = tuple(line for line in lines if line.index != removed.index)
        if config.version == "semantic_consensus_v2":
            kept = jacobian[labels != removed.index]
            spectrum = np.linalg.svd(kept, compute_uv=False)
            if len(spectrum) < 8 or spectrum[-1] <= spectrum[0] / config.max_normalized_condition:
                # Removing an essential measurement destroys identifiability;
                # the resulting arbitrary H is not measured influence. The
                # intact covariance still measures extrapolation uncertainty.
                essential.append(PAINT_NAMES[removed.index])
                continue
        alternative, _ = _fit(matrix, reduced, chart, config)
        if alternative is None:
            return {"stability_status": "unconstrained_leave_line_out", "normalized_condition": condition,
                    "projection_std_px": std.tolist(), "removed_line": PAINT_NAMES[removed.index]}
        projection = project_candidate(alternative, world)
        assert projection is not None
        change_names.append(PAINT_NAMES[removed.index])
        changes.append(float(np.max(np.linalg.norm(projection[visible] - base[visible], axis=1))))
    return {"stability_status": "ok", "normalized_condition": condition, "projection_std_px": std.tolist(),
            "max_visible_projection_std_px": float(std[visible].max()), "max_all_projection_std_px": float(std.max()),
            "leave_line_out_max_px": dict(zip(change_names, changes, strict=True)),
            "essential_lines": essential, "max_leave_line_out_px": max(changes, default=0.)}


def _semantic_alternatives(matrix: FloatArray, coarse: tuple[PaintLine, ...], fine: tuple[PaintLine, ...],
                           chart: _Chart, config: PaintConfig) -> list[str]:
    """A nearby stripe is ambiguous only if the other court lines cannot decide."""
    ambiguous = []
    base = project_candidate(matrix, court_template_xy().astype(np.float64))
    assert base is not None
    for line in coarse:
        for points in line.alternatives:
            changed = tuple(replace(other, points=points) if other.index == line.index else other for other in fine)
            alternate, _ = _fit(matrix, changed, chart, config)
            if alternate is None:
                continue
            projected_lines = image_lines(alternate)
            errors = [float(np.percentile(np.abs(np.c_[line.points, np.ones(len(line.points))] @ projected_lines[line.index]), 95))
                      for line in changed if line.status == "ok"]
            projection = project_candidate(alternate, court_template_xy().astype(np.float64))
            assert projection is not None
            if max(errors) <= config.max_line_residual_px and float(np.max(np.linalg.norm(projection - base, axis=1))) >= config.ambiguity_separation_px:
                ambiguous.append(PAINT_NAMES[line.index])
                break
    return ambiguous


def refine_paint(image: NDArray[np.uint8], initial: FloatArray | None,
                  config: PaintConfig = DEFAULT_PAINT_CONFIG) -> PaintResult:
    """Coarse/fine stripe fitting; missing support and ambiguity are failures.

    Only semantically confirmed initial lines are mandatory in v2. Every raw
    stripe and its disposition remains diagnostic. No rejected H is published.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("Paint refinement requires uint8 RGB")
    metrics: dict[str, object] = {"schema": PAINT_QUALITY_SCHEMA, "method": config.version,
        "score_units": "dimensionless_paint_cost", "coordinate_units": "source_image_pixels"}
    fitted: FloatArray | None = None

    def finish(reasons: list[str], score: float | None = None) -> PaintResult:
        accepted = fitted is not None and not reasons
        projection = project_candidate(fitted, court_template_xy().astype(np.float64)) if accepted and fitted is not None else None
        quality = CourtQuality(accepted, score, tuple(dict.fromkeys(reasons)), metrics)
        return PaintResult(fitted if accepted else None, np.full((14, 2), np.nan) if projection is None else projection, quality, fitted, initial)

    projected = None if initial is None else project_candidate(initial, court_template_xy().astype(np.float64))
    if projected is None:
        return finish(["no_model_homography"])
    assert initial is not None
    brightness = image.min(axis=2).astype(np.float64)
    coarse = extract_paint_lines(brightness, projected, config.radius_px, config)
    initial_stripes = {line.index for line in coarse if line.status == "ok"}
    required = initial_stripes.copy()
    chart = _Chart(image.shape[:2])
    coarse_start = initial
    coarse_observations = coarse
    if config.version == "semantic_consensus_v2":
        near = extract_paint_lines(brightness, projected, config.fine_radius_px, config)
        coarse_start, coarse_observations, consensus = court_consensus(initial, coarse, config, near)
        metrics["initial_correspondence"] = consensus
        # The fine-search band is the existing localization capture range. A
        # distant net/parallel stripe is not a mandatory semantic observation.
        initial_errors = line_errors(initial, coarse_observations)
        required = {line.index for line in coarse_observations if line.status == "ok"
                    and initial_errors[line.index] <= config.fine_radius_px}
        metrics["initial_model_residual_px"] = {PAINT_NAMES[line.index]: float(initial_errors[line.index])
            if np.isfinite(initial_errors[line.index]) else None for line in coarse}
        if coarse_start is None:
            metrics["initial_stripe_lines"] = [PAINT_NAMES[i] for i in sorted(initial_stripes)]
            return finish(["insufficient_paint_distribution" if len(initial_stripes) < config.min_court_lines
                           else "unconfirmed_paint_correspondence"])
        ambiguous_initial = set(_semantic_alternatives(coarse_start, coarse, coarse_observations, chart, config))
        required = {i for i in required if PAINT_NAMES[i] not in ambiguous_initial}
        metrics["initial_unresolved_alternatives"] = sorted(ambiguous_initial)
    metrics["initial_stripe_lines"] = [PAINT_NAMES[i] for i in sorted(initial_stripes)]
    metrics["initial_supported_lines"] = [PAINT_NAMES[i] for i in sorted(required)]
    metrics["initial_ambiguous_lines"] = [PAINT_NAMES[line.index] for line in coarse if line.ambiguous and line.status == "ok"]
    transverse = {0, 1, 6, 7}
    coarse_supported = {line.index for line in coarse_observations if line.status == "ok"}
    if len(coarse_supported) < config.min_court_lines or min(len(coarse_supported & transverse), len(coarse_supported - transverse)) < 2:
        metrics["lines"] = [{"name": PAINT_NAMES[line.index], "status": line.status, "coverage": line.coverage} for line in coarse]
        return finish(["insufficient_paint_distribution"])
    assert coarse_start is not None
    coarse_h, coarse_diagnostic = _fit(coarse_start, coarse_observations, chart, config)
    metrics["coarse_fit"] = coarse_diagnostic
    if coarse_h is None:
        return finish(["paint_optimization_failed"])
    coarse_projection = project_candidate(coarse_h, court_template_xy().astype(np.float64))
    assert coarse_projection is not None
    fine = extract_paint_lines(brightness, coarse_projection, config.fine_radius_px, config)
    supported = {line.index for line in fine if line.status == "ok"}
    lost = required - supported
    metrics["lost_initial_lines"] = [PAINT_NAMES[i] for i in sorted(lost)]
    metrics["lost_unconfirmed_stripes"] = [PAINT_NAMES[i] for i in sorted(initial_stripes - required - supported)]
    # Retain failures as reasons even when enough other lines could fit a H.
    reasons = ["lost_initial_paint_lines"] if lost else []
    if len(supported) < config.min_court_lines or min(len(supported & transverse), len(supported - transverse)) < 2:
        return finish([*reasons, "insufficient_paint_distribution"])
    fitted, diagnostic = _fit(coarse_h, fine, chart, config)
    metrics["fine_fit"] = diagnostic
    if fitted is None:
        return finish([*reasons, "paint_optimization_failed"])
    projected_lines = image_lines(fitted)
    if config.version == "semantic_consensus_v2":
        retained_errors = line_errors(fitted, coarse_observations)
        metrics["confirmed_initial_residual_px"] = {PAINT_NAMES[i]: float(retained_errors[i]) for i in sorted(required)}
        shifted = [PAINT_NAMES[i] for i in sorted(required) if retained_errors[i] > config.max_line_residual_px]
        metrics["shifted_confirmed_lines"] = shifted
        if shifted:
            reasons.append("confirmed_paint_line_shifted")
    reports = []
    residuals = []
    for line in fine:
        errors = np.abs(np.c_[line.points, np.ones(len(line.points))] @ projected_lines[line.index])
        p95 = float(np.percentile(errors, 95)) if len(errors) else None
        reports.append({"name": PAINT_NAMES[line.index], "status": line.status, "supported": line.index in supported,
            "required_initially": line.index in required, "support_locations": len(line.points), "sample_locations": line.locations,
            "coverage": line.coverage, "ambiguous": line.ambiguous,
            "median_residual_px": float(np.median(errors)) if len(errors) else None, "p95_residual_px": p95})
        if line.index in supported and p95 is not None:
            residuals.append(p95)
    metrics["lines"] = reports
    if max(residuals) > config.max_line_residual_px:
        reasons.append("paint_residual_too_large")
    ambiguous = _semantic_alternatives(fitted, coarse, fine, chart, config)
    metrics["unresolved_semantic_alternatives"] = ambiguous
    if ambiguous:
        reasons.append("ambiguous_paint_correspondence")
    final_projection = project_candidate(fitted, court_template_xy().astype(np.float64))
    assert final_projection is not None
    height, width = image.shape[:2]
    visible = (final_projection[:, 0] >= 0) & (final_projection[:, 0] <= width - 1) & (final_projection[:, 1] >= 0) & (final_projection[:, 1] <= height - 1)
    if not visible.any():
        return finish([*reasons, "no_visible_court"])
    stability = _uncertainty(fitted, fine, chart, visible, config)
    metrics["stability"] = stability
    if stability["stability_status"] != "ok":
        return finish([*reasons, str(stability["stability_status"])])
    std = float(stability["max_visible_projection_std_px"])  # type: ignore[arg-type]
    influence = float(stability["max_leave_line_out_px"])  # type: ignore[arg-type]
    if config.version == "semantic_consensus_v2":
        unconfirmed_essential = sorted(set(cast(list[str], stability["essential_lines"])) - {PAINT_NAMES[i] for i in required})
        metrics["unconfirmed_essential_lines"] = unconfirmed_essential
        if unconfirmed_essential:
            reasons.append("unconfirmed_essential_paint")
    if std > config.max_projection_std_px:
        reasons.append("uncertain_paint_extrapolation")
    if influence > config.max_leave_line_out_px:
        reasons.append("unstable_paint_line_influence")
    score = float(np.mean(residuals) / config.max_line_residual_px + std / config.max_projection_std_px + influence / config.max_leave_line_out_px)
    return finish(reasons, score)
