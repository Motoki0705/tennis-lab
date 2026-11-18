"""Balanced multi-frame court fit with explicit semantic/temporal support.

Model predictions initialize the search; their coordinates never enter the
final RGB objective. In particular two transverse stripes cannot independently
identify baselines when one might be a static net band.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ..homography import court_template_xy
from ..hybrid_homography import project_candidate
from ..line_evidence import FloatArray
from ..paint_observations import PAINT_NAMES, PaintConfig, image_lines

TRANSVERSE = frozenset((0, 1, 6, 7))


@dataclass(frozen=True)
class SharedPaintConfig:
    """Development profile; pixel limits refer to the supplied source images."""

    paint: PaintConfig = field(default_factory=PaintConfig)
    min_lines: int = 6
    min_transverse: int = 2
    min_longitudinal: int = 3
    min_clips_per_line: int = 3
    samples_per_group: int = 24
    max_common_line_p95_px: float = 6.
    equivalent_projection_px: float = 3.
    initializer_cluster_px: float = 12.
    max_nfev: int = 200
    max_condition: float = 10000.

    def __post_init__(self) -> None:
        integers = (self.min_lines, self.min_transverse, self.min_longitudinal,
                    self.min_clips_per_line, self.samples_per_group, self.max_nfev)
        if any(type(v) is not int or v <= 0 for v in integers):
            raise ValueError("Shared court counts must be positive integers")
        if not 2 <= self.min_transverse <= 4 or not 2 <= self.min_longitudinal <= 5 or not 5 <= self.min_lines <= 9:
            raise ValueError("Insufficient independent court line families")
        if self.samples_per_group < 4:
            raise ValueError("Insufficient spatial samples")
        for v in (self.max_common_line_p95_px, self.equivalent_projection_px, self.initializer_cluster_px, self.max_condition):
            if not np.isfinite(v) or v <= 0:
                raise ValueError("Shared court tolerances must be positive and finite")


@dataclass(frozen=True)
class PaintGroup:
    clip_id: str
    frame_index: int
    line_index: int
    points: FloatArray
    endings: tuple[str, str] = ("unobservable", "unobservable")

    def __post_init__(self) -> None:
        if (not self.clip_id or type(self.frame_index) is not int or self.frame_index < 0
                or not 0 <= self.line_index < 9 or self.points.ndim != 2 or self.points.shape[1] != 2
                or len(self.points) < 4 or not np.isfinite(self.points).all()):
            raise ValueError("Invalid paint observation group")
        if len(self.endings) != 2 or any(value not in {"unobservable", "ended", "continues", "ambiguous"} for value in self.endings):
            raise ValueError("Invalid endpoint evidence")


@dataclass(frozen=True)
class SharedFit:
    matrix: FloatArray | None
    fitted_matrix: FloatArray | None
    reasons: tuple[str, ...]
    diagnostic: dict[str, Any]


class HomographyChart:
    def __init__(self, size_wh: tuple[int, int]):
        world = court_template_xy().astype(float)
        centre = world.mean(axis=0)
        scale = float(np.sqrt(np.mean((world - centre) ** 2)))
        self.world = np.array([[1 / scale, 0, -centre[0] / scale], [0, 1 / scale, -centre[1] / scale], [0, 0, 1.]])
        diagonal = float(np.hypot(*size_wh))
        self.image = np.array([[1 / diagonal, 0, -size_wh[0] / (2 * diagonal)],
                               [0, 1 / diagonal, -size_wh[1] / (2 * diagonal)], [0, 0, 1.]])

    def pack(self, matrix: FloatArray) -> FloatArray:
        h = self.image @ matrix @ np.linalg.inv(self.world)
        return np.asarray((h / h[2, 2]).ravel()[:8], dtype=np.float64)

    def unpack(self, parameters: FloatArray) -> FloatArray:
        return np.asarray(np.linalg.solve(self.image, np.r_[parameters, 1.].reshape(3, 3)) @ self.world, dtype=np.float64)


def _fit_group_geometry(initial: FloatArray, groups: tuple[PaintGroup, ...], size_wh: tuple[int, int],
                        config: SharedPaintConfig) -> SharedFit:
    """Use equal physical-line, clip, frame weights; never trim a whole epoch."""
    keys = [(g.clip_id, g.frame_index, g.line_index) for g in groups]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate frame/physical-line observation")
    if min(size_wh) < 2 or np.asarray(initial).shape != (3, 3) or not np.isfinite(initial).all():
        raise ValueError("Invalid shared court initializer/image size")
    by_line: dict[int, set[str]] = defaultdict(set)
    for group in groups:
        by_line[group.line_index].add(group.clip_id)
    supported = {i for i, clips in by_line.items() if len(clips) >= config.min_clips_per_line}
    diagnostic: dict[str, Any] = {"schema": "video_court_fit_v3", "settings": asdict(config),
        "supported_lines": [PAINT_NAMES[i] for i in sorted(supported)],
        "line_clip_counts": {PAINT_NAMES[i]: len(clips) for i, clips in sorted(by_line.items())},
        "observed_groups": len(groups), "excluded_groups": [list(k) for k, g in zip(keys, groups, strict=True) if g.line_index not in supported]}
    if (len(supported) < config.min_lines or len(supported & TRANSVERSE) < config.min_transverse
            or len(supported - TRANSVERSE) < config.min_longitudinal):
        return SharedFit(None, None, ("insufficient_independent_line_support",), diagnostic)
    selected = [g for g in groups if g.line_index in supported]
    counts = Counter((g.clip_id, g.line_index) for g in selected)
    points, labels, weights = [], [], []
    for group in selected:
        # Increasing pixel density within one frame must not increase its vote.
        order = np.argsort(group.points[:, np.argmax(np.ptp(group.points, axis=0))], kind="stable")
        chosen = order[np.linspace(0, len(order) - 1, min(config.samples_per_group, len(order))).round().astype(int)]
        samples = group.points[chosen]
        points.extend(np.c_[samples, np.ones(len(samples))])
        labels.extend([group.line_index] * len(samples))
        weight = 1 / (len(samples) * counts[group.clip_id, group.line_index] * len(by_line[group.line_index]))
        weights.extend([weight] * len(samples))
    xy, indices = np.asarray(points), np.asarray(labels)
    weights_array = np.asarray(weights)
    chart = HomographyChart(size_wh)

    def residual(parameters: FloatArray) -> FloatArray:
        matrix = chart.unpack(parameters)
        if abs(np.linalg.det(matrix)) < 1e-12:
            return np.full(len(xy), 1e6)
        return np.asarray(np.einsum("ij,ij->i", xy, image_lines(matrix)[indices]), dtype=np.float64)

    parameters = chart.pack(initial)
    # IRLS keeps the robust scale in unweighted source pixels. Applying soft-L1
    # to already weighted residuals would give densely sampled groups a different
    # outlier threshold. The observation set is fixed throughout these solves.
    records = []
    for _ in range(4):
        errors = residual(parameters)
        robust = 1 / np.sqrt(1 + (errors / config.paint.line_inlier_distance_px) ** 2)
        scale = np.sqrt(weights_array * robust)
        optimized = least_squares(lambda p, scale=scale: residual(p) * scale, parameters, max_nfev=config.max_nfev,
                                  ftol=1e-9, xtol=1e-9, gtol=1e-9)
        parameters = optimized.x
        records.append({"success": bool(optimized.success), "status": int(optimized.status), "nfev": int(optimized.nfev)})
        if not optimized.success:
            diagnostic["solves"] = records
            return SharedFit(None, None, ("shared_optimization_failed",), diagnostic)
    matrix = chart.unpack(parameters)
    matrix /= matrix[2, 2]
    diagnostic["solves"] = records
    if project_candidate(matrix, court_template_xy().astype(float)) is None:
        return SharedFit(None, matrix, ("invalid_shared_projection",), diagnostic)
    # Count two spatial constraints per physical line. Correlated frames or
    # thousands of pixels cannot manufacture eight independent constraints.
    endpoints = []
    endpoint_labels = []
    for index in sorted(supported):
        candidates = xy[indices == index, :2]
        axis = int(np.argmax(np.ptp(candidates, axis=0)))
        order = np.argsort(candidates[:, axis], kind="stable")
        endpoints.extend(candidates[order[np.rint(np.array([.1, .9]) * (len(order) - 1)).astype(int)]])
        endpoint_labels.extend([index, index])

    def constraints(p: FloatArray) -> FloatArray:
        return np.asarray(np.einsum("ij,ij->i", np.c_[endpoints, np.ones(len(endpoints))],
                                    image_lines(chart.unpack(p))[endpoint_labels]), dtype=np.float64)

    jacobian = np.column_stack([(constraints(parameters + np.eye(8)[i] * 1e-6)
                                - constraints(parameters - np.eye(8)[i] * 1e-6)) / 2e-6 for i in range(8)])
    spectrum = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(spectrum[0] / spectrum[-1]) if spectrum[-1] > 1e-10 else None
    reasons = []
    if condition is None or condition > config.max_condition:
        reasons.append("unidentifiable_shared_court")
    reports = []
    for group in groups:
        r = np.c_[group.points, np.ones(len(group.points))] @ image_lines(matrix)[group.line_index]
        reports.append({"clip_id": group.clip_id, "frame_index": group.frame_index, "line": PAINT_NAMES[group.line_index],
                        "used": group.line_index in supported, "signed_median_px": float(np.median(r)),
                        "p95_px": float(np.percentile(np.abs(r), 95)), "points": len(r)})
    common_residual = max(float(r["p95_px"]) for r in reports if r["used"])
    distinct_frames = {(g.clip_id, g.frame_index) for g in groups}
    if common_residual > config.max_common_line_p95_px and len(distinct_frames) == 1:
        reasons.append("inconsistent_local_court")
    # A coherent temporal displacement belongs to the explicit 3/10/25px drift
    # policy. It must not silently become a stricter whole-camera 6px gate.
    diagnostic.update(normalized_condition=condition, group_residuals=reports, max_group_p95_px=common_residual,
                      requires_clip_drift_inspection=True,
                      large_common_residual_groups=[r for r in reports if r['used'] and r['p95_px'] > config.max_common_line_p95_px])
    return SharedFit(None if reasons else matrix, matrix, tuple(reasons), diagnostic)


def fit_common_court(initial: FloatArray, groups: tuple[PaintGroup, ...], size_wh: tuple[int, int],
                     config: SharedPaintConfig) -> SharedFit:
    """Certify finite baseline/service identities as well as an identifiable grid."""
    # Unverified transverse stripes (notably the far net band) remain evidence,
    # but cannot constrain the final common H just because other lines match.
    trusted = tuple(g for g in groups if g.line_index not in TRANSVERSE or ('ended' in g.endings and 'continues' not in g.endings))
    numerical = _fit_group_geometry(initial, trusted, size_wh, config)
    endings: dict[int, set[str]] = defaultdict(set)
    continued = []
    for group in groups:
        if group.line_index not in TRANSVERSE:
            continue
        if "continues" in group.endings:
            continued.append((group.clip_id, group.frame_index, PAINT_NAMES[group.line_index]))
        if "ended" in group.endings and "continues" not in group.endings:
            endings[group.line_index].add(group.clip_id)
    verified = {index for index, clips in endings.items() if len(clips) >= config.min_clips_per_line}
    reasons = list(numerical.reasons)
    if not verified & {0, 1} or not verified & {6, 7}:
        reasons.append("unverified_finite_line_identity")
    supported_indices = {i for i, name in enumerate(PAINT_NAMES) if name in numerical.diagnostic['supported_lines']}
    if len(supported_indices & TRANSVERSE) == 2 and 8 not in supported_indices:
        # A baseline and one service stripe, even with four sidelines and clear
        # endpoints, admit a near/far-service alias. The finite centre segment
        # or an independently supported third transverse marking resolves it.
        reasons.append('unverified_service_order')
    if continued:
        reasons.append("paint_continues_beyond_template")
    diagnostic = {**numerical.diagnostic, "endpoint_checked": True,
                  "verified_finite_transverse_lines": [PAINT_NAMES[i] for i in sorted(verified)],
                  "continued_transverse_groups": continued,
                  "unverified_transverse_groups": [{'clip_id': g.clip_id, 'frame_index': g.frame_index,
                                                      'line': PAINT_NAMES[g.line_index], 'endings': list(g.endings)}
                                                     for g in groups if g.line_index in TRANSVERSE
                                                     and ('ended' not in g.endings or 'continues' in g.endings)]}
    return SharedFit(None if reasons else numerical.matrix, numerical.fitted_matrix, tuple(reasons), diagnostic)


def choose_common_solution(results: tuple[SharedFit, ...], config: SharedPaintConfig) -> SharedFit:
    """A better scalar score does not resolve a physically different valid H."""
    accepted = [r for r in results if r.matrix is not None]
    summary: dict[str, Any] = {"candidates": len(results), "accepted_candidates": len(accepted),
                                 "candidate_reasons": [list(r.reasons) for r in results]}
    if not accepted:
        return SharedFit(None, None, ("no_acceptable_shared_court",), summary)
    projections = [project_candidate(r.matrix, court_template_xy().astype(float)) for r in accepted]
    for i, a in enumerate(projections):
        assert a is not None
        for b in projections[i + 1:]:
            assert b is not None
            if float(np.linalg.norm(a - b, axis=1).max()) > config.equivalent_projection_px:
                summary["alternative_homographies"] = [r.matrix.tolist() for r in accepted if r.matrix is not None]
                return SharedFit(None, None, ("ambiguous_line_assignment",), summary)
    best = min(accepted, key=lambda r: float(r.diagnostic["max_group_p95_px"]))
    return SharedFit(best.matrix, best.fitted_matrix, (), {**best.diagnostic, **summary})
