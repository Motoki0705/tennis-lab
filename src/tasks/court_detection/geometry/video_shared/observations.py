"""Frame-local stripe proposals and independent projective-grid verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..homography import COURT_HOMOGRAPHY_EDGES, court_template_xy
from ..hybrid_homography import project_candidate
from ..line_evidence import FloatArray
from ..paint_observations import (
    PAINT_NAMES,
    PaintLine,
    extract_paint_lines,
    image_lines,
)
from .candidates import CourtSeed
from .endpoints import measure_endings
from .fit import TRANSVERSE, PaintGroup, SharedPaintConfig, _fit_group_geometry


@dataclass(frozen=True)
class FramePaint:
    groups: tuple[PaintGroup, ...]
    diagnostic: dict[str, Any]


def clustered_initializers(seeds: tuple[CourtSeed, ...], config: SharedPaintConfig
                          ) -> tuple[tuple[CourtSeed, ...], dict[str, Any]]:
    """Deterministic geometric neighborhoods, not a majority vote for a court."""
    representatives: list[CourtSeed] = []
    projections: list[FloatArray] = []
    assignments = {}
    invalid = []
    for seed in sorted(seeds, key=lambda s: s.candidate_id):
        projected = project_candidate(seed.matrix, court_template_xy().astype(float))
        if projected is None:
            invalid.append(seed.candidate_id)
            continue
        match = next((i for i, old in enumerate(projections)
                      if np.linalg.norm(projected - old, axis=1).max() <= config.initializer_cluster_px), None)
        if match is None:
            match = len(representatives)
            representatives.append(seed)
            projections.append(projected)
        assignments[seed.candidate_id] = representatives[match].candidate_id
    return tuple(representatives), {"input_candidates": len(seeds), "initializers": len(representatives),
                                    "assignments": assignments, "invalid_projection": invalid,
                                    "cluster_radius_px": config.initializer_cluster_px}


def _options(lines: tuple[PaintLine, ...]) -> dict[int, list[PaintLine]]:
    result = {}
    for line in lines:
        if line.status != "ok":
            continue
        values = [line]
        for points in line.alternatives:
            centre = points.mean(axis=0)
            _, _, vh = np.linalg.svd(points - centre, full_matrices=False)
            equation = np.r_[vh[-1], -vh[-1] @ centre]
            values.append(replace(line, points=points, line=equation, alternatives=()))
        result[line.index] = values
    return result


def _finite_coverage(matrix: FloatArray, line: PaintLine) -> float:
    projected = project_candidate(matrix, court_template_xy().astype(float))
    if projected is None:
        return 0.
    a, b = projected[np.asarray(COURT_HOMOGRAPHY_EDGES[line.index])]
    direction = b - a
    t = (line.points - a) @ direction / (direction @ direction)
    return float(np.mean((t >= -.02) & (t <= 1.02)))


def measure_frame(image: NDArray[np.uint8], initial: FloatArray, *, clip_id: str,
                  frame_index: int, config: SharedPaintConfig) -> FramePaint:
    """Generate mappings from stripe proposals; verify using the remaining lines.

    Every physical line receives at most one observed stripe, and one observed
    stripe cannot certify two physical lines. Candidate counts and alternatives
    survive even when no semantic assignment can be certified.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Shared court measurement requires source uint8 RGB")
    projected = project_candidate(initial, court_template_xy().astype(float))
    if projected is None:
        return FramePaint((), {"reason": "invalid_initializer"})
    lines = extract_paint_lines(image.min(axis=2).astype(float), projected, config.paint.radius_px, config.paint)
    options = _options(lines)
    diagnostic: dict[str, Any] = {"clip_id": clip_id, "frame_index": frame_index,
        "stripes": [{"line": PAINT_NAMES[line.index], "status": line.status, "coverage": line.coverage,
                     "points": len(line.points), "alternatives": len(line.alternatives),
                     "equation": None if line.line is None else line.line.tolist()} for line in lines]}
    world = image_lines(np.eye(3))
    matrices = [initial]
    for rows in combinations(sorted(options.keys() & TRANSVERSE), 2):
        for columns in combinations(sorted(options.keys() - TRANSVERSE), 2):
            pairs = [(a, b) for a in rows for b in columns]
            source = np.asarray([np.cross(world[a], world[b]) for a, b in pairs])
            for grid_values in product(*(options[i] for i in (*rows, *columns))):
                chosen = dict(zip((*rows, *columns), grid_values, strict=True))
                target = np.asarray([np.cross(chosen[a].line, chosen[b].line) for a, b in pairs])
                if np.any(np.abs(target[:, 2]) < 1e-9):
                    continue
                matrix, _ = cv2.findHomography(source[:, :2] / source[:, 2, None], target[:, :2] / target[:, 2, None], method=0)
                if matrix is not None and project_candidate(matrix, court_template_xy().astype(float)) is not None:
                    matrices.append(matrix)
    # Score the same observation proposals for every mapping, including the
    # alternatives. A local model anchor receives no extra weight.
    samples = {(i, j): np.c_[value.points, np.ones(len(value.points))]
               for i, values in options.items() for j, value in enumerate(values)}
    choices_by_h = []
    best = None
    best_key = None
    for matrix in matrices:
        predicted = image_lines(matrix)
        chosen_lines = {}
        errors = []
        for index, values in options.items():
            distances = [float(np.percentile(np.abs(samples[index, j] @ predicted[index]), 95)) for j in range(len(values))]
            j = int(np.argmin(distances))
            if distances[j] <= config.paint.max_line_residual_px and _finite_coverage(matrix, values[j]) >= config.paint.min_coverage:
                chosen_lines[index] = values[j]
                errors.append(distances[j])
        supports = set(chosen_lines)
        # Keep weaker mappings as diagnostic evidence; they are not permission
        # to use an ambiguous two-transverse-line fit downstream.
        key = (len(supports), len(supports & TRANSVERSE), -float(np.mean(errors)) if errors else -np.inf)
        if best_key is None or key > best_key:
            best_key, best = key, (matrix, chosen_lines)
        if (len(supports) < config.min_lines or len(supports & TRANSVERSE) < config.min_transverse
                or len(supports - TRANSVERSE) < config.min_longitudinal):
            continue
        duplicate = False
        for a, b in combinations(chosen_lines.values(), 2):
            assert b.line is not None
            if np.percentile(np.abs(np.c_[a.points, np.ones(len(a.points))] @ b.line), 95) <= config.paint.line_inlier_distance_px:
                duplicate = True
                break
        if not duplicate:
            choices_by_h.append((key, matrix, chosen_lines))
    diagnostic["hypotheses"] = len(matrices)
    diagnostic["certifiable_hypotheses"] = len(choices_by_h)
    if best is not None:
        diagnostic["best_supported_lines"] = [PAINT_NAMES[i] for i in sorted(best[1])]
        diagnostic["best_fitted_homography"] = best[0].tolist()
    if not choices_by_h:
        diagnostic["reason"] = "insufficient_independent_line_support"
        return FramePaint((), diagnostic)
    choices_by_h.sort(key=lambda value: value[0], reverse=True)
    # Different four-line constructions of the same correspondence set are
    # numerical initializers, not different physical courts. Refit each unique
    # set with all its observations before testing geometric ambiguity.
    refined = []
    extent_attempts: list[dict[str, Any]] = []
    seen_assignments = set()
    for key, matrix, choices in choices_by_h:
        if key[:2] != choices_by_h[0][0][:2]:
            continue
        assignment = tuple((i, id(value)) for i, value in sorted(choices.items()))
        if assignment in seen_assignments:
            continue
        seen_assignments.add(assignment)
        active = dict(choices)
        rejected_extents: dict[str, Any] = {}
        for _ in range(3):
            groups = tuple(PaintGroup(clip_id, frame_index, i, value.points) for i, value in sorted(active.items()))
            fitted = _fit_group_geometry(matrix, groups, (image.shape[1], image.shape[0]), replace(config, min_clips_per_line=1))
            if fitted.matrix is None:
                extent_attempts.append({'status': 'reduced_geometry_failed', 'reasons': list(fitted.reasons),
                                        'removed': rejected_extents})
                break
            matrix = fitted.matrix
            endings = {}
            extent = {}
            for index in sorted(active.keys() & TRANSVERSE):
                endings[index], extent[PAINT_NAMES[index]] = measure_endings(image, matrix, index, config.paint)
            continued = {i for i, status in endings.items() if 'continues' in status}
            if continued:
                rejected_extents.update({PAINT_NAMES[i]: extent[PAINT_NAMES[i]] for i in continued})
                active = {i: value for i, value in active.items() if i not in continued}
                continue
            verified = {i for i, status in endings.items() if 'ended' in status}
            detail = {'status': 'verified' if verified & {0, 1} and verified & {6, 7} else 'unverified_finite_line_identity',
                      'evidence': extent, 'removed': rejected_extents}
            extent_attempts.append(detail)
            if detail['status'] == 'verified':
                refined.append((float(fitted.diagnostic['max_group_p95_px']), matrix, active, endings, detail))
            break
        else:
            extent_attempts.append({'status': 'extent_selection_did_not_converge', 'removed': rejected_extents})
    diagnostic['distinct_correspondence_sets'] = len(seen_assignments)
    diagnostic['extent_attempts'] = extent_attempts
    if not refined:
        diagnostic['reason'] = 'frame_correspondence_fit_failed'
        return FramePaint((), diagnostic)
    refined.sort(key=lambda item: item[0])
    _, matrix, choices, endings, extent_detail = refined[0]
    final = project_candidate(matrix, court_template_xy().astype(float))
    assert final is not None
    alternatives = []
    for _, other, _, _, _ in refined[1:]:
        other_points = project_candidate(other, court_template_xy().astype(float))
        assert other_points is not None
        if float(np.linalg.norm(final - other_points, axis=1).max()) > config.equivalent_projection_px:
            alternatives.append(other.tolist())
    diagnostic['alternative_homographies'] = alternatives
    diagnostic['finite_line_evidence'] = extent_detail
    if alternatives:
        diagnostic.update(reason="ambiguous_frame_line_assignment")
        return FramePaint((), diagnostic)
    diagnostic.update(reason="measured", chosen_homography=matrix.tolist(),
                      supported_lines=[PAINT_NAMES[i] for i in sorted(choices)])
    return FramePaint(tuple(PaintGroup(clip_id, frame_index, i, value.points, endings.get(i, ('unobservable', 'unobservable')))
                            for i, value in sorted(choices.items())), diagnostic)
