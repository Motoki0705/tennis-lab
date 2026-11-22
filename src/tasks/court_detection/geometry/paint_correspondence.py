"""Verify stripe identities against the complete planar court, before fitting.

An extended bright stripe is only a measurement proposal. Two lines from each
court direction define a projective grid; independent remaining markings test
that grid. This excludes a net/other stripe that no common court can explain.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, product

import cv2
import numpy as np

from .homography import court_template_xy
from .hybrid_homography import project_candidate
from .line_evidence import FloatArray
from .paint_observations import PAINT_NAMES, PaintConfig, PaintLine, image_lines

TRANSVERSE = frozenset((0, 1, 6, 7))


def line_errors(matrix: FloatArray, lines: tuple[PaintLine, ...]) -> FloatArray:
    predicted = image_lines(matrix)
    return np.asarray([
        np.percentile(np.abs(np.c_[line.points, np.ones(len(line.points))] @ predicted[line.index]), 95)
        if line.status == "ok" else np.inf for line in lines
    ], dtype=np.float64)


def court_consensus(initial: FloatArray, lines: tuple[PaintLine, ...], config: PaintConfig,
                    near: tuple[PaintLine, ...] | None = None
                    ) -> tuple[FloatArray | None, tuple[PaintLine, ...], dict[str, object]]:
    """Enumerate physical grids; close stripes compete with coarse proposals.

    A model-local stripe is an identity anchor only when a distributed court
    verifies it. Coarse coverage cannot outweigh those verified identities.
    """
    options: dict[int, list[PaintLine]] = {}
    for index, line in enumerate(lines):
        values = [value for value in (() if near is None else (near[index],)) + (line,) if value.status == "ok"]
        distinct: list[PaintLine] = []
        for value in values:
            if all(float(np.percentile(np.abs(np.c_[value.points, np.ones(len(value.points))] @ old.line), 95))
                   > config.line_inlier_distance_px for old in distinct):
                distinct.append(value)
        if distinct:
            options[index] = distinct
    world_lines = image_lines(np.eye(3))
    matrices = [initial]
    for rows in combinations(sorted(options.keys() & TRANSVERSE), 2):
        for columns in combinations(sorted(options.keys() - TRANSVERSE), 2):
            pairs = [(a, b) for a in rows for b in columns]
            source = np.asarray([np.cross(world_lines[a], world_lines[b]) for a, b in pairs])
            for grid_values in product(*(options[i] for i in (*rows, *columns))):
                chosen = dict(zip((*rows, *columns), grid_values, strict=True))
                target = np.asarray([np.cross(chosen[a].line, chosen[b].line) for a, b in pairs])
                if np.any(np.abs(target[:, 2]) < 1e-9):
                    continue
                matrix, _ = cv2.findHomography(source[:, :2] / source[:, 2, None], target[:, :2] / target[:, 2, None], method=0)
                if matrix is not None and project_candidate(matrix, court_template_xy().astype(float)) is not None:
                    matrices.append(matrix)
    # Precompute homogeneous samples: scoring hypotheses performs no extraction.
    samples = {(i, j): np.c_[line.points, np.ones(len(line.points))] for i, values in options.items() for j, line in enumerate(values)}
    best = None
    best_choices: dict[int, PaintLine] = {}
    best_anchors: set[int] = set()
    best_key = None
    for matrix in matrices:
        predicted = image_lines(matrix)
        choices = {}
        errors = []
        anchors = set()
        for i, values in options.items():
            distances = [float(np.percentile(np.abs(samples[i, j] @ predicted[i]), 95)) for j in range(len(values))]
            choice = int(np.argmin(distances))
            if distances[choice] <= config.max_line_residual_px:
                choices[i] = values[choice]
                errors.append(distances[choice])
            if near is not None and near[i].status == "ok" and distances[0] <= config.max_line_residual_px:
                anchors.add(i)
                choices[i] = values[0]
        support = set(choices)
        if len(support) < config.min_court_lines or min(len(support & TRANSVERSE), len(support - TRANSVERSE)) < 2:
            continue
        key = (-len(anchors), -len(support), float(np.mean(errors)))
        if best_key is None or key < best_key:
            best, best_choices, best_anchors, best_key = matrix, choices, anchors, key
    verified = tuple(best_choices[line.index] if line.index in best_choices else replace(line, status="unconfirmed_correspondence")
                     if line.status == "ok" else line for line in lines)
    return best, verified, {"hypotheses": len(matrices), "supported_lines": [PAINT_NAMES[i] for i in sorted(best_choices)],
        "confirmed_model_local_lines": [PAINT_NAMES[i] for i in sorted(best_anchors)],
        "excluded_stripes": [PAINT_NAMES[i] for i in sorted(options.keys() - best_choices.keys())]}
