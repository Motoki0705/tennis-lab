"""Produce frame-local 3D candidates and exclusive raw-row assignments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.assignment import Assignment, assign
from src.tennis_scene.multiview_pose.geometry import (
    PairFits,
    ViewRays,
    compare_rays,
    fit_groups,
    view_rays,
)
from src.tennis_scene.multiview_pose.inputs import PoseInputs
from src.tennis_scene.multiview_pose.schema import (
    FramePoseCandidates,
    PoseConfig,
    RowRejection,
)
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.schema.player import COCO17_TORSO_TWIST_JOINTS

DEFAULT_CONFIG = PoseConfig()


def _compatible_groups(
    pairs: dict[tuple[int, int], PairFits], view_count: int
) -> NDArray[np.int64]:
    neighbours: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for (left, right), pair in pairs.items():
        for row_left, row_right in np.argwhere(pair.compatible_rows):
            neighbours[(left, int(row_left))].add((right, int(row_right)))
    groups = []

    def extend(
        prefix: tuple[tuple[int, int], ...], reachable: set[tuple[int, int]]
    ) -> None:
        for following in sorted(reachable):
            path = (*prefix, following)
            rows = [-1] * view_count
            for camera, row in path:
                rows[camera] = row
            groups.append(rows)
            further = neighbours.get(following)
            if further:
                extend(path, reachable & further)

    for node, reachable in sorted(neighbours.items()):
        extend((node,), reachable)
    return np.asarray(groups, dtype=np.int64).reshape(-1, view_count)


@dataclass(frozen=True)
class _Frame:
    rows: NDArray[np.int64]
    positions: NDArray[np.float32]
    valid: NDArray[np.bool_]
    joint_reasons: NDArray[np.uint8]
    inliers: NDArray[np.bool_]
    errors: NDArray[np.float32]
    error_valid: NDArray[np.bool_]
    scores: NDArray[np.float32]
    assignment: Assignment


def _reconstruct_frame(
    views: tuple[ViewRays, ...],
    cameras: tuple[PinholeCamera, ...],
    source_offsets: NDArray[np.int64],
    codes: tuple[NDArray[np.uint8], ...],
    config: PoseConfig,
    pixel_scale: float,
) -> _Frame:
    pairs: dict[tuple[int, int], PairFits] = {}
    torso = list(COCO17_TORSO_TWIST_JOINTS)
    for left, right in combinations(range(len(views)), 2):
        if not len(views[left].uv) or not len(views[right].uv):
            continue
        pair = compare_rays(
            views[left],
            views[right],
            cameras[left],
            cameras[right],
            config,
            config.reprojection_px * pixel_scale,
        )
        pairs[(left, right)] = pair
        possible = pair.observed[..., torso].sum(axis=-1) >= 3
        separated = pair.angular[..., torso].sum(axis=-1) >= 3
        for camera, axis in ((left, 1), (right, 0)):
            possible_row, separated_row = (
                possible.any(axis=axis),
                separated.any(axis=axis),
            )
            codes[camera][possible_row] = np.maximum(
                codes[camera][possible_row], RowRejection.DEGENERATE_RAYS
            )
            codes[camera][separated_row] = RowRejection.INCONSISTENT_GEOMETRY

    groups = _compatible_groups(pairs, len(views))
    fits = fit_groups(groups, views, cameras, pairs, config, pixel_scale)
    viable = np.flatnonzero(fits.reasons == RowRejection.ASSIGNED)
    result = assign(groups[viable], fits.scores[viable], config.ambiguity_relative_gap)
    selected = viable[result.selected]
    for camera, codes_camera in enumerate(codes):
        present = groups[:, camera] >= 0
        rows = groups[present, camera]
        # An unsuccessful proposal can explain a row only up to the most
        # specific rejection it reached. An accepted group takes precedence.
        why = fits.reasons[present]
        np.maximum.at(
            codes_camera,
            rows,
            np.where(why == RowRejection.ASSIGNED, RowRejection.CONFLICT, why),
        )
        assigned = groups[selected, camera]
        codes_camera[assigned[assigned >= 0]] = RowRejection.ASSIGNED
    rows = groups[selected]
    rows = np.where(rows >= 0, rows + source_offsets, -1)
    return _Frame(
        rows,
        fits.positions_m[selected].astype(np.float32),
        fits.joint_valid[selected],
        fits.joint_reasons[selected],
        fits.inliers[selected],
        fits.errors[selected].astype(np.float32),
        fits.error_valid[selected],
        fits.scores[selected].astype(np.float32),
        result,
    )


def reconstruct(
    inputs: PoseInputs, config: PoseConfig = DEFAULT_CONFIG
) -> FramePoseCandidates:
    source, cameras = inputs.source, inputs.cameras
    rays = tuple(
        view_rays(pose.keypoints, camera, source.size, config)
        for pose, camera in zip(inputs.poses, cameras, strict=True)
    )
    offsets = tuple(detection.frame_offsets for detection in inputs.detections)
    rejection = tuple(
        np.full(axis[-1], RowRejection.INSUFFICIENT_VIEWS, np.uint8) for axis in offsets
    )
    # Clip-local prepared observations are reused; geometry and MILP are frame-local.
    axis_grid = np.stack(offsets)
    results = []
    for frame in range(source.num_frames):
        starts, ends = axis_grid[:, frame : frame + 2].T
        views = tuple(
            view.slice(int(start), int(end))
            for view, start, end in zip(rays, starts, ends, strict=True)
        )
        codes = tuple(
            code[start:end]
            for code, start, end in zip(rejection, starts, ends, strict=True)
        )
        results.append(
            _reconstruct_frame(
                views, cameras, starts, codes, config, source.pixel_threshold_scale
            )
        )

    def collect(field: str) -> NDArray[Any]:
        return np.concatenate([getattr(frame, field) for frame in results], axis=0)

    return FramePoseCandidates(
        source.camera_ids,
        np.cumsum([0, *[len(result.rows) for result in results]], dtype=np.int64),
        collect("rows"),
        collect("positions"),
        collect("valid"),
        collect("joint_reasons"),
        collect("inliers"),
        collect("errors"),
        collect("error_valid"),
        collect("scores"),
        np.concatenate([result.assignment.competing for result in results]),
        np.concatenate([result.assignment.relative_gap for result in results]).astype(
            np.float32
        ),
        np.concatenate([result.assignment.ambiguous for result in results]),
        offsets,
        rejection,
    )
