"""Deterministic regulation geometry and measured manual-alignment diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentResult,
    CandidateAlignment,
    MetricSceneAdapter,
    PartitionAssessment,
)
from src.synthetic_data_generation.alignment.evaluation import metrics_from_residuals
from src.synthetic_data_generation.alignment.heatmaps import AlignmentLineHeatmaps
from src.synthetic_data_generation.alignment.manual.models import LayoutEdit
from src.synthetic_data_generation.alignment.manual.source import (
    ManualSource,
    validate_source_heatmaps,
)
from src.synthetic_data_generation.alignment.whole_court import COURT_LINE_SEGMENTS
from src.synthetic_data_generation.scene_contract import (
    MultiCourtLayout,
    RigidTransform,
)


def court_segments() -> list[list[list[float]]]:
    """Share the canonical regulation line template with the browser."""
    return [[list(segment.start), list(segment.end)] for segment in COURT_LINE_SEGMENTS]


def build_manual_result(
    source: ManualSource, heatmaps: AlignmentLineHeatmaps, edit: LayoutEdit
) -> AlignmentResult:
    """Keep actual measured scores while making human confirmation authoritative."""
    if not edit.courts:
        raise ValueError("At least one court is required for formal application.")
    validate_source_heatmaps(source, heatmaps)
    initial = source.initial
    plane = source.plane
    basis = np.column_stack(
        (
            plane.basis_u_metric_scene,
            plane.basis_v_metric_scene,
            plane.normal_metric_scene,
        )
    )
    origin = np.asarray(plane.origin_metric_scene)
    forward = initial.metric_adapter.nht_matrix().copy()
    forward[:3, :3] *= edit.scale
    adapter = MetricSceneAdapter.from_nht_scene_from_metric_scene(forward)
    template = np.concatenate(
        [np.linspace(segment.start, segment.end, 64) for segment in COURT_LINE_SEGMENTS]
    )
    trees = {}
    observed_ids = {}
    for name, ids in (
        ("fit", initial.partitions.fit_camera_ids),
        ("holdout", initial.partitions.holdout_camera_ids),
    ):
        selected = [
            view
            for view in heatmaps.views
            if view.camera_id in ids and len(view.points_uv)
        ]
        if not selected:
            raise ValueError(f"No measured projected evidence for {name} diagnostics.")
        trees[name] = cKDTree(np.concatenate([view.points_uv for view in selected]))
        observed_ids[name] = tuple(
            camera_id
            for camera_id in ids
            if any(view.camera_id == camera_id for view in selected)
        )
    candidates = []
    bound_points = [
        np.asarray(initial.layout.complex_bounds_scene).reshape(2, 3) / edit.scale
    ]
    for placement in edit.courts:
        angle = np.radians(placement.angle_degrees)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        matrix = np.eye(4)
        matrix[:3, :3] = basis @ rotation
        matrix[:3, 3] = (
            origin + basis @ np.asarray([placement.u, placement.v, 0])
        ) / edit.scale
        transform = RigidTransform.from_matrix(matrix)
        predicted_uv = template @ rotation[:2, :2].T * edit.scale + [
            placement.u,
            placement.v,
        ]
        assessments = {}
        for name in ("fit", "holdout"):
            distances, _ = trees[name].query(predicted_uv, workers=1)
            thresholds = getattr(initial.policy, name)
            metrics = metrics_from_residuals(
                distances / edit.scale,
                camera_ids=observed_ids[name],
                inlier_distance_m=thresholds.inlier_distance_m,
            )
            assessments[name] = PartitionAssessment.evaluate(metrics, thresholds)
        candidate = CandidateAlignment(
            court_instance_id=placement.court_id,
            candidate_id=f"manual-{placement.court_id}",
            scene_from_court=transform,
            court_from_scene=transform.inverse(),
            fit=assessments["fit"],
            holdout=assessments["holdout"],
            human_confirmed=True,
        )
        candidates.append(candidate)
        bound_points.append(
            transform.apply(np.column_stack((template, np.zeros(len(template)))))
        )
    all_points = np.concatenate(bound_points)
    lower, upper = all_points.min(axis=0) - 0.01, all_points.max(axis=0) + 0.01
    layout = MultiCourtLayout(
        courts=tuple(candidate.to_court_instance() for candidate in candidates),
        complex_bounds_scene=tuple(np.concatenate((lower, upper))),
        primary_court_instance_id=edit.primary_court_id,
    )
    return AlignmentResult(
        partitions=initial.partitions,
        policy=initial.policy,
        candidates=tuple(candidates),
        layout=layout,
        metric_adapter=adapter,
    )
