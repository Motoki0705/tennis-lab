"""Court sides confirmed geometrically against an imported, human-reviewed ball.

Every half-turn hypothesis (reference camera fixed) is scored with the same
geometric test that ``camera_alignment`` applies, using only the ball
observations bound to the ``court_side`` node. This is a confirmation of the
reviewed ball, not a side model: the artifact records ``model_inference=False``.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np

from src.tennis_scene.pipeline.components.ball_reconstruction import (
    single_ball_observations,
)
from src.tennis_scene.pipeline.components.camera_geometry import (
    CameraGeometryConfig,
    SideEvidence,
    resolve_camera_geometry,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.identity import COURT_SIDE, CourtSideOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
from src.tennis_scene.pipeline.imports.publish import bind_import, publish_import
from src.tennis_scene.pipeline.input_assembly.observations import gather_balls
from src.tennis_scene.pipeline.runner import ComponentNode
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore

IMPORTER = "ball_confirmed_court_side"
IMPORTER_VERSION = 1


def confirm_sides_with_ball(source: ClipSource, calibration: CourtCalibrationOutput, balls: dict[str, Any], *,
                            ball_threshold: float, ball_reprojection_px: float, max_frames: int,
                            config: CameraGeometryConfig) -> tuple[tuple[bool, ...], dict[str, Any]]:
    """The unique half-turn assignment the ball supports, or ``ReconstructionUnavailable``.

    ``balls`` maps ``ball_<camera>`` to every source camera's ball artifact.
    Thresholds are the pipeline's own (scaled by ``pixel_threshold_scale``).
    """
    active = tuple(view.source_index for view in calibration.calibration.views)
    grouped = single_ball_observations(gather_balls(source, balls).select_views(active), threshold=ball_threshold)
    sample = sampled_frame_indices(source.num_frames, source.fps, max_frames=max_frames)
    evidence = SideEvidence("ball", grouped.uv_px[:, :, sample], grouped.visibility[:, :, sample],
                            ball_reprojection_px * source.pixel_threshold_scale)
    ids = calibration.calibration.camera_ids
    reference = ids.index(calibration.reference_camera)
    choices = [(False,) if view == reference else (False, True) for view in range(len(ids))]
    hypotheses = tuple(np.asarray(turns, bool) for turns in product(*choices))
    geometry = resolve_camera_geometry(calibration.calibration, calibration.reference_camera, hypotheses, (evidence,), config=config)
    confirmation = {"method": "all_half_turn_hypotheses_against_the_bound_ball_observations",
        "reference_camera": calibration.reference_camera, "camera_ids": list(ids),
        "view_half_turns": list(geometry.view_half_turns), "candidates": geometry.document["side_candidates"],
        "sampled_frames": len(sample), "independent_model_accuracy_evaluation": False}
    return geometry.view_half_turns, confirmation


def import_ball_confirmed_sides(nodes: Sequence[ComponentNode], store: ClipStore, source: ClipSource, *,
                                ball_threshold: float, ball_reprojection_px: float, max_frames: int,
                                config: CameraGeometryConfig) -> tuple[ArtifactRef, dict[str, Any]]:
    """Confirm and publish the load-only ``court_side`` node from its bound artifacts."""
    bound = bind_import(nodes, COURT_SIDE, store)
    calibration: CourtCalibrationOutput = bound.artifacts["calibration"]
    balls = {port: value for port, value in bound.artifacts.items() if port.startswith("ball_")}
    half_turns, confirmation = confirm_sides_with_ball(source, calibration, balls, ball_threshold=ball_threshold,
        ball_reprojection_px=ball_reprojection_px, max_frames=max_frames, config=config)
    value = CourtSideOutput(calibration.calibration.camera_ids, calibration.reference_camera, half_turns)
    reference = publish_import(bound, store, value, importer=IMPORTER, version=IMPORTER_VERSION,
        identity={"half_turns": list(half_turns), "ball_threshold": ball_threshold,
                  "ball_reprojection_px": ball_reprojection_px, "max_frames": max_frames, "geometry": config},
        provenance={"model_inference": False, "confirmation": confirmation})
    return reference, confirmation
