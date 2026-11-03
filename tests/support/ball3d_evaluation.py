"""Small asymmetric synthetic contracts; these values are never real-clip GT."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.inference.trajectory_gate import TrajectoryGateConfig
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.court_side.hypothesis import HypothesisScore
from src.tennis_scene.evaluation.ball3d.annotations import Ball2DAnnotations
from src.tennis_scene.evaluation.ball3d.baseline import PipelineBaseline
from src.tennis_scene.evaluation.ball3d.bundle import FixedInputBundle
from src.tennis_scene.evaluation.ball3d.contracts import (
    DetectorReplayConfig,
    MethodInput,
    ViewInput,
)
from src.tennis_scene.evaluation.ball3d.court import (
    CourtLineObservations,
    CourtViewObservations,
)
from src.tennis_scene.ground_truth import (
    ReferenceIdentity,
    ReferencePosition,
    UserReference,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_evidence import BallHeatmapEvidence
from src.tennis_scene.pipeline.components.ball_points import (
    BallPointsInput,
    BallPointsModule,
)
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.camera_geometry import (
    CalibrationSet,
    CameraGeometryResult,
    LocalCourtCalibration,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.components.triangulation import BallTriangulationOutput
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.geometry.temporal_smoothing import TemporalSmoothingConfig
from src.utils.geometry.triangulation import PinholeCamera, TriangulatedPoints


def make_scoring_bundle(inputs, digest="x"):
    """Caller-owned synthetic source identity for pure scoring contract tests."""
    source = ClipSource(
        "synthetic_source_id",
        tuple(
            SourceVideo(
                camera,
                Path(f"{camera}.mp4"),
                str(i + 1) * 64,
                inputs.num_frames,
                inputs.fps,
                *inputs.source_size_wh,
            )
            for i, camera in enumerate(inputs.camera_ids)
        ),
    )
    return FixedInputBundle(
        inputs, BallTriangulationOutput(None), source, digest, {}, {}
    )


def make_court(camera_id, index, size, frames):
    raw = np.column_stack(
        (np.arange(14) + 21.25 + index * 47, np.arange(14) + 50.5 + index * 29)
    ).astype(np.float64)[None]
    valid: NDArray[np.bool_] = np.ones((1, 14), bool)
    valid[0, index + 7] = False
    selected: NDArray[np.bool_] = np.zeros((1, 14), bool)
    selected[0, [0, index + 2, 13]] = True
    lines = CourtLineObservations(
        np.zeros(2, np.int64),
        np.array([[20.0 + index, 70.0], [25.0, 80.0 + index]]),
        np.array([0.8 - index / 10, 0.5]),
        np.array([True, False]),
    )
    return CourtViewObservations(
        camera_id,
        size,
        frames,
        np.array([0], np.int64),
        raw,
        np.linspace(0.2, 0.9, 14)[None] + index * 0.02,
        valid,
        selected,
        "synthetic_camera_view_kp14_fixture",
        ("x_reflected",),
        lines,
        None,
    )


def court_diagnostics(inputs):
    return {
        "output_keypoint_contract": "camera_view_v2",
        "temporal_policy": "static_first_frame_broadcast",
        "observed_frame_indices": [0],
        "source_frame_count": inputs.num_frames,
        "cameras": [
            {
                "camera_id": view.camera.camera_id,
                "frames": [
                    {
                        "frame_index": 0,
                        "raw_keypoints_px": view.court.keypoints_px[0].tolist(),
                        "raw_scores": view.court.scores[0].tolist(),
                        "raw_valid": view.court.valid[0].tolist(),
                        "selected_keypoints": np.flatnonzero(
                            view.court.selected_for_homography[0]
                        ).tolist(),
                        "keypoint_schema": view.court.keypoint_schema,
                        "kp_only_prosac_orientation": view.court.template_orientations[
                            0
                        ],
                        "original_size_hw": list(reversed(inputs.source_size_wh)),
                        "output_keypoint_contract": "camera_view_v2",
                        "raw_line_observations": {
                            "uv_px": view.court.require_lines().uv_px.tolist(),
                            "scores": view.court.require_lines().scores.tolist(),
                            "valid": view.court.require_lines().valid.tolist(),
                        },
                    }
                ],
            }
            for view in inputs.views
        ],
    }


def make_annotations(inputs, uv, kinds, *, events=(), estimation_segments=()):
    """Hand-written typed reference for scoring tests, never real-clip labels."""
    identity = ReferenceIdentity(
        inputs.clip_id,
        inputs.camera_ids,
        inputs.num_frames,
        inputs.fps,
        inputs.source_size_wh,
        {c: str(i + 1) * 64 for i, c in enumerate(inputs.camera_ids)},
    )
    positions = {}
    for i, camera in enumerate(inputs.camera_ids):
        for frame in range(inputs.num_frames):
            kind = str(kinds[i, frame])
            valid = kind in {"observed", "estimated"}
            positions[camera, frame] = ReferencePosition(
                camera,
                frame,
                kind if kind in {"observed", "occluded"} else "unknown",
                kind,
                valid,
                tuple(uv[i, frame]) if valid else None,
                "synthetic.csv",
                {"method": "hand_calculation"} if kind == "estimated" else None,
            )
    return Ball2DAnnotations(
        UserReference(
            identity,
            "f" * 64,
            positions,
            tuple(events),
            tuple(estimation_segments),
            {"provenance": {"kind": "synthetic_test_fixture"}},
        )
    )


def make_inputs(frames: int = 4) -> MethodInput:
    ids, size = ("right", "near", "far"), (320, 240)
    centers = [(-2.0, -4.0, 8.0), (3.0, -2.0, 10.0), (1.0, 5.0, 9.0)]
    focal = [(120.0, 130.0), (170.0, 145.0), (155.0, 180.0)]
    rotation = np.diag([1.0, -1.0, -1.0])
    xyz = np.column_stack(
        (
            1 + np.arange(frames) / 10,
            2 - np.arange(frames) / 5,
            1 + np.arange(frames) / 20,
        )
    )
    views = []
    for i, camera_id in enumerate(ids):
        fx, fy = focal[i]
        center = np.array(centers[i])
        camera = PinholeCamera(
            camera_id,
            np.array([[fx, 0.0, 160.0], [0.0, fy, 120.0], [0.0, 0.0, 1.0]]),
            rotation,
            -rotation @ center,
        )
        # Independent axis-aligned pinhole formulas, rather than Camera.project.
        depth = center[2] - xyz[:, 2]
        uv = np.column_stack(
            (
                fx * (xyz[:, 0] - center[0]) / depth + 160,
                fy * (center[1] - xyz[:, 1]) / depth + 120,
            )
        ).astype(np.float32)
        observed: NDArray[np.bool_] = np.ones(frames, bool)
        observed[(i + 1) % frames] = False
        uv[~observed] = 0
        maps: NDArray[np.float32] = np.full((frames, 5, 7), 0.01, np.float32)
        cells = np.tile(np.array([i + 1, 2], np.int64), (frames, 1, 1))
        maps[:, 2, i + 1] = 0.7 + i / 10
        patches = maps[:, 1:4, i : i + 3][:, None].copy()
        evidence = BallHeatmapEvidence(
            size,
            BallCandidateConfig(max_candidates=1, nms_kernel=3, patch_size=3),
            maps,
            (cells / [6, 4] * [319, 239]).astype(np.float32),
            np.full((frames, 1), 0.7 + i / 10, np.float32),
            np.ones((frames, 1), bool),
            cells,
            patches,
            np.ones((frames, 1, 3, 3), bool),
            np.arange(frames, dtype=np.int64),
            np.zeros(frames, np.int64),
        )
        detections = BallDetectionOutput(
            camera_id,
            np.arange(frames, dtype=np.int64),
            uv,
            np.where(observed, 0.8 + 0.05 * i, 0).astype(np.float32),
            observed,
            observed.astype(np.uint8),
            "model_score",
            evidence,
        )
        config = TemporalSmoothingConfig()
        points = BallPointsModule(config).process(
            BallPointsInput(detections, size, 10.0)
        )
        half_turn = i == 2
        local = LocalCourtCalibration(
            camera.half_turned(half_turn), i, 0, np.eye(3), float(i + 1) / 10
        )
        views.append(
            ViewInput(
                camera,
                local,
                half_turn,
                detections,
                points,
                DetectorReplayConfig(
                    0.5, False, TrajectoryGateConfig(False, 60.0, 2, 5, 2)
                ),
                config,
                make_court(camera_id, i, size, frames),
            )
        )
    return MethodInput(
        "synthetic_contract_fixture",
        np.arange(frames, dtype=np.int64),
        10.0,
        size,
        tuple(views),
    )


def make_store(
    root: Path,
    inputs: MethodInput,
    *,
    detector_origin: str = "component",
    court_diagnostics_override=None,
) -> ClipStore:
    source = ClipSource(
        "historical_fixture_id",
        tuple(
            SourceVideo(
                camera,
                Path(f"{camera}.mp4"),
                str(i + 1) * 64,
                inputs.num_frames,
                inputs.fps,
                *inputs.source_size_wh,
            )
            for i, camera in enumerate(inputs.camera_ids)
        ),
    )
    store = ClipStore(root, json_value(source))
    refs = {}

    def publish(name, value, schema, version=1, deps=None, settings=None):
        deps = deps or {}
        identity = {"dependencies": json_value(deps), "settings": settings or {}}
        refs[name] = store.publish(
            name,
            value,
            ArtifactCodec(type(value)),
            schema=schema,
            version=version,
            identity=identity,
            dependencies=deps,
            provenance={
                "origin": detector_origin
                if name.startswith("ball_detection/")
                else "component",
                "fixture": True,
            },
        )
        return refs[name]

    for view in inputs.views:
        camera = view.camera.camera_id
        detection = publish(
            f"ball_detection/{camera}",
            view.detections,
            "ball_detections",
            2,
            settings={"config": asdict(view.detector_config)},
        )
        publish(
            f"ball_points/{camera}",
            view.points,
            "ball_points",
            4,
            {"detections": detection},
            {"smoothing": asdict(view.smoothing_config)},
        )
    calibration = CourtCalibrationOutput(
        CourtKPResult(
            np.zeros((3, inputs.num_frames, 14, 2), np.float32),
            np.ones((3, inputs.num_frames, 14), bool),
            np.arange(inputs.num_frames, dtype=np.int32),
            court_diagnostics(inputs)
            if court_diagnostics_override is None
            else court_diagnostics_override,
        ),
        CalibrationSet(tuple(v.local_calibration for v in inputs.views), {}),
        "right",
        {},
    )
    cal = publish("court_calibration", calibration, "local_court_calibration")
    turns = (False, False, True)
    hypotheses = tuple(
        HypothesisScore(t, 0.1 + i / 10, 0.9 - i / 10, inputs.num_frames)
        for i, t in enumerate(
            [turns, (False, True, False), (False, False, False), (False, True, True)]
        )
    )
    side = publish(
        "court_side",
        CourtSideOutput(
            inputs.camera_ids, "right", turns, hypotheses, 0.1, inputs.num_frames
        ),
        "court_side",
        3,
        {"calibration": cal},
    )
    contract = resolve_court_keypoint_contract("camera_view_v2")
    records = tuple(
        build_court_view_record(
            camera_id=v.camera.camera_id,
            camera_center_court_m=v.camera.center.tolist(),
            contract=contract,
        )
        for v in inputs.views
    )
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=StableCameraIdTable.from_complete_scene_camera_ids(
            inputs.camera_ids
        ),
        selected_views=records,
        reference_camera_id="right",
    )
    geometry = CameraGeometryResult(
        inputs.camera_ids,
        "right",
        turns,
        tuple(v.camera for v in inputs.views),
        selection,
        {},
    )
    alignment = publish(
        "camera_alignment",
        CameraAlignmentOutput(geometry),
        "aligned_cameras",
        deps={"calibration": cal, "side": side},
    )
    result = PipelineBaseline(min_frames=1)(inputs)
    assert result.diagnostics is not None
    trajectory = TriangulatedPoints(
        result.positions,
        result.valid,
        result.diagnostics["rejection_codes"],
        result.used_views,
        np.zeros((3, inputs.num_frames), np.float32),
    )
    ball = BallReconstructionResult(
        np.stack([v.points.uv_px for v in inputs.views]),
        np.stack([v.points.observed for v in inputs.views]),
        trajectory,
        "ok",
    )
    publish(
        "ball_triangulation",
        BallTriangulationOutput(ball),
        "ball_trajectory",
        deps={
            "alignment": alignment,
            **{c: refs[f"ball_points/{c}"] for c in inputs.camera_ids},
        },
        settings={
            "config": {"enabled": True, "min_frames": 1, "reprojection_px": 20.0},
            "threshold": 0.0,
        },
    )
    return store
