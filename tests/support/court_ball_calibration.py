"""Independent non-symmetric metric court and moving-ball projections."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.tennis_scene.ball_calibration.court import CourtViewObservations
from src.tennis_scene.pipeline.observation_types import GroupedObservations
from src.utils.geometry.triangulation import PinholeCamera


def calibration_fixture(frames=32):
    size, fps = (1920, 1080), 30.0
    centres = ([-7., -22., 8.], [10., -19., 6.], [2., 24., 9.])
    focal = (900., 1100., 980.)
    ids = ("a_near", "b_right", "c_far")
    offsets = np.array([0., .28, -.37], np.float64)
    # Hand-specified semantic CourtKP14, independently of the implementation.
    court = np.array([
        [-5.485, 11.885, 0], [5.485, 11.885, 0],
        [-5.485, -11.885, 0], [5.485, -11.885, 0],
        [-4.115, 11.885, 0], [-4.115, -11.885, 0],
        [4.115, 11.885, 0], [4.115, -11.885, 0],
        [-4.115, 6.4, 0], [4.115, 6.4, 0],
        [-4.115, -6.4, 0], [4.115, -6.4, 0],
        [0, 6.4, 0], [0, -6.4, 0],
    ])
    truth, perturbed, courts, uv = [], [], [], []
    turns = (False, False, True)
    for v, centre_value in enumerate(centres):
        centre = np.asarray(centre_value)
        forward = -centre / np.linalg.norm(centre)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        r = np.stack((right, down, forward))
        k = np.array([[focal[v], 0, 960], [0, focal[v], 540], [0, 0, 1.]])

        def project(xyz, centre=centre, r=r, f=focal[v]):
            camera = (xyz - centre) @ r.T
            return (camera[:, :2] / camera[:, 2:]) * f + [960, 540]

        camera = PinholeCamera(ids[v], k, r, -r @ centre)
        truth.append(camera)
        kp_xyz = court * ([-1, -1, 1] if turns[v] else [1, 1, 1])
        kp = project(kp_xyz)[None]
        selected: NDArray[np.bool_] = np.ones((1, 14), bool)
        selected[0, [4 + v, 9 + v]] = False
        courts.append(CourtViewObservations(
            ids[v], size, frames, np.array([0], np.int64), kp,
            np.full((1, 14), .9), np.ones((1, 14), bool), selected,
            "synthetic_camera_view_kp14_fixture", ("x_reflected",),
            None, "no line inference in synthetic fixture",
        ))
        t = (np.arange(frames) + offsets[v]) / fps
        xyz = np.column_stack((2 * np.sin(t * 2), -4 + 8*t + t*t, 1.2 + 2*t - t*t))
        uv.append(project(xyz))
        kk = k.copy()
        kk[:2, :2] *= 1 + [.017, -.023, .012][v]
        rr = Rotation.from_rotvec(np.array([.003, -.004, .002]) * (v + 1) / 2).as_matrix() @ r
        cc = centre + np.array([.06, -.08, .04]) * (v + 1) / 2
        perturbed.append(PinholeCamera(ids[v], kk, rr, -rr @ cc))
    uv = np.array(uv, np.float32)[None, :, :, None]
    obs = GroupedObservations(
        np.array([0], np.int64), uv, np.ones(uv.shape[:-1], np.float32),
        np.ones(uv.shape[:-1], bool), np.zeros((1, 3, frames), np.int64),
    )
    return obs, tuple(perturbed), tuple(courts), turns, tuple(truth), offsets, fps


def destroyed_holdout(courts):
    result = []
    for court in courts:
        uv = court.keypoints_px.copy()
        uv[~court.selected_for_homography] += [77., 31.]
        result.append(replace(court, keypoints_px=uv))
    return tuple(result)


def calibration_method_inputs(frames=32):
    from src.tennis_scene.evaluation.ball3d.contracts import MethodInput, ViewInput
    from src.tennis_scene.pipeline.components.ball_points import (
        BallPointsInput,
        BallPointsModule,
    )
    from src.tennis_scene.pipeline.components.camera_geometry import (
        LocalCourtCalibration,
    )
    from src.utils.geometry.temporal_smoothing import TemporalSmoothingConfig
    from tests.support.ball3d_evaluation import make_inputs

    obs, cameras, courts, turns, truth, offsets, fps = calibration_fixture(frames)
    template = make_inputs(frames)
    size = (1920, 1080)
    views = []
    smoothing = TemporalSmoothingConfig(enabled=False)
    for v, row in enumerate(template.views):
        camera = cameras[v]
        evidence = replace(row.detections.evidence, source_size_wh=size,
            candidate_uv_px=(row.detections.evidence.candidate_uv_px / [319, 239] * [1919, 1079]).astype(np.float32))
        detection = replace(row.detections, camera_id=camera.camera_id,
                            uv_px=obs.uv_px[0, v, :, 0].copy(),
                            observed=np.ones(frames, bool),
                            confidence=np.ones(frames, np.float32),
                            point_kind=np.ones(frames, np.uint8), evidence=evidence)
        points = BallPointsModule(smoothing).process(BallPointsInput(detection, size, fps))
        views.append(ViewInput(
            camera, LocalCourtCalibration(camera.half_turned(turns[v]), v, 0, np.eye(3), 0.),
            turns[v], detection, points, row.detector_config, smoothing, courts[v],
        ))
    return MethodInput("synthetic_calibration", np.arange(frames, dtype=np.int64), fps, size, tuple(views))


def main_regression_inputs():
    from src.tennis_scene.pipeline.components.ball_points import (
        BallPointsInput,
        BallPointsModule,
    )

    inputs = calibration_method_inputs(frames=12)
    rows = list(inputs.views)
    observed = rows[0].detections.observed.copy()
    observed[3] = False
    detection = replace(rows[0].detections, observed=observed,
        uv_px=np.where(observed[:, None], rows[0].detections.uv_px, 0).astype(np.float32),
        confidence=observed.astype(np.float32), point_kind=observed.astype(np.uint8))
    points = BallPointsModule(rows[0].smoothing_config).process(
        BallPointsInput(detection, inputs.source_size_wh, inputs.fps))
    rows[0] = replace(rows[0], detections=detection, points=points)
    mask = rows[1].detections.observed.copy()
    mask[3] = False
    uv = rows[1].detections.uv_px.copy()
    uv[6] += [45., -13.]
    detection = replace(rows[1].detections, observed=mask, uv_px=np.where(mask[:, None], uv, 0).astype(np.float32),
        confidence=mask.astype(np.float32), point_kind=mask.astype(np.uint8))
    points = BallPointsModule(rows[1].smoothing_config).process(
        BallPointsInput(detection, inputs.source_size_wh, inputs.fps))
    rows[1] = replace(rows[1], detections=detection, points=points)
    return replace(inputs, views=tuple(rows))


def frozen_main_ball(inputs) -> Any:
    """Execute checksum-pinned main kernels on this backend for exact regression.

    Golden float arrays from another NumPy/SciPy build can differ by an ULP.
    The reference is historical source, independent of the current kernels;
    both executions use this machine's numerical libraries without tolerance.
    """
    import ast
    import hashlib
    import json
    import sys
    import types

    root = Path(__file__).parent / "fixtures"
    manifest = json.loads((root / "court_ball_main.json").read_text())

    def read(name):
        record = manifest["reference_sources"][name]
        data = (root / record["snapshot"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Historical main reference source checksum mismatch")
        return ast.parse(data.decode(), filename=record["source"])

    geometry = types.ModuleType("_court_ball_frozen_main_geometry")
    sys.modules[geometry.__name__] = geometry
    exec(compile(read("geometry"), "frozen_main_geometry", "exec"), geometry.__dict__)
    reconstruction = types.ModuleType("_court_ball_frozen_main_reconstruction")
    sys.modules[reconstruction.__name__] = reconstruction
    tree = read("reconstruction")
    # Bind the original source imports to its original geometric kernel.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "src.utils.geometry.triangulation")]
    for name in ("PinholeCamera", "PointTriangulationConfig", "TriangulatedPoints",
                 "reject_excessive_speed", "triangulate_points"):
        reconstruction.__dict__[name] = getattr(geometry, name)
    exec(compile(tree, "frozen_main_reconstruction", "exec"), reconstruction.__dict__)
    # Independent source-frame single-ball grouping, before the selectable stage.
    uv = np.stack([v.points.uv_px for v in inputs.views])
    confidence = np.stack([v.points.confidence for v in inputs.views])
    visible = np.stack([v.points.observed for v in inputs.views])
    observations = GroupedObservations(
        np.array([0], np.int64), uv[None, :, :, None], confidence[None, :, :, None],
        visible[None, :, :, None], np.where(visible[None], 0, -1).astype(np.int64))
    cameras = tuple(geometry.PinholeCamera(v.camera.camera_id, v.camera.intrinsic.copy(),
        v.camera.rotation.copy(), v.camera.translation.copy()) for v in inputs.views)
    return reconstruction.reconstruct_ball(observations, cameras, fps=inputs.fps,
        reprojection_px=20., min_frames=5)


def calibration_pipeline_artifacts(inputs, directory):
    from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
    from src.tasks.base.generate_dataset import (
        build_court_view_record,
        resolve_court_keypoint_contract,
    )
    from src.tennis_scene.pipeline.components.camera_alignment import (
        CameraAlignmentOutput,
    )
    from src.tennis_scene.pipeline.components.camera_geometry import (
        CalibrationSet,
        CameraGeometryResult,
    )
    from src.tennis_scene.pipeline.components.court_calibration import (
        CourtCalibrationOutput,
    )
    from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
    from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo

    ids = inputs.camera_ids
    source = ClipSource(inputs.clip_id, tuple(
        SourceVideo(c, directory / f"{c}.mp4", str(v + 1) * 64, inputs.num_frames, inputs.fps, *inputs.source_size_wh)
        for v, c in enumerate(ids)))
    contract = resolve_court_keypoint_contract("camera_view_v2")
    records = tuple(build_court_view_record(camera_id=view.camera.camera_id,
        camera_center_court_m=view.camera.center.tolist(), contract=contract) for view in inputs.views)
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=StableCameraIdTable.from_complete_scene_camera_ids(ids),
        selected_views=records, reference_camera_id=ids[0])
    geometry = CameraGeometryResult(ids, ids[0], tuple(v.half_turn for v in inputs.views),
        tuple(v.camera for v in inputs.views), selection, {})
    calibration = CourtCalibrationOutput(
        CourtKPResult(np.zeros((3, inputs.num_frames, 14, 2), np.float32),
            np.ones((3, inputs.num_frames, 14), bool), np.arange(inputs.num_frames, dtype=np.int32),
            {"output_keypoint_contract": "camera_view_v2", "temporal_policy": "static_first_frame_broadcast",
             "observed_frame_indices": [0], "source_frame_count": inputs.num_frames,
             "cameras": [{"camera_id": v.camera.camera_id, "frames": [{
                 "frame_index": 0, "raw_keypoints_px": v.court.keypoints_px[0].tolist(),
                 "raw_scores": v.court.scores[0].tolist(), "raw_valid": v.court.valid[0].tolist(),
                 "selected_keypoints": np.flatnonzero(v.court.selected_for_homography[0]).tolist(),
                 "keypoint_schema": v.court.keypoint_schema, "kp_only_prosac_orientation": "x_reflected",
                 "original_size_hw": list(reversed(inputs.source_size_wh)), "output_keypoint_contract": "camera_view_v2",
             }]} for v in inputs.views]}),
        CalibrationSet(tuple(v.local_calibration for v in inputs.views), {}), ids[0], {})
    artifacts = {"calibration": calibration, "alignment": CameraAlignmentOutput(geometry),
                 **{f"ball_{v.camera.camera_id}": v.points for v in inputs.views}}
    return source, artifacts
