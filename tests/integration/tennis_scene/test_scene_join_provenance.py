"""Non-neural four-camera observations -> excluded view -> grouped people -> scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_points import (
    BallPointsInput,
    BallPointsModule,
)
from src.tennis_scene.pipeline.components.body_placement import BodyPlacementOutput
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.camera_geometry import (
    CameraGeometryConfig,
    SideEvidence,
    resolve_camera_geometry,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationInput,
    CourtCalibrationModule,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.components.identity import PlayerIdentitiesOutput
from src.tennis_scene.pipeline.components.scene_assembly import SceneAssemblyModule
from src.tennis_scene.pipeline.components.triangulation import (
    BallTriangulationModule,
    PlayerTriangulationModule,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.input_assembly.observations import (
    gather_people,
    identified_people,
)
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    BallTriangulationInputAssembler,
    PlayerTriangulationInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.scene import SceneAssemblyInputAssembler
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.utils.geometry.temporal_smoothing import TemporalSmoothingConfig
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.schema.court import CourtConfig, court_keypoints_3d

CAMERAS = ("west", "blocked", "east", "high")
ACTIVE = (0, 2, 3)
TRACKS = ((41,), (301, 70, 4, 99), (88, 6), (9, 105, 52))
FRAMES = 4
SIZE = (1280, 720)
# Ground truth by source camera, carrier, frame. Local IDs deliberately do not sort.
PEOPLE = (((17, 17, 5, 5),), ((5,) * 4,) * 4,
          ((5, 5, 17, 17), (17, 17, 5, 5)),
          ((17, 17, 5, 5), (31, 31, 31, 31), (5, 5, 17, 17)))


def camera(name: str, center: tuple[float, float, float], focal: float) -> PinholeCamera:
    c = np.array(center)
    forward = np.array([0., 0., 1.]) - c
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    return PinholeCamera(name, np.array([[focal, 0., 640.], [0., focal, 360.], [0., 0., 1.]]), rotation, -rotation @ c)


def project(model: PinholeCamera, xyz: np.ndarray) -> NDArray[np.float32]:
    """Independent pinhole equation; avoid the production projection/join helpers."""
    q = np.einsum("ij,...j->...i", model.rotation, xyz - model.center)
    return np.asarray(q[..., :2] / q[..., 2, None] * [model.intrinsic[0, 0], model.intrinsic[1, 1]] + [640., 360.], dtype=np.float32)


def test_excluded_middle_camera_preserves_observations_id_switch_and_world_points() -> None:
    source = ClipSource("join", tuple(SourceVideo(c, Path(f"{c}.mp4"), "synthetic", FRAMES, 30., *SIZE) for c in CAMERAS))
    models = (camera("west", (-8., -16., 9.), 900.), camera("blocked", (2., -22., 11.), 920.),
              camera("east", (8., -15., 10.), 940.), camera("high", (-2., -20., 14.), 880.))
    template = court_keypoints_3d(CourtConfig(.914, None)).numpy()[:14]
    courts = []
    for v, model in enumerate(models):
        pixels = project(model, template)
        homography = model.intrinsic @ np.column_stack((model.rotation[:, :2], model.translation))
        courts.append(CourtKPResult((pixels / [1279, 719]).astype(np.float32)[None, None],
            ((pixels >= 0) & (pixels < SIZE)).all(-1).astype(np.float32)[None, None], np.array([0], np.int32),
            {"output_keypoint_contract": "camera_view_v2", "temporal_policy": "static_first_frame", "cameras": [{"frames": [{
                "frame_index": 0, "status": "joint_optimization_failed" if v == 1 else "ok",
                "homography_court_metres_to_image_pixels": None if v == 1 else homography.tolist(),
            }]}]}))
    config = CameraGeometryConfig(reference_camera="west", side_min_frames=1)
    calibration = CourtCalibrationModule(CAMERAS, config, roi_margins=(0., 0.)).process(CourtCalibrationInput(source, tuple(courts)))
    assert tuple(view.source_index for view in calibration.calibration.views) == ACTIVE
    assert calibration.calibration.excluded == {"blocked": "no_accepted_homography"}

    # Known world points identify player, frame and joint independently of carrier order.
    xyz: NDArray[np.float32] = np.zeros((3, FRAMES, 17, 3), np.float32)
    for p, t, j in np.ndindex(3, FRAMES, 17):
        xyz[p, t, j] = [-1.2 + 1.1 * p + .02 * t + .015 * j, -4. + 3. * p + .06 * t - .01 * j, .5 + .07 * j]
    rows = []
    for v, model in enumerate(models):
        count = len(TRACKS[v])
        uv: NDArray[np.float32] = np.empty((1, FRAMES, count, 17, 2), np.float32)
        confidence = np.empty(uv.shape[:-1], np.float32)
        observed: NDArray[np.bool_] = np.ones((1, FRAMES, count), bool)
        for d, player_ids in enumerate(PEOPLE[v]):
            for t, identity in enumerate(player_ids):
                uv[0, t, d] = project(model, xyz[(5, 17, 31).index(identity), t])
                confidence[0, t, d] = .5 + .04 * v + .02 * d + .005 * t + .001 * np.arange(17)
                if identity == 17 and t == 1:
                    observed[0, t, d] = False
        rows.append(ObjectObservations((CAMERAS[v],), SIZE, 30., uv, confidence, observed,
            np.array([TRACKS[v]], np.int64), np.full((1, FRAMES, count, 3), 50. + v, np.float32)))
    rows[2].observed[0, 3, 1] = False
    rows[3].confidence[0, 0, 2, 0] = .1
    rows[0].uv_px[0, 0, 0, 4, 0] = 1280  # invisible joint; other cameras still support it
    artifacts: dict[str, Any] = {f"pose_{c}": row for c, row in zip(CAMERAS, rows, strict=True)}
    raw = gather_people(source, artifacts)
    np.testing.assert_array_equal(raw.local_track_ids, [[41, -1, -1, -1], [301, 70, 4, 99], [88, 6, -1, -1], [9, 105, 52, -1]])
    for v, row in enumerate(rows):
        count = len(TRACKS[v])
        for name in ("uv_px", "confidence", "observed", "boxes_xys"):
            np.testing.assert_array_equal(getattr(raw, name)[v, :, :count], getattr(row, name)[0])
            np.testing.assert_array_equal(getattr(raw, name)[v, :, count:], 0)
    local = np.array([[41, -1, -1, -1], [88, 6, -1, -1], [9, 105, 52, -1]], np.int64)
    ids: NDArray[np.int64] = np.full((3, 4, FRAMES), -1, np.int64)
    for v, original in enumerate(ACTIVE):
        ids[v, :len(TRACKS[original])] = PEOPLE[original]
    identities = PlayerIdentitiesOutput(("west", "east", "high"), local, ids, {})
    grouped = identified_people(raw.select_views(ACTIVE), identities, .3)
    np.testing.assert_array_equal(grouped.identities, [5, 17, 31])
    expected_raw = np.array([
        [[-1, -1, 0, 0], [0, 0, 1, -1], [2, 2, 0, 0]],
        [[0, -1, -1, -1], [1, -1, 0, 0], [0, -1, 2, 2]],
        [[-1, -1, -1, -1], [-1, -1, -1, -1], [1, 1, 1, 1]],
    ], np.int64)
    np.testing.assert_array_equal(grouped.raw_indices, expected_raw)
    expected_uv: NDArray[np.float32] = np.zeros((3, 4, FRAMES, 17, 2), np.float32)
    expected_confidence = np.zeros(expected_uv.shape[:-1], np.float32)
    for p, v, t, j in np.ndindex(3, 3, FRAMES, 17):
        original, d = ACTIVE[v], expected_raw[p, v, t]
        visible = d >= 0 and (p, v, t, j) not in {(0, 2, 0, 0), (1, 0, 0, 4)}
        assert grouped.visibility[p, v, t, j] == visible
        if visible:
            expected_uv[p, original, t, j] = rows[original].uv_px[0, t, d, j]
            expected_confidence[p, original, t, j] = .5 + .04 * original + .02 * d + .005 * t + .001 * j
    np.testing.assert_array_equal(grouped.uv_px, expected_uv[:, list(ACTIVE)])
    np.testing.assert_allclose(grouped.confidence, expected_confidence[:, list(ACTIVE)], atol=1e-7)

    geometry = resolve_camera_geometry(calibration.calibration, "west", (False,) * 3,
        (SideEvidence("plcs", grouped.uv_px, grouped.visibility, 2.),), config=config)
    artifacts.update(calibration=calibration, identities=identities, alignment=CameraAlignmentOutput(geometry))
    context = AssemblyContext(source)
    skeleton = PlayerTriangulationModule(CAMERAS, reprojection_px=2., joint_confidence=.3).process(
        PlayerTriangulationInputAssembler(.3).assemble(context, artifacts))
    expected_valid: NDArray[np.bool_] = np.ones((3, FRAMES, 17), bool)
    expected_valid[0, 0, 0] = False  # only east has a confident joint
    expected_valid[1, 1] = False  # no observations, despite finite raw pixels
    expected_valid[2] = False  # player 31 has only one camera
    expected_xyz = np.where(expected_valid[..., None], xyz, 0)
    assert skeleton.skeleton is not None
    np.testing.assert_array_equal(skeleton.skeleton.valid, expected_valid)
    np.testing.assert_allclose(skeleton.skeleton.positions, expected_xyz, atol=2e-4)

    ball_xyz = np.array([[.4 + .02 * t, -2. + .08 * t, 1.4 + .01 * t] for t in range(FRAMES)], np.float32)
    ball_uv = np.stack([project(model, ball_xyz) for model in models])
    ball_vis: NDArray[np.bool_] = np.ones((4, FRAMES), bool)
    ball_vis[2, 1] = False
    for v, c in enumerate(CAMERAS):
        detections = BallDetectionOutput(c, np.arange(FRAMES, dtype=np.int64), ball_uv[v], np.full(FRAMES, .8, np.float32),
            ball_vis[v], ball_vis[v].astype(np.uint8), "annotation_acceptance_not_probability", None)
        artifacts[f"ball_{c}"] = BallPointsModule(TemporalSmoothingConfig(enabled=False)).process(BallPointsInput(detections, SIZE, 30.))
    ball = BallTriangulationModule(CAMERAS, reprojection_px=2., min_frames=1).process(
        BallTriangulationInputAssembler(.3).assemble(context, artifacts))
    artifacts.update(skeleton=skeleton, ball=ball, placement=BodyPlacementOutput(None))
    scene = SceneAssemblyModule(CAMERAS, require_people=True, require_ball=True, require_body=False).process(
        SceneAssemblyInputAssembler(.3).assemble(context, artifacts))

    assert scene.metadata["camera_ids"] == list(CAMERAS)
    assert scene.metadata["court_reference"]["camera_ids"] == ["west", "east", "high"]
    assert scene.metadata["status"] == "ok"
    np.testing.assert_array_equal(scene.player_track_ids, [5, 17, 31])
    np.testing.assert_array_equal(scene.player_observed, [[True] * 4, [True, False, True, True], [True] * 4])
    np.testing.assert_allclose(scene.human_kp_2d, expected_uv / SIZE, atol=1e-7)
    np.testing.assert_allclose(scene.human_kp_vis, expected_confidence, atol=1e-7)
    np.testing.assert_array_equal(scene.player_kp_3d_vis, expected_valid)
    np.testing.assert_allclose(scene.player_kp_3d, expected_xyz, atol=2e-4)
    expected_ball_uv: NDArray[np.float32] = np.zeros((4, FRAMES, 2), np.float32)
    expected_ball_uv[list(ACTIVE)] = np.where(ball_vis[list(ACTIVE), :, None], ball_uv[list(ACTIVE)], 0) / SIZE
    ball_vis[1] = False
    np.testing.assert_allclose(scene.ball_uv, expected_ball_uv, atol=1e-7)
    np.testing.assert_array_equal(scene.ball_vis, ball_vis)
    np.testing.assert_array_equal(scene.ball_3d_valid, True)
    np.testing.assert_allclose(scene.ball_3d, ball_xyz, atol=2e-4)
    for v, court in enumerate(courts):
        np.testing.assert_allclose(scene.court_kp[v], np.repeat(court.keypoints[0] * [1279, 719] / SIZE, FRAMES, axis=0), atol=1e-7)
