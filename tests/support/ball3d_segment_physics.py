"""Independent DOP853 data with a known bounce, drag and signed clocks."""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tennis_scene.pipeline.components.ball_evidence import BallHeatmapEvidence
from src.tennis_scene.pipeline.components.ball_points import (
    BallPointsInput,
    BallPointsModule,
)
from src.utils.geometry.triangulation import PinholeCamera, TriangulatedPoints
from tests.support.ball3d_evaluation import make_court, make_inputs


def synthetic_flights():
    fps, frames, bounce, drag = 60.0, 76, 31, 0.021
    offsets = np.array([0.27, -0.46, 0.0]) / fps

    def derivative(t, state):
        velocity = state[3:]
        return np.r_[
            velocity, -drag * np.linalg.norm(velocity) * velocity - [0.0, 0.0, 9.81]
        ]

    first = np.array([-3.0, -5.0, 0.0, 8.0, 16.0, -2.8])
    trial = solve_ivp(
        derivative, (0, bounce / fps), first, method="DOP853", rtol=1e-12, atol=1e-13
    )
    first[2] = 0.0335 - trial.y[2, -1]
    flight1 = solve_ivp(
        derivative,
        (-0.02, bounce / fps),
        solve_ivp(
            derivative, (0, -0.02), first, method="DOP853", rtol=1e-12, atol=1e-13
        ).y[:, -1],
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
        dense_output=True,
    )
    second = flight1.sol(bounce / fps)
    second[3:] *= [0.88, 0.9, -0.73]
    flight2 = solve_ivp(
        derivative,
        (bounce / fps, frames / fps + 0.02),
        second,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
        dense_output=True,
    )

    def world(times):
        return np.where(
            (times < bounce / fps)[:, None], flight1.sol(times).T, flight2.sol(times).T
        )

    xyz = world(np.arange(frames) / fps)[:, :3]
    cameras, evidence = [], []
    centers = [(-6.0, -8.0, 14.0), (6.0, -2.0, 16.0), (1.0, 9.0, 15.0)]
    rotation = np.diag([1.0, -1.0, -1.0])
    size, grid = (640, 480), (129, 97)
    gy, gx = np.mgrid[: grid[1], : grid[0]]
    for i, (name, center) in enumerate(
        zip(("right", "near", "far"), centers, strict=True)
    ):
        center = np.array(center)
        fx, fy = 230 + 10 * i, 125 + 10 * i
        camera = PinholeCamera(
            name,
            np.array([[fx, 0.0, 320.0], [0.0, fy, 240.0], [0.0, 0.0, 1.0]]),
            rotation,
            -rotation @ center,
        )
        point = world(np.arange(frames) / fps + offsets[i])[:, :3]
        # Independent pinhole equation, not Camera.project or the fitted integrator.
        uv = np.column_stack(
            (
                fx * (point[:, 0] - center[0]) / (center[2] - point[:, 2]) + 320,
                fy * (center[1] - point[:, 1]) / (center[2] - point[:, 2]) + 240,
            )
        )
        native = uv / [639, 479] * [128, 96]
        maps = (
            0.9
            * np.exp(
                -(
                    (gx - native[:, 0, None, None]) ** 2
                    + (gy - native[:, 1, None, None]) ** 2
                )
                / (2 * 0.85**2)
            )
            + 1e-8
        ).astype(np.float32)
        cells = np.rint(native).astype(np.int64)[:, None]
        patches = np.stack(
            [
                maps[t, y - 1 : y + 2, x - 1 : x + 2]
                for t, (x, y) in enumerate(cells[:, 0])
            ]
        )[:, None]
        scores = maps[np.arange(frames), cells[:, 0, 1], cells[:, 0, 0]][:, None]
        evidence.append(
            BallHeatmapEvidence(
                size,
                BallCandidateConfig(max_candidates=1, nms_kernel=3, patch_size=3),
                maps,
                uv.astype(np.float32)[:, None],
                scores,
                np.ones((frames, 1), bool),
                cells,
                patches,
                np.ones_like(patches, bool),
                np.arange(frames, dtype=np.int64),
                np.zeros(frames, np.int64),
            )
        )
        cameras.append(camera)
    valid: NDArray[np.bool_] = np.ones(frames, bool)
    initial = TriangulatedPoints(
        xyz.astype(np.float32),
        valid,
        np.zeros(frames, np.uint8),
        np.ones((3, frames), bool),
        np.zeros((3, frames), np.float32),
    )
    return tuple(cameras), tuple(evidence), initial, fps, drag, offsets, bounce


def synthetic_method_inputs():
    cameras, evidence, initial, fps, drag, offsets, bounce = synthetic_flights()
    template = make_inputs(frames=len(initial.valid))
    views = []
    for i, (view, camera, row) in enumerate(
        zip(template.views, cameras, evidence, strict=True)
    ):
        detections = replace(
            view.detections,
            uv_px=row.candidate_uv_px[:, 0].copy(),
            confidence=row.candidate_scores[:, 0].copy(),
            observed=np.ones(len(initial.valid), bool),
            point_kind=np.ones(len(initial.valid), np.uint8),
            evidence=row,
        )
        points = BallPointsModule(view.smoothing_config).process(
            BallPointsInput(detections, row.source_size_wh, fps)
        )
        local = replace(
            view.local_calibration, camera=camera.half_turned(view.half_turn)
        )
        views.append(
            replace(
                view,
                camera=camera,
                local_calibration=local,
                detections=detections,
                points=points,
                court=make_court(
                    camera.camera_id, i, row.source_size_wh, len(initial.valid)
                ),
            )
        )
    return (
        replace(
            template,
            fps=fps,
            source_size_wh=evidence[0].source_size_wh,
            views=tuple(views),
        ),
        initial,
        drag,
        offsets,
        bounce,
    )
