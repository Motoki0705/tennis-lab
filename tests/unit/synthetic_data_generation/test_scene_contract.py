"""Unit tests for renderer-independent semantic scene contracts."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import src.synthetic_data_generation.scene_contract as scene_contract_module
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)


def _rigid(*, translation: tuple[float, float, float]) -> RigidTransform:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return RigidTransform.from_matrix(matrix)


def _camera() -> SceneCamera:
    return SceneCamera(
        camera_id="camera-001",
        source_frame_index=12,
        width=960,
        height=540,
        intrinsics=(500.0, 0.0, 480.0, 0.0, 500.0, 270.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path="images/frame-000012.png",
    )


def _court(
    court_instance_id: str = "court-001",
    candidate_id: str = "candidate-001",
) -> CourtInstance:
    scene_from_court = _rigid(translation=(1.0, 2.0, 3.0))
    return CourtInstance(
        court_instance_id=court_instance_id,
        candidate_id=candidate_id,
        scene_from_court=scene_from_court,
        court_from_scene=scene_from_court.inverse(),
        fit_status="accepted",
        fit_metrics={"fit_camera_count": 20},
        holdout_status="accepted",
        holdout_metrics={"holdout_camera_count": 10},
    )


def test_rigid_transform_round_trips_points() -> None:
    transform = _rigid(translation=(1.0, 2.0, 3.0))
    points = np.asarray([[0.0, 0.0, 0.0], [5.0, -2.0, 1.0]])

    recovered = transform.inverse().apply(transform.apply(points))

    np.testing.assert_allclose(recovered, points, atol=1.0e-12)


def test_scene_camera_strict_json_round_trip_and_projection() -> None:
    camera = _camera()

    loaded = SceneCamera.from_dict(camera.to_dict())
    pixels, depth = loaded.project_scene_points(np.asarray([[0.0, 0.0, 2.0]]))

    assert loaded == camera
    np.testing.assert_allclose(pixels, [[480.0, 270.0]])
    np.testing.assert_allclose(depth, [2.0])


def test_scene_camera_rejects_unknown_schema_field() -> None:
    payload = _camera().to_dict()
    payload["artifact_hash"] = "not-semantic-authority"

    with pytest.raises(ValueError, match="keys do not match"):
        SceneCamera.from_dict(payload)


def test_rigid_transform_rejects_reflection() -> None:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = -1.0

    with pytest.raises(ValueError, match="proper rotation"):
        RigidTransform.from_matrix(matrix)


def test_court_instance_requires_fit_and_holdout_acceptance() -> None:
    transform = _rigid(translation=(1.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="fit- and holdout-accepted"):
        CourtInstance(
            court_instance_id="court-001",
            candidate_id="candidate-001",
            scene_from_court=transform,
            court_from_scene=transform.inverse(),
            fit_status="accepted",
            fit_metrics={},
            holdout_status="rejected",
            holdout_metrics={},
        )


def test_court_instance_rejects_inconsistent_inverse() -> None:
    transform = _rigid(translation=(1.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="must be reciprocal"):
        CourtInstance(
            court_instance_id="court-001",
            candidate_id="candidate-001",
            scene_from_court=transform,
            court_from_scene=RigidTransform.identity(),
            fit_status="accepted",
            fit_metrics={},
            holdout_status="accepted",
            holdout_metrics={},
        )


def test_multi_court_layout_preserves_all_accepted_courts_without_fallback() -> None:
    first = _court()
    second = _court("court-002", "candidate-002")
    layout = MultiCourtLayout(
        courts=(first, second),
        complex_bounds_scene=(-20.0, -40.0, -2.0, 20.0, 40.0, 10.0),
        primary_court_instance_id=None,
    )

    assert layout.court("court-002") == second
    assert layout.to_dict()["courts"] == [first.to_dict(), second.to_dict()]
    with pytest.raises(KeyError, match="Unknown court_instance_id"):
        layout.court("missing")


def test_multi_court_layout_rejects_duplicate_candidate_identity() -> None:
    with pytest.raises(ValueError, match="candidate_id values must be unique"):
        MultiCourtLayout(
            courts=(_court(), _court("court-002", "candidate-001")),
            complex_bounds_scene=(-20.0, -40.0, -2.0, 20.0, 40.0, 10.0),
            primary_court_instance_id=None,
        )


def test_scene_contract_has_no_task_or_renderer_backend_imports() -> None:
    tree = ast.parse(inspect.getsource(scene_contract_module))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    forbidden = ("gsplat", "src.tasks")
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported_modules
        for prefix in forbidden
    )
