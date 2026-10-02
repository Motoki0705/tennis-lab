"""Unit tests for fixed publication overview geometry."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.synthetic_data_generation.visualization.publication.cameras import (
    PublicationCameraCollection,
)
from src.synthetic_data_generation.visualization.publication.figures import (
    camera_render_indices,
    overview_panel_bounds,
)


def _camera_collection(
    owner: str,
    matrices: tuple[np.ndarray, ...],
    *,
    camera_ids: tuple[str, ...] | None = None,
) -> PublicationCameraCollection:
    ids = (
        tuple(f"camera-{index}" for index in range(len(matrices)))
        if camera_ids is None
        else camera_ids
    )
    cameras = tuple(
        SceneCamera(
            camera_id=camera_id,
            source_frame_index=index,
            width=64,
            height=64,
            intrinsics=(50.0, 0.0, 32.0, 0.0, 50.0, 32.0, 0.0, 0.0, 1.0),
            camera_to_scene=RigidTransform.from_matrix(matrix),
            image_path=f"images/{camera_id}.png",
        )
        for index, (camera_id, matrix) in enumerate(zip(ids, matrices, strict=True))
    )
    return PublicationCameraCollection(
        owner=owner,
        schema=f"{owner}_fixture_v1",
        scene_id="scene-0",
        logical_scene_id=None if owner == "reconstruction" else "logical-0",
        camera_ids=ids,
        cameras=cameras,
        camera_to_metric_scene=np.stack(matrices),
    )


def _layout() -> MultiCourtLayout:
    transform = RigidTransform.from_matrix(np.eye(4, dtype=np.float64))
    return MultiCourtLayout(
        courts=(
            CourtInstance(
                court_instance_id="court-0",
                candidate_id="candidate-0",
                scene_from_court=transform,
                court_from_scene=transform,
                fit_status="accepted",
                fit_metrics={},
                holdout_status="accepted",
                holdout_metrics={},
            ),
        ),
        complex_bounds_scene=(-20.0, -20.0, -1.0, 20.0, 20.0, 10.0),
        primary_court_instance_id="court-0",
    )


def _translated_matrices(count: int, *, offset: float = 0.0) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for index in range(count):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (float(index) + offset, 0.0, 4.0)
        result.append(matrix)
    return tuple(result)


def _canonical_drift_matrix() -> NDArray[np.float64]:
    angle = 1.1884684684684685
    forward = np.asarray(
        (0.6 * np.sin(angle), 0.8 * np.sin(angle), np.cos(angle)),
        dtype=np.float64,
    )
    right = np.cross(np.asarray((0.0, 1.0, 0.0)), forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix: NDArray[np.float64] = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward * (1.0 + 5.0e-8)))
    return matrix


def test_overview_panel_bounds_are_ordered_and_inside_canvas() -> None:
    width, height = 600, 400
    bounds = overview_panel_bounds((width, height))

    assert tuple(label for label, _ in bounds) == (
        "Court dataset",
        "Alignment evidence",
        "Captured cameras",
    )
    for _, (left, top, right, bottom) in bounds:
        assert 0 <= left < right <= width
        assert 0 <= top < bottom <= height

    row = [rectangle for _, rectangle in bounds]
    assert len({rectangle[1] for rectangle in row}) == 1
    assert row[0][0] < row[1][0] < row[2][0]


@pytest.mark.parametrize("size", [(599, 400), (600, 399), (64, 64)])
def test_overview_panel_bounds_require_minimum_canvas(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="at least 600x400"):
        overview_panel_bounds(size)


def test_captured_render_indices_are_deterministic_bounded_and_endpoint_inclusive() -> (
    None
):
    first = camera_render_indices(491, maximum_rendered_cameras=24)
    second = camera_render_indices(491, maximum_rendered_cameras=24)

    assert first == second
    assert len(first) == 24
    assert first[0] == 0
    assert first[-1] == 490
    assert len(set(first)) == len(first)
    assert set(np.diff(first)) <= {21, 22}
