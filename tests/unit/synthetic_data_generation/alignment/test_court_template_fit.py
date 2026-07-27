"""Tests for metric multi-court fitting on ground-line evidence."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.court_template_fit import (
    COURT_GEOMETRY_SCHEMA,
    CourtLocalRefitSettings,
    CourtTemplateFitSettings,
    fit_court_instance_near_reference,
    fit_court_instances,
    load_court_geometry_artifact,
    publish_court_geometry_artifact,
    scene_from_court_matrix,
)
from src.synthetic_data_generation.alignment.ground_plane import GroundPlaneEstimate
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
)


def _plane() -> GroundPlaneEstimate:
    return GroundPlaneEstimate(
        normal=(0.0, 0.0, 1.0),
        offset=0.0,
        origin=(0.0, 0.0, 0.0),
        basis_u=(1.0, 0.0, 0.0),
        basis_v=(0.0, 1.0, 0.0),
        support_uv_bounds=(-2.0, 2.0, -2.0, 2.0),
        metrics={},
    )


def _draw_court(
    image: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    spacing: float,
    center: tuple[float, float],
    orientation: float,
    scale: float,
) -> None:
    xd = HALF_DOUBLES_WIDTH
    xs = HALF_SINGLES_WIDTH
    yb = HALF_LENGTH
    ys = SERVICE_LINE_DISTANCE
    segments = (
        ((-xd, -yb), (-xd, yb)),
        ((xd, -yb), (xd, yb)),
        ((-xs, -yb), (-xs, yb)),
        ((xs, -yb), (xs, yb)),
        ((-xd, -yb), (xd, -yb)),
        ((-xd, yb), (xd, yb)),
        ((-xs, -ys), (xs, -ys)),
        ((-xs, ys), (xs, ys)),
        ((0.0, -ys), (0.0, ys)),
    )
    cosine = math.cos(orientation)
    sine = math.sin(orientation)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    u_min, _, v_min, _ = bounds
    for start, end in segments:
        uv = np.asarray(
            (start, end), dtype=np.float64
        ) @ rotation.T * scale + np.asarray(center)
        pixels = np.rint(
            np.column_stack(
                (
                    (uv[:, 0] - u_min) / spacing,
                    (uv[:, 1] - v_min) / spacing,
                )
            )
        ).astype(np.int32)
        cv2.line(image, pixels[0], pixels[1], 5.0, 2, cv2.LINE_AA)


def test_fit_keeps_two_adjacent_courts_as_distinct_instances() -> None:
    bounds = (-1.3, 1.3, -1.1, 1.3)
    spacing = 0.005
    width = int(np.ceil((bounds[1] - bounds[0]) / spacing)) + 1
    height = int(np.ceil((bounds[3] - bounds[2]) / spacing)) + 1
    evidence: NDArray[np.float32] = np.zeros((height, width), dtype=np.float32)
    expected_centers = ((-0.08, 0.67), (-0.02, -0.34))
    for center in expected_centers:
        _draw_court(
            evidence,
            bounds=bounds,
            spacing=spacing,
            center=center,
            orientation=-1.50,
            scale=0.070,
        )

    candidates = fit_court_instances(
        evidence,
        bounds=bounds,
        grid_spacing=spacing,
        plane=_plane(),
        settings=CourtTemplateFitSettings(
            seed=19,
            optimizer_max_iterations=70,
            optimizer_population_size=10,
            optimizer_tolerance=1.0e-6,
        ),
    )

    assert len(candidates) == 2
    fitted_centers = np.asarray([candidate.center_uv for candidate in candidates])
    for expected in expected_centers:
        distance = np.linalg.norm(fitted_centers - np.asarray(expected), axis=1)
        assert float(distance.min()) < 0.03
    for candidate in candidates:
        assert candidate.scale_scene_per_metre == pytest.approx(0.070, abs=0.002)
        matrix = np.asarray(candidate.scene_from_court).reshape(4, 4)
        assert np.linalg.det(matrix[:3, :3]) > 0.0


def test_scene_from_court_maps_ground_and_positive_height() -> None:
    transform = scene_from_court_matrix(
        _plane(),
        center_uv=(0.3, -0.2),
        orientation_radians=0.4,
        scale_scene_per_metre=0.07,
    )

    np.testing.assert_allclose(transform[:3, 3], (0.3, -0.2, 0.0))
    np.testing.assert_allclose(transform[:3, 2], (0.0, 0.0, 0.07))
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(0.07**3)


def test_local_refit_keeps_the_selected_physical_court_cluster() -> None:
    bounds = (-1.3, 1.3, -1.1, 1.3)
    spacing = 0.005
    width = int(np.ceil((bounds[1] - bounds[0]) / spacing)) + 1
    height = int(np.ceil((bounds[3] - bounds[2]) / spacing)) + 1
    evidence: NDArray[np.float32] = np.zeros((height, width), dtype=np.float32)
    selected_center = (-0.08, 0.67)
    _draw_court(
        evidence,
        bounds=bounds,
        spacing=spacing,
        center=selected_center,
        orientation=-1.50,
        scale=0.070,
    )
    _draw_court(
        evidence,
        bounds=bounds,
        spacing=spacing,
        center=(-0.02, -0.34),
        orientation=-1.50,
        scale=0.070,
    )

    candidate = fit_court_instance_near_reference(
        evidence,
        bounds=bounds,
        grid_spacing=spacing,
        plane=_plane(),
        reference_center_uv=(-0.06, 0.65),
        reference_orientation_radians=-1.48,
        reference_scale_scene_per_metre=0.069,
        settings=CourtLocalRefitSettings(
            seed=23,
            centre_radius_m=1.0,
            optimizer_max_iterations=60,
            optimizer_population_size=10,
            optimizer_tolerance=1.0e-6,
        ),
    )

    assert (
        np.linalg.norm(np.asarray(candidate.center_uv) - np.asarray(selected_center))
        < 0.03
    )
    assert candidate.orientation_radians == pytest.approx(-1.50, abs=0.03)


def test_local_refit_enforces_circular_centre_radius() -> None:
    bounds = (-1.3, 1.3, -1.1, 1.3)
    spacing = 0.005
    width = int(np.ceil((bounds[1] - bounds[0]) / spacing)) + 1
    height = int(np.ceil((bounds[3] - bounds[2]) / spacing)) + 1
    evidence: NDArray[np.float32] = np.zeros((height, width), dtype=np.float32)
    _draw_court(
        evidence,
        bounds=bounds,
        spacing=spacing,
        center=(0.08, 0.08),
        orientation=-1.50,
        scale=0.070,
    )
    radius_m = 0.5
    reference_scale = 0.070

    candidate = fit_court_instance_near_reference(
        evidence,
        bounds=bounds,
        grid_spacing=spacing,
        plane=_plane(),
        reference_center_uv=(0.0, 0.0),
        reference_orientation_radians=-1.50,
        reference_scale_scene_per_metre=reference_scale,
        settings=CourtLocalRefitSettings(
            seed=7,
            centre_radius_m=radius_m,
            optimizer_max_iterations=40,
            optimizer_population_size=8,
            optimizer_tolerance=1.0e-6,
        ),
    )

    assert np.linalg.norm(
        np.asarray(candidate.center_uv)
    ) <= radius_m * reference_scale * (1.0 + 1.0e-8)


def test_geometry_artifact_round_trip_and_holdout_status(tmp_path: Path) -> None:
    payload = {
        "schema": COURT_GEOMETRY_SCHEMA,
        "artifact_id": "synthetic-court-fit-v1",
        "created_at_utc": "2026-07-25T00:00:00+00:00",
        "ground_line_artifact": {},
        "fit_settings": {},
        "candidates": [
            {
                "candidate_id": "court-0",
                "template_score": 1.0,
            }
        ],
        "selection": {
            "selected_candidate_id": "court-0",
            "rule": "highest score",
        },
        "acceptance_status": "fit_candidate_holdout_not_run",
        "provenance": {},
    }

    path = publish_court_geometry_artifact(payload, output_dir=tmp_path)

    loaded = load_court_geometry_artifact(path)
    assert loaded["selection"]["selected_candidate_id"] == "court-0"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_court_geometry_artifact(payload, output_dir=tmp_path)
