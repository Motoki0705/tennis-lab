"""Tests for bold smooth SfM-envelope orbit families."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.artifacts.layout import (
    MultiCourtLayout,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.orbit import (
    derive_orbit_families,
    sample_orbit_families,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


class _OrbitSamplingKwargs(TypedDict):
    cameras: tuple[SceneCamera, ...]
    layout: MultiCourtLayout
    support_points_scene: NDArray[np.float64]
    seed: int
    samples_per_orbit: int


def test_family_derivation_contains_nested_circles_ellipses_and_targets(
    captured_cameras: tuple[SceneCamera, ...],
    two_court_layout: MultiCourtLayout,
) -> None:
    families = derive_orbit_families(
        captured_cameras,
        two_court_layout,
        seed=11,
        samples_per_orbit=16,
    )

    assert len(families) == 18
    assert {family.shape for family in families} == {"circle", "ellipse"}
    assert {family.target_court_instance_id for family in families} == {
        None,
        "court_0",
        "court_1",
    }
    ellipse_radii = {
        (round(family.radius_x_m, 4), round(family.radius_y_m, 4))
        for family in families
        if family.shape == "ellipse"
    }
    assert len(ellipse_radii) == 3


def test_sampling_is_deterministic_bold_and_keeps_partial_supervision(
    captured_cameras: tuple[SceneCamera, ...],
    two_court_layout: MultiCourtLayout,
    support_points_scene: NDArray[np.float64],
) -> None:
    kwargs: _OrbitSamplingKwargs = {
        "cameras": captured_cameras,
        "layout": two_court_layout,
        "support_points_scene": support_points_scene,
        "seed": 19,
        "samples_per_orbit": 16,
    }

    first = sample_orbit_families(**kwargs)
    second = sample_orbit_families(**kwargs)

    assert first == second
    assert len(first.frames) > 100
    assert max(frame.nearest_captured_translation_m for frame in first.frames) > 2.0
    assert max(frame.nearest_captured_rotation_deg for frame in first.frames) > 5.0
    buckets = {
        court.coverage_bucket
        for frame in first.frames
        for court in frame.projection.courts
    }
    assert "partial" in buckets
    assert "full" in buckets

    by_family: dict[str, list[NDArray[np.float64]]] = {}
    for frame in first.frames:
        matrix = np.asarray(frame.camera.camera_to_scene).reshape(4, 4)
        by_family.setdefault(frame.family_id, []).append(matrix[:3, 3])
    maximum_step = max(
        np.linalg.norm(np.diff(np.stack(centers), axis=0), axis=1).max()
        for centers in by_family.values()
        if len(centers) > 1
    )
    assert maximum_step < 12.0
