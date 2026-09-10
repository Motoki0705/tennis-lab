import json

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    build_court_dataset_plan,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sfm_bounds import (
    bound_trajectory_candidates,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.targeting import (
    sample_look_at_offset,
    validate_camera_looks_at_resolved_binding,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.trajectory import (
    derive_orbit_centers,
    generate_trajectory_candidates,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    CourtDatasetPlanV3,
    OrbitSamplingPolicy,
    OrbitTrajectorySpec,
    OrbitViewSpecV2,
)
from src.synthetic_data_generation.scene_contract import MultiCourtLayout, SceneCamera
from tests.unit.synthetic_data_generation.dataset.court.components.camera_sampling.test_selection import (
    _composed_configuration,
)


@pytest.mark.parametrize("expansion_percent", [0.0, 5.0, 15.0])
def test_complete_shapes_stay_inside_inset_sfm_hull(
    captured_cameras: tuple[SceneCamera, ...], multi_court_layout: MultiCourtLayout,
    expansion_percent: float,
) -> None:
    config = _composed_configuration("sfm_bounded")
    policy = OrbitSamplingPolicy.from_configuration(config.sampling)
    centers = derive_orbit_centers(captured_cameras, multi_court_layout)
    candidates = generate_trajectory_candidates(
        config.trajectory,
        centers,
        seed=policy.seed,
        stable_field_order=policy.stable_field_order,
    )
    bounded = bound_trajectory_candidates(
        candidates, centers=centers, cameras=captured_cameras, margin_m=0.5,
        expansion_percent=expansion_percent,
    )
    hull = ConvexHull(
        np.array([c.camera_to_scene.matrix()[:2, 3] for c in captured_cameras])
    )
    origin = hull.points[hull.vertices].mean(axis=0)
    expanded_hull = ConvexHull(origin + (hull.points - origin) * (1 + expansion_percent / 100))
    hull = expanded_hull
    by_key = {c.key(): c for c in centers}
    # Every family, including rotated elongated ellipses and off-centre courts.
    for candidate in bounded:
        path = sample_uniform_arc_length(
            candidate,
            by_key[candidate.center_kind, candidate.center_court_instance_id],
            policy,
        )
        distances = (
            path.points_scene_m[:, :2] @ hull.equations[:, :2].T + hull.equations[:, 2]
        )
        assert np.max(distances) <= -0.5 + 1e-9
    assert len(bounded) == len(candidates)
    assert any(
        b.base_radius_m < c.base_radius_m
        for b, c in zip(bounded, candidates, strict=True)
    )
    assert {b.radius_scale for b in bounded} == {0.65, 0.8, 0.95}


def test_unobserved_or_degenerate_boundary_fails(
    captured_cameras: tuple[SceneCamera, ...], multi_court_layout: MultiCourtLayout
) -> None:
    centers = derive_orbit_centers(captured_cameras, multi_court_layout)
    with pytest.raises(ValueError, match="outside"):
        bound_trajectory_candidates(
            (), centers=centers, cameras=captured_cameras, margin_m=100.0
        )
    repeated = (captured_cameras[0],) * 3
    with pytest.raises(ValueError, match="planar support"):
        bound_trajectory_candidates((), centers=centers, cameras=repeated, margin_m=0.0)


def test_jitter_is_reproducible_uniform_area_and_bounded() -> None:
    offsets = np.stack(
        [
            sample_look_at_offset(radius_m=1.0, seed=695, sample_index=i)
            for i in range(2000)
        ]
    )
    np.testing.assert_array_equal(
        offsets[100], sample_look_at_offset(radius_m=1.0, seed=695, sample_index=100)
    )
    squared = np.sum(offsets[:, :2] ** 2, axis=1)
    assert squared.max() <= 1.0
    assert squared.mean() == pytest.approx(0.5, abs=0.03)
    np.testing.assert_allclose(offsets.mean(axis=0), 0.0, atol=0.04)
    assert not np.array_equal(
        offsets[100], sample_look_at_offset(radius_m=1.0, seed=696, sample_index=100)
    )
    np.testing.assert_array_equal(
        sample_look_at_offset(radius_m=0.0, seed=695, sample_index=100), np.zeros(3)
    )


def test_bounded_plan_roundtrip_and_jitter_tamper_detection(
    captured_cameras: tuple[SceneCamera, ...],
    multi_court_layout: MultiCourtLayout,
    identity_metric_adapter: MetricSceneAdapter,
) -> None:
    config = _composed_configuration("sfm_bounded")
    plan = build_court_dataset_plan(
        scene_id="fixture",
        profile="test",
        cameras=captured_cameras,
        layout=multi_court_layout,
        configuration=config,
        metric_adapter=identity_metric_adapter,
    )
    assert isinstance(plan, CourtDatasetPlanV3)
    groups = {g.trajectory_group_id: g for g in plan.groups}
    for sample in plan.samples[::71]:
        view = groups[sample.trajectory_group_id].views[0]
        assert OrbitViewSpecV2.from_mapping(view.to_dict()) == view
        validate_camera_looks_at_resolved_binding(
            camera=sample.camera,
            target_court=sample.target_court,
            look_at_height_m=view.look_at_height_m,
            look_at_jitter_radius_m=view.look_at_jitter_radius_m,
            sample_index=sample.sample_index,
        )
        with pytest.raises(ValueError, match="forward axis"):
            validate_camera_looks_at_resolved_binding(
                camera=sample.camera,
                target_court=sample.target_court,
                look_at_height_m=view.look_at_height_m,
                look_at_jitter_radius_m=0.0,
                sample_index=sample.sample_index,
            )


def test_bounds_are_invariant_to_scene_frame(
    captured_cameras: tuple[SceneCamera, ...], multi_court_layout: MultiCourtLayout
) -> None:
    from dataclasses import replace

    from scipy.spatial.transform import Rotation

    from src.synthetic_data_generation.scene_contract import RigidTransform

    config = _composed_configuration("sfm_bounded")
    policy = OrbitSamplingPolicy.from_configuration(config.sampling)
    centers = derive_orbit_centers(captured_cameras, multi_court_layout)
    candidates = generate_trajectory_candidates(
        config.trajectory,
        centers,
        seed=policy.seed,
        stable_field_order=policy.stable_field_order,
    )
    expected = bound_trajectory_candidates(
        candidates, centers=centers, cameras=captured_cameras, margin_m=0.5
    )
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", [25, -15, 70], degrees=True
    ).as_matrix()
    transform[:3, 3] = [100, -40, 12]
    rotated_centers = tuple(
        replace(
            c,
            scene_from_center=RigidTransform.from_matrix(
                transform @ c.scene_from_center.matrix()
            ),
        )
        for c in centers
    )
    rotated_cameras = tuple(
        replace(
            c,
            camera_to_scene=RigidTransform.from_matrix(
                transform @ c.camera_to_scene.matrix()
            ),
        )
        for c in captured_cameras
    )
    actual = bound_trajectory_candidates(
        candidates, centers=rotated_centers, cameras=rotated_cameras, margin_m=0.5
    )
    np.testing.assert_allclose(
        [c.base_radius_m for c in actual],
        [c.base_radius_m for c in expected],
        atol=1e-10,
        rtol=0,
    )


def test_spatial_selection_expands_shared_coverage_without_losing_shapes(
    captured_cameras: tuple[SceneCamera, ...],
    multi_court_layout: MultiCourtLayout,
    identity_metric_adapter: MetricSceneAdapter,
) -> None:
    from collections import Counter
    from dataclasses import replace

    from src.synthetic_data_generation.dataset.court.contracts import OrbitShape

    config = _composed_configuration("sfm_bounded")
    baseline_config = replace(config, trajectory=replace(config.trajectory, spatial_coverage_cell_m=None))
    from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
    from src.synthetic_data_generation.dataset.court.contracts import (
        CourtDatasetPlanAny,
    )

    def make_plan(configuration: CourtDatasetConfiguration) -> CourtDatasetPlanAny:
        return build_court_dataset_plan(scene_id="fixture", profile="test", cameras=captured_cameras, layout=multi_court_layout, metric_adapter=identity_metric_adapter, configuration=configuration)

    baseline = make_plan(baseline_config)
    spatial = make_plan(config)
    common = multi_court_layout.courts[0].court_from_scene
    occupied = []
    for plan in (baseline, spatial):
        points = common.apply(np.asarray([sample.camera_center_scene_m for sample in plan.samples]))
        occupied.append(len(np.unique(np.floor(points[:, :2]).astype(int), axis=0)))
    assert occupied[1] > occupied[0]
    counts = Counter(group.trajectory.shape for group in spatial.groups)
    assert set(counts) == set(OrbitShape)
    assert min(counts.values()) >= 2
    assert spatial.to_dict() == make_plan(config).to_dict()


@pytest.mark.parametrize("expansion_percent", [-1.0, float("nan"), float("inf")])
def test_invalid_expansion_rejected(expansion_percent: float) -> None:
    with pytest.raises(ValueError, match="expansion"):
        bound_trajectory_candidates((), centers=(), cameras=(), margin_m=0.0, expansion_percent=expansion_percent)


def test_expansion_grows_bounds_and_zero_is_compatible(
    captured_cameras: tuple[SceneCamera, ...], multi_court_layout: MultiCourtLayout
) -> None:
    config = _composed_configuration("sfm_bounded")
    policy = OrbitSamplingPolicy.from_configuration(config.sampling)
    centers = derive_orbit_centers(captured_cameras, multi_court_layout)
    candidates = generate_trajectory_candidates(config.trajectory, centers, seed=policy.seed, stable_field_order=policy.stable_field_order)
    def bound(percent: float = 0.0) -> tuple[OrbitTrajectorySpec, ...]:
        return tuple(bound_trajectory_candidates(candidates, centers=centers, cameras=captured_cameras, margin_m=0.0, expansion_percent=percent))
    baseline = bound()
    expanded = bound(5.0)
    assert baseline == bound_trajectory_candidates(candidates, centers=centers, cameras=captured_cameras, margin_m=0.0)
    assert all(b.base_radius_m > a.base_radius_m for a,b in zip(baseline, expanded, strict=True))


@pytest.mark.parametrize("percent", [-1.0, float("inf"), float("nan")])
def test_config_rejects_invalid_expansion(percent: float) -> None:
    from dataclasses import asdict

    from src.synthetic_data_generation.configuration import (
        CourtTrajectoryPolicy,
        SemanticConfigurationError,
    )
    raw = json.loads(json.dumps(asdict(_composed_configuration("sfm_bounded").trajectory)))
    raw["sfm_boundary_expansion_percent"] = percent
    with pytest.raises(SemanticConfigurationError):
        CourtTrajectoryPolicy.from_mapping(raw)


def test_config_expansion_requires_bounds_and_accepts_percent() -> None:
    from dataclasses import asdict

    from src.synthetic_data_generation.configuration import (
        CourtTrajectoryPolicy,
        SemanticConfigurationError,
    )
    raw = json.loads(json.dumps(asdict(_composed_configuration("sfm_bounded").trajectory)))
    raw["sfm_boundary_expansion_percent"] = 5.0
    assert CourtTrajectoryPolicy.from_mapping(raw).sfm_boundary_expansion_percent == 5.0
    raw["sfm_boundary_margin_m"] = None
    with pytest.raises(SemanticConfigurationError, match="expansion"):
        CourtTrajectoryPolicy.from_mapping(raw)


def test_bounded_complex_center_ignores_background_bounds(
    captured_cameras: tuple[SceneCamera, ...], multi_court_layout: MultiCourtLayout
) -> None:
    from dataclasses import replace

    changed = replace(multi_court_layout, complex_bounds_scene=(-300.0, -200.0, -10.0, -100.0, 300.0, 20.0))
    expected = derive_orbit_centers(captured_cameras, multi_court_layout, use_court_centroid=True)
    actual = derive_orbit_centers(captured_cameras, changed, use_court_centroid=True)
    assert actual == expected
    reference = multi_court_layout.court(actual[0].reference_court_instance_id)
    local = reference.court_from_scene.apply(np.array([c.scene_from_court.matrix()[:3, 3] for c in multi_court_layout.courts]))
    center = reference.court_from_scene.apply(actual[0].scene_from_center.matrix()[:3, 3][None, :])[0]
    np.testing.assert_allclose(center[:2], local[:, :2].mean(axis=0))
    assert center[2] == pytest.approx(0.0, abs=1e-10)


def test_captured_hull_center_moves_only_complex_orbits(
    captured_cameras: tuple[SceneCamera, ...], multi_court_layout: MultiCourtLayout
) -> None:
    ordinary = derive_orbit_centers(captured_cameras, multi_court_layout, use_court_centroid=True)
    shifted = derive_orbit_centers(captured_cameras, multi_court_layout, use_court_centroid=True, use_captured_hull_centroid=True)
    ref = multi_court_layout.court(shifted[0].reference_court_instance_id)
    xy = ref.court_from_scene.apply(np.array([c.camera_to_scene.matrix()[:3, 3] for c in captured_cameras]))[:, :2]
    hull = ConvexHull(xy)
    actual = ref.court_from_scene.apply(shifted[0].scene_from_center.matrix()[:3, 3][None, :])[0]
    np.testing.assert_allclose(actual[:2], xy[hull.vertices].mean(axis=0))
    assert actual[2] == pytest.approx(0.0, abs=1e-10)
    assert shifted[1:] == ordinary[1:]
    # Repeated interior/captured samples must not bias the hull-owned origin.
    repeated = derive_orbit_centers((*captured_cameras, *captured_cameras[:2]), multi_court_layout, use_captured_hull_centroid=True)
    assert repeated[0].scene_from_center == shifted[0].scene_from_center


def test_captured_hull_center_requires_explicit_bounds() -> None:
    from dataclasses import asdict

    from src.synthetic_data_generation.configuration import (
        CourtTrajectoryPolicy,
        SemanticConfigurationError,
    )

    raw = json.loads(json.dumps(asdict(_composed_configuration("sfm_bounded").trajectory)))
    raw["sfm_complex_center_on_hull"] = True
    assert CourtTrajectoryPolicy.from_mapping(raw).sfm_complex_center_on_hull
    raw["sfm_boundary_margin_m"] = None
    with pytest.raises(SemanticConfigurationError, match="Captured-hull"):
        CourtTrajectoryPolicy.from_mapping(raw)
