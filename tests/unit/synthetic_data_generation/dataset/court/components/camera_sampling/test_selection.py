from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    build_court_dataset_plan,
    select_budgeted_coverage,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.trajectory import (
    generate_trajectory_candidates,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitCenterKind,
    OrbitSamplingPolicy,
)
from src.synthetic_data_generation.scene_contract import (
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)


def _configuration() -> CourtDatasetConfiguration:
    raw = OmegaConf.to_container(
        OmegaConf.load(
            Path("src/synthetic_data_generation/configs/dataset/court/train.yaml")
        ),
        resolve=True,
    )
    return CourtDatasetConfiguration.from_mapping(raw)


def test_production_plan_is_budgeted_diverse_group_disjoint_and_deterministic(
    captured_cameras: tuple[SceneCamera, ...],
    multi_court_layout: MultiCourtLayout,
    identity_metric_adapter: MetricSceneAdapter,
) -> None:
    configuration = _configuration()
    first = build_court_dataset_plan(
        scene_id="B00",
        profile="train",
        cameras=captured_cameras,
        layout=multi_court_layout,
        configuration=configuration,
        metric_adapter=identity_metric_adapter,
    )
    second = build_court_dataset_plan(
        scene_id="B00",
        profile="train",
        cameras=captured_cameras,
        layout=multi_court_layout,
        configuration=configuration,
        metric_adapter=identity_metric_adapter,
    )
    assert len(first.groups) >= 24
    assert 2_000 <= first.proposal_count <= 4_800
    assert max(group.maximum_adjacent_step_m for group in first.groups) <= 1.05
    assert {group.trajectory.shape.value for group in first.groups} == {
        "circle",
        "ellipse",
    }
    assert any(
        group.trajectory.axis_ratio <= 0.8
        for group in first.groups
        if group.trajectory.shape.value == "ellipse"
    )
    assert {
        round(np.degrees(group.trajectory.orientation_radians))
        for group in first.groups
    } >= {0, 45, 90}
    assert {group.trajectory.center_kind for group in first.groups} == set(
        OrbitCenterKind
    )
    assert len({group.trajectory.base_height_m for group in first.groups}) >= 3
    assert any(group.trajectory.vertical_amplitude_m > 0.0 for group in first.groups)
    variant_group = next(group for group in first.groups if len(group.views) == 2)
    assert len({view.target_kind for view in variant_group.views}) == 2
    assert all(
        len({sample.split for sample in first.samples if sample.trajectory_group_id == group_id})
        == 1
        for group_id in {sample.trajectory_group_id for sample in first.samples}
    )
    court_counts = {
        court_id: sum(
            group.target_court.court_instance_id == court_id for group in first.groups
        )
        for court_id in ("court-0", "court-1")
    }
    assert max(court_counts.values()) - min(court_counts.values()) <= 1
    assert all(
        group.trajectory.center_court_instance_id is None
        or group.trajectory.center_court_instance_id
        == group.target_court.court_instance_id
        for group in first.groups
    )
    assert first.to_dict() == second.to_dict()


def test_selector_reserves_group_budget_for_a_long_captured_complex_orbit() -> None:
    configuration = _configuration()
    policy = OrbitSamplingPolicy.from_configuration(configuration.sampling)
    centers = (
        OrbitCenter(
            center_kind=OrbitCenterKind.COMPLEX,
            court_instance_id=None,
            reference_court_instance_id="court-000",
            scene_from_center=RigidTransform.identity(),
            base_radius_m=174.15,
            captured_offset_median_m=159.50,
            captured_offset_q90_m=174.15,
            captured_camera_count=491,
        ),
        OrbitCenter(
            center_kind=OrbitCenterKind.COURT,
            court_instance_id="court-000",
            reference_court_instance_id="court-000",
            scene_from_center=RigidTransform.identity(),
            base_radius_m=20.65,
            captured_offset_median_m=15.43,
            captured_offset_q90_m=20.65,
            captured_camera_count=491,
        ),
    )
    candidates = generate_trajectory_candidates(
        configuration.trajectory,
        centers,
        seed=policy.seed,
        stable_field_order=policy.stable_field_order,
    )

    first = select_budgeted_coverage(candidates, centers=centers, policy=policy)
    second = select_budgeted_coverage(candidates, centers=centers, policy=policy)

    proposal_count = sum(
        len(item.path.theta_radians) * (2 if index == 0 else 1)
        for index, item in enumerate(first)
    )
    assert len(first) >= policy.minimum_trajectory_groups
    assert proposal_count >= np.ceil(
        policy.minimum_accepted_frames / policy.minimum_accepted_fraction
    )
    assert proposal_count <= policy.proposal_budget
    assert {item.trajectory.center_kind for item in first} == set(OrbitCenterKind)
    assert [item.trajectory.trajectory_group_id for item in first] == [
        item.trajectory.trajectory_group_id for item in second
    ]
