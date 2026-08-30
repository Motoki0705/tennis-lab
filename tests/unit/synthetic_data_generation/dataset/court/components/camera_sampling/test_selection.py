from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.configuration import (
    CourtDatasetConfiguration,
    CourtTrajectoryPolicyV4,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling import (
    selection as selection_module,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    SafeCandidateExhaustionError,
    SelectedTrajectory,
    assign_group_shards,
    build_court_dataset_plan,
    select_budgeted_coverage,
    select_safe_budgeted_coverage,
    select_semantic_phase_budgeted_coverage,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.trajectory import (
    derive_orbit_centers,
    generate_trajectory_candidates,
    generate_trajectory_candidates_v4,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    CourtDatasetPlan,
    CourtDatasetPlanV2,
    CourtDatasetPlanV3,
    OrbitCenter,
    OrbitCenterKind,
    OrbitCoverageObjective,
    OrbitSamplingPolicy,
    OrbitStableFieldV4,
    OrbitViewSpecV2,
    PathFamilyV4,
    RequiredTrajectoryCoverage,
    TargetCourtResolutionPolicy,
    TrajectorySafetyEvaluation,
    TrajectorySemanticPhaseEvaluation,
    VerticalProfileV4,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
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


def _composed_configuration(selector: str) -> CourtDatasetConfiguration:
    config_root = Path("src/synthetic_data_generation/configs").resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
        config = compose(
            config_name="run_scene_pipeline",
            overrides=[f"dataset/court={selector}"],
        )
    raw = OmegaConf.to_container(config.dataset.court, resolve=True)
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
    assert isinstance(first, CourtDatasetPlan)
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
    assert {group.trajectory.curve_mode for group in first.groups} == set(
        configuration.trajectory.curve_modes
    )
    assert {view.target_mode for group in first.groups for view in group.views} == set(
        configuration.view.target_modes
    )
    assert {
        view.coverage_mode for group in first.groups for view in group.views
    } == set(configuration.view.coverage_modes)
    variant_group = next(group for group in first.groups if len(group.views) == 2)
    assert len({view.target_kind for view in variant_group.views}) == 2
    assert all(
        len(
            {
                sample.split
                for sample in first.samples
                if sample.trajectory_group_id == group_id
            }
        )
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


def test_v2_v3_plans_share_unchanged_sampling_and_per_sample_geometric_targets(
    captured_cameras: tuple[SceneCamera, ...],
    multi_court_layout: MultiCourtLayout,
    identity_metric_adapter: MetricSceneAdapter,
) -> None:
    configuration = _composed_configuration("v2")
    first = build_court_dataset_plan(
        scene_id="B00",
        profile="v2",
        cameras=captured_cameras,
        layout=multi_court_layout,
        configuration=configuration,
        metric_adapter=identity_metric_adapter,
    )
    v3_configuration = _composed_configuration("v3")
    second = build_court_dataset_plan(
        scene_id="B00",
        profile="v3",
        cameras=captured_cameras,
        layout=multi_court_layout,
        configuration=v3_configuration,
        metric_adapter=identity_metric_adapter,
    )

    assert isinstance(first, CourtDatasetPlanV2)
    assert isinstance(second, CourtDatasetPlanV3)
    assert first.schema_version is CourtDatasetSchemaVersion.V2
    assert second.schema_version is CourtDatasetSchemaVersion.V3
    assert first.to_dict()["schema"] == "canonical_court_orbit_plan_v2"
    assert second.to_dict()["schema"] == "canonical_court_orbit_plan_v3"
    assert first.policy.minimum_accepted_fraction == 0.9
    assert first.policy == second.policy
    assert first.groups == second.groups
    assert first.samples == second.samples
    assert all(len(group.views) == 1 for group in first.groups)
    assert {
        view.target_mode.value for group in first.groups for view in group.views
    } == {"court_center"}

    group_by_id = {group.trajectory_group_id: group for group in first.groups}
    targets_by_group: dict[str, list[str]] = {}
    court_centres = {
        court.court_instance_id: court.scene_from_court.apply(
            np.zeros((1, 3), dtype=np.float64)
        )[0]
        for court in multi_court_layout.courts
    }
    for sample in first.samples:
        group = group_by_id[sample.trajectory_group_id]
        target_id = sample.target_court.binding.court_instance_id
        targets_by_group.setdefault(group.trajectory_group_id, []).append(target_id)
        if (
            group.target_court_policy.mode
            is TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT
        ):
            assert target_id == group.trajectory.center_court_instance_id
        else:
            distances = {
                court_id: float(
                    np.linalg.norm(np.asarray(sample.camera_center_scene_m) - centre)
                )
                for court_id, centre in court_centres.items()
            }
            minimum = min(distances.values())
            expected = min(
                court_id
                for court_id, distance in distances.items()
                if distance <= minimum + 1.0e-9
            )
            assert target_id == expected
            assert (
                sample.target_court.camera_to_court_center_distance_m
                == pytest.approx(distances[target_id], abs=1.0e-9)
            )

        view = group.views[0]
        target_scene = sample.target_court.binding.scene_from_court.apply(
            np.asarray(((0.0, 0.0, view.look_at_height_m),), dtype=np.float64)
        )[0]
        camera_matrix = sample.camera.camera_to_scene.matrix()
        expected_forward = target_scene - camera_matrix[:3, 3]
        expected_forward /= np.linalg.norm(expected_forward)
        np.testing.assert_allclose(
            camera_matrix[:3, 2], expected_forward, atol=1.0e-9, rtol=0.0
        )

    complex_group_ids = {
        group.trajectory_group_id
        for group in first.groups
        if group.trajectory.center_kind is OrbitCenterKind.COMPLEX
    }
    assert any(
        len(set(targets_by_group[group_id])) > 1 for group_id in complex_group_ids
    )


def test_v4_hydra_selector_is_explicit_and_missing_support_fails_closed(
    captured_cameras: tuple[SceneCamera, ...],
    multi_court_layout: MultiCourtLayout,
    identity_metric_adapter: MetricSceneAdapter,
) -> None:
    default = _composed_configuration("train")
    configuration = _composed_configuration("v4")

    assert default.schema_version is CourtDatasetSchemaVersion.V1
    assert configuration.schema_version is CourtDatasetSchemaVersion.V4
    assert isinstance(configuration.trajectory, CourtTrajectoryPolicyV4)
    assert set(configuration.trajectory.shapes) == {
        PathFamilyV4.CIRCLE,
        PathFamilyV4.ELLIPSE,
        PathFamilyV4.ROUNDED_RECTANGLE,
    }
    assert set(configuration.trajectory.curve_modes) == {
        VerticalProfileV4.PLANAR,
        VerticalProfileV4.SINUSOIDAL_HEIGHT,
        VerticalProfileV4.RAISED_PHASES,
    }
    assert PathFamilyV4.FREE_SPACE_CYCLE not in configuration.trajectory.shapes
    assert set(configuration.sampling.stable_field_order) == set(OrbitStableFieldV4)
    assert configuration.support is not None
    assert configuration.benchmark_decision_id == configuration.support.decision_id

    with pytest.raises(ValueError, match="missing_support_capability"):
        build_court_dataset_plan(
            scene_id="B00",
            profile="v4",
            cameras=captured_cameras,
            layout=multi_court_layout,
            configuration=configuration,
            metric_adapter=identity_metric_adapter,
            points_scene=None,
        )


def test_v4_safe_selection_samples_each_candidate_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _composed_configuration("v4")
    assert isinstance(configuration.trajectory, CourtTrajectoryPolicyV4)
    policy = OrbitSamplingPolicy.from_configuration(configuration.sampling)
    centers = (
        OrbitCenter(
            center_kind=OrbitCenterKind.COMPLEX,
            court_instance_id=None,
            reference_court_instance_id="court-0",
            scene_from_center=RigidTransform.identity(),
            base_radius_m=20.0,
            captured_offset_median_m=18.0,
            captured_offset_q90_m=20.0,
            captured_camera_count=100,
        ),
        OrbitCenter(
            center_kind=OrbitCenterKind.COURT,
            court_instance_id="court-0",
            reference_court_instance_id="court-0",
            scene_from_center=RigidTransform.identity(),
            base_radius_m=20.0,
            captured_offset_median_m=18.0,
            captured_offset_q90_m=20.0,
            captured_camera_count=100,
        ),
    )
    candidates = generate_trajectory_candidates_v4(
        configuration.trajectory,
        centers,
        seed=policy.seed,
        stable_field_order=policy.stable_field_order,
    )[: policy.minimum_trajectory_groups]
    sampled_ids: list[str] = []
    real_sample = selection_module.sample_uniform_arc_length

    def counting_sample(candidate, center, sampling_policy):
        sampled_ids.append(candidate.trajectory_id)
        return real_sample(candidate, center, sampling_policy)

    def safe_evaluation(*, trajectory_id, trajectory_group_id, path, support_model):
        del support_model
        point_count = len(path.theta_radians)
        return TrajectorySafetyEvaluation(
            trajectory_id=trajectory_id,
            trajectory_group_id=trajectory_group_id,
            support_input_digest="a" * 64,
            safe=True,
            reasons=(),
            path_point_count=point_count,
            closed_segment_count=point_count,
            swept_sample_count=point_count,
            violating_point_indices=(),
            violating_segment_indices=(),
            minimum_support_margin_m=1.0,
            minimum_obstacle_clearance_m=1.0,
        )

    monkeypatch.setattr(selection_module, "sample_uniform_arc_length", counting_sample)
    monkeypatch.setattr(selection_module, "evaluate_trajectory_safety", safe_evaluation)

    def semantic_evaluations(items, **_kwargs):
        result = []
        for item in items:
            frame_count = len(item.path.theta_radians)
            for phase_index in range(6):
                result.append(
                    TrajectorySemanticPhaseEvaluation(
                        trajectory_id=item.trajectory.trajectory_id,
                        trajectory_group_id=item.trajectory.trajectory_group_id,
                        phase_index=phase_index,
                        phase_count=6,
                        view=OrbitViewSpecV2(
                            view_id=(
                                f"view-{item.trajectory.trajectory_group_id}-{phase_index}"
                            ),
                            target_kind=selection_module.OrbitTargetKind.COURT,
                            target_mode=selection_module.OrbitTargetMode.COURT_CENTER,
                            coverage_mode=configuration.view.coverage_modes[
                                phase_index % 3
                            ],
                            look_at_height_m=configuration.view.look_at_height_m[
                                phase_index % 2
                            ],
                            hfov_degrees=75.0,
                        ),
                        expected_frame_count=frame_count,
                        expected_valid_frame_count=frame_count,
                        semantically_viable=True,
                        rejection_counts=(),
                        disposition_digest=(f"{phase_index:x}" * 64)[:64],
                    )
                )
        return tuple(result)

    monkeypatch.setattr(
        selection_module,
        "_evaluate_semantic_phases",
        semantic_evaluations,
    )

    with pytest.raises(
        SafeCandidateExhaustionError,
        match="anchored_rounded_rectangle",
    ):
        select_safe_budgeted_coverage(
            candidates,
            centers=centers,
            policy=policy,
            support_model=object(),  # type: ignore[arg-type]
            cameras=(),
            layout=object(),  # type: ignore[arg-type]
            configuration=configuration,
        )

    assert sampled_ids == [candidate.trajectory_id for candidate in candidates]


def test_joint_semantic_phase_assignment_is_order_independent_and_phase_sensitive() -> (
    None
):
    items, evaluations, policy, required_coverage = _semantic_phase_fixture()

    errors: list[SafeCandidateExhaustionError] = []
    for ordered_items, ordered_evaluations in (
        (items, evaluations),
        (tuple(reversed(items)), tuple(reversed(evaluations))),
    ):
        with pytest.raises(SafeCandidateExhaustionError) as caught:
            select_semantic_phase_budgeted_coverage(
                ordered_items,
                semantic_phase_evaluations=ordered_evaluations,
                policy=policy,
                required_coverage=required_coverage,
            )
        errors.append(caught.value)
    assert errors[0].coverage_shortfall == errors[1].coverage_shortfall
    assert "minimum_anchored_rounded_rectangle_groups" in (
        errors[0].coverage_shortfall
    )


def test_joint_semantic_phase_assignment_fails_closed_when_too_few_candidates_are_compatible() -> (
    None
):
    items, evaluations, policy, required_coverage = _semantic_phase_fixture()
    compatible_groups = {
        item.trajectory.trajectory_group_id
        for item in items[: policy.minimum_trajectory_groups - 1]
    }
    incompatible = tuple(
        replace(
            evaluation,
            expected_valid_frame_count=0,
            semantically_viable=False,
            rejection_counts=(
                (
                    "insufficient_pre_render_semantic_coverage",
                    evaluation.expected_frame_count,
                ),
            ),
        )
        if evaluation.trajectory_group_id not in compatible_groups
        else evaluation
        for evaluation in evaluations
    )

    with pytest.raises(
        SafeCandidateExhaustionError,
        match="insufficient candidates have a compatible semantic phase",
    ):
        select_semantic_phase_budgeted_coverage(
            items,
            semantic_phase_evaluations=incompatible,
            policy=policy,
            required_coverage=required_coverage,
        )


def _semantic_phase_fixture() -> tuple[
    tuple[SelectedTrajectory, ...],
    tuple[TrajectorySemanticPhaseEvaluation, ...],
    OrbitSamplingPolicy,
    RequiredTrajectoryCoverage,
]:
    configuration = _composed_configuration("v4")
    assert isinstance(configuration.trajectory, CourtTrajectoryPolicyV4)
    assert configuration.required_coverage is not None
    policy = OrbitSamplingPolicy.from_configuration(configuration.sampling)
    centers = (
        OrbitCenter(
            center_kind=OrbitCenterKind.COMPLEX,
            court_instance_id=None,
            reference_court_instance_id="court-0",
            scene_from_center=RigidTransform.identity(),
            base_radius_m=20.0,
            captured_offset_median_m=18.0,
            captured_offset_q90_m=20.0,
            captured_camera_count=100,
        ),
        OrbitCenter(
            center_kind=OrbitCenterKind.COURT,
            court_instance_id="court-0",
            reference_court_instance_id="court-0",
            scene_from_center=RigidTransform.identity(),
            base_radius_m=20.0,
            captured_offset_median_m=18.0,
            captured_offset_q90_m=20.0,
            captured_camera_count=100,
        ),
    )
    candidates = generate_trajectory_candidates_v4(
        configuration.trajectory,
        centers,
        seed=policy.seed,
        stable_field_order=policy.stable_field_order,
    )[: policy.minimum_trajectory_groups]
    center_by_key = {center.key(): center for center in centers}
    items = tuple(
        SelectedTrajectory(
            trajectory=candidate,
            center=center_by_key[
                candidate.center_kind, candidate.center_court_instance_id
            ],
            path=sample_uniform_arc_length(
                candidate,
                center_by_key[
                    candidate.center_kind, candidate.center_court_instance_id
                ],
                policy,
            ),
        )
        for candidate in candidates
    )
    evaluations: list[TrajectorySemanticPhaseEvaluation] = []
    for candidate_index, item in enumerate(items):
        frame_count = len(item.path.theta_radians)
        compatible_phase = candidate_index % 6
        for phase_index in range(6):
            valid_count = frame_count if phase_index == compatible_phase else 0
            evaluations.append(
                TrajectorySemanticPhaseEvaluation(
                    trajectory_id=item.trajectory.trajectory_id,
                    trajectory_group_id=item.trajectory.trajectory_group_id,
                    phase_index=phase_index,
                    phase_count=6,
                    view=OrbitViewSpecV2(
                        view_id=(
                            f"view-{item.trajectory.trajectory_group_id}-{phase_index}"
                        ),
                        target_kind=selection_module.OrbitTargetKind.COURT,
                        target_mode=selection_module.OrbitTargetMode.COURT_CENTER,
                        coverage_mode=configuration.view.coverage_modes[
                            phase_index % 3
                        ],
                        look_at_height_m=configuration.view.look_at_height_m[
                            phase_index % 2
                        ],
                        hfov_degrees=75.0,
                    ),
                    expected_frame_count=frame_count,
                    expected_valid_frame_count=valid_count,
                    semantically_viable=valid_count > 0,
                    rejection_counts=(
                        ()
                        if valid_count
                        else (
                            (
                                "insufficient_pre_render_semantic_coverage",
                                frame_count,
                            ),
                        )
                    ),
                    disposition_digest=(f"{candidate_index:02x}{phase_index:02x}" * 16),
                )
            )
    return items, tuple(evaluations), policy, configuration.required_coverage


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
    permuted = select_budgeted_coverage(
        tuple(reversed(candidates)),
        centers=centers,
        policy=policy,
    )

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
    assert [item.trajectory.trajectory_group_id for item in first] == [
        item.trajectory.trajectory_group_id for item in permuted
    ]


def test_candidate_generation_consumes_every_configured_typed_mode_exactly(
    captured_cameras: tuple[SceneCamera, ...],
    multi_court_layout: MultiCourtLayout,
) -> None:
    configuration = _configuration()
    policy = OrbitSamplingPolicy.from_configuration(configuration.sampling)
    centers = derive_orbit_centers(captured_cameras, multi_court_layout)

    candidates = generate_trajectory_candidates(
        configuration.trajectory,
        centers,
        seed=policy.seed,
        stable_field_order=policy.stable_field_order,
    )

    assert {candidate.shape for candidate in candidates} == set(
        configuration.trajectory.shapes
    )
    assert {candidate.center_kind for candidate in candidates} == set(
        configuration.trajectory.center_kinds
    )
    assert {candidate.curve_mode for candidate in candidates} == set(
        configuration.trajectory.curve_modes
    )


def test_distinct_coverage_objectives_change_greedy_selection_behavior(
    captured_cameras: tuple[SceneCamera, ...],
    multi_court_layout: MultiCourtLayout,
) -> None:
    configuration = _configuration()
    policy = OrbitSamplingPolicy.from_configuration(configuration.sampling)
    centers = derive_orbit_centers(captured_cameras, multi_court_layout)
    candidates = generate_trajectory_candidates(
        configuration.trajectory,
        centers,
        seed=policy.seed,
        stable_field_order=policy.stable_field_order,
    )
    coverage_first = replace(
        policy,
        coverage_objective=(OrbitCoverageObjective.COVERAGE_MODE,),
    )
    trajectory_first = replace(
        policy,
        coverage_objective=(OrbitCoverageObjective.TRAJECTORY_GROUP,),
    )

    coverage_selected = select_budgeted_coverage(
        candidates,
        centers=centers,
        policy=coverage_first,
    )
    trajectory_selected = select_budgeted_coverage(
        candidates,
        centers=centers,
        policy=trajectory_first,
    )

    assert tuple(
        item.trajectory.trajectory_group_id for item in coverage_selected
    ) != tuple(item.trajectory.trajectory_group_id for item in trajectory_selected)


def test_selector_rejects_duplicate_and_short_candidate_inventories(
    captured_cameras: tuple[SceneCamera, ...],
    multi_court_layout: MultiCourtLayout,
) -> None:
    configuration = _configuration()
    policy = OrbitSamplingPolicy.from_configuration(configuration.sampling)
    centers = derive_orbit_centers(captured_cameras, multi_court_layout)
    candidates = generate_trajectory_candidates(
        configuration.trajectory,
        centers,
        seed=policy.seed,
        stable_field_order=policy.stable_field_order,
    )
    duplicate = replace(
        candidates[0],
        trajectory_id="duplicate-trajectory",
        trajectory_group_id="duplicate-group",
    )

    with pytest.raises(ValueError, match="Duplicate typed trajectory"):
        select_budgeted_coverage(
            (*candidates, duplicate),
            centers=centers,
            policy=policy,
        )
    with pytest.raises(ValueError, match="minimum trajectory groups"):
        select_budgeted_coverage(
            candidates[: policy.minimum_trajectory_groups - 1],
            centers=centers,
            policy=policy,
        )


def test_sampling_policy_rejects_budget_overflow_and_impossible_shortage() -> None:
    policy = OrbitSamplingPolicy.from_configuration(_configuration().sampling)

    with pytest.raises(ValueError, match="must not exceed 5,000"):
        replace(policy, proposal_budget=5_001)
    with pytest.raises(ValueError, match="cannot satisfy accepted frames"):
        replace(policy, proposal_budget=2_000)


def test_group_shards_are_deterministic_whole_and_batch_bounded() -> None:
    counts = {
        "group-a": 300,
        "group-b": 250,
        "group-c": 200,
        "group-d": 150,
    }

    first = assign_group_shards(
        counts,
        shard_count=2,
        seed=11,
        maximum_shard_samples=500,
    )
    second = assign_group_shards(
        dict(reversed(tuple(counts.items()))),
        shard_count=2,
        seed=11,
        maximum_shard_samples=500,
    )
    loads = {
        shard_id: sum(
            counts[group_id]
            for group_id, assigned_shard in first.items()
            if assigned_shard == shard_id
        )
        for shard_id in set(first.values())
    }

    assert first == second
    assert set(first) == set(counts)
    assert max(loads.values()) <= 500
    with pytest.raises(ValueError, match="cannot satisfy"):
        assign_group_shards(
            {"group-a": 4, "group-b": 4, "group-c": 4},
            shard_count=2,
            seed=11,
            maximum_shard_samples=6,
        )
