"""Deterministic budgeted coverage selection and Court render planning."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.configuration import (
    CourtDatasetConfiguration,
    CourtTrajectoryPolicyV4,
)
from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court.components.camera_sampling.anchored_paths import (
    generate_anchored_rounded_rectangle_candidates,
    validate_anchored_trajectory_provenance,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.constructive_paths import (
    generate_free_space_cycle_candidates,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.support import (
    TrajectorySupportModel,
    build_trajectory_support_model,
    evaluate_trajectory_safety,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.targeting import (
    resolve_target_court,
    resolved_court_look_at_scene,
    target_court_policy_for_trajectory,
    validate_camera_looks_at_resolved_court,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.trajectory import (
    derive_orbit_centers,
    generate_trajectory_candidates,
    generate_trajectory_candidates_v4,
    identify_trajectory_candidates_v4,
    trajectory_field_value,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    CourtDatasetPlan,
    CourtDatasetPlanAny,
    CourtDatasetPlanV2,
    CourtDatasetPlanV3,
    CourtDatasetPlanV4,
    DatasetSplit,
    OrbitCenter,
    OrbitCoverageMode,
    OrbitCoverageObjective,
    OrbitPathSamples,
    OrbitSamplingPolicy,
    OrbitStableField,
    OrbitStableFieldV4,
    OrbitTargetKind,
    OrbitTargetMode,
    OrbitTrajectorySpec,
    OrbitTrajectorySpecV4,
    OrbitViewSpec,
    OrbitViewSpecV2,
    PathConstructorV4,
    PathFamilyV4,
    PlannedCourtSample,
    PlannedCourtSampleV2,
    PlannedCourtSampleV4,
    RequiredTrajectoryCoverage,
    ResolvedTargetCourtV2,
    SelectedTrajectoryCoverage,
    TrajectoryGroupPlan,
    TrajectoryGroupPlanV2,
    TrajectoryGroupPlanV4,
    TrajectorySafetyEvaluation,
    TrajectorySafetyReason,
    TrajectorySemanticPhaseEvaluation,
    VerticalProfileV4,
    build_selected_coverage_from_records,
    required_coverage_shortfall,
    semantic_phase_inventory_digest,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.dataset.court.semantic_pre_render import (
    CourtSemanticFrameDisposition,
    court_semantic_phase_disposition_digest,
    evaluate_court_semantic_pre_render,
)
from src.synthetic_data_generation.dataset.court_assignment import CourtAssignment
from src.synthetic_data_generation.scene_contract import (
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.utils.schema.court import HALF_LENGTH


@dataclass(frozen=True, slots=True)
class SelectedTrajectory:
    """One selected path and its already validated uniform samples."""

    trajectory: OrbitTrajectorySpec
    center: OrbitCenter
    path: OrbitPathSamples


@dataclass(frozen=True, slots=True)
class SafeSelectionResult:
    """Complete candidate decisions plus the safe-only selected inventory."""

    selected: tuple[SelectedTrajectory, ...]
    evaluations: tuple[TrajectorySafetyEvaluation, ...]
    semantic_phase_evaluations: tuple[TrajectorySemanticPhaseEvaluation, ...]
    selected_semantic_phases: tuple[TrajectorySemanticPhaseEvaluation, ...]
    selected_coverage: SelectedTrajectoryCoverage
    required_coverage_shortfall: tuple[str, ...]
    optional_candidate_coverage_shortfall: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticPhaseSelection:
    """One geometry-safe path paired with its frozen semantic-view phase."""

    selected: SelectedTrajectory
    semantic_phase: TrajectorySemanticPhaseEvaluation

    def __post_init__(self) -> None:
        trajectory = self.selected.trajectory
        phase = self.semantic_phase
        if (
            phase.trajectory_id != trajectory.trajectory_id
            or phase.trajectory_group_id != trajectory.trajectory_group_id
            or phase.expected_frame_count != len(self.selected.path.theta_radians)
            or not phase.semantically_viable
        ):
            raise ValueError(
                "Semantic-phase selection disagrees with its sampled trajectory."
            )


class SafeCandidateExhaustionError(ValueError):
    """Structured V4 exhaustion without a legacy-orbit fallback."""

    reason = TrajectorySafetyReason.SAFE_CANDIDATE_EXHAUSTION

    def __init__(self, detail: str, *, coverage_shortfall: Sequence[str]) -> None:
        self.coverage_shortfall = tuple(sorted(set(coverage_shortfall)))
        super().__init__(
            f"{self.reason.value}: {detail}; coverage_shortfall={list(self.coverage_shortfall)}"
        )


_OBJECTIVE_FIELDS: Mapping[
    OrbitCoverageObjective,
    tuple[OrbitStableField, ...],
] = {
    # Path footprint controls framing coverage before a renderer is invoked.
    OrbitCoverageObjective.COVERAGE_MODE: (
        OrbitStableField.SHAPE,
        OrbitStableField.RADIUS_SCALE,
        OrbitStableField.AXIS_RATIO,
        OrbitStableField.ORIENTATION_DEGREES,
    ),
    # Centre/elevation are the pre-render geometric visibility authorities.
    OrbitCoverageObjective.SEMANTIC_VISIBILITY: (
        OrbitStableField.CENTER_KIND,
        OrbitStableField.BASE_HEIGHT_M,
    ),
    # Vertical profile distinguishes the motion of one camera-centre path.
    OrbitCoverageObjective.TRAJECTORY_GROUP: (
        OrbitStableField.VERTICAL_MODULATION_M,
        OrbitStableField.CURVE_MODE,
    ),
}

StableFieldAny: TypeAlias = OrbitStableField | OrbitStableFieldV4


def build_court_dataset_plan(
    *,
    scene_id: str,
    profile: str,
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
    configuration: CourtDatasetConfiguration,
    metric_adapter: MetricSceneAdapter,
    points_scene: NDArray[np.floating] | None = None,
) -> CourtDatasetPlanAny:
    """Plan in metres after adapting captured public NHT-export cameras."""
    policy = OrbitSamplingPolicy.from_configuration(configuration.sampling)
    nht_camera_tuple = tuple(cameras)
    if not nht_camera_tuple:
        raise ValueError("Court dataset planning requires captured cameras.")
    camera_tuple = _metric_cameras_from_nht_export(
        nht_camera_tuple,
        metric_adapter=metric_adapter,
    )
    centers = derive_orbit_centers(camera_tuple, layout)
    support_model: TrajectorySupportModel | None = None
    safe_result: SafeSelectionResult | None = None
    candidates: tuple[OrbitTrajectorySpec, ...]
    if configuration.schema_version is CourtDatasetSchemaVersion.V4:
        if not isinstance(configuration.trajectory, CourtTrajectoryPolicyV4):
            raise TypeError("Court V4 planning requires CourtTrajectoryPolicyV4.")
        if configuration.support is None or points_scene is None:
            raise ValueError(
                f"{TrajectorySafetyReason.MISSING_SUPPORT_CAPABILITY.value}: "
                "V4 planning requires public points and an explicit support policy"
            )
        point_array = np.asarray(points_scene)
        if point_array.ndim != 2 or point_array.shape[1] not in (3, 6):
            raise ValueError(
                f"{TrajectorySafetyReason.MISSING_SUPPORT_CAPABILITY.value}: "
                "public points must have shape (N,3) or (N,6)"
            )
        metric_points = metric_adapter.metric_from_nht_points(point_array[:, :3])
        support_model = build_trajectory_support_model(
            cameras=camera_tuple,
            points_scene_m=metric_points,
            policy=configuration.support,
        )
        analytic_candidates = generate_trajectory_candidates_v4(
            configuration.trajectory,
            centers,
            seed=policy.seed,
            stable_field_order=policy.stable_field_order,
        )
        constructive_candidates = generate_free_space_cycle_candidates(
            support_model=support_model,
            centers=tuple(centers),
            policy=configuration.trajectory,
        )
        anchored_candidates = generate_anchored_rounded_rectangle_candidates(
            support_model=support_model,
            cameras=camera_tuple,
            centers=centers,
            policy=configuration.trajectory,
            seed=policy.seed,
        )
        candidates = identify_trajectory_candidates_v4(
            (*analytic_candidates, *constructive_candidates, *anchored_candidates),
            stable_field_order=policy.stable_field_order,
        )
    else:
        candidates = generate_trajectory_candidates(
            configuration.trajectory,
            centers,
            seed=policy.seed,
            stable_field_order=policy.stable_field_order,
        )
    first_group_view_count = len(
        _target_modes_for_group(group_index=0, configuration=configuration)
    )
    if support_model is None:
        selected = select_budgeted_coverage(
            candidates,
            centers=centers,
            policy=policy,
            first_group_view_count=first_group_view_count,
        )
    else:
        safe_result = select_safe_budgeted_coverage(
            tuple(
                candidate
                for candidate in candidates
                if isinstance(candidate, OrbitTrajectorySpecV4)
            ),
            centers=centers,
            policy=policy,
            support_model=support_model,
            cameras=camera_tuple,
            layout=layout,
            configuration=configuration,
        )
        selected = safe_result.selected
    split_by_group = assign_group_disjoint_splits(
        tuple(item.trajectory.trajectory_group_id for item in selected),
        fractions=policy.split_fractions,
        seed=policy.seed,
    )
    shard_by_group = assign_group_shards(
        {
            item.trajectory.trajectory_group_id: len(item.path.theta_radians)
            * len(
                _target_modes_for_group(
                    group_index=index,
                    configuration=configuration,
                )
            )
            for index, item in enumerate(selected)
        },
        shard_count=policy.shard_count,
        seed=policy.seed,
        maximum_shard_samples=configuration.performance.maximum_batch_frames,
    )
    if configuration.schema_version is CourtDatasetSchemaVersion.V4:
        if support_model is None or safe_result is None:
            raise RuntimeError("V4 support selection was not constructed.")
        return _build_v4_plan(
            scene_id=scene_id,
            profile=profile,
            selected=selected,
            split_by_group=split_by_group,
            shard_by_group=shard_by_group,
            cameras=camera_tuple,
            layout=layout,
            centers=centers,
            configuration=configuration,
            policy=policy,
            support_model=support_model,
            safe_result=safe_result,
        )
    if configuration.schema_version in (
        CourtDatasetSchemaVersion.V2,
        CourtDatasetSchemaVersion.V3,
    ):
        return _build_v2_plan(
            scene_id=scene_id,
            profile=profile,
            selected=selected,
            split_by_group=split_by_group,
            shard_by_group=shard_by_group,
            cameras=camera_tuple,
            layout=layout,
            centers=centers,
            configuration=configuration,
            policy=policy,
        )
    if configuration.schema_version is not CourtDatasetSchemaVersion.V1:
        raise TypeError("Unsupported Court dataset schema version.")
    assignments = assign_court_targets_for_groups(
        selected,
        split_by_group=split_by_group,
        layout=layout,
        seed=policy.seed,
    )
    assignment_by_group = {
        assignment.scene_id: assignment for assignment in assignments
    }
    groups: list[TrajectoryGroupPlan] = []
    paths_by_group: dict[str, OrbitPathSamples] = {}
    for group_index, selected_item in enumerate(selected):
        trajectory = selected_item.trajectory
        group_id = trajectory.trajectory_group_id
        assignment = assignment_by_group[group_id]
        court = layout.court(assignment.court_instance_id)
        binding = TargetCourtBinding(
            court_instance_id=court.court_instance_id,
            candidate_id=court.candidate_id,
            scene_from_court=court.scene_from_court,
            selection_seed=assignment.selection_seed,
        )
        views = _views_for_group(
            group_index=group_index,
            group_id=group_id,
            target_court_instance_id=court.court_instance_id,
            configuration=configuration,
        )
        groups.append(
            TrajectoryGroupPlan(
                trajectory=trajectory,
                center=selected_item.center,
                views=views,
                split=split_by_group[group_id],
                shard_id=shard_by_group[group_id],
                target_court=binding,
                sample_count=len(selected_item.path.theta_radians),
                maximum_adjacent_step_m=float(
                    np.max(selected_item.path.adjacent_steps_m)
                ),
                total_arc_length_m=selected_item.path.total_arc_length_m,
            )
        )
        paths_by_group[group_id] = selected_item.path
    generated_target_modes = {
        view.target_mode for group in groups for view in group.views
    }
    if generated_target_modes != set(configuration.view.target_modes):
        raise ValueError(
            "Court view generation did not consume configured targets exactly."
        )
    generated_coverage_modes = {
        view.coverage_mode for group in groups for view in group.views
    }
    if generated_coverage_modes != set(configuration.view.coverage_modes):
        raise ValueError(
            "Court view generation did not consume configured coverage modes exactly."
        )
    groups.sort(key=lambda item: item.trajectory_group_id)
    samples = _plan_samples(
        groups=groups,
        paths_by_group=paths_by_group,
        cameras=camera_tuple,
        layout=layout,
        centers=centers,
    )
    return CourtDatasetPlan(
        scene_id=scene_id,
        profile=profile,
        policy=policy,
        groups=tuple(groups),
        samples=samples,
    )


def _build_v4_plan(
    *,
    scene_id: str,
    profile: str,
    selected: Sequence[SelectedTrajectory],
    split_by_group: Mapping[str, DatasetSplit],
    shard_by_group: Mapping[str, str],
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
    centers: Sequence[OrbitCenter],
    configuration: CourtDatasetConfiguration,
    policy: OrbitSamplingPolicy,
    support_model: TrajectorySupportModel,
    safe_result: SafeSelectionResult,
) -> CourtDatasetPlanV4:
    """Build a V4 plan whose selected groups carry their safety authority."""
    evaluation_by_group = {
        item.trajectory_group_id: item for item in safe_result.evaluations
    }
    phase_by_group = {
        item.trajectory_group_id: item for item in safe_result.selected_semantic_phases
    }
    groups: list[TrajectoryGroupPlanV4] = []
    paths_by_group: dict[str, OrbitPathSamples] = {}
    for selected_item in selected:
        trajectory = selected_item.trajectory
        if not isinstance(trajectory, OrbitTrajectorySpecV4):
            raise TypeError("V4 selection returned a legacy trajectory contract.")
        group_id = trajectory.trajectory_group_id
        semantic_phase = phase_by_group[group_id]
        groups.append(
            TrajectoryGroupPlanV4(
                trajectory=trajectory,
                center=selected_item.center,
                views=(semantic_phase.view,),
                split=split_by_group[group_id],
                shard_id=shard_by_group[group_id],
                target_court_policy=target_court_policy_for_trajectory(trajectory),
                sample_count=len(selected_item.path.theta_radians),
                maximum_adjacent_step_m=float(
                    np.max(selected_item.path.adjacent_steps_m)
                ),
                total_arc_length_m=selected_item.path.total_arc_length_m,
                safety_evaluation=evaluation_by_group[group_id],
                semantic_phase_evaluation=semantic_phase,
            )
        )
        paths_by_group[group_id] = selected_item.path
    groups.sort(key=lambda item: item.trajectory_group_id)
    base_samples = _plan_samples_v2(
        groups=groups,
        paths_by_group=paths_by_group,
        cameras=cameras,
        layout=layout,
        centers=centers,
        selection_seed=policy.seed,
    )
    samples = tuple(
        PlannedCourtSampleV4(
            sample_index=sample.sample_index,
            sample_id=sample.sample_id,
            trajectory_group_id=sample.trajectory_group_id,
            trajectory_id=sample.trajectory_id,
            view_id=sample.view_id,
            trajectory_frame_index=sample.trajectory_frame_index,
            split=sample.split,
            shard_id=sample.shard_id,
            camera_center_scene_m=sample.camera_center_scene_m,
            camera=sample.camera,
            target_court=sample.target_court,
            safety_support_input_digest=support_model.summary.input_digest,
            semantic_phase_index=phase_by_group[sample.trajectory_group_id].phase_index,
            semantic_phase_disposition_digest=phase_by_group[
                sample.trajectory_group_id
            ].disposition_digest,
        )
        for sample in base_samples
    )
    if {view.target_mode for group in groups for view in group.views} != set(
        configuration.view.target_modes
    ):
        raise ValueError("V4 view generation omitted a configured target mode.")
    if {view.coverage_mode for group in groups for view in group.views} != set(
        configuration.view.coverage_modes
    ):
        raise ValueError("V4 view generation omitted a configured coverage mode.")
    if configuration.support is None:
        raise TypeError("V4 configuration support policy is missing.")
    if configuration.required_coverage is None:
        raise TypeError("V4 configuration required coverage is missing.")
    return CourtDatasetPlanV4(
        scene_id=scene_id,
        profile=profile,
        policy=policy,
        groups=tuple(groups),
        samples=samples,
        support_policy=configuration.support,
        support_summary=support_model.summary,
        candidate_safety_evaluations=safe_result.evaluations,
        candidate_semantic_phase_evaluations=(safe_result.semantic_phase_evaluations),
        semantic_phase_inventory_digest=semantic_phase_inventory_digest(
            safe_result.semantic_phase_evaluations
        ),
        required_coverage=configuration.required_coverage,
        selected_coverage=safe_result.selected_coverage,
        required_coverage_shortfall=safe_result.required_coverage_shortfall,
        optional_candidate_coverage_shortfall=(
            safe_result.optional_candidate_coverage_shortfall
        ),
    )


def _build_v2_plan(
    *,
    scene_id: str,
    profile: str,
    selected: Sequence[SelectedTrajectory],
    split_by_group: Mapping[str, DatasetSplit],
    shard_by_group: Mapping[str, str],
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
    centers: Sequence[OrbitCenter],
    configuration: CourtDatasetConfiguration,
    policy: OrbitSamplingPolicy,
) -> CourtDatasetPlanV2 | CourtDatasetPlanV3:
    """Build V2/V3 groups, then resolve every target inside its sample loop."""
    groups: list[TrajectoryGroupPlanV2] = []
    paths_by_group: dict[str, OrbitPathSamples] = {}
    for group_index, selected_item in enumerate(selected):
        trajectory = selected_item.trajectory
        group_id = trajectory.trajectory_group_id
        groups.append(
            TrajectoryGroupPlanV2(
                trajectory=trajectory,
                center=selected_item.center,
                views=_views_for_group_v2(
                    group_index=group_index,
                    group_id=group_id,
                    configuration=configuration,
                ),
                split=split_by_group[group_id],
                shard_id=shard_by_group[group_id],
                target_court_policy=target_court_policy_for_trajectory(trajectory),
                sample_count=len(selected_item.path.theta_radians),
                maximum_adjacent_step_m=float(
                    np.max(selected_item.path.adjacent_steps_m)
                ),
                total_arc_length_m=selected_item.path.total_arc_length_m,
            )
        )
        paths_by_group[group_id] = selected_item.path
    generated_target_modes = {
        view.target_mode for group in groups for view in group.views
    }
    if generated_target_modes != set(configuration.view.target_modes):
        raise ValueError(
            "Court view generation did not consume configured targets exactly."
        )
    generated_coverage_modes = {
        view.coverage_mode for group in groups for view in group.views
    }
    if generated_coverage_modes != set(configuration.view.coverage_modes):
        raise ValueError(
            "Court view generation did not consume configured coverage modes exactly."
        )
    groups.sort(key=lambda item: item.trajectory_group_id)
    samples = _plan_samples_v2(
        groups=groups,
        paths_by_group=paths_by_group,
        cameras=cameras,
        layout=layout,
        centers=centers,
        selection_seed=policy.seed,
    )
    plan_type: type[CourtDatasetPlanV2]
    if configuration.schema_version is CourtDatasetSchemaVersion.V2:
        plan_type = CourtDatasetPlanV2
    elif configuration.schema_version is CourtDatasetSchemaVersion.V3:
        plan_type = CourtDatasetPlanV3
    else:
        raise TypeError("V2/V3 plan builder received an unsupported schema version.")
    return plan_type(
        scene_id=scene_id,
        profile=profile,
        policy=policy,
        groups=tuple(groups),
        samples=samples,
    )


def _metric_cameras_from_nht_export(
    cameras: Sequence[SceneCamera],
    *,
    metric_adapter: MetricSceneAdapter,
) -> tuple[SceneCamera, ...]:
    """Convert captured public-export poses to the metric planning frame."""
    camera_tuple = tuple(cameras)
    if not camera_tuple:
        raise ValueError("Court camera conversion requires captured NHT cameras.")
    return tuple(
        SceneCamera(
            camera_id=camera.camera_id,
            source_frame_index=camera.source_frame_index,
            width=camera.width,
            height=camera.height,
            intrinsics=camera.intrinsics,
            camera_to_scene=metric_adapter.metric_from_nht_camera(
                camera.camera_to_scene
            ),
            image_path=camera.image_path,
        )
        for camera in camera_tuple
    )


def select_budgeted_coverage(
    candidates: Sequence[OrbitTrajectorySpec],
    *,
    centers: Sequence[OrbitCenter],
    policy: OrbitSamplingPolicy,
    first_group_view_count: int = 2,
    _pre_sampled: Sequence[SelectedTrajectory] | None = None,
) -> tuple[SelectedTrajectory, ...]:
    """Maximize ordered typed token families within one explicit frame budget.

    Each configured objective contributes a lexicographic ``(novelty, balance)``
    score in declared order. Stable typed fields, the explicit seed, and the
    canonical semantic identity then resolve ties deterministically.
    """
    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        raise ValueError("Trajectory candidate inventory must not be empty.")
    if len({candidate.trajectory_id for candidate in candidate_tuple}) != len(
        candidate_tuple
    ) or len({candidate.trajectory_group_id for candidate in candidate_tuple}) != len(
        candidate_tuple
    ):
        raise ValueError("Trajectory candidate IDs must be unique.")
    if len({candidate.semantic_key() for candidate in candidate_tuple}) != len(
        candidate_tuple
    ):
        raise ValueError("Duplicate typed trajectory candidates are forbidden.")
    if len(candidate_tuple) < policy.minimum_trajectory_groups:
        raise ValueError(
            "Candidate inventory cannot satisfy minimum trajectory groups."
        )
    if (
        isinstance(first_group_view_count, bool)
        or not isinstance(first_group_view_count, int)
        or first_group_view_count < 1
    ):
        raise ValueError("first_group_view_count must be a positive integer.")
    center_by_key = {center.key(): center for center in centers}
    if len(center_by_key) != len(tuple(centers)):
        raise ValueError("Resolved orbit centres must be unique.")
    resolved: list[SelectedTrajectory]
    if _pre_sampled is None:
        resolved = []
        for candidate in candidate_tuple:
            try:
                center = center_by_key[
                    candidate.center_kind,
                    candidate.center_court_instance_id,
                ]
            except KeyError as error:
                raise ValueError(
                    "Candidate references an unknown orbit centre."
                ) from error
            resolved.append(
                SelectedTrajectory(
                    trajectory=candidate,
                    center=center,
                    path=sample_uniform_arc_length(candidate, center, policy),
                )
            )
    else:
        resolved = list(_pre_sampled)
        if tuple(item.trajectory for item in resolved) != candidate_tuple or any(
            center_by_key.get(item.center.key()) != item.center
            or item.path.trajectory_group_id != item.trajectory.trajectory_group_id
            for item in resolved
        ):
            raise ValueError("Pre-sampled candidate paths disagree with the inventory.")
    all_tokens = set().union(
        *(
            _objective_tokens(item.trajectory, policy.coverage_objective)
            for item in resolved
        )
    )
    required_proposals = math.ceil(
        policy.minimum_accepted_frames / policy.minimum_accepted_fraction
    )
    canonical_order = sorted(resolved, key=_canonical_candidate_identity)
    stable_order = sorted(
        resolved,
        key=lambda item: (
            tuple(
                repr(trajectory_field_value(item.trajectory, field))
                for field in policy.stable_field_order
            ),
            _canonical_candidate_identity(item),
        ),
    )
    stable_rank = {
        item.trajectory.trajectory_group_id: len(stable_order) - rank
        for rank, item in enumerate(stable_order)
    }
    seeded_order = list(canonical_order)
    random.Random(policy.seed).shuffle(seeded_order)
    seeded_rank = {
        item.trajectory.trajectory_group_id: rank
        for rank, item in enumerate(seeded_order, start=1)
    }
    canonical_rank = {
        item.trajectory.trajectory_group_id: len(canonical_order) - rank
        for rank, item in enumerate(canonical_order)
    }
    remaining = list(resolved)
    selected: list[SelectedTrajectory] = []
    covered: set[tuple[OrbitCoverageObjective, StableFieldAny, object]] = set()
    token_counts: Counter[tuple[OrbitCoverageObjective, StableFieldAny, object]] = (
        Counter()
    )
    stable_token_counts: Counter[tuple[StableFieldAny, object]] = Counter()
    proposal_count = 0
    while remaining:
        requirements_met = (
            len(selected) >= policy.minimum_trajectory_groups
            and proposal_count >= required_proposals
            and covered == all_tokens
        )
        if requirements_met:
            break
        feasible: list[tuple[SelectedTrajectory, int]] = []
        for item in remaining:
            view_count = first_group_view_count if not selected else 1
            cost = len(item.path.theta_radians) * view_count
            remaining_group_count = max(
                0,
                policy.minimum_trajectory_groups - len(selected) - 1,
            )
            completion_costs = sorted(
                len(other.path.theta_radians)
                for other in remaining
                if other is not item
            )
            if len(completion_costs) < remaining_group_count:
                continue
            minimum_completion_cost = sum(completion_costs[:remaining_group_count])
            if (
                proposal_count + cost + minimum_completion_cost
                <= policy.proposal_budget
            ):
                feasible.append((item, cost))
        if not feasible:
            break

        def score(entry: tuple[SelectedTrajectory, int]) -> tuple[int, ...]:
            item, _cost = entry
            trajectory = item.trajectory
            parts: list[int] = []
            for objective in policy.coverage_objective:
                family_tokens = _objective_tokens(trajectory, (objective,))
                parts.extend(
                    (
                        len(family_tokens - covered),
                        -sum(token_counts[token] for token in family_tokens),
                    )
                )
            for field in policy.stable_field_order:
                stable_token = (field, trajectory_field_value(trajectory, field))
                parts.extend(
                    (
                        int(stable_token_counts[stable_token] == 0),
                        -stable_token_counts[stable_token],
                    )
                )
            group_id = trajectory.trajectory_group_id
            parts.extend(
                (
                    stable_rank[group_id],
                    seeded_rank[group_id],
                    canonical_rank[group_id],
                )
            )
            return tuple(parts)

        chosen, cost = max(feasible, key=score)
        selected.append(chosen)
        proposal_count += cost
        tokens = _objective_tokens(
            chosen.trajectory,
            policy.coverage_objective,
        )
        covered.update(tokens)
        token_counts.update(tokens)
        stable_token_counts.update(
            (field, trajectory_field_value(chosen.trajectory, field))
            for field in policy.stable_field_order
        )
        remaining.remove(chosen)
    if len(selected) < policy.minimum_trajectory_groups:
        raise ValueError(
            "Candidate inventory cannot satisfy minimum trajectory groups within budget."
        )
    if proposal_count < required_proposals:
        raise ValueError(
            "Candidate inventory cannot satisfy the accepted-frame objective within budget."
        )
    if covered != all_tokens:
        missing = sorted(repr(token) for token in all_tokens - covered)
        raise ValueError(
            f"Candidate budget cannot cover all typed field values: {missing}."
        )
    if proposal_count > policy.proposal_budget:
        raise ValueError("Coverage selector exceeded proposal_budget.")
    if not any(len(item.path.theta_radians) > 0 for item in selected):
        raise ValueError("Coverage selector produced no samples.")
    return tuple(selected)


def _semantic_phase_count(configuration: CourtDatasetConfiguration) -> int:
    """Return the deterministic period of every configured V4 view schedule."""
    return math.lcm(
        len(configuration.view.target_modes),
        len(configuration.view.coverage_modes),
        len(configuration.view.look_at_height_m),
    )


def _view_for_semantic_phase(
    *,
    trajectory_group_id: str,
    phase_index: int,
    phase_count: int,
    configuration: CourtDatasetConfiguration,
) -> OrbitViewSpecV2:
    """Resolve one explicit phase without consulting selected-group order."""
    if not 0 <= phase_index < phase_count:
        raise ValueError("Semantic phase index is outside its configured period.")
    target = configuration.view.target_modes[
        phase_index % len(configuration.view.target_modes)
    ]
    coverage = configuration.view.coverage_modes[
        phase_index % len(configuration.view.coverage_modes)
    ]
    look_at_height_m = configuration.view.look_at_height_m[
        phase_index % len(configuration.view.look_at_height_m)
    ]
    low_hfov, high_hfov = configuration.view.hfov_degrees
    hfov_degrees = {
        OrbitCoverageMode.FULL: high_hfov,
        OrbitCoverageMode.NEAR_FULL: (low_hfov + high_hfov) / 2.0,
        OrbitCoverageMode.PARTIAL: low_hfov,
    }[coverage]
    return OrbitViewSpecV2(
        view_id=(f"view-{trajectory_group_id}-semantic-phase-{phase_index:02d}"),
        target_kind=target.target_kind,
        target_mode=target,
        coverage_mode=coverage,
        look_at_height_m=look_at_height_m,
        hfov_degrees=hfov_degrees,
    )


def _evaluate_semantic_phases(
    safe_items: Sequence[SelectedTrajectory],
    *,
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
    centers: Sequence[OrbitCenter],
    configuration: CourtDatasetConfiguration,
    selection_seed: int,
) -> tuple[TrajectorySemanticPhaseEvaluation, ...]:
    """Apply the public pre-render oracle to each safe candidate/phase pair."""
    if configuration.schema_version is not CourtDatasetSchemaVersion.V4:
        raise TypeError("Semantic phase evaluation is a strict V4 operation.")
    phase_count = _semantic_phase_count(configuration)
    evaluations: list[TrajectorySemanticPhaseEvaluation] = []
    for item in sorted(
        safe_items,
        key=lambda value: value.trajectory.trajectory_group_id,
    ):
        trajectory = item.trajectory
        if not isinstance(trajectory, OrbitTrajectorySpecV4):
            raise TypeError("Semantic phase evaluation requires a V4 trajectory.")
        for phase_index in range(phase_count):
            view = _view_for_semantic_phase(
                trajectory_group_id=trajectory.trajectory_group_id,
                phase_index=phase_index,
                phase_count=phase_count,
                configuration=configuration,
            )
            provisional_group = TrajectoryGroupPlanV2(
                trajectory=trajectory,
                center=item.center,
                views=(view,),
                split=DatasetSplit.TRAIN,
                shard_id="semantic-phase-evaluation",
                target_court_policy=target_court_policy_for_trajectory(trajectory),
                sample_count=len(item.path.theta_radians),
                maximum_adjacent_step_m=float(np.max(item.path.adjacent_steps_m)),
                total_arc_length_m=item.path.total_arc_length_m,
            )
            samples = _plan_samples_v2(
                groups=(provisional_group,),
                paths_by_group={trajectory.trajectory_group_id: item.path},
                cameras=cameras,
                layout=layout,
                centers=centers,
                selection_seed=selection_seed,
            )
            dispositions: list[CourtSemanticFrameDisposition] = []
            rejection_counts: Counter[str] = Counter()
            valid_count = 0
            for sample in samples:
                decision = evaluate_court_semantic_pre_render(
                    sample.camera,
                    layout,
                    schema_version=CourtDatasetSchemaVersion.V4,
                )
                if decision.accepted:
                    valid_count += 1
                else:
                    if len(decision.rejection_reasons) != 1:
                        raise ValueError(
                            "Semantic phase accounting requires one canonical reason per frame."
                        )
                    rejection_counts.update(decision.rejection_reasons)
                dispositions.append(
                    CourtSemanticFrameDisposition(
                        trajectory_frame_index=sample.trajectory_frame_index,
                        camera=sample.camera,
                        decision=decision,
                    )
                )
            evaluations.append(
                TrajectorySemanticPhaseEvaluation(
                    trajectory_id=trajectory.trajectory_id,
                    trajectory_group_id=trajectory.trajectory_group_id,
                    phase_index=phase_index,
                    phase_count=phase_count,
                    view=view,
                    expected_frame_count=len(samples),
                    expected_valid_frame_count=valid_count,
                    semantically_viable=valid_count > 0,
                    rejection_counts=tuple(sorted(rejection_counts.items())),
                    disposition_digest=court_semantic_phase_disposition_digest(
                        dispositions,
                        schema_version=CourtDatasetSchemaVersion.V4,
                        trajectory_group_id=trajectory.trajectory_group_id,
                        phase_index=phase_index,
                        phase_count=phase_count,
                    ),
                )
            )
    return tuple(evaluations)


def select_semantic_phase_budgeted_coverage(
    safe_items: Sequence[SelectedTrajectory],
    *,
    semantic_phase_evaluations: Sequence[TrajectorySemanticPhaseEvaluation],
    policy: OrbitSamplingPolicy,
    required_coverage: RequiredTrajectoryCoverage,
) -> tuple[SemanticPhaseSelection, ...]:
    """Jointly select unique safe candidates and explicit semantic phases.

    Typed geometry coverage and phase diversity are hard constraints.  Within
    those constraints the deterministic score prefers the phase with stronger
    projected pre-render validity, and the final inventory must satisfy the
    same accepted-count and accepted-fraction gates as assembly.
    """
    items = tuple(safe_items)
    evaluations = tuple(semantic_phase_evaluations)
    if not items or not evaluations:
        raise SafeCandidateExhaustionError(
            "semantic candidate/phase inventory is empty",
            coverage_shortfall=("semantic_phase",),
        )
    if not isinstance(required_coverage, RequiredTrajectoryCoverage):
        raise TypeError("Semantic selection requires typed release coverage.")
    item_by_group = {item.trajectory.trajectory_group_id: item for item in items}
    if len(item_by_group) != len(items):
        raise ValueError("Geometry-safe candidate group IDs must be unique.")
    if len({item.trajectory.semantic_key() for item in items}) != len(items):
        raise ValueError("Geometry-safe candidate semantics must be unique.")
    phases_by_group: dict[str, list[TrajectorySemanticPhaseEvaluation]] = defaultdict(
        list
    )
    for evaluation in evaluations:
        try:
            item = item_by_group[evaluation.trajectory_group_id]
        except KeyError as error:
            raise ValueError(
                "Semantic phase references a non-safe trajectory group."
            ) from error
        if (
            evaluation.trajectory_id != item.trajectory.trajectory_id
            or evaluation.expected_frame_count != len(item.path.theta_radians)
        ):
            raise ValueError(
                "Semantic phase identity/count disagrees with its safe candidate."
            )
        phases_by_group[evaluation.trajectory_group_id].append(evaluation)
    if set(phases_by_group) != set(item_by_group):
        raise ValueError("Every geometry-safe candidate requires semantic phases.")
    phase_counts = {phase.phase_count for phase in evaluations}
    if len(phase_counts) != 1:
        raise ValueError("All semantic candidates must use one phase count.")
    phase_count = next(iter(phase_counts))
    for group_id, phases in phases_by_group.items():
        phases.sort(key=lambda item: item.phase_index)
        if tuple(phase.phase_index for phase in phases) != tuple(range(phase_count)):
            raise ValueError(
                f"Semantic phase inventory is incomplete for {group_id!r}."
            )
    viable_by_group = {
        group_id: tuple(phase for phase in phases if phase.semantically_viable)
        for group_id, phases in phases_by_group.items()
    }
    viable_by_group = {
        group_id: phases for group_id, phases in viable_by_group.items() if phases
    }
    if len(viable_by_group) < policy.minimum_trajectory_groups:
        raise SafeCandidateExhaustionError(
            "insufficient candidates have a compatible semantic phase",
            coverage_shortfall=("semantic_phase",),
        )
    canonical_items = sorted(
        (item_by_group[group_id] for group_id in viable_by_group),
        key=_canonical_candidate_identity,
    )
    stable_items = sorted(
        canonical_items,
        key=lambda item: (
            tuple(
                repr(trajectory_field_value(item.trajectory, field))
                for field in policy.stable_field_order
            ),
            _canonical_candidate_identity(item),
        ),
    )
    stable_rank = {
        item.trajectory.trajectory_group_id: len(stable_items) - index
        for index, item in enumerate(stable_items)
    }
    seeded_items = list(canonical_items)
    random.Random(policy.seed).shuffle(seeded_items)
    seeded_rank = {
        item.trajectory.trajectory_group_id: index
        for index, item in enumerate(seeded_items, start=1)
    }
    canonical_rank = {
        item.trajectory.trajectory_group_id: len(canonical_items) - index
        for index, item in enumerate(canonical_items)
    }
    remaining_groups = set(viable_by_group)
    selected: list[SemanticPhaseSelection] = []
    stable_token_counts: Counter[tuple[StableFieldAny, object]] = Counter()
    covered_phases: set[int] = set()
    planned_count = 0
    expected_valid_count = 0
    while remaining_groups:
        release_shortfall = _semantic_selection_required_shortfall(
            selected,
            required_coverage=required_coverage,
        )
        requirements_met = (
            len(selected) >= policy.minimum_trajectory_groups
            and not release_shortfall
            and covered_phases == set(range(phase_count))
            and expected_valid_count >= policy.minimum_accepted_frames
            and expected_valid_count
            >= math.ceil(policy.minimum_accepted_fraction * planned_count)
        )
        if requirements_met:
            break
        feasible: list[
            tuple[SelectedTrajectory, TrajectorySemanticPhaseEvaluation]
        ] = []
        for group_id in sorted(remaining_groups):
            item = item_by_group[group_id]
            cost = len(item.path.theta_radians)
            remaining_group_requirement = max(
                0,
                policy.minimum_trajectory_groups - len(selected) - 1,
            )
            completion_costs = sorted(
                len(item_by_group[other].path.theta_radians)
                for other in remaining_groups
                if other != group_id
            )
            if len(completion_costs) < remaining_group_requirement:
                continue
            if (
                planned_count
                + cost
                + sum(completion_costs[:remaining_group_requirement])
                > policy.proposal_budget
            ):
                continue
            feasible.extend((item, phase) for phase in viable_by_group[group_id])
        if not feasible:
            break
        release_shortfall_count = len(release_shortfall)
        release_deficit = _semantic_selection_required_deficit(
            selected,
            required_coverage=required_coverage,
        )

        def score(
            entry: tuple[SelectedTrajectory, TrajectorySemanticPhaseEvaluation],
            release_shortfall_count: int = release_shortfall_count,
            release_deficit: int = release_deficit,
        ) -> tuple[int, ...]:
            item, phase = entry
            trajectory = item.trajectory
            preview = (
                *selected,
                SemanticPhaseSelection(selected=item, semantic_phase=phase),
            )
            preview_shortfall = _semantic_selection_required_shortfall(
                preview,
                required_coverage=required_coverage,
            )
            preview_deficit = _semantic_selection_required_deficit(
                preview,
                required_coverage=required_coverage,
            )
            if not isinstance(trajectory, OrbitTrajectorySpecV4):
                raise TypeError("Semantic V4 selection contains a legacy trajectory.")
            anchor = trajectory.anchor_provenance
            selected_anchor_indices: set[int] = set()
            for selected_item in selected:
                selected_trajectory = selected_item.selected.trajectory
                if not isinstance(selected_trajectory, OrbitTrajectorySpecV4):
                    raise TypeError(
                        "Semantic V4 selection contains a legacy trajectory."
                    )
                selected_anchor = selected_trajectory.anchor_provenance
                if selected_anchor is not None:
                    selected_anchor_indices.add(
                        selected_anchor.ordered_camera_index
                    )
            parts: list[int] = [
                release_deficit - preview_deficit,
                release_shortfall_count - len(preview_shortfall),
                int(phase.phase_index not in covered_phases),
                int(
                    trajectory.constructor
                    is PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE
                    and anchor is not None
                    and anchor.ordered_camera_index not in selected_anchor_indices
                ),
            ]
            parts.extend(
                (
                    round(phase.expected_valid_fraction * 1_000_000_000),
                    phase.expected_valid_frame_count,
                    -phase.expected_rejected_frame_count,
                )
            )
            for field in policy.stable_field_order:
                stable_token = (field, trajectory_field_value(trajectory, field))
                parts.extend(
                    (
                        int(stable_token_counts[stable_token] == 0),
                        -stable_token_counts[stable_token],
                    )
                )
            group_id = trajectory.trajectory_group_id
            parts.extend(
                (
                    stable_rank[group_id],
                    seeded_rank[group_id],
                    canonical_rank[group_id],
                    phase_count - phase.phase_index,
                )
            )
            return tuple(parts)

        chosen_item, chosen_phase = max(feasible, key=score)
        selected.append(
            SemanticPhaseSelection(
                selected=chosen_item,
                semantic_phase=chosen_phase,
            )
        )
        group_id = chosen_item.trajectory.trajectory_group_id
        remaining_groups.remove(group_id)
        planned_count += chosen_phase.expected_frame_count
        expected_valid_count += chosen_phase.expected_valid_frame_count
        covered_phases.add(chosen_phase.phase_index)
        stable_token_counts.update(
            (
                field,
                trajectory_field_value(chosen_item.trajectory, field),
            )
            for field in policy.stable_field_order
        )
    shortfall: list[str] = []
    shortfall.extend(
        _semantic_selection_required_shortfall(
            selected,
            required_coverage=required_coverage,
        )
    )
    shortfall.extend(
        f"semantic_phase:{phase_index}"
        for phase_index in sorted(set(range(phase_count)) - covered_phases)
    )
    if len(selected) < policy.minimum_trajectory_groups:
        shortfall.append("trajectory_group")
    if expected_valid_count < policy.minimum_accepted_frames:
        shortfall.append("minimum_accepted_frames")
    if not selected or expected_valid_count < math.ceil(
        policy.minimum_accepted_fraction * planned_count
    ):
        shortfall.append("minimum_accepted_fraction")
    if planned_count > policy.proposal_budget:
        shortfall.append("proposal_budget")
    if shortfall:
        raise SafeCandidateExhaustionError(
            "joint semantic-phase selection cannot satisfy release gates",
            coverage_shortfall=shortfall,
        )
    return tuple(selected)


def _semantic_selection_required_deficit(
    selected: Sequence[SemanticPhaseSelection],
    *,
    required_coverage: RequiredTrajectoryCoverage,
) -> int:
    """Measure numeric release deficits so greedy choice makes monotonic progress."""
    values = tuple(selected)
    constructors: Counter[PathConstructorV4] = Counter()
    families: set[PathFamilyV4] = set()
    profiles: set[VerticalProfileV4] = set()
    targets: set[OrbitTargetMode] = set()
    anchors: set[int] = set()
    anchored_planar = 0
    anchored_raised = 0
    anchored_required_lift = 0
    total_frames = 0
    anchored_frames = 0
    for item in values:
        trajectory = item.selected.trajectory
        if not isinstance(trajectory, OrbitTrajectorySpecV4):
            raise TypeError("Semantic V4 selection contains a legacy trajectory.")
        if not isinstance(trajectory.shape, PathFamilyV4) or not isinstance(
            trajectory.curve_mode, VerticalProfileV4
        ):
            raise TypeError("Semantic V4 selection vocabulary is invalid.")
        frame_count = item.semantic_phase.expected_frame_count
        constructors[trajectory.constructor] += 1
        families.add(trajectory.shape)
        profiles.add(trajectory.curve_mode)
        targets.add(item.semantic_phase.view.target_mode)
        total_frames += frame_count
        if (
            trajectory.constructor
            is not PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE
        ):
            continue
        anchor = trajectory.anchor_provenance
        if anchor is None:  # pragma: no cover - strict trajectory excludes this
            raise ValueError("Selected anchored trajectory lacks provenance.")
        anchors.add(anchor.ordered_camera_index)
        anchored_frames += frame_count
        if trajectory.curve_mode is VerticalProfileV4.PLANAR:
            anchored_planar += 1
        elif trajectory.curve_mode is VerticalProfileV4.RAISED_PHASES:
            anchored_raised += 1
            if math.isclose(
                anchor.lift_m,
                required_coverage.required_raised_lift_m,
                abs_tol=1.0e-12,
                rel_tol=0.0,
            ):
                anchored_required_lift += 1
    anchored_groups = constructors.get(
        PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE, 0
    )
    frame_share_deficit = max(
        0,
        math.ceil(
            required_coverage.minimum_anchored_frame_share * total_frames
            - anchored_frames
            - 1.0e-12
        ),
    )
    deficits = (
        max(0, required_coverage.minimum_total_groups - len(values)),
        max(
            0,
            required_coverage.minimum_free_space_cycle_groups
            - constructors.get(PathConstructorV4.FREE_SPACE_CYCLE, 0),
        ),
        max(
            0,
            required_coverage.minimum_anchored_rounded_rectangle_groups
            - anchored_groups,
        ),
        max(0, required_coverage.minimum_unique_anchors - len(anchors)),
        max(
            0,
            required_coverage.minimum_anchored_planar_groups - anchored_planar,
        ),
        max(
            0,
            required_coverage.minimum_anchored_raised_groups - anchored_raised,
        ),
        max(
            0,
            required_coverage.minimum_anchored_raised_groups
            - anchored_required_lift,
        ),
        frame_share_deficit,
        sum(item not in constructors for item in required_coverage.constructors),
        sum(item not in families for item in required_coverage.path_families),
        sum(item not in profiles for item in required_coverage.vertical_profiles),
        sum(item not in targets for item in required_coverage.target_modes),
    )
    return sum(deficits)


def _semantic_selection_required_shortfall(
    selected: Sequence[SemanticPhaseSelection],
    *,
    required_coverage: RequiredTrajectoryCoverage,
) -> tuple[str, ...]:
    """Use the shared typed coverage authority during pre-group selection."""
    values = tuple(selected)
    if not values:
        return (
            "constructor:anchored_rounded_rectangle",
            "constructor:free_space_cycle",
            "family:rounded_rectangle",
            "minimum_anchored_frame_share",
            "minimum_anchored_planar_groups",
            "minimum_anchored_raised_groups",
            "minimum_anchored_rounded_rectangle_groups",
            "minimum_free_space_cycle_groups",
            "minimum_total_groups",
            "minimum_unique_anchors",
            "profile:planar",
            "profile:raised_phases",
            "required_raised_lift_m",
            "target:court_center",
        )
    coverage = build_selected_coverage_from_records(
        tuple(
            (
                item.selected.trajectory,
                item.semantic_phase.view.target_mode,
                item.semantic_phase.expected_frame_count,
            )
            for item in values
            if isinstance(item.selected.trajectory, OrbitTrajectorySpecV4)
        ),
        required_raised_lift_m=required_coverage.required_raised_lift_m,
    )
    shortfall: tuple[str, ...] = required_coverage_shortfall(
        required_coverage, coverage
    )
    return shortfall


def select_safe_budgeted_coverage(
    candidates: Sequence[OrbitTrajectorySpecV4],
    *,
    centers: Sequence[OrbitCenter],
    policy: OrbitSamplingPolicy,
    support_model: TrajectorySupportModel,
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
    configuration: CourtDatasetConfiguration,
) -> SafeSelectionResult:
    """Evaluate geometry once, then jointly freeze safe candidate/view phases."""
    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        raise SafeCandidateExhaustionError(
            "candidate inventory is empty", coverage_shortfall=("all",)
        )
    required_coverage = configuration.required_coverage
    if required_coverage is None:
        raise TypeError("V4 safe selection requires required_coverage configuration.")
    center_by_key = {center.key(): center for center in centers}
    evaluations: list[TrajectorySafetyEvaluation] = []
    safe_candidates: list[OrbitTrajectorySpecV4] = []
    safe_items: list[SelectedTrajectory] = []
    for candidate in candidate_tuple:
        try:
            center = center_by_key[
                candidate.center_kind,
                candidate.center_court_instance_id,
            ]
        except KeyError as error:
            raise ValueError(
                "V4 candidate references an unknown orbit centre."
            ) from error
        if (
            candidate.constructor
            is PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE
        ):
            validate_anchored_trajectory_provenance(
                candidate,
                center=center,
                support_model=support_model,
            )
        path = sample_uniform_arc_length(candidate, center, policy)
        evaluation = evaluate_trajectory_safety(
            trajectory_id=candidate.trajectory_id,
            trajectory_group_id=candidate.trajectory_group_id,
            path=path,
            support_model=support_model,
        )
        evaluations.append(evaluation)
        if evaluation.safe:
            safe_candidates.append(candidate)
            safe_items.append(
                SelectedTrajectory(
                    trajectory=candidate,
                    center=center,
                    path=path,
                )
            )
    all_tokens = set().union(
        *(
            _objective_tokens(candidate, policy.coverage_objective)
            for candidate in candidate_tuple
        )
    )
    safe_tokens = (
        set().union(
            *(
                _objective_tokens(candidate, policy.coverage_objective)
                for candidate in safe_candidates
            )
        )
        if safe_candidates
        else set()
    )
    shortfall = tuple(sorted(repr(token) for token in all_tokens - safe_tokens))
    if len(safe_candidates) < policy.minimum_trajectory_groups:
        raise SafeCandidateExhaustionError(
            "safe candidate count cannot satisfy minimum trajectory groups",
            coverage_shortfall=shortfall or ("trajectory_group",),
        )
    semantic_evaluations = _evaluate_semantic_phases(
        safe_items,
        cameras=cameras,
        layout=layout,
        centers=centers,
        configuration=configuration,
        selection_seed=policy.seed,
    )
    phase_selections = select_semantic_phase_budgeted_coverage(
        safe_items,
        semantic_phase_evaluations=semantic_evaluations,
        policy=policy,
        required_coverage=required_coverage,
    )
    selected = tuple(item.selected for item in phase_selections)
    selected_phases = tuple(item.semantic_phase for item in phase_selections)
    selected_coverage = build_selected_coverage_from_records(
        tuple(
            (
                item.selected.trajectory,
                item.semantic_phase.view.target_mode,
                item.semantic_phase.expected_frame_count,
            )
            for item in phase_selections
            if isinstance(item.selected.trajectory, OrbitTrajectorySpecV4)
        ),
        required_raised_lift_m=required_coverage.required_raised_lift_m,
    )
    release_shortfall = required_coverage_shortfall(
        required_coverage,
        selected_coverage,
    )
    if release_shortfall:
        raise SafeCandidateExhaustionError(
            "selected candidates do not satisfy required release coverage",
            coverage_shortfall=release_shortfall,
        )
    return SafeSelectionResult(
        selected=selected,
        evaluations=tuple(evaluations),
        semantic_phase_evaluations=semantic_evaluations,
        selected_semantic_phases=selected_phases,
        selected_coverage=selected_coverage,
        required_coverage_shortfall=release_shortfall,
        optional_candidate_coverage_shortfall=shortfall,
    )


def _objective_tokens(
    candidate: OrbitTrajectorySpec,
    objectives: Sequence[OrbitCoverageObjective],
) -> set[tuple[OrbitCoverageObjective, StableFieldAny, object]]:
    return {
        (objective, field, trajectory_field_value(candidate, field))
        for objective in objectives
        for field in _objective_fields(candidate, objective)
    }


def _objective_fields(
    candidate: OrbitTrajectorySpec,
    objective: OrbitCoverageObjective,
) -> tuple[StableFieldAny, ...]:
    legacy = _OBJECTIVE_FIELDS[objective]
    if not isinstance(candidate, OrbitTrajectorySpecV4):
        return legacy
    if candidate.shape is PathFamilyV4.FREE_SPACE_CYCLE:
        return {
            OrbitCoverageObjective.COVERAGE_MODE: (
                OrbitStableField.SHAPE,
                OrbitStableField.ORIENTATION_DEGREES,
            ),
            OrbitCoverageObjective.SEMANTIC_VISIBILITY: (
                OrbitStableField.CENTER_KIND,
                OrbitStableField.BASE_HEIGHT_M,
            ),
            OrbitCoverageObjective.TRAJECTORY_GROUP: (OrbitStableField.CURVE_MODE,),
        }[objective]
    additions = {
        OrbitCoverageObjective.COVERAGE_MODE: (OrbitStableFieldV4.CORNER_RADIUS_RATIO,),
        OrbitCoverageObjective.SEMANTIC_VISIBILITY: (),
        OrbitCoverageObjective.TRAJECTORY_GROUP: (OrbitStableFieldV4.VERTICAL_PHASE,),
    }
    return (*legacy, *additions[objective])


def _canonical_candidate_identity(item: SelectedTrajectory) -> str:
    """Return a canonical semantic identity independent of input sequence order."""
    return repr(item.trajectory.semantic_key())


def assign_group_disjoint_splits(
    group_ids: Sequence[str],
    *,
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, DatasetSplit]:
    """Assign each trajectory group exactly once using configured fractions."""
    identifiers = tuple(group_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("group_ids must be non-empty and unique.")
    if len(identifiers) < len(DatasetSplit):
        raise ValueError("At least three groups are required for non-empty splits.")
    exact = np.asarray(fractions, dtype=np.float64) * len(identifiers)
    counts = np.floor(exact).astype(np.int64)
    for index in np.argsort(-(exact - counts))[: len(identifiers) - int(counts.sum())]:
        counts[index] += 1
    for index in range(len(counts)):
        if counts[index] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                raise ValueError(
                    "Configured fractions cannot produce non-empty splits."
                )
            counts[donor] -= 1
            counts[index] += 1
    shuffled = sorted(identifiers)
    random.Random(seed).shuffle(shuffled)
    result: dict[str, DatasetSplit] = {}
    offset = 0
    for split, count in zip(DatasetSplit, counts.tolist(), strict=True):
        for group_id in shuffled[offset : offset + count]:
            result[group_id] = split
        offset += count
    if set(result) != set(identifiers):
        raise ValueError("Split assignment did not cover every trajectory group.")
    return result


def assign_group_shards(
    group_sample_counts: Mapping[str, int],
    *,
    shard_count: int,
    seed: int,
    maximum_shard_samples: int,
) -> dict[str, str]:
    """Deterministically balance whole groups within the render batch limit."""
    identifiers = tuple(group_sample_counts)
    if not identifiers or any(
        not isinstance(group_id, str) or not group_id for group_id in identifiers
    ):
        raise ValueError("group_sample_counts must use non-empty group IDs.")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in group_sample_counts.values()
    ):
        raise ValueError("group sample counts must be positive integers.")
    if isinstance(shard_count, bool) or not 1 <= shard_count <= len(identifiers):
        raise ValueError("shard_count must lie in [1, group_count].")
    if (
        isinstance(maximum_shard_samples, bool)
        or not isinstance(maximum_shard_samples, int)
        or maximum_shard_samples <= 0
    ):
        raise ValueError("maximum_shard_samples must be a positive integer.")
    if sum(group_sample_counts.values()) > shard_count * maximum_shard_samples:
        raise ValueError("Group sample inventory exceeds the aggregate shard budget.")
    if max(group_sample_counts.values()) > maximum_shard_samples:
        raise ValueError("A trajectory group exceeds the per-shard sample budget.")

    seeded_groups = sorted(identifiers)
    random.Random(seed + 1).shuffle(seeded_groups)
    group_rank = {group_id: rank for rank, group_id in enumerate(seeded_groups)}
    ordered_groups = sorted(
        identifiers,
        key=lambda group_id: (
            -group_sample_counts[group_id],
            group_rank[group_id],
            group_id,
        ),
    )
    seeded_shards = list(range(shard_count))
    random.Random(seed + 2).shuffle(seeded_shards)
    shard_rank = {shard_index: rank for rank, shard_index in enumerate(seeded_shards)}
    shard_loads = [0] * shard_count
    assigned: dict[str, str] = {}
    for group_id in ordered_groups:
        count = group_sample_counts[group_id]
        feasible = [
            shard_index
            for shard_index, load in enumerate(shard_loads)
            if load + count <= maximum_shard_samples
        ]
        if not feasible:
            raise ValueError(
                "Whole trajectory groups cannot satisfy the per-shard sample budget."
            )
        shard_index = min(
            feasible,
            key=lambda index: (shard_loads[index], shard_rank[index], index),
        )
        shard_loads[shard_index] += count
        assigned[group_id] = f"shard-{shard_index:03d}"
    return assigned


def assign_court_targets_for_groups(
    selected: Sequence[SelectedTrajectory],
    *,
    split_by_group: Mapping[str, DatasetSplit],
    layout: MultiCourtLayout,
    seed: int,
) -> tuple[CourtAssignment, ...]:
    """Balance targets while preserving every court-centred path's transform."""
    selected_tuple = tuple(selected)
    group_ids = {item.trajectory.trajectory_group_id for item in selected_tuple}
    if not selected_tuple or set(split_by_group) != group_ids:
        raise ValueError(
            "Court target assignment requires every selected group and split."
        )
    court_ids = sorted(court.court_instance_id for court in layout.courts)
    ranked_courts = list(court_ids)
    random.Random(seed).shuffle(ranked_courts)
    court_rank = {court_id: index for index, court_id in enumerate(ranked_courts)}
    global_counts: Counter[str] = Counter()
    split_counts: dict[DatasetSplit, Counter[str]] = defaultdict(Counter)
    assigned: dict[str, str] = {}
    complex_groups: dict[DatasetSplit, list[str]] = defaultdict(list)
    for item in selected_tuple:
        group_id = item.trajectory.trajectory_group_id
        split = split_by_group[group_id]
        center_court_id = item.trajectory.center_court_instance_id
        if center_court_id is None:
            complex_groups[split].append(group_id)
            continue
        if center_court_id not in court_ids:
            raise ValueError("Court-centred trajectory references an unaccepted court.")
        assigned[group_id] = center_court_id
        global_counts[center_court_id] += 1
        split_counts[split][center_court_id] += 1
    split_totals = Counter(split_by_group.values())
    global_floor = len(selected_tuple) // len(court_ids)
    global_ceiling = math.ceil(len(selected_tuple) / len(court_ids))
    split_limits = {
        split: (
            split_totals[split] // len(court_ids),
            math.ceil(split_totals[split] / len(court_ids)),
        )
        for split in DatasetSplit
    }
    for split in DatasetSplit:
        _floor, ceiling = split_limits[split]
        if any(split_counts[split][court_id] > ceiling for court_id in court_ids):
            raise ValueError(
                f"Fixed court-centred paths cannot be balanced in {split.value}."
            )
    if any(global_counts[court_id] > global_ceiling for court_id in court_ids):
        raise ValueError("Fixed court-centred paths cannot be balanced globally.")
    pending: list[tuple[DatasetSplit, str]] = []
    for split in DatasetSplit:
        group_list = sorted(complex_groups[split])
        random.Random(f"{seed}:{split.value}").shuffle(group_list)
        pending.extend((split, group_id) for group_id in group_list)

    def solve(index: int) -> bool:
        if index == len(pending):
            global_values = [global_counts[court_id] for court_id in court_ids]
            if any(
                not global_floor <= value <= global_ceiling for value in global_values
            ):
                return False
            return all(
                all(
                    floor <= split_counts[split][court_id] <= ceiling
                    for court_id in court_ids
                )
                for split, (floor, ceiling) in split_limits.items()
            )
        split, group_id = pending[index]
        _floor, split_ceiling = split_limits[split]
        options = sorted(
            court_ids,
            key=lambda value: (
                split_counts[split][value],
                global_counts[value],
                court_rank[value],
            ),
        )
        for court_id in options:
            if (
                split_counts[split][court_id] >= split_ceiling
                or global_counts[court_id] >= global_ceiling
            ):
                continue
            assigned[group_id] = court_id
            split_counts[split][court_id] += 1
            global_counts[court_id] += 1
            if solve(index + 1):
                return True
            del assigned[group_id]
            split_counts[split][court_id] -= 1
            global_counts[court_id] -= 1
        return False

    if not solve(0):
        raise ValueError(
            "Complex-centred paths cannot satisfy balanced court assignment."
        )
    if set(assigned) != group_ids:
        raise ValueError("Court target assignment did not cover every selected group.")
    if len(selected_tuple) >= len(court_ids) and set(global_counts) != set(court_ids):
        raise ValueError("Every accepted court must be used by trajectory groups.")
    if max(global_counts.values()) - min(global_counts.values()) > 1:
        raise ValueError("Court target group counts differ by more than one globally.")
    for split, counts in split_counts.items():
        values = [counts.get(court_id, 0) for court_id in court_ids]
        if max(values) - min(values) > 1:
            raise ValueError(
                f"Court target group counts differ by more than one in {split.value}."
            )
    return tuple(
        CourtAssignment(
            scene_id=group_id,
            split=split_by_group[group_id].value,
            court_instance_id=assigned[group_id],
            candidate_id=layout.court(assigned[group_id]).candidate_id,
            selection_seed=seed,
        )
        for group_id in sorted(group_ids)
    )


def _views_for_group(
    *,
    group_index: int,
    group_id: str,
    target_court_instance_id: str,
    configuration: CourtDatasetConfiguration,
) -> tuple[OrbitViewSpec, ...]:
    coverage_modes = configuration.view.coverage_modes
    coverage = coverage_modes[group_index % len(coverage_modes)]
    low_hfov, high_hfov = configuration.view.hfov_degrees
    hfov_by_coverage = {
        OrbitCoverageMode.FULL: high_hfov,
        OrbitCoverageMode.NEAR_FULL: (low_hfov + high_hfov) / 2.0,
        OrbitCoverageMode.PARTIAL: low_hfov,
    }
    low_height, high_height = configuration.view.look_at_height_m
    targets = _target_modes_for_group(
        group_index=group_index,
        configuration=configuration,
    )
    views = tuple(
        _view_for_target(
            group_index=group_index,
            target=target,
            target_court_instance_id=target_court_instance_id,
            coverage=coverage,
            look_at_height_m=(
                low_height if (group_index + target_index) % 2 == 0 else high_height
            ),
            hfov_degrees=hfov_by_coverage[coverage],
        )
        for target_index, target in enumerate(targets)
    )
    if group_id in {view.view_id for view in views}:
        raise ValueError("Opaque group and view IDs unexpectedly collide.")
    return views


def _views_for_group_v2(
    *,
    group_index: int,
    group_id: str,
    configuration: CourtDatasetConfiguration,
) -> tuple[OrbitViewSpecV2, ...]:
    coverage_modes = configuration.view.coverage_modes
    coverage = coverage_modes[group_index % len(coverage_modes)]
    low_hfov, high_hfov = configuration.view.hfov_degrees
    hfov_by_coverage = {
        OrbitCoverageMode.FULL: high_hfov,
        OrbitCoverageMode.NEAR_FULL: (low_hfov + high_hfov) / 2.0,
        OrbitCoverageMode.PARTIAL: low_hfov,
    }
    low_height, high_height = configuration.view.look_at_height_m
    targets = _target_modes_for_group(
        group_index=group_index,
        configuration=configuration,
    )
    views = tuple(
        OrbitViewSpecV2(
            view_id=f"view-{group_index:05d}-{target.value}",
            target_kind=target.target_kind,
            target_mode=target,
            coverage_mode=coverage,
            look_at_height_m=(
                low_height if (group_index + target_index) % 2 == 0 else high_height
            ),
            hfov_degrees=hfov_by_coverage[coverage],
        )
        for target_index, target in enumerate(targets)
    )
    if group_id in {view.view_id for view in views}:
        raise ValueError("Opaque group and view IDs unexpectedly collide.")
    return views


def _target_modes_for_group(
    *,
    group_index: int,
    configuration: CourtDatasetConfiguration,
) -> tuple[OrbitTargetMode, ...]:
    """Return configured variants, adding a cross-kind view only when present."""
    configured = configuration.view.target_modes
    primary_target = configured[group_index % len(configured)]
    if group_index != 0:
        return (primary_target,)
    variant = next(
        (
            target
            for target in configured
            if target.target_kind is not primary_target.target_kind
        ),
        None,
    )
    if variant is None:
        return (primary_target,)
    return primary_target, variant


def _view_for_target(
    *,
    group_index: int,
    target: OrbitTargetMode,
    target_court_instance_id: str,
    coverage: OrbitCoverageMode,
    look_at_height_m: float,
    hfov_degrees: float,
) -> OrbitViewSpec:
    """Materialize exactly one configured typed target without a fallback."""
    return OrbitViewSpec(
        view_id=f"view-{group_index:05d}-{target.value}",
        target_kind=target.target_kind,
        target_court_instance_id=(
            target_court_instance_id
            if target.target_kind is OrbitTargetKind.COURT
            else None
        ),
        target_mode=target,
        coverage_mode=coverage,
        look_at_height_m=look_at_height_m,
        hfov_degrees=hfov_degrees,
    )


def _plan_samples(
    *,
    groups: Sequence[TrajectoryGroupPlan],
    paths_by_group: Mapping[str, OrbitPathSamples],
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
    centers: Sequence[OrbitCenter],
) -> tuple[PlannedCourtSample, ...]:
    template = cameras[0]
    complex_center = next(
        center for center in centers if center.court_instance_id is None
    )
    samples: list[PlannedCourtSample] = []
    for group in groups:
        path = paths_by_group[group.trajectory_group_id]
        vertical_scene = group.center.scene_from_center.matrix()[:3, 2]
        for view in group.views:
            target_scene = _target_scene(
                view,
                layout=layout,
                complex_center=complex_center,
            )
            intrinsics = _intrinsics_from_hfov(
                width=template.width,
                height=template.height,
                hfov_degrees=view.hfov_degrees,
            )
            for frame_index, center_scene in enumerate(path.points_scene_m):
                camera_to_scene = _look_at_opencv(
                    center_scene,
                    target_scene,
                    vertical_scene=vertical_scene,
                )
                sample_index = len(samples)
                sample_id = f"court-sample-{sample_index:06d}"
                camera = SceneCamera(
                    camera_id=sample_id,
                    source_frame_index=sample_index,
                    width=template.width,
                    height=template.height,
                    intrinsics=intrinsics,
                    camera_to_scene=RigidTransform.from_matrix(camera_to_scene),
                    image_path=f"generated/court/{sample_id}.png",
                )
                samples.append(
                    PlannedCourtSample(
                        sample_index=sample_index,
                        sample_id=sample_id,
                        trajectory_group_id=group.trajectory_group_id,
                        trajectory_id=group.trajectory.trajectory_id,
                        view_id=view.view_id,
                        trajectory_frame_index=frame_index,
                        split=group.split,
                        shard_id=group.shard_id,
                        camera_center_scene_m=(
                            float(center_scene[0]),
                            float(center_scene[1]),
                            float(center_scene[2]),
                        ),
                        camera=camera,
                    )
                )
    return tuple(samples)


def _plan_samples_v2(
    *,
    groups: Sequence[TrajectoryGroupPlanV2],
    paths_by_group: Mapping[str, OrbitPathSamples],
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
    centers: Sequence[OrbitCenter],
    selection_seed: int,
) -> tuple[PlannedCourtSampleV2, ...]:
    """Resolve sample target, then construct its look-at pose in that order."""
    template = cameras[0]
    complex_center = next(
        center for center in centers if center.court_instance_id is None
    )
    samples: list[PlannedCourtSampleV2] = []
    for group in groups:
        path = paths_by_group[group.trajectory_group_id]
        vertical_scene = group.center.scene_from_center.matrix()[:3, 2]
        for view in group.views:
            intrinsics = _intrinsics_from_hfov(
                width=template.width,
                height=template.height,
                hfov_degrees=view.hfov_degrees,
            )
            for frame_index, center_scene in enumerate(path.points_scene_m):
                resolved_target = resolve_target_court(
                    policy=group.target_court_policy,
                    camera_center_scene_m=center_scene,
                    layout=layout,
                    selection_seed=selection_seed,
                )
                target_scene = _target_scene_v2(
                    view,
                    target_court=resolved_target,
                    layout=layout,
                    complex_center=complex_center,
                )
                camera_to_scene = _look_at_opencv(
                    center_scene,
                    target_scene,
                    vertical_scene=vertical_scene,
                )
                sample_index = len(samples)
                sample_id = f"court-sample-{sample_index:06d}"
                camera = SceneCamera(
                    camera_id=sample_id,
                    source_frame_index=sample_index,
                    width=template.width,
                    height=template.height,
                    intrinsics=intrinsics,
                    camera_to_scene=RigidTransform.from_matrix(camera_to_scene),
                    image_path=f"generated/court/{sample_id}.png",
                )
                if view.target_mode is OrbitTargetMode.COURT_CENTER:
                    validate_camera_looks_at_resolved_court(
                        camera=camera,
                        target_court=resolved_target,
                        layout=layout,
                        look_at_height_m=view.look_at_height_m,
                    )
                samples.append(
                    PlannedCourtSampleV2(
                        sample_index=sample_index,
                        sample_id=sample_id,
                        trajectory_group_id=group.trajectory_group_id,
                        trajectory_id=group.trajectory.trajectory_id,
                        view_id=view.view_id,
                        trajectory_frame_index=frame_index,
                        split=group.split,
                        shard_id=group.shard_id,
                        camera_center_scene_m=(
                            float(center_scene[0]),
                            float(center_scene[1]),
                            float(center_scene[2]),
                        ),
                        camera=camera,
                        target_court=resolved_target,
                    )
                )
    return tuple(samples)


def _target_scene(
    view: OrbitViewSpec,
    *,
    layout: MultiCourtLayout,
    complex_center: OrbitCenter,
) -> NDArray[np.float64]:
    if view.target_kind is OrbitTargetKind.COMPLEX:
        local = np.asarray(((0.0, 0.0, view.look_at_height_m),), dtype=np.float64)
        transformed = complex_center.scene_from_center.apply(local)
        return np.asarray(
            (
                float(transformed[0, 0]),
                float(transformed[0, 1]),
                float(transformed[0, 2]),
            ),
            dtype=np.float64,
        )
    assert view.target_court_instance_id is not None
    court = layout.court(view.target_court_instance_id)
    y = {
        OrbitTargetMode.COURT_CENTER: 0.0,
        OrbitTargetMode.NEAR_BASELINE: -HALF_LENGTH,
        OrbitTargetMode.FAR_BASELINE: HALF_LENGTH,
    }[view.target_mode]
    local = np.asarray(((0.0, y, view.look_at_height_m),), dtype=np.float64)
    transformed = court.scene_from_court.apply(local)
    return np.asarray(
        (
            float(transformed[0, 0]),
            float(transformed[0, 1]),
            float(transformed[0, 2]),
        ),
        dtype=np.float64,
    )


def _target_scene_v2(
    view: OrbitViewSpecV2,
    *,
    target_court: ResolvedTargetCourtV2,
    layout: MultiCourtLayout,
    complex_center: OrbitCenter,
) -> NDArray[np.float64]:
    """Resolve a v2 view without reading a static court from the view."""
    if not isinstance(target_court, ResolvedTargetCourtV2):
        raise TypeError("target_court must be a ResolvedTargetCourtV2.")
    if view.target_kind is OrbitTargetKind.COMPLEX:
        local = np.asarray(((0.0, 0.0, view.look_at_height_m),), dtype=np.float64)
        return np.asarray(
            complex_center.scene_from_center.apply(local)[0],
            dtype=np.float64,
        )
    if view.target_mode is OrbitTargetMode.COURT_CENTER:
        return resolved_court_look_at_scene(
            target_court=target_court,
            layout=layout,
            look_at_height_m=view.look_at_height_m,
        )
    court = layout.court(target_court.binding.court_instance_id)
    y = {
        OrbitTargetMode.NEAR_BASELINE: -HALF_LENGTH,
        OrbitTargetMode.FAR_BASELINE: HALF_LENGTH,
    }[view.target_mode]
    local = np.asarray(((0.0, y, view.look_at_height_m),), dtype=np.float64)
    return np.asarray(court.scene_from_court.apply(local)[0], dtype=np.float64)


def _intrinsics_from_hfov(
    *,
    width: int,
    height: int,
    hfov_degrees: float,
) -> tuple[float, ...]:
    focal = width / (2.0 * math.tan(math.radians(hfov_degrees) / 2.0))
    return (
        focal,
        0.0,
        (width - 1.0) / 2.0,
        0.0,
        focal,
        (height - 1.0) / 2.0,
        0.0,
        0.0,
        1.0,
    )


def _look_at_opencv(
    center_scene: NDArray[np.float64],
    target_scene: NDArray[np.float64],
    *,
    vertical_scene: NDArray[np.float64],
) -> NDArray[np.float64]:
    forward = np.asarray(target_scene - center_scene, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-9:
        raise ValueError("Camera centre and look-at target must differ.")
    forward /= norm
    up = np.asarray(vertical_scene, dtype=np.float64)
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1.0e-9:
        raise ValueError("Orbit local vertical axis must be non-zero.")
    up /= up_norm
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-8:
        raise ValueError("Look-at direction is parallel to the local vertical axis.")
    right /= right_norm
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    matrix: NDArray[np.float64] = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward))
    matrix[:3, 3] = center_scene
    return matrix


__all__ = [
    "SafeCandidateExhaustionError",
    "SafeSelectionResult",
    "SelectedTrajectory",
    "assign_group_disjoint_splits",
    "assign_group_shards",
    "assign_court_targets_for_groups",
    "build_court_dataset_plan",
    "select_budgeted_coverage",
    "select_safe_budgeted_coverage",
]
