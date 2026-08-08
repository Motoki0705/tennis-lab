"""Deterministic budgeted coverage selection and Court render planning."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.trajectory import (
    derive_orbit_centers,
    generate_trajectory_candidates,
    trajectory_field_value,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    CourtDatasetPlan,
    DatasetSplit,
    OrbitCenter,
    OrbitCoverageMode,
    OrbitPathSamples,
    OrbitSamplingPolicy,
    OrbitTargetKind,
    OrbitTrajectorySpec,
    OrbitViewSpec,
    PlannedCourtSample,
    TrajectoryGroupPlan,
)
from src.synthetic_data_generation.scene_contract import (
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.tasks.base.generate_dataset.court_assignment import CourtAssignment
from src.utils.schema.court import HALF_LENGTH


@dataclass(frozen=True, slots=True)
class SelectedTrajectory:
    """One selected path and its already validated uniform samples."""

    trajectory: OrbitTrajectorySpec
    center: OrbitCenter
    path: OrbitPathSamples


def build_court_dataset_plan(
    *,
    scene_id: str,
    profile: str,
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
    configuration: CourtDatasetConfiguration,
    metric_adapter: MetricSceneAdapter,
) -> CourtDatasetPlan:
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
    candidates = generate_trajectory_candidates(
        configuration.trajectory,
        centers,
        seed=policy.seed,
        stable_field_order=policy.stable_field_order,
    )
    selected = select_budgeted_coverage(candidates, centers=centers, policy=policy)
    split_by_group = assign_group_disjoint_splits(
        tuple(item.trajectory.trajectory_group_id for item in selected),
        fractions=policy.split_fractions,
        seed=policy.seed,
    )
    shard_by_group = assign_group_shards(
        tuple(item.trajectory.trajectory_group_id for item in selected),
        shard_count=policy.shard_count,
        seed=policy.seed,
    )
    assignments = assign_court_targets_for_groups(
        selected,
        split_by_group=split_by_group,
        layout=layout,
        seed=policy.seed,
    )
    assignment_by_group = {assignment.scene_id: assignment for assignment in assignments}
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
) -> tuple[SelectedTrajectory, ...]:
    """Greedily maximize typed-field coverage within one explicit frame budget."""
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
    center_by_key = {center.key(): center for center in centers}
    if len(center_by_key) != len(tuple(centers)):
        raise ValueError("Resolved orbit centres must be unique.")
    resolved: list[SelectedTrajectory] = []
    for candidate in candidate_tuple:
        try:
            center = center_by_key[
                candidate.center_kind,
                candidate.center_court_instance_id,
            ]
        except KeyError as error:
            raise ValueError("Candidate references an unknown orbit centre.") from error
        resolved.append(
            SelectedTrajectory(
                trajectory=candidate,
                center=center,
                path=sample_uniform_arc_length(candidate, center, policy),
            )
        )
    all_tokens = set().union(
        *(_coverage_tokens(item.trajectory, policy.stable_field_order) for item in resolved)
    )
    required_proposals = math.ceil(
        policy.minimum_accepted_frames / policy.minimum_accepted_fraction
    )
    rng = random.Random(policy.seed)
    tie_break = {
        item.trajectory.trajectory_group_id: rng.random() for item in resolved
    }
    remaining = list(resolved)
    selected: list[SelectedTrajectory] = []
    covered: set[tuple[str, object]] = set()
    token_counts: Counter[tuple[str, object]] = Counter()
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
            view_count = 2 if not selected else 1
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
            minimum_completion_cost = sum(
                completion_costs[:remaining_group_count]
            )
            if (
                proposal_count + cost + minimum_completion_cost
                <= policy.proposal_budget
            ):
                feasible.append((item, cost))
        if not feasible:
            break

        def score(entry: tuple[SelectedTrajectory, int]) -> tuple[float, ...]:
            item, cost = entry
            tokens = _coverage_tokens(item.trajectory, policy.stable_field_order)
            new_count = len(tokens - covered)
            balance = -sum(token_counts[token] for token in tokens)
            efficiency = new_count / cost
            return (
                float(new_count),
                float(balance),
                efficiency,
                tie_break[item.trajectory.trajectory_group_id],
            )

        chosen, cost = max(feasible, key=score)
        selected.append(chosen)
        proposal_count += cost
        tokens = _coverage_tokens(chosen.trajectory, policy.stable_field_order)
        covered.update(tokens)
        token_counts.update(tokens)
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
        raise ValueError(f"Candidate budget cannot cover all typed field values: {missing}.")
    if proposal_count > policy.proposal_budget:
        raise ValueError("Coverage selector exceeded proposal_budget.")
    if not any(len(item.path.theta_radians) > 0 for item in selected):
        raise ValueError("Coverage selector produced no samples.")
    return tuple(selected)


def _coverage_tokens(
    candidate: OrbitTrajectorySpec,
    field_order: Sequence[str],
) -> set[tuple[str, object]]:
    tokens = {
        (field, trajectory_field_value(candidate, field)) for field in field_order
    }
    tokens.add(
        (
            "resolved_center",
            (candidate.center_kind.value, candidate.center_court_instance_id),
        )
    )
    return tokens


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
                raise ValueError("Configured fractions cannot produce non-empty splits.")
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
    group_ids: Sequence[str],
    *,
    shard_count: int,
    seed: int,
) -> dict[str, str]:
    """Deterministically assign whole trajectory groups to render shards."""
    identifiers = tuple(group_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("group_ids must be non-empty and unique.")
    if isinstance(shard_count, bool) or not 1 <= shard_count <= len(identifiers):
        raise ValueError("shard_count must lie in [1, group_count].")
    shuffled = sorted(identifiers)
    random.Random(seed + 1).shuffle(shuffled)
    return {
        group_id: f"shard-{index % shard_count:03d}"
        for index, group_id in enumerate(shuffled)
    }


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
        raise ValueError("Court target assignment requires every selected group and split.")
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
            if any(not global_floor <= value <= global_ceiling for value in global_values):
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
        raise ValueError("Complex-centred paths cannot satisfy balanced court assignment.")
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
    coverage_modes = tuple(OrbitCoverageMode(value) for value in configuration.view.coverage_modes)
    coverage = coverage_modes[group_index % len(coverage_modes)]
    configured_target = configuration.view.target_modes[
        group_index % len(configuration.view.target_modes)
    ]
    target_mode = "center" if configured_target == "court_center" else configured_target
    low_hfov, high_hfov = configuration.view.hfov_degrees
    hfov_by_coverage = {
        OrbitCoverageMode.FULL: high_hfov,
        OrbitCoverageMode.NEAR_FULL: (low_hfov + high_hfov) / 2.0,
        OrbitCoverageMode.PARTIAL: low_hfov,
    }
    low_height, high_height = configuration.view.look_at_height_m
    court_view = OrbitViewSpec(
        view_id=f"view-{group_index:05d}-court",
        target_kind=OrbitTargetKind.COURT,
        target_court_instance_id=target_court_instance_id,
        target_mode=target_mode,
        coverage_mode=coverage,
        look_at_height_m=(low_height if group_index % 2 == 0 else high_height),
        hfov_degrees=hfov_by_coverage[coverage],
    )
    if group_index != 0:
        if group_index % 2 == 0:
            return (court_view,)
        return (
            OrbitViewSpec(
                view_id=f"view-{group_index:05d}-complex",
                target_kind=OrbitTargetKind.COMPLEX,
                target_court_instance_id=None,
                target_mode="center",
                coverage_mode=coverage,
                look_at_height_m=(
                    low_height if group_index % 2 == 0 else high_height
                ),
                hfov_degrees=hfov_by_coverage[coverage],
            ),
        )
    complex_view = OrbitViewSpec(
        view_id=f"view-{group_index:05d}-complex",
        target_kind=OrbitTargetKind.COMPLEX,
        target_court_instance_id=None,
        target_mode="center",
        coverage_mode=OrbitCoverageMode.NEAR_FULL,
        look_at_height_m=high_height,
        hfov_degrees=hfov_by_coverage[OrbitCoverageMode.NEAR_FULL],
    )
    if group_id != court_view.view_id:
        return court_view, complex_view
    raise ValueError("Opaque group and view IDs unexpectedly collide.")


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
        "center": 0.0,
        "near_baseline": -HALF_LENGTH,
        "far_baseline": HALF_LENGTH,
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
    "SelectedTrajectory",
    "assign_group_disjoint_splits",
    "assign_group_shards",
    "assign_court_targets_for_groups",
    "build_court_dataset_plan",
    "select_budgeted_coverage",
]
