"""Machine-readable and human-readable diagnostics for Court production output."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.camera_sampling.targeting import (
    nearest_court_tie_ids,
)
from src.synthetic_data_generation.dataset.court.components.camera_view import (
    AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    CourtDatasetPlan,
    CourtDatasetPlanAny,
    CourtDatasetPlanV2,
    PlannedCourtSampleV2,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
    court_schema_for_version,
)
from src.synthetic_data_generation.scene_contract import MultiCourtLayout
from src.utils.io import save_json_atomic

DIAGNOSTIC_FILES: tuple[str, ...] = (
    "trajectory-plan.json",
    "sample-points.npy",
    "arc-step-distribution.json",
    "acceptance.json",
    "splits.json",
    "parameter-table.json",
    "semantic-visibility.json",
    "semantic-manifest.json",
    "performance.json",
    "summary.txt",
)


def write_court_diagnostics(
    root: Path,
    *,
    plan: CourtDatasetPlanAny,
    accepted_sample_ids: Sequence[str],
    rejected: Sequence[Mapping[str, object]],
    coverage_counts: Mapping[str, int],
    visible_by_class: Mapping[str, int],
    layout: MultiCourtLayout | None = None,
) -> tuple[str, ...]:
    """Persist the complete required diagnostic inventory at fixed names."""
    accepted = tuple(accepted_sample_ids)
    rejected_tuple = tuple(dict(value) for value in rejected)
    _validate_acceptance_inventory(
        plan,
        accepted_sample_ids=accepted,
        rejected=rejected_tuple,
        coverage_counts=coverage_counts,
    )
    root.mkdir(parents=True, exist_ok=False)
    save_json_atomic(plan.to_dict(), root / "trajectory-plan.json")
    points = np.asarray(
        [sample.camera_center_scene_m for sample in plan.samples],
        dtype=np.float32,
    )
    np.save(root / "sample-points.npy", points, allow_pickle=False)
    steps_by_group = {
        group.trajectory_group_id: {
            "sample_count": group.sample_count,
            "total_arc_length_m": group.total_arc_length_m,
            "maximum_adjacent_step_m": group.maximum_adjacent_step_m,
            "mean_arc_step_m": group.total_arc_length_m / group.sample_count,
        }
        for group in plan.groups
    }
    maximum_steps = np.asarray(
        [group.maximum_adjacent_step_m for group in plan.groups],
        dtype=np.float64,
    )
    definition = court_schema_for_version(plan.schema_version)
    save_json_atomic(
        {
            "schema": definition.arc_step_diagnostics_schema,
            "policy_maximum_m": plan.policy.max_arc_step_m,
            "observed_maximum_m": float(maximum_steps.max()),
            "observed_quantiles_m": _quantiles(maximum_steps),
            "groups": steps_by_group,
        },
        root / "arc-step-distribution.json",
    )
    save_json_atomic(
        {
            "schema": definition.acceptance_diagnostics_schema,
            "proposal_count": plan.proposal_count,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected_tuple),
            "accepted_fraction": len(accepted) / plan.proposal_count,
            "accepted_sample_ids": list(accepted),
            "rejected": list(rejected_tuple),
            "coverage_counts": dict(sorted(coverage_counts.items())),
        },
        root / "acceptance.json",
    )
    if plan.schema_version in (
        CourtDatasetSchemaVersion.V2,
        CourtDatasetSchemaVersion.V3,
    ):
        if not isinstance(plan, CourtDatasetPlanV2):
            raise TypeError("V2/V3 Court diagnostics require a resolved-target plan.")
        if layout is None:
            raise ValueError("V2/V3 Court diagnostics require the accepted layout.")
        resolution = _v2_target_resolution_diagnostics(plan, layout=layout)
        split_groups = {
            group.trajectory_group_id: {
                "split": group.split.value,
                "shard_id": group.shard_id,
                "target_court_policy": group.target_court_policy.to_dict(),
            }
            for group in plan.groups
        }
        parameter_rows = [
            {
                **group.trajectory.to_dict(),
                "view_ids": [view.view_id for view in group.views],
                "split": group.split.value,
                "shard_id": group.shard_id,
                "sample_count_per_view": group.sample_count,
                "target_court_policy": group.target_court_policy.to_dict(),
            }
            for group in plan.groups
        ]
    elif plan.schema_version is CourtDatasetSchemaVersion.V1:
        if not isinstance(plan, CourtDatasetPlan):
            raise TypeError("V1 Court diagnostics require a V1 plan.")
        resolution = None
        split_groups = {
            group.trajectory_group_id: {
                "split": group.split.value,
                "shard_id": group.shard_id,
                "target_court_instance_id": group.target_court.court_instance_id,
            }
            for group in plan.groups
        }
        parameter_rows = [
            {
                **group.trajectory.to_dict(),
                "view_ids": [view.view_id for view in group.views],
                "split": group.split.value,
                "shard_id": group.shard_id,
                "sample_count_per_view": group.sample_count,
                "target_court_instance_id": group.target_court.court_instance_id,
                "candidate_id": group.target_court.candidate_id,
            }
            for group in plan.groups
        ]
    else:  # pragma: no cover - exact plan versions are exhaustive
        raise TypeError("Unsupported Court diagnostic plan version.")
    split_payload: dict[str, object] = {
        "schema": definition.split_diagnostics_schema,
        "groups": split_groups,
    }
    if resolution is not None:
        split_payload["target_resolution"] = resolution
    save_json_atomic(split_payload, root / "splits.json")
    save_json_atomic(
        {
            "schema": definition.parameter_table_schema,
            "rows": parameter_rows,
        },
        root / "parameter-table.json",
    )
    save_json_atomic(
        {
            "schema": definition.semantic_visibility_diagnostics_schema,
            "renderer_visible_points_by_class": dict(sorted(visible_by_class.items())),
            "all_classes_visible": all(
                value > 0 for value in visible_by_class.values()
            ),
        },
        root / "semantic-visibility.json",
    )
    summary = (
        "Canonical Court dataset diagnostics\n"
        f"scene: {plan.scene_id}\n"
        f"profile: {plan.profile}\n"
        f"trajectory groups: {len(plan.groups)}\n"
        f"proposals: {plan.proposal_count}\n"
        f"accepted: {len(accepted)}\n"
        f"rejected: {len(rejected_tuple)}\n"
        f"accepted fraction: {len(accepted) / plan.proposal_count:.6f}\n"
        f"maximum 3-D adjacent step: {maximum_steps.max():.6f} m\n"
        f"coverage: {dict(sorted(coverage_counts.items()))}\n"
        f"renderer-visible semantic classes: {dict(sorted(visible_by_class.items()))}\n"
    )
    (root / "summary.txt").write_text(summary, encoding="utf-8")
    return tuple(f"diagnostics/{name}" for name in DIAGNOSTIC_FILES)


def _validate_acceptance_inventory(
    plan: CourtDatasetPlanAny,
    *,
    accepted_sample_ids: Sequence[str],
    rejected: Sequence[Mapping[str, object]],
    coverage_counts: Mapping[str, int],
) -> None:
    """Fail before writing a mixed-version or incomplete disposition payload."""
    accepted = tuple(accepted_sample_ids)
    rejected_tuple = tuple(rejected)
    if not accepted or len(accepted) != len(set(accepted)) or any(
        not isinstance(sample_id, str)
        or not sample_id
        or sample_id != sample_id.strip()
        for sample_id in accepted
    ):
        raise ValueError("Court acceptance IDs must be unique trimmed strings.")
    rejected_keys = {
        "sample_index",
        "sample_id",
        "trajectory_group_id",
        "trajectory_id",
        "view_id",
        "trajectory_frame_index",
        "split",
        "shard_id",
        "width",
        "height",
        "camera",
        "projection",
        "metadata",
        "reasons",
    }
    if plan.schema_version in (
        CourtDatasetSchemaVersion.V2,
        CourtDatasetSchemaVersion.V3,
    ):
        rejected_keys.add("target_court")
    planned_by_id = {sample.sample_id: sample for sample in plan.samples}
    rejected_ids: list[str] = []
    for record in rejected_tuple:
        if set(record) != rejected_keys:
            raise ValueError(
                "Court acceptance rejection record schema is mixed or invalid."
            )
        sample_id = record["sample_id"]
        reasons = record["reasons"]
        projection = record["projection"]
        if not isinstance(sample_id, str) or sample_id not in planned_by_id:
            raise ValueError("Court acceptance rejection references an unknown sample.")
        if not isinstance(reasons, list) or not reasons or any(
            not isinstance(reason, str)
            or not reason
            or reason != reason.strip()
            for reason in reasons
        ):
            raise ValueError("Court acceptance rejection reasons are incomplete.")
        if projection is None:
            if plan.schema_version not in (
                CourtDatasetSchemaVersion.V2,
                CourtDatasetSchemaVersion.V3,
            ) or not any(
                reason.startswith(f"{AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON}:")
                for reason in reasons
            ):
                raise ValueError(
                    "Only an explicit v2/v3 mid-plane ambiguity may omit projection."
                )
        elif not isinstance(projection, Mapping):
            raise TypeError("Court acceptance rejection projection must be a mapping.")
        planned = planned_by_id[sample_id]
        if (
            record["sample_index"] != planned.sample_index
            or record["trajectory_group_id"] != planned.trajectory_group_id
            or record["trajectory_id"] != planned.trajectory_id
            or record["view_id"] != planned.view_id
            or record["trajectory_frame_index"] != planned.trajectory_frame_index
            or record["split"] != planned.split.value
            or record["shard_id"] != planned.shard_id
            or record["camera"] != planned.camera.to_dict()
        ):
            raise ValueError(
                "Court acceptance rejection disagrees with its planned sample."
            )
        if isinstance(planned, PlannedCourtSampleV2) and (
            record["target_court"] != planned.target_court.to_dict()
        ):
            raise ValueError("Court v2/v3 acceptance rejection target is invalid.")
        rejected_ids.append(sample_id)
    if len(rejected_ids) != len(set(rejected_ids)):
        raise ValueError("Court acceptance rejection IDs must be unique.")
    accepted_set = set(accepted)
    rejected_set = set(rejected_ids)
    expected_ids = [sample.sample_id for sample in plan.samples]
    if accepted_set & rejected_set or accepted_set | rejected_set != set(expected_ids):
        raise ValueError(
            "Court acceptance dispositions do not partition the planned inventory."
        )
    if not isinstance(coverage_counts, Mapping) or any(
        not isinstance(name, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in coverage_counts.items()
    ):
        raise TypeError("Court acceptance coverage counts are invalid.")


def _v2_target_resolution_diagnostics(
    plan: CourtDatasetPlanV2,
    *,
    layout: MultiCourtLayout,
) -> dict[str, object]:
    """Recompute sample-owned v2 targeting evidence for diagnostics."""
    target_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    switch_counts: dict[str, int] = {}
    tie_count = 0
    samples_by_variant: dict[tuple[str, str], list[PlannedCourtSampleV2]] = {}
    for sample in plan.samples:
        target_counts[sample.target_court.binding.court_instance_id] += 1
        policy_counts[sample.target_court.resolution_policy.value] += 1
        samples_by_variant.setdefault(
            (sample.trajectory_group_id, sample.view_id), []
        ).append(sample)
        if (
            sample.target_court.resolution_policy.value == "nearest_camera"
            and len(
                nearest_court_tie_ids(
                    camera_center_scene_m=sample.camera_center_scene_m,
                    layout=layout,
                )
            )
            > 1
        ):
            tie_count += 1
    for group in plan.groups:
        switches = 0
        for view in group.views:
            samples = samples_by_variant[(group.trajectory_group_id, view.view_id)]
            target_ids = [
                sample.target_court.binding.court_instance_id for sample in samples
            ]
            switches += sum(
                current != previous
                for previous, current in zip(target_ids, target_ids[1:], strict=False)
            )
        switch_counts[group.trajectory_group_id] = switches
    return {
        "sample_counts_by_target_court": dict(sorted(target_counts.items())),
        "target_switch_counts_by_trajectory": dict(sorted(switch_counts.items())),
        "resolution_policy_counts": dict(sorted(policy_counts.items())),
        "nearest_court_tie_count": tie_count,
    }


def _quantiles(values: NDArray[np.float64]) -> dict[str, float]:
    quantiles: NDArray[np.float64] = np.asarray(
        np.quantile(values, (0.0, 0.1, 0.5, 0.9, 1.0)),
        dtype=np.float64,
    ).reshape(5)
    return {
        name: float(quantiles[index])
        for index, name in enumerate(("minimum", "p10", "median", "p90", "maximum"))
    }


__all__ = ["DIAGNOSTIC_FILES", "write_court_diagnostics"]
