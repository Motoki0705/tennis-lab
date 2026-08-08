"""Machine-readable and human-readable diagnostics for Court production output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.contracts import CourtDatasetPlan
from src.utils.io import save_json_atomic

DIAGNOSTIC_FILES: tuple[str, ...] = (
    "trajectory-plan.json",
    "sample-points.npy",
    "arc-step-distribution.json",
    "acceptance.json",
    "splits.json",
    "parameter-table.json",
    "semantic-visibility.json",
    "performance.json",
    "summary.txt",
)


def write_court_diagnostics(
    root: Path,
    *,
    plan: CourtDatasetPlan,
    accepted_sample_ids: Sequence[str],
    rejected: Sequence[Mapping[str, object]],
    coverage_counts: Mapping[str, int],
    visible_by_class: Mapping[str, int],
) -> tuple[str, ...]:
    """Persist the complete required diagnostic inventory at fixed names."""
    root.mkdir(parents=True, exist_ok=False)
    accepted = tuple(accepted_sample_ids)
    rejected_tuple = tuple(dict(value) for value in rejected)
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
    save_json_atomic(
        {
            "schema": "court_arc_step_diagnostics_v1",
            "policy_maximum_m": plan.policy.max_arc_step_m,
            "observed_maximum_m": float(maximum_steps.max()),
            "observed_quantiles_m": _quantiles(maximum_steps),
            "groups": steps_by_group,
        },
        root / "arc-step-distribution.json",
    )
    save_json_atomic(
        {
            "schema": "court_acceptance_diagnostics_v1",
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
    save_json_atomic(
        {
            "schema": "court_split_diagnostics_v1",
            "groups": {
                group.trajectory_group_id: {
                    "split": group.split.value,
                    "shard_id": group.shard_id,
                    "target_court_instance_id": group.target_court.court_instance_id,
                }
                for group in plan.groups
            },
        },
        root / "splits.json",
    )
    save_json_atomic(
        {
            "schema": "court_parameter_table_v1",
            "rows": [
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
            ],
        },
        root / "parameter-table.json",
    )
    save_json_atomic(
        {
            "schema": "court_semantic_visibility_diagnostics_v1",
            "renderer_visible_points_by_class": dict(sorted(visible_by_class.items())),
            "all_classes_visible": all(value > 0 for value in visible_by_class.values()),
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
