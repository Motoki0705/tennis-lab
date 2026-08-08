"""Machine-readable and human-readable diagnostics for full BLCS timelines."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from src.synthetic_data_generation.dataset.blcs.contracts import BLCSSampleRecord
from src.synthetic_data_generation.dataset.blcs.timeline import BLCSTrajectoryPlan
from src.synthetic_data_generation.dataset.contracts import FrameInventory
from src.tasks.base.generate_dataset.continuity import FrameContinuityReport


def write_blcs_diagnostics(
    output_directory: Path,
    *,
    plans: Sequence[BLCSTrajectoryPlan],
    inventory: FrameInventory,
    continuity: FrameContinuityReport,
    records: Sequence[BLCSSampleRecord],
    rendered_visible_object_views: int,
) -> tuple[str, ...]:
    """Write exact timeline, camera, visibility, and court-balance evidence."""
    diagnostics = output_directory / "diagnostics"
    if diagnostics.exists():
        raise FileExistsError(
            "BLCS diagnostics must be written once per stage attempt."
        )
    diagnostics.mkdir(parents=True, exist_ok=False)
    court_counts = Counter(plan.target_court.court_instance_id for plan in plans)
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for plan in plans:
        split_counts[plan.source.split][plan.target_court.court_instance_id] += 1
    geometric_visible = sum(int(plan.geometric_visible.sum()) for plan in plans)
    metrics = {
        "schema": "canonical_blcs_diagnostics_v1",
        "frame_inventory": inventory.to_dict(),
        "trajectory_count": len(plans),
        "object_count": sum(plan.source.object_count for plan in plans),
        "sample_count": len(records),
        "chunk_count": continuity.chunk_count,
        "track_count": continuity.track_count,
        "camera_count": continuity.camera_count,
        "continuity_record_count": continuity.record_count,
        "camera_profiles": sorted({plan.camera_rig.profile for plan in plans}),
        "camera_counts_per_trajectory": {
            plan.source.trajectory_id: len(plan.camera_rig.cameras) for plan in plans
        },
        "court_counts": dict(sorted(court_counts.items())),
        "court_count_difference": max(court_counts.values())
        - min(court_counts.values()),
        "court_counts_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_counts.items())
        },
        "geometric_visible_object_views": geometric_visible,
        "rendered_visible_object_views": rendered_visible_object_views,
        "trajectories": [
            {
                "trajectory_id": plan.source.trajectory_id,
                "split": plan.source.split,
                "source": plan.source.frame_count,
                "planned": len(plan.global_frame_indices),
                "rendered": plan.source.frame_count,
                "labelled": plan.source.frame_count,
                "first_source_frame": 0,
                "last_source_frame": plan.source.frame_count - 1,
                "chunk_count": len(plan.chunks),
                "target_court": plan.target_court.court_instance_id,
                "candidate_id": plan.target_court.candidate_id,
                "target_court_transform_metric": (
                    plan.target_court.scene_from_court.to_list()
                ),
                "camera_profile": plan.camera_rig.profile,
                "camera_count": len(plan.camera_rig.cameras),
                "camera_poses_metric": {
                    camera.scene_camera.camera_id: (
                        camera.scene_camera.camera_to_scene.to_list()
                    )
                    for camera in plan.camera_rig.cameras
                },
                "seed": {
                    "court_assignment": plan.target_court.selection_seed,
                    "camera_sampling": plan.camera_rig.seed,
                },
            }
            for plan in plans
        ],
    }
    metrics_path = diagnostics / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    camera_profiles = sorted({plan.camera_rig.profile for plan in plans})
    summary = (
        "Canonical BLCS full-timeline diagnostics\n"
        f"trajectories: {len(plans)}\n"
        f"source/planned/rendered/labelled: {inventory.source_count}/"
        f"{len(inventory.planned_indices)}/{len(inventory.rendered_indices)}/"
        f"{len(inventory.labelled_indices)}\n"
        f"samples: {len(records)}\n"
        f"chunks: {continuity.chunk_count}\n"
        f"camera profiles: {', '.join(camera_profiles)}\n"
        f"court count difference: {metrics['court_count_difference']}\n"
        f"rendered visible object views: {rendered_visible_object_views}\n"
    )
    (diagnostics / "summary.txt").write_text(summary, encoding="utf-8")
    return ("diagnostics/metrics.json", "diagnostics/summary.txt")


__all__ = ["write_blcs_diagnostics"]
