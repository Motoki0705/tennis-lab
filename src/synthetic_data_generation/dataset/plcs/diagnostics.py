"""Machine-readable and human PLCS motion/camera/court diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.synthetic_data_generation.dataset.plcs.composition import PreparedAvatar
from src.synthetic_data_generation.dataset.plcs.timeline import PLCSGlobalTimeline
from src.tasks.base.generate_dataset.camera_profiles import SampledCameraRig
from src.tasks.base.generate_dataset.court_assignment import CourtAssignment


def write_plcs_diagnostics(
    *,
    staging_directory: Path,
    timeline: PLCSGlobalTimeline,
    rig: SampledCameraRig,
    avatars: dict[str, PreparedAvatar],
    assignments: tuple[CourtAssignment, ...],
    court_instance_ids: tuple[str, ...],
    clip_load_count: int,
    model_load_count: int,
    execution_device: str,
) -> tuple[str, ...]:
    """Persist all required motion, frame, camera, and court-balance evidence."""
    if set(avatars) != {track.object_id for track in timeline.tracks}:
        raise ValueError("Diagnostic avatars differ from the PLCS timeline.")
    selected_source_count = len({track.clip.source_path for track in timeline.tracks})
    selected_gender_count = len({track.clip.gender for track in timeline.tracks})
    if clip_load_count != selected_source_count:
        raise ValueError("PLCS stage did not load each selected source exactly once.")
    if model_load_count != selected_gender_count:
        raise ValueError(
            "PLCS stage did not load each selected gender model exactly once."
        )
    if not execution_device.startswith("cuda"):
        raise ValueError("PLCS production diagnostics require a CUDA execution device.")
    diagnostics = staging_directory / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    if not court_instance_ids or len(court_instance_ids) != len(
        set(court_instance_ids)
    ):
        raise ValueError("Diagnostic court_instance_ids must be non-empty and unique.")
    counts = Counter(item.court_instance_id for item in assignments)
    unknown_courts = set(counts).difference(court_instance_ids)
    if unknown_courts:
        raise ValueError(
            f"Court assignments reference unknown courts: {sorted(unknown_courts)}."
        )
    court_count_values = [counts.get(court_id, 0) for court_id in court_instance_ids]
    balance_difference = (
        max(court_count_values) - min(court_count_values) if court_count_values else 0
    )
    motion_records = []
    for track in timeline.tracks:
        avatar = avatars[track.object_id]
        root = track.clip.root_translation_m.astype(np.float64, copy=False)
        root_relative = root - root[:1]
        motion_records.append(
            {
                **track.clip.metadata(),
                "object_id": track.object_id,
                "root_translation_min_m": root.min(axis=0).tolist(),
                "root_translation_max_m": root.max(axis=0).tolist(),
                "root_relative_extent_m": np.ptp(root_relative, axis=0).tolist(),
                "articulation": avatar.articulation.to_dict(),
                "gaussian_count": avatar.surface_asset.gaussian_count,
            }
        )
    machine = {
        "schema": "tennis_plcs_diagnostics_v2",
        "scene_id": timeline.scene_id,
        "mode": timeline.mode,
        "amass_compatible": True,
        "global_frame_count": timeline.frame_count,
        "source_frame_counts": {
            track.object_id: track.clip.frame_count for track in timeline.tracks
        },
        "motion": motion_records,
        "camera_distribution": {
            "profile": rig.profile,
            "camera_count": len(rig.cameras),
            "camera_ids": [camera.scene_camera.camera_id for camera in rig.cameras],
            "sampled_parameters": [camera.to_metadata() for camera in rig.cameras],
        },
        "target_court": timeline.target_court.to_dict(),
        "stage_cache": {
            "clip_load_count": clip_load_count,
            "selected_source_count": selected_source_count,
            "model_load_count": model_load_count,
            "selected_gender_count": selected_gender_count,
            "execution_device": execution_device,
        },
        "court_balance": {
            "scene_count": len(assignments),
            "counts": {
                court_id: counts.get(court_id, 0)
                for court_id in sorted(court_instance_ids)
            },
            "maximum_count_difference": balance_difference,
        },
    }
    machine_path = diagnostics / "motion-camera-court.json"
    machine_path.write_text(
        json.dumps(machine, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "PLCS production diagnostics",
        f"scene: {timeline.scene_id}",
        f"mode: {timeline.mode}",
        f"global frames: {timeline.frame_count}",
        f"camera profile/count: {rig.profile}/{len(rig.cameras)}",
        f"target court: {timeline.target_court.court_instance_id}",
        f"court maximum count difference: {balance_difference}",
        f"clip/model loads: {clip_load_count}/{model_load_count}",
        f"execution device: {execution_device}",
    ]
    for record in motion_records:
        articulation = record["articulation"]
        summary_lines.append(
            "motion "
            f"{record['object_id']}: {record['category']} {record['frame_count']} frames; "
            f"local residual={articulation['gaussian_nonrigid_residual_m']:.6f} m; "
            f"regions={articulation['region_displacement_m']}"
        )
    summary_path = diagnostics / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return (
        str(machine_path.relative_to(staging_directory)),
        str(summary_path.relative_to(staging_directory)),
    )


__all__ = ["write_plcs_diagnostics"]
