"""Machine-readable and human multi-scene PLCS diagnostics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np

from src.synthetic_data_generation.dataset.camera_profiles import SampledCameraRig
from src.synthetic_data_generation.dataset.plcs.timeline import PLCSSceneInventory


class DiagnosticAvatar(Protocol):
    """Prepared-avatar evidence consumed without constraining test devices."""

    @property
    def surface_asset(self) -> object:
        """Return an object exposing ``gaussian_count``."""

    @property
    def articulation(self) -> object:
        """Return an object exposing ``to_dict``."""


def write_plcs_diagnostics(
    *,
    staging_directory: Path,
    inventory: PLCSSceneInventory,
    rigs: Mapping[str, SampledCameraRig],
    avatars: Mapping[str, DiagnosticAvatar],
    clip_load_count: int,
    model_load_count: int,
    execution_device: str,
    allow_test_cpu_oracle: bool = False,
) -> tuple[str, ...]:
    """Persist complete motion, frame, camera, and court-balance evidence."""
    reference = inventory.scenes[0].timeline
    expected_object_ids = {track.object_id for track in reference.tracks}
    if set(avatars) != expected_object_ids:
        raise ValueError("Diagnostic avatars differ from the PLCS source inventory.")
    if set(rigs) != {scene.timeline.scene_id for scene in inventory.scenes}:
        raise ValueError("Diagnostic camera rigs differ from PLCS logical scenes.")
    selected_source_count = len({track.clip.source_path for track in reference.tracks})
    selected_gender_count = len({track.clip.gender for track in reference.tracks})
    if clip_load_count != selected_source_count:
        raise ValueError("PLCS stage did not load each selected source exactly once.")
    if model_load_count != selected_gender_count:
        raise ValueError(
            "PLCS stage did not load each selected gender model exactly once."
        )
    if execution_device == "test-cpu-oracle":
        if not allow_test_cpu_oracle:
            raise ValueError("PLCS test CPU diagnostics require explicit injection.")
    elif not execution_device.startswith("cuda"):
        raise ValueError("PLCS production diagnostics require a CUDA execution device.")

    diagnostics = staging_directory / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    motion_records = []
    for track in reference.tracks:
        avatar = avatars[track.object_id]
        root = track.clip.root_translation_m.astype(np.float64, copy=False)
        root_relative = root - root[:1]
        surface = avatar.surface_asset
        articulation = avatar.articulation
        gaussian_count = getattr(surface, "gaussian_count", None)
        to_dict = getattr(articulation, "to_dict", None)
        if not isinstance(gaussian_count, int) or not callable(to_dict):
            raise TypeError("Diagnostic avatar evidence is incomplete.")
        motion_records.append(
            {
                **track.clip.metadata(),
                "object_id": track.object_id,
                "root_translation_min_m": root.min(axis=0).tolist(),
                "root_translation_max_m": root.max(axis=0).tolist(),
                "root_relative_extent_m": np.ptp(root_relative, axis=0).tolist(),
                "articulation": to_dict(),
                "gaussian_count": gaussian_count,
            }
        )

    court_counts = Counter(
        scene.timeline.target_court.court_instance_id for scene in inventory.scenes
    )
    per_split: dict[str, Counter[str]] = defaultdict(Counter)
    for scene in inventory.scenes:
        per_split[scene.split][scene.timeline.target_court.court_instance_id] += 1
    court_values = [
        court_counts[court_id] for court_id in inventory.accepted_court_instance_ids
    ]
    logical_scenes = []
    for scene in inventory.scenes:
        timeline = scene.timeline
        rig = rigs[timeline.scene_id]
        if rig.court_instance_id != timeline.target_court.court_instance_id:
            raise ValueError("Diagnostic camera rig and logical-scene court disagree.")
        logical_scenes.append(
            {
                "scene_id": timeline.scene_id,
                "split": scene.split,
                "mode": timeline.mode,
                "global_frame_count": timeline.frame_count,
                "source_frame_counts": {
                    track.object_id: track.clip.frame_count for track in timeline.tracks
                },
                "motion_categories": [
                    track.clip.category.value for track in timeline.tracks
                ],
                "target_court": timeline.target_court.to_dict(),
                "camera_distribution": {
                    "profile": rig.profile,
                    "camera_count": len(rig.cameras),
                    "camera_ids": [
                        camera.scene_camera.camera_id for camera in rig.cameras
                    ],
                    "sampled_parameters": [
                        camera.to_metadata() for camera in rig.cameras
                    ],
                },
            }
        )
    machine = {
        "schema": "tennis_plcs_diagnostics_v3",
        "scene_id": inventory.dataset_scene_id,
        "amass_compatible": True,
        "logical_scene_count": inventory.scene_count,
        "aggregate_global_frame_count": inventory.aggregate_global_frame_count,
        "aggregate_source_frame_count": inventory.aggregate_source_frame_count,
        "motion": motion_records,
        "logical_scenes": logical_scenes,
        "stage_cache": {
            "clip_load_count": clip_load_count,
            "selected_source_count": selected_source_count,
            "model_load_count": model_load_count,
            "selected_gender_count": selected_gender_count,
            "execution_device": execution_device,
        },
        "court_balance": {
            "scene_count": inventory.scene_count,
            "accepted_court_instance_ids": list(inventory.accepted_court_instance_ids),
            "counts": {
                court_id: court_counts[court_id]
                for court_id in inventory.accepted_court_instance_ids
            },
            "maximum_count_difference": max(court_values) - min(court_values),
            "per_split_counts": {
                split: {
                    court_id: counts[court_id]
                    for court_id in inventory.accepted_court_instance_ids
                }
                for split, counts in sorted(per_split.items())
            },
        },
    }
    machine_path = diagnostics / "motion-camera-court.json"
    machine_path.write_text(
        json.dumps(machine, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "PLCS production diagnostics",
        f"scene: {inventory.dataset_scene_id}",
        f"logical scenes: {inventory.scene_count}",
        f"aggregate global frames: {inventory.aggregate_global_frame_count}",
        f"aggregate source frames: {inventory.aggregate_source_frame_count}",
        f"accepted courts: {', '.join(inventory.accepted_court_instance_ids)}",
        f"court maximum count difference: {max(court_values) - min(court_values)}",
        f"clip/model loads: {clip_load_count}/{model_load_count}",
        f"execution device: {execution_device}",
    ]
    for scene in inventory.scenes:
        summary_lines.append(
            "logical scene "
            f"{scene.timeline.scene_id}: {scene.timeline.frame_count} frames; "
            f"split={scene.split}; "
            f"court={scene.timeline.target_court.court_instance_id}; "
            f"cameras={len(rigs[scene.timeline.scene_id].cameras)}"
        )
    for record in motion_records:
        articulation = record["articulation"]
        if not isinstance(articulation, dict):
            raise TypeError("Articulation diagnostic must be a mapping.")
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


__all__ = ["DiagnosticAvatar", "write_plcs_diagnostics"]
