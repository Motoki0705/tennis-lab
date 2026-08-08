"""Exact multi-scene compact PLCS inventory validation and assembly."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.synthetic_data_generation.dataset.contracts import (
    DatasetDomain,
    DatasetManifest,
    FrameInventory,
)
from src.synthetic_data_generation.dataset.plcs.timeline import (
    PLCSFrameEntry,
    PLCSGlobalTimeline,
    PLCSSceneInventory,
)
from src.synthetic_data_generation.dataset.runtime import (
    ChunkReader,
    FinalDatasetAssembler,
)
from src.tasks.base.generate_dataset.camera_profiles import SampledCameraRig
from src.tasks.base.generate_dataset.continuity import (
    FrameContinuityReport,
    TimelineFrameRecord,
    validate_frame_continuity,
)

PLCS_DATASET_SCHEMA = "tennis_plcs_compact_dataset_v3"
PLCS_FRAME_LABEL_SCHEMA = "tennis_plcs_frame_label_v3"


@dataclass(frozen=True, slots=True)
class PLCSSceneAssemblyInput:
    """Attempt-local compact chunks for one intact logical scene."""

    timeline: PLCSGlobalTimeline
    split: str
    rig: SampledCameraRig
    chunk_readers: tuple[ChunkReader, ...]
    attempt_token: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("PLCS scene assembly split is unsupported.")
        if self.rig.court_instance_id != self.timeline.target_court.court_instance_id:
            raise ValueError("PLCS logical scene camera and timeline courts disagree.")
        if not self.chunk_readers:
            raise ValueError("PLCS logical scene requires compact chunk readers.")
        if not self.attempt_token.strip():
            raise ValueError("PLCS logical scene attempt token must be non-empty.")


@dataclass(frozen=True, slots=True)
class PLCSSceneAssemblyResult:
    """Validated inventory evidence for one complete logical scene."""

    scene_id: str
    continuity: FrameContinuityReport
    sample_count: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class PLCSAssemblyResult:
    """Validated canonical multi-scene output and aggregate evidence."""

    manifest: DatasetManifest
    manifest_path: Path
    scenes: tuple[PLCSSceneAssemblyResult, ...]
    sample_count: int
    chunk_count: int

    @property
    def continuity_record_count(self) -> int:
        """Return all per-scene frame/track/camera continuity records."""
        return sum(scene.continuity.record_count for scene in self.scenes)


def build_frame_label(
    *,
    timeline: PLCSGlobalTimeline,
    rig: SampledCameraRig,
    frame_index: int,
    camera_index: int,
    visibility: dict[int, int],
    seed: int,
) -> dict[str, object]:
    """Build complete provenance for every present and absent track."""
    frame = timeline.frames[frame_index]
    camera = rig.cameras[camera_index]
    objects = [
        _object_label(
            entry,
            timeline=timeline,
            visible_pixel_count=visibility.get(entry.instance_id, 0),
        )
        for entry in frame.entries
    ]
    return {
        "schema": PLCS_FRAME_LABEL_SCHEMA,
        "scene_id": timeline.scene_id,
        "frame_index": frame_index,
        "camera_id": camera.scene_camera.camera_id,
        "camera_profile": rig.profile,
        "camera_parameters": camera.to_metadata(),
        "target_court": timeline.target_court.to_dict(),
        "seed": seed,
        "objects": objects,
    }


def _object_label(
    entry: PLCSFrameEntry,
    *,
    timeline: PLCSGlobalTimeline,
    visible_pixel_count: int,
) -> dict[str, object]:
    track = next(
        track for track in timeline.tracks if track.object_id == entry.object_id
    )
    if not entry.present and visible_pixel_count != 0:
        raise ValueError(
            f"Absent object {entry.object_id!r} has visible labelled pixels."
        )
    return {
        "label_id": (
            f"{timeline.scene_id}:{entry.frame_index}:"
            f"{entry.object_id}:{entry.instance_id}"
        ),
        "object_id": entry.object_id,
        "instance_id": entry.instance_id,
        "present": entry.present,
        "source_frame_index": entry.source_frame_index,
        "motion_source": track.clip.source_path,
        "motion_category": track.clip.category.value,
        "gender": track.clip.gender,
        "native_fps": track.clip.fps,
        "scene_from_asset": (
            entry.scene_from_asset.to_dict()
            if entry.scene_from_asset is not None
            else None
        ),
        "visible_pixel_count": visible_pixel_count,
    }


def assemble_plcs_dataset(
    *,
    staging_directory: Path,
    inventory: PLCSSceneInventory,
    scene_inputs: tuple[PLCSSceneAssemblyInput, ...],
    chunk_size: int,
    diagnostics: tuple[str, ...],
    seed: int,
) -> PLCSAssemblyResult:
    """Validate and publish every logical scene without splitting its timeline."""
    if staging_directory.name != "staging" or not staging_directory.is_dir():
        raise ValueError(
            "PLCS assembly requires the runner-provided staging directory."
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    background_directory = staging_directory / "backgrounds"
    if not background_directory.is_dir() or background_directory.is_symlink():
        raise FileNotFoundError(
            "PLCS compact dataset requires one shared background store."
        )
    by_scene = {value.timeline.scene_id: value for value in scene_inputs}
    expected_scene_ids = tuple(
        logical.timeline.scene_id for logical in inventory.scenes
    )
    if len(by_scene) != len(scene_inputs) or set(by_scene) != set(expected_scene_ids):
        raise ValueError(
            "PLCS assembly inputs must cover every logical scene exactly once."
        )

    aggregate_offset = 0
    scene_results: list[PLCSSceneAssemblyResult] = []
    logical_metadata: list[dict[str, object]] = []
    storage_scenes: list[dict[str, object]] = []
    for logical in inventory.scenes:
        value = by_scene[logical.timeline.scene_id]
        if value.timeline is not logical.timeline or value.split != logical.split:
            raise ValueError("PLCS assembly input disagrees with its scene inventory.")
        result = _validate_scene(value, chunk_size=chunk_size, seed=seed)
        scene_results.append(result)
        timeline = value.timeline
        frame_indices = tuple(range(timeline.frame_count))
        local_inventory = FrameInventory(
            source_count=timeline.frame_count,
            planned_indices=frame_indices,
            rendered_indices=frame_indices,
            labelled_indices=frame_indices,
        )
        logical_metadata.append(
            {
                "scene_id": timeline.scene_id,
                "split": value.split,
                "aggregate_frame_offset": aggregate_offset,
                "frame_inventory": local_inventory.to_dict(),
                "mode": timeline.mode,
                "target_court": timeline.target_court.to_dict(),
                "camera_profile": value.rig.profile,
                "cameras": [camera.to_metadata() for camera in value.rig.cameras],
                "motion_sources": [
                    track.clip.metadata() for track in timeline.tracks
                ],
                "tracks": [
                    {
                        "object_id": track.object_id,
                        "instance_id": track.instance_id,
                        "asset_id": track.asset_id,
                        "start_frame": track.start_frame,
                        "stop_frame": track.stop_frame,
                        "anchor_position_court_m": list(
                            track.anchor_position_court_m
                        ),
                        "yaw_radians": track.yaw_radians,
                    }
                    for track in timeline.tracks
                ],
                "continuity": {
                    "frame_count": result.continuity.frame_count,
                    "chunk_count": result.continuity.chunk_count,
                    "track_count": result.continuity.track_count,
                    "camera_count": result.continuity.camera_count,
                    "record_count": result.continuity.record_count,
                },
            }
        )
        storage_scenes.append(
            {
                "scene_id": timeline.scene_id,
                "chunks": [
                    str(reader.directory.relative_to(staging_directory))
                    for reader in value.chunk_readers
                ],
                "attempt_token": value.attempt_token,
                "sample_order": "scene-frame-then-configured-camera",
            }
        )
        aggregate_offset += timeline.frame_count

    aggregate_indices = tuple(range(aggregate_offset))
    frame_inventory = FrameInventory(
        source_count=aggregate_offset,
        planned_indices=aggregate_indices,
        rendered_indices=aggregate_indices,
        labelled_indices=aggregate_indices,
    )
    manifest = DatasetManifest(
        scene_id=inventory.dataset_scene_id,
        domain=DatasetDomain.PLCS,
        schema=PLCS_DATASET_SCHEMA,
        frame_inventory=frame_inventory,
        target_courts=inventory.target_courts,
        metadata={
            "seed": seed,
            "logical_scene_count": inventory.scene_count,
            "aggregate_global_frame_count": inventory.aggregate_global_frame_count,
            "aggregate_source_frame_count": inventory.aggregate_source_frame_count,
            "required_motion_categories": sorted(
                inventory.required_motion_categories
            ),
            "accepted_court_instance_ids": list(
                inventory.accepted_court_instance_ids
            ),
            "logical_scenes": logical_metadata,
        },
        diagnostics=diagnostics,
    )
    manifest_path = staging_directory / "dataset.json"
    payload = {
        "schema": manifest.schema,
        "scene_id": manifest.scene_id,
        "domain": manifest.domain.value,
        "frame_inventory": manifest.frame_inventory.to_dict(),
        "target_courts": [court.to_dict() for court in manifest.target_courts],
        "metadata": manifest.metadata,
        "diagnostics": list(manifest.diagnostics),
        "storage": {
            "layout": "shared-background-plus-per-scene-foreground-delta",
            "background_store": "backgrounds",
            "scenes": storage_scenes,
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return PLCSAssemblyResult(
        manifest=manifest,
        manifest_path=manifest_path,
        scenes=tuple(scene_results),
        sample_count=sum(result.sample_count for result in scene_results),
        chunk_count=sum(result.chunk_count for result in scene_results),
    )


def _validate_scene(
    value: PLCSSceneAssemblyInput,
    *,
    chunk_size: int,
    seed: int,
) -> PLCSSceneAssemblyResult:
    timeline = value.timeline
    camera_ids = tuple(
        camera.scene_camera.camera_id for camera in value.rig.cameras
    )
    validated = FinalDatasetAssembler(
        frame_count=timeline.frame_count,
        camera_ids=camera_ids,
        attempt_token=value.attempt_token,
    ).validate(value.chunk_readers)
    continuity_records: list[TimelineFrameRecord] = []
    for _chunk, reader in zip(validated, value.chunk_readers, strict=True):
        deltas = reader.deltas()
        metadata = reader.metadata()
        for delta, label in zip(deltas, metadata, strict=True):
            frame_index = delta.key.frame_index
            camera_index = camera_ids.index(delta.key.camera_id)
            expected_label = build_frame_label(
                timeline=timeline,
                rig=value.rig,
                frame_index=frame_index,
                camera_index=camera_index,
                visibility=delta.visible_instance_counts,
                seed=seed,
            )
            if label != expected_label:
                raise ValueError(
                    "PLCS compact label semantics disagree with its delta."
                )
            raw_objects = expected_label["objects"]
            if not isinstance(raw_objects, Sequence) or isinstance(
                raw_objects, (str, bytes)
            ):
                raise TypeError("PLCS object labels must be a sequence.")
            for entry, raw_object in zip(
                timeline.frames[frame_index].entries,
                raw_objects,
                strict=True,
            ):
                if not isinstance(raw_object, dict):
                    raise TypeError("PLCS object label must be a mapping.")
                continuity_records.append(
                    TimelineFrameRecord(
                        frame_index=frame_index,
                        chunk_index=frame_index // chunk_size,
                        track_id=entry.object_id,
                        present=entry.present,
                        source_frame_index=entry.source_frame_index,
                        camera_id=delta.key.camera_id,
                        label_id=f"{raw_object['label_id']}:{delta.key.camera_id}",
                        court_instance_id=timeline.target_court.court_instance_id,
                    )
                )
    continuity = validate_frame_continuity(
        continuity_records,
        frame_count=timeline.frame_count,
    )
    return PLCSSceneAssemblyResult(
        scene_id=timeline.scene_id,
        continuity=continuity,
        sample_count=timeline.frame_count * len(camera_ids),
        chunk_count=len(validated),
    )


__all__ = [
    "PLCS_DATASET_SCHEMA",
    "PLCS_FRAME_LABEL_SCHEMA",
    "PLCSAssemblyResult",
    "PLCSSceneAssemblyInput",
    "PLCSSceneAssemblyResult",
    "assemble_plcs_dataset",
    "build_frame_label",
]
