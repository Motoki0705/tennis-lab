"""Exact compact PLCS chunk/label inventory validation and assembly."""

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

PLCS_DATASET_SCHEMA = "tennis_plcs_compact_dataset_v2"
PLCS_FRAME_LABEL_SCHEMA = "tennis_plcs_frame_label_v2"


@dataclass(frozen=True, slots=True)
class PLCSAssemblyResult:
    """Validated canonical compact output and continuity evidence."""

    manifest: DatasetManifest
    manifest_path: Path
    continuity: FrameContinuityReport
    sample_count: int
    chunk_count: int


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
    timeline: PLCSGlobalTimeline,
    rig: SampledCameraRig,
    chunk_readers: tuple[ChunkReader, ...],
    attempt_token: str,
    chunk_size: int,
    diagnostics: tuple[str, ...],
    seed: int,
) -> PLCSAssemblyResult:
    """Validate compact chunks once as a complete global frame-camera stream."""
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
    camera_ids = tuple(camera.scene_camera.camera_id for camera in rig.cameras)
    validated = FinalDatasetAssembler(
        frame_count=timeline.frame_count,
        camera_ids=camera_ids,
        attempt_token=attempt_token,
    ).validate(chunk_readers)
    continuity_records: list[TimelineFrameRecord] = []
    rendered_frames: set[int] = set()
    labelled_frames: set[int] = set()
    for _chunk, reader in zip(validated, chunk_readers, strict=True):
        deltas = reader.deltas()
        metadata = reader.metadata()
        for delta, label in zip(deltas, metadata, strict=True):
            frame_index = delta.key.frame_index
            camera_index = camera_ids.index(delta.key.camera_id)
            expected_label = build_frame_label(
                timeline=timeline,
                rig=rig,
                frame_index=frame_index,
                camera_index=camera_index,
                visibility=delta.visible_instance_counts,
                seed=seed,
            )
            if label != expected_label:
                raise ValueError(
                    "PLCS compact label semantics disagree with its delta."
                )
            rendered_frames.add(frame_index)
            labelled_frames.add(frame_index)
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
    frame_indices = tuple(range(timeline.frame_count))
    inventory = FrameInventory(
        source_count=timeline.frame_count,
        planned_indices=frame_indices,
        rendered_indices=tuple(sorted(rendered_frames)),
        labelled_indices=tuple(sorted(labelled_frames)),
    )
    manifest = DatasetManifest(
        scene_id=timeline.scene_id,
        domain=DatasetDomain.PLCS,
        schema=PLCS_DATASET_SCHEMA,
        frame_inventory=inventory,
        target_courts=(timeline.target_court,),
        metadata={
            "mode": timeline.mode,
            "seed": seed,
            "camera_profile": rig.profile,
            "cameras": [camera.to_metadata() for camera in rig.cameras],
            "motion_sources": [track.clip.metadata() for track in timeline.tracks],
            "tracks": [
                {
                    "object_id": track.object_id,
                    "instance_id": track.instance_id,
                    "asset_id": track.asset_id,
                    "start_frame": track.start_frame,
                    "stop_frame": track.stop_frame,
                    "anchor_position_court_m": list(track.anchor_position_court_m),
                    "yaw_radians": track.yaw_radians,
                }
                for track in timeline.tracks
            ],
            "continuity": {
                "frame_count": continuity.frame_count,
                "chunk_count": continuity.chunk_count,
                "track_count": continuity.track_count,
                "camera_count": continuity.camera_count,
                "record_count": continuity.record_count,
            },
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
            "layout": "shared-background-plus-foreground-delta",
            "background_store": "backgrounds",
            "chunks": [
                str(reader.directory.relative_to(staging_directory))
                for reader in chunk_readers
            ],
            "attempt_token": attempt_token,
            "sample_order": "global-frame-then-configured-camera",
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return PLCSAssemblyResult(
        manifest=manifest,
        manifest_path=manifest_path,
        continuity=continuity,
        sample_count=timeline.frame_count * len(camera_ids),
        chunk_count=len(validated),
    )


__all__ = [
    "PLCS_DATASET_SCHEMA",
    "PLCS_FRAME_LABEL_SCHEMA",
    "PLCSAssemblyResult",
    "assemble_plcs_dataset",
    "build_frame_label",
]
