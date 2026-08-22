"""Strict, frame-streaming sources for canonical generated datasets."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.blcs.assembler import (
    validate_blcs_dataset_envelope,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCS_DATASET_SCHEMA,
    BLCS_SAMPLE_SCHEMA,
    BLCSSampleRecord,
)
from src.synthetic_data_generation.dataset.court.assembler import (
    CourtArrayValidationMode,
    validate_court_dataset,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
    court_schema_from_dataset_schema,
)
from src.synthetic_data_generation.dataset.plcs.assembler import (
    PLCS_DATASET_SCHEMA,
    PLCS_FRAME_LABEL_SCHEMA,
    PLCSSupervisionArrays,
)
from src.synthetic_data_generation.dataset.plcs.coordinates import (
    PLCSCoordinateContract,
    PLCSSourceSupportPlane,
)
from src.synthetic_data_generation.dataset.runtime import (
    ChunkReader,
    ForegroundDelta,
    LogicalRenderSample,
    RenderSampleKey,
    SharedBackgroundStore,
    materialize_logical_sample,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


@dataclass(frozen=True, slots=True)
class CourtSourceFrame:
    """One manifest-ordered Court image and its semantic label payload."""

    rgb: NDArray[np.float32]
    sample_id: str
    view_id: str
    trajectory_frame_index: int
    projection: Mapping[str, object]
    schema_version: CourtDatasetSchemaVersion = CourtDatasetSchemaVersion.V1


@dataclass(frozen=True, slots=True)
class BLCSSourceFrame:
    """One exact compact BLCS view reconstructed on demand."""

    render: LogicalRenderSample
    source_frame_index: int
    global_frame_index: int
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PLCSSourceFrame:
    """One exact compact PLCS view plus selected dense supervision."""

    render: LogicalRenderSample
    frame_index: int
    label: Mapping[str, object]
    human_kp: NDArray[np.float32]
    human_vis: NDArray[np.bool_]
    court_kp: NDArray[np.float32]
    court_vis: NDArray[np.bool_]
    present: NDArray[np.bool_]


class CourtVisualizationSource:
    """Validated Court trajectory reader preserving canonical manifest order."""

    def __init__(self, root: Path, *, trajectory_id: str) -> None:
        validate_court_dataset(
            root,
            array_validation=CourtArrayValidationMode.FULL,
        )
        manifest = _object(
            _load_json(_contained_file(root, "dataset.json")),
            name="Court dataset",
        )
        self.dataset_schema = _text(manifest.get("schema"), name="Court schema")
        self.schema_definition = court_schema_from_dataset_schema(self.dataset_schema)
        self.dataset_scene_id = _text(manifest.get("scene_id"), name="Court scene_id")
        groups = tuple(
            _object(value, name="Court trajectory group")
            for value in _array(
                manifest.get("trajectory_groups"), name="Court trajectory_groups"
            )
        )
        selected_group: Mapping[str, object] | None = None
        for group in groups:
            trajectory = _object(group.get("trajectory"), name="Court trajectory")
            if trajectory.get("trajectory_id") == trajectory_id:
                selected_group = group
                break
        if selected_group is None:
            raise KeyError(f"Unknown Court trajectory_id: {trajectory_id!r}.")
        views = tuple(
            _text(
                _object(value, name="Court view").get("view_id"),
                name="Court view_id",
            )
            for value in _array(selected_group.get("views"), name="Court views")
        )
        if not views or len(views) != len(set(views)):
            raise ValueError("Court trajectory view inventory is invalid.")
        sample_count = _positive_integer(
            selected_group.get("sample_count"), name="Court sample_count"
        )
        records = tuple(
            record
            for value in _array(manifest.get("samples"), name="Court samples")
            for record in (_object(value, name="Court sample"),)
            if record.get("trajectory_id") == trajectory_id
        )
        if not records:
            raise ValueError(
                f"Court trajectory {trajectory_id!r} has no accepted rendered frames."
            )
        expected_order = tuple(
            record
            for view_id in views
            for record in records
            if record.get("view_id") == view_id
        )
        if records != expected_order:
            raise ValueError(
                "Court trajectory frames do not follow canonical view/frame order."
            )
        for view_id in views:
            indices = tuple(
                _nonnegative_integer(
                    record.get("trajectory_frame_index"),
                    name="Court trajectory_frame_index",
                )
                for record in records
                if record.get("view_id") == view_id
            )
            if not indices or indices != tuple(sorted(set(indices))):
                raise ValueError(
                    f"Court view {view_id!r} source-frame ordering is inconsistent."
                )
            if indices[-1] >= sample_count:
                raise ValueError("Court frame index exceeds its trajectory inventory.")
        dimensions = {
            (
                _positive_integer(record.get("width"), name="Court width"),
                _positive_integer(record.get("height"), name="Court height"),
            )
            for record in records
        }
        if len(dimensions) != 1:
            raise ValueError("Court trajectory frame dimensions are inconsistent.")
        self.width, self.height = next(iter(dimensions))
        self.root = root
        self.trajectory_id = trajectory_id
        self._records = records
        self.frame_order = tuple(
            {
                "sample_id": _text(record.get("sample_id"), name="sample_id"),
                "view_id": _text(record.get("view_id"), name="view_id"),
                "trajectory_frame_index": _nonnegative_integer(
                    record.get("trajectory_frame_index"),
                    name="trajectory_frame_index",
                ),
            }
            for record in records
        )

    @property
    def frame_count(self) -> int:
        """Return the accepted rendered-frame count for the selected trajectory."""
        return len(self._records)

    def frames(self) -> Iterator[CourtSourceFrame]:
        """Stream each NHT RGB array and corresponding label in manifest order."""
        for record in self._records:
            rgb = _float32_rgb(
                _contained_file(
                    self.root,
                    _text(record.get("rgb"), name="Court rgb path"),
                ),
                width=self.width,
                height=self.height,
            )
            label = _object(
                _load_json(
                    _contained_file(
                        self.root,
                        _text(record.get("labels"), name="Court labels path"),
                    )
                ),
                name="Court labels",
            )
            label_schema = label.get("schema")
            if (
                not isinstance(label_schema, str)
                or label_schema != self.schema_definition.sample_schema
            ):
                raise ValueError("Court labels schema changed after validation.")
            for field in (
                "sample_id",
                "view_id",
                "trajectory_frame_index",
                "projection",
            ):
                if label.get(field) != record.get(field):
                    raise ValueError(
                        f"Court labels changed after validation at field {field!r}."
                    )
            yield CourtSourceFrame(
                rgb=rgb,
                sample_id=_text(label["sample_id"], name="Court sample_id"),
                view_id=_text(label["view_id"], name="Court view_id"),
                trajectory_frame_index=_nonnegative_integer(
                    label["trajectory_frame_index"],
                    name="Court trajectory_frame_index",
                ),
                projection=_object(label["projection"], name="Court projection"),
                schema_version=self.schema_definition.version,
            )


class BLCSVisualizationSource:
    """Bounded-memory compact BLCS reader for one trajectory-camera view."""

    def __init__(
        self,
        root: Path,
        *,
        logical_scene_id: str,
        camera_id: str,
    ) -> None:
        validate_blcs_dataset_envelope(root)
        manifest = _exact_object(
            _load_json(_contained_file(root, "dataset.json")),
            name="BLCS dataset",
            keys={
                "schema",
                "scene_id",
                "domain",
                "frame_inventory",
                "target_courts",
                "metadata",
                "diagnostics",
                "performance",
                "trajectories",
                "samples",
            },
        )
        if (
            manifest.get("schema") != BLCS_DATASET_SCHEMA
            or manifest.get("domain") != "blcs"
        ):
            raise ValueError("Unsupported canonical compact BLCS schema/domain.")
        self.dataset_schema = BLCS_DATASET_SCHEMA
        self.dataset_scene_id = _text(manifest.get("scene_id"), name="BLCS scene_id")
        trajectories = tuple(
            _exact_object(
                value,
                name="BLCS trajectory",
                keys={
                    "trajectory_id",
                    "split",
                    "source_frame_count",
                    "global_frame_offset",
                    "frame_inventory",
                    "target_court",
                    "candidate_id",
                    "transform",
                    "camera_profile",
                    "camera_seed",
                    "camera_ids",
                    "attempt_token",
                    "chunk_count",
                    "chunk_directories",
                    "background_store",
                    "plan_json",
                    "plan_npz",
                },
            )
            for value in _array(manifest.get("trajectories"), name="BLCS trajectories")
        )
        trajectory_ids = tuple(
            _text(value.get("trajectory_id"), name="BLCS trajectory_id")
            for value in trajectories
        )
        if not trajectory_ids or len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("BLCS trajectory IDs must be non-empty and unique.")
        try:
            trajectory = next(
                value
                for value in trajectories
                if value.get("trajectory_id") == logical_scene_id
            )
        except StopIteration as error:
            raise KeyError(
                f"Unknown BLCS logical_scene_id/trajectory_id: {logical_scene_id!r}."
            ) from error
        camera_ids = tuple(
            _text(value, name="BLCS camera_id")
            for value in _array(trajectory.get("camera_ids"), name="BLCS camera_ids")
        )
        if not camera_ids or len(camera_ids) != len(set(camera_ids)):
            raise ValueError("BLCS camera IDs must be non-empty and unique.")
        if camera_id not in camera_ids:
            raise KeyError(
                f"Unknown BLCS camera_id {camera_id!r} for {logical_scene_id!r}."
            )
        frame_count = _positive_integer(
            trajectory.get("source_frame_count"), name="BLCS source_frame_count"
        )
        all_records = tuple(
            _blcs_sample_record(value)
            for value in _array(manifest.get("samples"), name="BLCS samples")
        )
        trajectory_records = tuple(
            record for record in all_records if record.trajectory_id == logical_scene_id
        )
        records = tuple(
            record for record in trajectory_records if record.camera_id == camera_id
        )
        if tuple(record.source_frame_index for record in records) != tuple(
            range(frame_count)
        ):
            raise ValueError(
                "BLCS selected view does not cover source frames exactly in order."
            )
        plan = _exact_object(
            _load_json(
                _contained_file(
                    root,
                    _text(trajectory.get("plan_json"), name="BLCS plan_json"),
                )
            ),
            name="BLCS plan",
            keys={
                "trajectory_id",
                "split",
                "fps",
                "source_frame_count",
                "global_frame_offset",
                "global_frame_indices",
                "tracks",
                "target_court",
                "camera_profile",
                "camera_seed",
                "cameras",
                "chunks",
                "composition",
                "source_metadata",
            },
        )
        if (
            plan.get("trajectory_id") != logical_scene_id
            or plan.get("source_frame_count") != frame_count
        ):
            raise ValueError("BLCS plan identity/frame inventory is inconsistent.")
        self.source_fps = _positive_number(plan.get("fps"), name="BLCS fps")
        planned_cameras = tuple(
            SceneCamera.from_dict(
                _exact_object(
                    value,
                    name="BLCS planned camera",
                    keys={
                        "slot_id",
                        "court_local_center_m",
                        "court_local_look_at_m",
                        "hfov_degrees",
                        "camera",
                    },
                )["camera"]
            )
            for value in _array(plan.get("cameras"), name="BLCS cameras")
        )
        if tuple(value.camera_id for value in planned_cameras) != camera_ids:
            raise ValueError("BLCS plan and manifest camera order differs.")
        tracks = tuple(
            _object(value, name="BLCS track")
            for value in _array(plan.get("tracks"), name="BLCS tracks")
        )
        self.object_ids = tuple(
            _text(track.get("object_id"), name="BLCS object_id") for track in tracks
        )
        if not self.object_ids or len(self.object_ids) != len(set(self.object_ids)):
            raise ValueError("BLCS object IDs must be non-empty and unique.")
        attempt_token = _text(
            trajectory.get("attempt_token"), name="BLCS attempt_token"
        )
        planned_chunks = tuple(
            _exact_object(
                value,
                name="BLCS plan chunk",
                keys={"chunk_index", "frame_indices"},
            )
            for value in _array(plan.get("chunks"), name="BLCS plan chunks")
        )
        chunk_paths = tuple(
            _contained_directory(root, _text(value, name="BLCS chunk path"))
            for value in _array(
                trajectory.get("chunk_directories"),
                name="BLCS chunk_directories",
            )
        )
        if len(chunk_paths) != _positive_integer(
            trajectory.get("chunk_count"), name="chunk_count"
        ) or len(chunk_paths) != len(planned_chunks):
            raise ValueError("BLCS chunk inventory is inconsistent.")
        expected_records: list[BLCSSampleRecord] = []
        global_offset = _nonnegative_integer(
            trajectory.get("global_frame_offset"), name="global_frame_offset"
        )
        split = _text(trajectory.get("split"), name="BLCS split")
        background_relative = _text(
            trajectory.get("background_store"), name="BLCS background_store"
        )
        for chunk_index, (chunk_path, planned_chunk) in enumerate(
            zip(chunk_paths, planned_chunks, strict=True)
        ):
            if planned_chunk.get("chunk_index") != chunk_index:
                raise ValueError("BLCS chunk indices are not exact contiguous order.")
            frame_indices = tuple(
                _nonnegative_integer(value, name="BLCS chunk frame_index")
                for value in _array(
                    planned_chunk.get("frame_indices"), name="BLCS frame_indices"
                )
            )
            reader = ChunkReader(chunk_path)
            validated = reader.validate(expected_attempt_token=attempt_token)
            expected_keys = tuple(
                RenderSampleKey(frame_index, planned_camera_id)
                for frame_index in frame_indices
                for planned_camera_id in camera_ids
            )
            if validated.keys != expected_keys:
                raise ValueError(
                    "BLCS compact chunk frame/camera order is inconsistent."
                )
            expected_records.extend(
                BLCSSampleRecord(
                    trajectory_id=logical_scene_id,
                    split=split,
                    global_frame_index=global_offset + key.frame_index,
                    source_frame_index=key.frame_index,
                    chunk_index=chunk_index,
                    camera_id=key.camera_id,
                    background_store=background_relative,
                    foreground_chunk=chunk_path.relative_to(
                        root.resolve(strict=True)
                    ).as_posix(),
                    chunk_sample_index=ordinal,
                )
                for ordinal, key in enumerate(validated.keys)
            )
        if tuple(expected_records) != trajectory_records:
            raise ValueError("BLCS manifest records differ from compact chunk order.")
        if any(record.background_store != background_relative for record in records):
            raise ValueError("BLCS selected view changes background stores.")
        self._backgrounds = SharedBackgroundStore(
            _contained_directory(root, background_relative)
        )
        if self._backgrounds.camera_ids != camera_ids:
            raise ValueError("BLCS camera and background inventories differ.")
        background = self._backgrounds.load(camera_id)
        self.width = background.width
        self.height = background.height
        camera_index = camera_ids.index(camera_id)
        camera = planned_cameras[camera_index]
        if (self.width, self.height) != (
            camera.width,
            camera.height,
        ):
            raise ValueError("BLCS selected camera/background dimensions differ.")
        with np.load(
            _contained_file(
                root,
                _text(trajectory.get("plan_npz"), name="BLCS plan_npz"),
            ),
            allow_pickle=False,
        ) as archive:
            expected_array_names = {
                "positions_court_m",
                "velocities_court_mps",
                "present",
                "positions_scene",
                "camera_uv",
                "camera_depth",
                "geometric_visible",
                "court_uv",
                "court_visible",
            }
            if set(archive.files) != expected_array_names:
                raise ValueError("BLCS plan array inventory is invalid.")
            court_uv = np.asarray(archive["court_uv"])[camera_index]
            court_visible = np.asarray(archive["court_visible"])[camera_index]
        if (
            court_uv.shape != (20, 2)
            or court_uv.dtype != np.float64
            or court_visible.shape != (20,)
            or court_visible.dtype != np.bool_
            or not np.isfinite(court_uv).all()
        ):
            raise ValueError("BLCS selected court supervision is invalid.")
        self.court_kp = np.asarray(court_uv, dtype=np.float32)
        self.court_vis = np.asarray(court_visible, dtype=np.bool_)
        self.root = root
        self.logical_scene_id = logical_scene_id
        self.camera_id = camera_id
        self._attempt_token = attempt_token
        self._records = records
        self.frame_order = tuple(
            {
                "source_frame_index": record.source_frame_index,
                "global_frame_index": record.global_frame_index,
            }
            for record in records
        )

    @property
    def frame_count(self) -> int:
        """Return the complete source trajectory length."""
        return len(self._records)

    def frames(self) -> Iterator[BLCSSourceFrame]:
        """Stream one compact chunk at a time and yield the selected camera."""
        active_path: Path | None = None
        deltas: tuple[ForegroundDelta, ...] = ()
        metadata: tuple[Mapping[str, object], ...] = ()
        for record in self._records:
            chunk_path = _contained_directory(self.root, record.foreground_chunk)
            if chunk_path != active_path:
                reader = ChunkReader(chunk_path)
                reader.validate(expected_attempt_token=self._attempt_token)
                deltas = reader.deltas()
                metadata = cast(tuple[Mapping[str, object], ...], reader.metadata())
                active_path = chunk_path
            ordinal = record.chunk_sample_index
            if ordinal >= len(deltas) or ordinal >= len(metadata):
                raise ValueError("BLCS sample ordinal exceeds its compact chunk.")
            delta = deltas[ordinal]
            expected = RenderSampleKey(record.source_frame_index, self.camera_id)
            if delta.key != expected:
                raise ValueError("BLCS source-frame/camera ordering changed in chunk.")
            sample_metadata = _exact_object(
                metadata[ordinal],
                name="BLCS sample metadata",
                keys={
                    "schema",
                    "scene_id",
                    "trajectory_id",
                    "split",
                    "global_frame_index",
                    "source_frame_index",
                    "chunk_index",
                    "source_trajectory",
                    "source_frame",
                    "target_court",
                    "candidate_id",
                    "transform",
                    "camera_profile",
                    "camera_parameters",
                    "seed",
                    "objects",
                    "semantic_arrays",
                },
            )
            if (
                sample_metadata["schema"] != BLCS_SAMPLE_SCHEMA
                or sample_metadata["scene_id"] != self.dataset_scene_id
                or sample_metadata["trajectory_id"] != self.logical_scene_id
                or sample_metadata["source_frame_index"] != record.source_frame_index
                or sample_metadata["global_frame_index"] != record.global_frame_index
            ):
                raise ValueError(
                    "BLCS sample metadata identity changed during streaming."
                )
            _exact_object(
                sample_metadata["semantic_arrays"],
                name="BLCS semantic arrays",
                keys={
                    "ball_uv",
                    "ball_depth",
                    "geometric_visible",
                    "rendered_visible",
                    "positions_court_m",
                    "positions_scene",
                    "velocities_court_mps",
                    "present",
                    "source_frame_indices",
                    "instance_ids",
                },
            )
            objects = tuple(
                _exact_object(
                    value,
                    name="BLCS object label",
                    keys={
                        "object_id",
                        "instance_id",
                        "present",
                        "source_trajectory",
                        "source_frame",
                        "geometric_visible",
                        "rendered_visible",
                    },
                )
                for value in _array(sample_metadata["objects"], name="BLCS objects")
            )
            if tuple(value.get("object_id") for value in objects) != self.object_ids:
                raise ValueError("BLCS label object order changed during streaming.")
            render = materialize_logical_sample(
                self._backgrounds.load(self.camera_id),
                delta,
            )
            yield BLCSSourceFrame(
                render=render,
                source_frame_index=record.source_frame_index,
                global_frame_index=record.global_frame_index,
                metadata=sample_metadata,
            )


@dataclass(frozen=True, slots=True)
class _PLCSChunkSelection:
    path: Path
    ordinals: tuple[int, ...]


class PLCSVisualizationSource:
    """Bounded-memory compact PLCS reader for one logical scene-camera view."""

    def __init__(
        self,
        root: Path,
        *,
        logical_scene_id: str,
        camera_id: str,
    ) -> None:
        if (
            root.is_symlink()
            or not root.is_dir()
            or {path.name for path in root.iterdir()}
            != {"dataset.json", "backgrounds", "scenes", "diagnostics"}
        ):
            raise ValueError("PLCS dataset owner inventory is not canonical.")
        manifest = _exact_object(
            _load_json(_contained_file(root, "dataset.json")),
            name="PLCS dataset",
            keys={
                "schema",
                "scene_id",
                "domain",
                "frame_inventory",
                "target_courts",
                "metadata",
                "diagnostics",
                "storage",
            },
        )
        if (
            manifest.get("schema") != PLCS_DATASET_SCHEMA
            or manifest.get("domain") != "plcs"
        ):
            raise ValueError("Unsupported canonical compact PLCS schema/domain.")
        self.dataset_schema = PLCS_DATASET_SCHEMA
        self.dataset_scene_id = _text(
            manifest.get("scene_id"), name="PLCS dataset scene_id"
        )
        metadata = _exact_object(
            manifest.get("metadata"),
            name="PLCS metadata",
            keys={
                "coordinate_contract",
                "seed",
                "logical_scene_count",
                "aggregate_global_frame_count",
                "aggregate_source_frame_count",
                "required_motion_categories",
                "accepted_court_instance_ids",
                "logical_scenes",
            },
        )
        PLCSCoordinateContract.from_dict(metadata.get("coordinate_contract"))
        logical_scenes = tuple(
            _exact_object(
                value,
                name="PLCS logical scene",
                keys={
                    "scene_id",
                    "split",
                    "aggregate_frame_offset",
                    "frame_inventory",
                    "mode",
                    "target_court",
                    "camera_profile",
                    "cameras",
                    "motion_sources",
                    "tracks",
                    "continuity",
                },
            )
            for value in _array(
                metadata.get("logical_scenes"), name="PLCS logical_scenes"
            )
        )
        try:
            logical = next(
                value
                for value in logical_scenes
                if value.get("scene_id") == logical_scene_id
            )
        except StopIteration as error:
            raise KeyError(
                f"Unknown PLCS logical_scene_id: {logical_scene_id!r}."
            ) from error
        local_inventory = _exact_object(
            logical.get("frame_inventory"),
            name="PLCS frame_inventory",
            keys={
                "source",
                "planned",
                "rendered",
                "labelled",
                "first_frame",
                "last_frame",
            },
        )
        frame_count = _positive_integer(
            local_inventory.get("source"), name="PLCS frame_count"
        )
        if local_inventory != {
            "source": frame_count,
            "planned": frame_count,
            "rendered": frame_count,
            "labelled": frame_count,
            "first_frame": 0,
            "last_frame": frame_count - 1,
        }:
            raise ValueError("PLCS logical frame inventory is not exact.")
        cameras = tuple(
            _exact_object(
                value,
                name="PLCS camera",
                keys={
                    "slot_id",
                    "court_local_center_m",
                    "court_local_look_at_m",
                    "hfov_degrees",
                    "camera",
                },
            )
            for value in _array(logical.get("cameras"), name="PLCS cameras")
        )
        scene_cameras = tuple(
            SceneCamera.from_dict(camera.get("camera")) for camera in cameras
        )
        camera_ids = tuple(camera.camera_id for camera in scene_cameras)
        if not camera_ids or len(camera_ids) != len(set(camera_ids)):
            raise ValueError("PLCS camera IDs must be non-empty and unique.")
        if camera_id not in camera_ids:
            raise KeyError(
                f"Unknown PLCS camera_id {camera_id!r} for {logical_scene_id!r}."
            )
        camera_index = camera_ids.index(camera_id)
        tracks = tuple(
            _exact_object(
                value,
                name="PLCS track",
                keys={
                    "object_id",
                    "instance_id",
                    "asset_id",
                    "support_plane",
                    "start_frame",
                    "stop_frame",
                    "anchor_position_court_m",
                    "yaw_radians",
                },
            )
            for value in _array(logical.get("tracks"), name="PLCS tracks")
        )
        for track in tracks:
            PLCSSourceSupportPlane.from_dict(track.get("support_plane"))
        self.object_ids = tuple(
            _text(track.get("object_id"), name="PLCS object_id") for track in tracks
        )
        if not self.object_ids or len(self.object_ids) != len(set(self.object_ids)):
            raise ValueError("PLCS object IDs must be non-empty and unique.")
        storage = _exact_object(
            manifest.get("storage"),
            name="PLCS storage",
            keys={"layout", "background_store", "scenes"},
        )
        if storage.get("layout") != "shared-background-plus-per-scene-foreground-delta":
            raise ValueError("PLCS compact storage layout is unsupported.")
        storage_scenes = tuple(
            _exact_object(
                value,
                name="PLCS storage scene",
                keys={
                    "scene_id",
                    "chunks",
                    "attempt_token",
                    "sample_order",
                    "supervision",
                    "camera_ids",
                    "object_ids",
                },
            )
            for value in _array(storage.get("scenes"), name="PLCS storage scenes")
        )
        logical_ids = tuple(
            _text(value.get("scene_id"), name="PLCS logical scene_id")
            for value in logical_scenes
        )
        storage_ids = tuple(
            _text(value.get("scene_id"), name="PLCS storage scene_id")
            for value in storage_scenes
        )
        if (
            logical_ids != storage_ids
            or len(logical_ids) != len(set(logical_ids))
            or _positive_integer(
                metadata.get("logical_scene_count"), name="PLCS logical_scene_count"
            )
            != len(logical_ids)
        ):
            raise ValueError("PLCS logical/storage scene inventories are inconsistent.")
        storage_scene = next(
            value
            for value in storage_scenes
            if value.get("scene_id") == logical_scene_id
        )
        if (
            tuple(_array(storage_scene.get("camera_ids"), name="camera_ids"))
            != camera_ids
        ):
            raise ValueError("PLCS logical and storage camera order differs.")
        if tuple(_array(storage_scene.get("object_ids"), name="object_ids")) != (
            self.object_ids
        ):
            raise ValueError("PLCS logical and storage object order differs.")
        supervision_path = _contained_file(
            root,
            _text(storage_scene.get("supervision"), name="PLCS supervision"),
        )
        with np.load(supervision_path, allow_pickle=False) as archive:
            expected_supervision_names = set(PLCSSupervisionArrays.__dataclass_fields__)
            if set(archive.files) != expected_supervision_names:
                raise ValueError("PLCS supervision array inventory is invalid.")
            human_kp = np.asarray(archive["human_kp"][:, camera_index]).copy()
            human_vis = np.asarray(archive["human_vis"][:, camera_index]).copy()
            court_kp = np.asarray(archive["court_kp"][:, camera_index]).copy()
            court_vis = np.asarray(archive["court_vis"][:, camera_index]).copy()
            present = np.asarray(archive["present"]).copy()
        expected_shapes = {
            "human_kp": (human_kp, (frame_count, len(tracks), 17, 2), np.float32),
            "human_vis": (human_vis, (frame_count, len(tracks), 17), np.bool_),
            "court_kp": (court_kp, (frame_count, 20, 2), np.float32),
            "court_vis": (court_vis, (frame_count, 20), np.bool_),
            "present": (present, (frame_count, len(tracks)), np.bool_),
        }
        for name, (array, shape, dtype) in expected_shapes.items():
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"PLCS selected supervision {name} is invalid.")
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"PLCS selected supervision {name} is non-finite.")
        if (
            np.any(human_kp[human_vis] < 0.0)
            or np.any(human_kp[human_vis] > 1.0)
            or np.any(court_kp[court_vis] < 0.0)
            or np.any(court_kp[court_vis] > 1.0)
        ):
            raise ValueError("PLCS selected visible projections leave normalized UV.")
        if np.any(human_kp[~human_vis] != 0.0):
            raise ValueError("PLCS invisible selected keypoints must be explicit zero.")
        if np.any(human_vis & ~present[..., None]):
            raise ValueError("Absent PLCS objects cannot have visible keypoints.")
        self._human_kp = cast(NDArray[np.float32], human_kp)
        self._human_vis = cast(NDArray[np.bool_], human_vis)
        self._court_kp = cast(NDArray[np.float32], court_kp)
        self._court_vis = cast(NDArray[np.bool_], court_vis)
        self._present = cast(NDArray[np.bool_], present)
        attempt_token = _text(
            storage_scene.get("attempt_token"), name="PLCS attempt_token"
        )
        if storage_scene.get("sample_order") != "scene-frame-then-configured-camera":
            raise ValueError("PLCS compact sample order is unsupported.")
        selections: list[_PLCSChunkSelection] = []
        observed_keys: list[RenderSampleKey] = []
        all_observed_keys: list[RenderSampleKey] = []
        for relative in _array(storage_scene.get("chunks"), name="PLCS chunks"):
            path = _contained_directory(root, _text(relative, name="PLCS chunk path"))
            validated = ChunkReader(path).validate(expected_attempt_token=attempt_token)
            ordinals = tuple(
                index
                for index, key in enumerate(validated.keys)
                if key.camera_id == camera_id
            )
            if not ordinals:
                raise ValueError("PLCS chunk omits the selected camera.")
            all_observed_keys.extend(validated.keys)
            observed_keys.extend(validated.keys[index] for index in ordinals)
            selections.append(_PLCSChunkSelection(path=path, ordinals=ordinals))
        expected_keys = [
            RenderSampleKey(index, camera_id) for index in range(frame_count)
        ]
        if observed_keys != expected_keys:
            raise ValueError(
                "PLCS selected view does not cover logical frames exactly in order."
            )
        expected_all_keys = [
            RenderSampleKey(index, expected_camera_id)
            for index in range(frame_count)
            for expected_camera_id in camera_ids
        ]
        if all_observed_keys != expected_all_keys:
            raise ValueError(
                "PLCS compact chunks do not cover exact camera/frame order."
            )
        background_relative = _text(
            storage.get("background_store"), name="PLCS background_store"
        )
        self._backgrounds = SharedBackgroundStore(
            _contained_directory(root, background_relative)
        )
        if not set(camera_ids).issubset(self._backgrounds.camera_ids):
            raise ValueError("PLCS selected scene cameras are missing backgrounds.")
        background = self._backgrounds.load(camera_id)
        self.width = background.width
        self.height = background.height
        camera = scene_cameras[camera_index]
        if (self.width, self.height) != (
            camera.width,
            camera.height,
        ):
            raise ValueError("PLCS selected camera/background dimensions differ.")
        self.root = root
        self.logical_scene_id = logical_scene_id
        self.camera_id = camera_id
        self._attempt_token = attempt_token
        self._selections = tuple(selections)
        self.frame_order = tuple(
            {"frame_index": frame_index} for frame_index in range(frame_count)
        )

    @property
    def frame_count(self) -> int:
        """Return the complete logical global timeline length."""
        return len(self.frame_order)

    def frames(self) -> Iterator[PLCSSourceFrame]:
        """Stream one compact chunk at a time in exact local-frame order."""
        expected_frame = 0
        for selection in self._selections:
            reader = ChunkReader(selection.path)
            reader.validate(expected_attempt_token=self._attempt_token)
            deltas = reader.deltas()
            labels = reader.metadata()
            for ordinal in selection.ordinals:
                if ordinal >= len(deltas) or ordinal >= len(labels):
                    raise ValueError("PLCS sample ordinal exceeds its compact chunk.")
                delta = deltas[ordinal]
                expected = RenderSampleKey(expected_frame, self.camera_id)
                if delta.key != expected:
                    raise ValueError(
                        "PLCS source-frame/camera ordering changed during streaming."
                    )
                label = _exact_object(
                    labels[ordinal],
                    name="PLCS compact label",
                    keys={
                        "schema",
                        "scene_id",
                        "frame_index",
                        "camera_id",
                        "camera_profile",
                        "camera_parameters",
                        "target_court",
                        "seed",
                        "objects",
                    },
                )
                if (
                    label.get("schema") != PLCS_FRAME_LABEL_SCHEMA
                    or label.get("scene_id") != self.logical_scene_id
                    or label.get("frame_index") != expected_frame
                    or label.get("camera_id") != self.camera_id
                ):
                    raise ValueError("PLCS label identity changed during streaming.")
                objects = tuple(
                    _exact_object(
                        value,
                        name="PLCS object label",
                        keys={
                            "label_id",
                            "object_id",
                            "instance_id",
                            "present",
                            "source_frame_index",
                            "motion_source",
                            "motion_category",
                            "gender",
                            "native_fps",
                            "scene_from_asset",
                            "visible_pixel_count",
                        },
                    )
                    for value in _array(label.get("objects"), name="PLCS objects")
                )
                if (
                    tuple(value.get("object_id") for value in objects)
                    != self.object_ids
                ):
                    raise ValueError(
                        "PLCS label object order changed during streaming."
                    )
                render = materialize_logical_sample(
                    self._backgrounds.load(self.camera_id), delta
                )
                yield PLCSSourceFrame(
                    render=render,
                    frame_index=expected_frame,
                    label=label,
                    human_kp=self._human_kp[expected_frame],
                    human_vis=self._human_vis[expected_frame],
                    court_kp=self._court_kp[expected_frame],
                    court_vis=self._court_vis[expected_frame],
                    present=self._present[expected_frame],
                )
                expected_frame += 1
        if expected_frame != self.frame_count:
            raise ValueError(
                "PLCS streaming ended before the complete logical timeline."
            )


def _load_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Required visualization JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed JSON object.")
    return cast(Mapping[str, object], value)


def _exact_object(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> Mapping[str, object]:
    result = _object(value, name=name)
    if set(result) != keys:
        raise ValueError(
            f"{name} keys differ; missing={sorted(keys - set(result))}, "
            f"unknown={sorted(set(result) - keys)}."
        )
    return result


def _array(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _blcs_sample_record(value: object) -> BLCSSampleRecord:
    raw = _exact_object(
        value,
        name="BLCS sample record",
        keys={
            "trajectory_id",
            "split",
            "global_frame_index",
            "source_frame_index",
            "chunk_index",
            "camera_id",
            "background_store",
            "foreground_chunk",
            "chunk_sample_index",
        },
    )
    return BLCSSampleRecord(
        trajectory_id=_text(raw.get("trajectory_id"), name="trajectory_id"),
        split=_text(raw.get("split"), name="split"),
        global_frame_index=_nonnegative_integer(
            raw.get("global_frame_index"), name="global_frame_index"
        ),
        source_frame_index=_nonnegative_integer(
            raw.get("source_frame_index"), name="source_frame_index"
        ),
        chunk_index=_nonnegative_integer(raw.get("chunk_index"), name="chunk_index"),
        camera_id=_text(raw.get("camera_id"), name="camera_id"),
        background_store=_text(raw.get("background_store"), name="background_store"),
        foreground_chunk=_text(raw.get("foreground_chunk"), name="foreground_chunk"),
        chunk_sample_index=_nonnegative_integer(
            raw.get("chunk_sample_index"), name="chunk_sample_index"
        ),
    )


def _contained_file(root: Path, relative: str) -> Path:
    return _contained(root, relative, directory=False)


def _contained_directory(root: Path, relative: str) -> Path:
    return _contained(root, relative, directory=True)


def _contained(root: Path, relative: str, *, directory: bool) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("Dataset reference must be a contained relative POSIX path.")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink():
        raise ValueError("Dataset references must not be symbolic links.")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("Dataset reference escapes its canonical root.")
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        raise ValueError("Dataset reference has the wrong file type.")
    return resolved


def _float32_rgb(path: Path, *, width: int, height: int) -> NDArray[np.float32]:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (height, width, 3):
        raise ValueError(f"NHT RGB frame has an invalid contract: {path}")
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError(f"NHT RGB frame is non-finite or outside [0,1]: {path}")
    return cast(NDArray[np.float32], value)


__all__ = [
    "BLCSSourceFrame",
    "BLCSVisualizationSource",
    "CourtSourceFrame",
    "CourtVisualizationSource",
    "PLCSSourceFrame",
    "PLCSVisualizationSource",
]
