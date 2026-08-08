"""Strict reader/validation for the sole multi-scene compact PLCS schema."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.contracts import (
    FrameInventory,
    TargetCourtBinding,
)
from src.synthetic_data_generation.dataset.plcs.assembler import (
    PLCS_DATASET_SCHEMA,
    PLCS_FRAME_LABEL_SCHEMA,
    PLCSSupervisionArrays,
)
from src.synthetic_data_generation.dataset.plcs.production import (
    PLCSProductionMode,
    validate_plcs_production_contract,
)
from src.synthetic_data_generation.dataset.runtime import (
    ChunkReader,
    DatasetPerformanceMetrics,
    FinalDatasetAssembler,
    LogicalRenderSample,
    RenderSampleKey,
    SharedBackgroundStore,
    materialize_logical_sample,
)
from src.synthetic_data_generation.scene_contract import SceneCamera
from src.tasks.base.generate_dataset.continuity import (
    TimelineFrameRecord,
    validate_frame_continuity,
)
from src.utils.projection.camera_projector import make_look_at_camera


@dataclass(frozen=True, slots=True)
class PLCSTrackIndex:
    """One manifest-authoritative person interval."""

    scene_id: str
    split: str
    object_id: str
    object_index: int
    start_frame: int
    stop_frame: int


@dataclass(frozen=True, slots=True)
class PLCSSceneIndex:
    """One complete logical scene and its generated view inventory."""

    scene_id: str
    split: str
    frame_count: int
    camera_ids: tuple[str, ...]
    object_ids: tuple[str, ...]
    tracks: tuple[PLCSTrackIndex, ...]


@dataclass(frozen=True, slots=True)
class PLCSAllViewScene:
    """All-view dense supervision for a complete source timeline."""

    index: PLCSSceneIndex
    supervision: PLCSSupervisionArrays


class PLCSCompactDatasetReader:
    """Cache-aware logical reader keyed by scene, local frame, and camera."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        manifest = _manifest(directory)
        metadata = _object(manifest["metadata"], name="metadata")
        storage = _object(manifest["storage"], name="storage")
        background_relative = _relative_path(
            storage["background_store"], name="background_store"
        )
        self.backgrounds = SharedBackgroundStore(directory / background_relative)
        logical_scenes = _logical_scene_map(metadata)
        storage_scenes = _storage_scene_map(storage)
        if set(logical_scenes) != set(storage_scenes):
            raise ValueError("PLCS reader scene metadata and storage differ.")
        self.cameras: dict[str, tuple[SceneCamera, ...]] = {}
        self._logical_scenes = logical_scenes
        self._storage_scenes = storage_scenes
        self._index: dict[tuple[str, RenderSampleKey], tuple[ChunkReader, int]] = {}
        for scene_id, scene in logical_scenes.items():
            cameras = _scene_cameras(scene)
            self.cameras[scene_id] = cameras
            record = storage_scenes[scene_id]
            readers = tuple(
                ChunkReader(directory / _relative_path(value, name="scene chunk"))
                for value in _array(record["chunks"], name="scene chunks")
            )
            attempt_token = _text(record["attempt_token"], name="attempt_token")
            for reader in readers:
                validated = reader.validate(expected_attempt_token=attempt_token)
                for ordinal, key in enumerate(validated.keys):
                    compound = (scene_id, key)
                    if compound in self._index:
                        raise ValueError(f"Duplicate compact PLCS sample: {compound}.")
                    self._index[compound] = (reader, ordinal)
        self.scenes = tuple(
            _scene_index(scene_id, scene, storage_scenes[scene_id])
            for scene_id, scene in logical_scenes.items()
        )
        self.tracks = tuple(track for scene in self.scenes for track in scene.tracks)

    def split_scenes(self, split: str) -> tuple[PLCSSceneIndex, ...]:
        """Return complete logical scenes for one manifest split."""
        canonical = "validation" if split == "val" else split
        if canonical not in {"train", "validation", "test"}:
            raise ValueError("PLCS split must be train, validation/val, or test.")
        return tuple(scene for scene in self.scenes if scene.split == canonical)

    def split_tracks(self, split: str) -> tuple[PLCSTrackIndex, ...]:
        """Return stable track intervals for one manifest split."""
        selected = {scene.scene_id for scene in self.split_scenes(split)}
        return tuple(track for track in self.tracks if track.scene_id in selected)

    def materialize_all_views(self, scene_id: str) -> PLCSAllViewScene:
        """Load every generated camera and every global frame without selection."""
        try:
            index = next(scene for scene in self.scenes if scene.scene_id == scene_id)
        except StopIteration as error:
            raise KeyError(f"Unknown PLCS logical scene: {scene_id!r}.") from error
        arrays = _supervision_arrays(
            self.directory,
            self._storage_scenes[scene_id],
            frame_count=index.frame_count,
            camera_count=len(index.camera_ids),
            object_count=len(index.object_ids),
        )
        return PLCSAllViewScene(index=index, supervision=arrays)

    def logical_sample(
        self,
        scene_id: str,
        frame_index: int,
        camera_id: str,
    ) -> LogicalRenderSample:
        """Materialize one exact logical scene/frame/camera sample on demand."""
        key = RenderSampleKey(frame_index, camera_id)
        try:
            reader, ordinal = self._index[(scene_id, key)]
        except KeyError as error:
            raise KeyError(
                f"Unknown compact PLCS sample: {(scene_id, key)}."
            ) from error
        delta = reader.deltas()[ordinal]
        return materialize_logical_sample(self.backgrounds.load(camera_id), delta)


def validate_plcs_dataset(directory: Path) -> dict[str, int | float | str]:
    """Validate every complete logical timeline, court, label, and budget fact."""
    manifest = _manifest(directory)
    scene_id = _text(manifest["scene_id"], name="scene_id")
    frame_inventory = FrameInventory.from_dict(manifest["frame_inventory"])
    metadata = _object(manifest["metadata"], name="metadata")
    _keys(
        metadata,
        {
            "seed",
            "logical_scene_count",
            "aggregate_global_frame_count",
            "aggregate_source_frame_count",
            "required_motion_categories",
            "accepted_court_instance_ids",
            "logical_scenes",
        },
        name="metadata",
    )
    seed = _nonnegative_integer(metadata["seed"], name="seed")
    required_categories = tuple(
        _text(value, name="required motion category")
        for value in _array(
            metadata["required_motion_categories"],
            name="required_motion_categories",
        )
    )
    accepted_courts = tuple(
        _text(value, name="accepted court")
        for value in _array(
            metadata["accepted_court_instance_ids"],
            name="accepted_court_instance_ids",
        )
    )
    if not accepted_courts or len(accepted_courts) != len(set(accepted_courts)):
        raise ValueError("PLCS accepted court inventory must be non-empty and unique.")
    bindings = tuple(
        TargetCourtBinding.from_dict(value)
        for value in _array(manifest["target_courts"], name="target_courts")
    )
    binding_by_court = {value.court_instance_id: value for value in bindings}
    if tuple(binding_by_court) != accepted_courts:
        raise ValueError("PLCS target bindings differ from accepted court inventory.")

    logical_scene_values = _array(metadata["logical_scenes"], name="logical_scenes")
    logical_scene_count = _positive_integer(
        metadata["logical_scene_count"], name="logical_scene_count"
    )
    if len(logical_scene_values) != logical_scene_count:
        raise ValueError("PLCS logical scene count disagrees with its inventory.")
    if logical_scene_count < len(accepted_courts):
        raise ValueError("PLCS logical scenes cannot cover every accepted court.")

    storage = _object(manifest["storage"], name="storage")
    _keys(
        storage,
        {"layout", "background_store", "scenes"},
        name="storage",
    )
    if storage["layout"] != "shared-background-plus-per-scene-foreground-delta":
        raise ValueError("PLCS dataset does not use the sole multi-scene layout.")
    logical_scenes = _logical_scene_map(metadata)
    storage_scenes = _storage_scene_map(storage)
    if set(logical_scenes) != set(storage_scenes):
        raise ValueError("PLCS logical-scene metadata and storage IDs differ.")

    backgrounds = SharedBackgroundStore(
        directory / _relative_path(storage["background_store"], name="background_store")
    )
    backgrounds.validate_all()
    expected_background_ids: list[str] = []
    seen_camera_ids: set[str] = set()
    reference_motion_signature: tuple[tuple[object, ...], ...] | None = None
    aggregate_offset = 0
    aggregate_source_frames = 0
    sample_count = 0
    chunk_count = 0
    scene_court_counts: Counter[str] = Counter()
    split_court_counts: dict[str, Counter[str]] = defaultdict(Counter)
    camera_count_per_scene: int | None = None
    production_mode: PLCSProductionMode | None = None
    for raw_scene in logical_scene_values:
        scene = _object(raw_scene, name="logical scene")
        _keys(
            scene,
            {
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
            name="logical scene",
        )
        logical_scene_id = _text(scene["scene_id"], name="logical scene_id")
        split = _text(scene["split"], name="logical split")
        if split not in {"train", "validation", "test"}:
            raise ValueError("PLCS logical scene split is unsupported.")
        if (
            _nonnegative_integer(
                scene["aggregate_frame_offset"], name="aggregate_frame_offset"
            )
            != aggregate_offset
        ):
            raise ValueError("PLCS aggregate logical-scene offsets are discontinuous.")
        local_inventory = FrameInventory.from_dict(scene["frame_inventory"])
        frame_count = local_inventory.source_count
        scene_mode = PLCSProductionMode.from_persisted_timeline_mode(scene["mode"])
        if production_mode is None:
            production_mode = scene_mode
        elif scene_mode is not production_mode:
            raise ValueError("PLCS logical scenes disagree on production mode.")
        binding = TargetCourtBinding.from_dict(scene["target_court"])
        if binding_by_court.get(binding.court_instance_id) != binding:
            raise ValueError("PLCS logical scene target binding is not canonical.")
        scene_court_counts[binding.court_instance_id] += 1
        split_court_counts[split][binding.court_instance_id] += 1

        cameras = _scene_cameras(scene)
        _validate_camera_binding(scene, cameras=cameras, binding=binding)
        camera_metadata = {
            camera.camera_id: value
            for camera, value in zip(
                cameras,
                _array(scene["cameras"], name="cameras"),
                strict=True,
            )
        }
        camera_profile = _text(scene["camera_profile"], name="camera_profile")
        expected_camera_count = 6 if camera_profile == "default" else 2
        if (
            camera_profile not in {"default", "broadcast"}
            or len(cameras) != expected_camera_count
        ):
            raise ValueError("PLCS logical scene violates the canonical 6/2 profile.")
        if camera_count_per_scene is None:
            camera_count_per_scene = len(cameras)
        elif len(cameras) != camera_count_per_scene:
            raise ValueError("PLCS logical scenes use inconsistent camera counts.")
        for camera in cameras:
            if camera.camera_id not in seen_camera_ids:
                seen_camera_ids.add(camera.camera_id)
                expected_background_ids.append(camera.camera_id)
        if any(
            not camera.camera_id.startswith(f"{binding.court_instance_id}-")
            for camera in cameras
        ):
            raise ValueError("PLCS camera identity differs from its target court.")

        sources = tuple(
            _motion_source(value)
            for value in _array(scene["motion_sources"], name="motion_sources")
        )
        tracks = tuple(
            _track(value) for value in _array(scene["tracks"], name="tracks")
        )
        if len(sources) != len(tracks) or not sources:
            raise ValueError("PLCS motion-source and track inventory is incomplete.")
        categories = {str(source["category"]) for source in sources}
        if categories != set(required_categories):
            raise ValueError(
                "PLCS logical scene does not retain every required motion category."
            )
        validate_plcs_production_contract(
            mode=scene_mode,
            configured_motion_categories=required_categories,
            object_motion_categories=(str(source["category"]) for source in sources),
            object_start_frames=(
                _nonnegative_integer(track["start_frame"], name="start_frame")
                for track in tracks
            ),
        )
        for source, track in zip(sources, tracks, strict=True):
            source_count = _positive_integer(
                source["frame_count"], name="source frame_count"
            )
            start_frame = _nonnegative_integer(
                track["start_frame"], name="start_frame"
            )
            stop_frame = _positive_integer(track["stop_frame"], name="stop_frame")
            if stop_frame - start_frame != source_count:
                raise ValueError(
                    "PLCS track interval does not cover every source frame exactly."
                )
        signature = tuple(
            (
                source["source_path"],
                source["category"],
                source["gender"],
                source["native_fps"],
                source["frame_count"],
                source["pose_dtype"],
                source["beta_count"],
                track["object_id"],
                track["instance_id"],
                track["asset_id"],
                track["start_frame"],
                track["stop_frame"],
                tuple(cast(Sequence[object], track["anchor_position_court_m"])),
                track["yaw_radians"],
            )
            for source, track in zip(sources, tracks, strict=True)
        )
        if reference_motion_signature is None:
            reference_motion_signature = signature
        elif signature != reference_motion_signature:
            raise ValueError(
                "PLCS logical scenes do not retain one exact source inventory."
            )
        expected_global = max(
            _positive_integer(track["stop_frame"], name="stop_frame")
            for track in tracks
        )
        if expected_global != frame_count:
            raise ValueError("PLCS logical scene omits part of its global timeline.")
        aggregate_source_frames += sum(
            _positive_integer(source["frame_count"], name="source frame_count")
            for source in sources
        )

        record = storage_scenes[logical_scene_id]
        supervision = _supervision_arrays(
            directory,
            record,
            frame_count=frame_count,
            camera_count=len(cameras),
            object_count=len(tracks),
        )
        if tuple(_array(record["camera_ids"], name="camera_ids")) != tuple(
            camera.camera_id for camera in cameras
        ) or tuple(_array(record["object_ids"], name="object_ids")) != tuple(
            _text(track["object_id"], name="object_id") for track in tracks
        ):
            raise ValueError("PLCS supervision axis identities are inconsistent.")
        expected_presence: NDArray[np.bool_] = np.zeros(
            (frame_count, len(tracks)), dtype=np.bool_
        )
        for object_index, track in enumerate(tracks):
            expected_presence[
                _nonnegative_integer(
                    track["start_frame"], name="start_frame"
                ) : _positive_integer(track["stop_frame"], name="stop_frame"),
                object_index,
            ] = True
        if not np.array_equal(supervision.present, expected_presence):
            raise ValueError("PLCS supervision omits or invents lifecycle frames.")
        readers = tuple(
            ChunkReader(directory / _relative_path(value, name="scene chunk"))
            for value in _array(record["chunks"], name="scene chunks")
        )
        attempt_token = _text(record["attempt_token"], name="attempt_token")
        if record["sample_order"] != "scene-frame-then-configured-camera":
            raise ValueError("PLCS logical scene sample order is unsupported.")
        validated = FinalDatasetAssembler(
            frame_count=frame_count,
            camera_ids=tuple(camera.camera_id for camera in cameras),
            attempt_token=attempt_token,
        ).validate(readers)
        continuity_records: list[TimelineFrameRecord] = []
        for chunk_index, (_chunk, reader) in enumerate(
            zip(validated, readers, strict=True)
        ):
            for delta, label in zip(reader.deltas(), reader.metadata(), strict=True):
                _validate_label(
                    label,
                    scene_id=logical_scene_id,
                    key=delta.key,
                    visible_counts=delta.visible_instance_counts,
                    binding=binding,
                    sources=sources,
                    tracks=tracks,
                    seed=seed,
                    camera_profile=camera_profile,
                    camera_metadata=camera_metadata[delta.key.camera_id],
                )
                objects = _array(label["objects"], name="objects")
                for track, raw_object in zip(tracks, objects, strict=True):
                    object_record = _object(raw_object, name="object")
                    source_frame = object_record["source_frame_index"]
                    continuity_records.append(
                        TimelineFrameRecord(
                            frame_index=delta.key.frame_index,
                            chunk_index=chunk_index,
                            track_id=_text(track["object_id"], name="object_id"),
                            present=cast(bool, object_record["present"]),
                            source_frame_index=(
                                None
                                if source_frame is None
                                else _nonnegative_integer(
                                    source_frame, name="source_frame_index"
                                )
                            ),
                            camera_id=delta.key.camera_id,
                            label_id=f"{object_record['label_id']}:{delta.key.camera_id}",
                            court_instance_id=binding.court_instance_id,
                        )
                    )
                sample_count += 1
        continuity = validate_frame_continuity(
            continuity_records,
            frame_count=frame_count,
        )
        persisted_continuity = _object(scene["continuity"], name="continuity")
        expected_continuity = {
            "frame_count": continuity.frame_count,
            "chunk_count": continuity.chunk_count,
            "track_count": continuity.track_count,
            "camera_count": continuity.camera_count,
            "record_count": continuity.record_count,
        }
        if persisted_continuity != expected_continuity:
            raise ValueError("PLCS logical-scene continuity metadata is stale.")
        chunk_count += len(validated)
        aggregate_offset += frame_count

    if tuple(expected_background_ids) != backgrounds.camera_ids:
        raise ValueError("PLCS background store differs from accepted-court cameras.")
    if (
        aggregate_offset != frame_inventory.source_count
        or aggregate_offset != metadata["aggregate_global_frame_count"]
    ):
        raise ValueError("PLCS aggregate global frame inventory is inexact.")
    if aggregate_source_frames != metadata["aggregate_source_frame_count"]:
        raise ValueError("PLCS aggregate source-motion frame inventory is inexact.")
    if (
        production_mode is PLCSProductionMode.SINGLE_OBJECT
        and aggregate_source_frames != aggregate_offset
    ):
        raise ValueError(
            "PLCS single-object source/planned/rendered/labelled inventory is inexact."
        )
    if set(scene_court_counts) != set(accepted_courts):
        raise ValueError("PLCS logical scenes do not use every accepted court.")
    court_values = [scene_court_counts[court] for court in accepted_courts]
    if max(court_values) - min(court_values) > 1:
        raise ValueError("PLCS logical-scene court count difference exceeds one.")
    for split, counts in split_court_counts.items():
        values = [counts[court] for court in accepted_courts]
        if max(values) - min(values) > 1:
            raise ValueError(f"PLCS court balance fails within split {split!r}.")

    _validate_diagnostics(
        directory,
        manifest=manifest,
        scene_court_counts=scene_court_counts,
        split_court_counts=split_court_counts,
        logical_scene_count=logical_scene_count,
        aggregate_global_frames=aggregate_offset,
        aggregate_source_frames=aggregate_source_frames,
    )
    performance = DatasetPerformanceMetrics.from_dict(
        _load_json(directory / "diagnostics" / "performance.json")
    )
    execution_device = performance.execution_device
    if execution_device == "test-cpu-oracle":
        if performance.cuda_peak_bytes != 0:
            raise ValueError("PLCS test CPU oracle cannot report CUDA allocation.")
    elif not execution_device.startswith("cuda") or performance.cuda_peak_bytes <= 0:
        raise ValueError("PLCS production performance lacks CUDA execution evidence.")
    if (
        performance.domain != "plcs"
        or performance.frame_count != aggregate_offset
        or performance.camera_count != len(expected_background_ids)
        or performance.sample_count != sample_count
        or performance.nht_invocations != 1
        or performance.background_cache_misses != len(expected_background_ids)
        or performance.dense_reference_bytes <= 0
        or performance.generated_bytes < performance.published_bytes
        or performance.published_bytes
        != sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    ):
        raise ValueError("PLCS measured performance violates the multi-scene schema.")
    return {
        "scene_id": scene_id,
        "logical_scene_count": logical_scene_count,
        "frame_count": aggregate_offset,
        "camera_count": len(expected_background_ids),
        "camera_count_per_scene": camera_count_per_scene or 0,
        "sample_count": sample_count,
        "chunk_count": chunk_count,
        "published_bytes": performance.published_bytes,
        "dense_reference_bytes": performance.dense_reference_bytes,
    }


def _validate_diagnostics(
    directory: Path,
    *,
    manifest: Mapping[str, object],
    scene_court_counts: Counter[str],
    split_court_counts: Mapping[str, Counter[str]],
    logical_scene_count: int,
    aggregate_global_frames: int,
    aggregate_source_frames: int,
) -> None:
    diagnostics = _array(manifest["diagnostics"], name="diagnostics")
    expected_diagnostics = {
        "diagnostics/motion-camera-court.json",
        "diagnostics/summary.txt",
        "diagnostics/performance.json",
    }
    if set(diagnostics) != expected_diagnostics:
        raise ValueError("PLCS diagnostic inventory differs from the compact schema.")
    for relative_value in diagnostics:
        path = directory / _relative_path(relative_value, name="diagnostic path")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"PLCS diagnostic is missing: {path}")
    machine = _object(
        _load_json(directory / "diagnostics" / "motion-camera-court.json"),
        name="motion-camera-court.json",
    )
    if (
        machine.get("schema") != "tennis_plcs_diagnostics_v3"
        or machine.get("scene_id") != manifest["scene_id"]
        or machine.get("logical_scene_count") != logical_scene_count
        or machine.get("aggregate_global_frame_count") != aggregate_global_frames
        or machine.get("aggregate_source_frame_count") != aggregate_source_frames
        or machine.get("amass_compatible") is not True
    ):
        raise ValueError("PLCS machine diagnostics disagree with the dataset.")
    metadata = _object(manifest["metadata"], name="metadata")
    dataset_scenes = _array(metadata["logical_scenes"], name="logical_scenes")
    diagnostic_scenes = _array(
        machine.get("logical_scenes"), name="diagnostic logical_scenes"
    )
    if len(diagnostic_scenes) != len(dataset_scenes):
        raise ValueError("PLCS diagnostics omit logical scenes.")
    for dataset_value, diagnostic_value in zip(
        dataset_scenes, diagnostic_scenes, strict=True
    ):
        dataset_scene = _object(dataset_value, name="dataset logical scene")
        diagnostic_scene = _object(diagnostic_value, name="diagnostic logical scene")
        sources = tuple(
            _motion_source(value)
            for value in _array(dataset_scene["motion_sources"], name="motion_sources")
        )
        tracks = tuple(
            _track(value) for value in _array(dataset_scene["tracks"], name="tracks")
        )
        cameras = _scene_cameras(dataset_scene)
        camera_distribution = _object(
            diagnostic_scene.get("camera_distribution"),
            name="diagnostic camera_distribution",
        )
        expected_source_counts = {
            _text(track["object_id"], name="object_id"): _positive_integer(
                source["frame_count"], name="source frame_count"
            )
            for source, track in zip(sources, tracks, strict=True)
        }
        if (
            diagnostic_scene.get("scene_id") != dataset_scene["scene_id"]
            or diagnostic_scene.get("split") != dataset_scene["split"]
            or diagnostic_scene.get("mode") != dataset_scene["mode"]
            or diagnostic_scene.get("global_frame_count")
            != _object(dataset_scene["frame_inventory"], name="frame_inventory")[
                "source"
            ]
            or diagnostic_scene.get("source_frame_counts") != expected_source_counts
            or diagnostic_scene.get("motion_categories")
            != [source["category"] for source in sources]
            or diagnostic_scene.get("target_court") != dataset_scene["target_court"]
            or camera_distribution.get("profile") != dataset_scene["camera_profile"]
            or camera_distribution.get("camera_count") != len(cameras)
            or camera_distribution.get("camera_ids")
            != [camera.camera_id for camera in cameras]
            or camera_distribution.get("sampled_parameters") != dataset_scene["cameras"]
        ):
            raise ValueError(
                "PLCS logical-scene diagnostics disagree with published metadata."
            )
    court_balance = _object(machine.get("court_balance"), name="court_balance")
    expected_counts = {
        court: scene_court_counts[court] for court in sorted(scene_court_counts)
    }
    count_values = list(expected_counts.values())
    if (
        court_balance.get("scene_count") != logical_scene_count
        or court_balance.get("accepted_court_instance_ids")
        != metadata["accepted_court_instance_ids"]
        or court_balance.get("counts") != expected_counts
        or court_balance.get("maximum_count_difference")
        != max(count_values) - min(count_values)
    ):
        raise ValueError("PLCS diagnostic court counts disagree with logical scenes.")
    expected_split_counts = {
        split: {court: counts[court] for court in sorted(scene_court_counts)}
        for split, counts in sorted(split_court_counts.items())
    }
    if court_balance.get("per_split_counts") != expected_split_counts:
        raise ValueError("PLCS diagnostic split balance disagrees with logical scenes.")
    summary = (directory / "diagnostics" / "summary.txt").read_text(encoding="utf-8")
    if (
        "PLCS production diagnostics" not in summary
        or f"logical scenes: {logical_scene_count}" not in summary
        or f"aggregate global frames: {aggregate_global_frames}" not in summary
    ):
        raise ValueError("PLCS human diagnostics disagree with the dataset.")


def _manifest(directory: Path) -> Mapping[str, object]:
    is_owner = directory.name == "plcs" and directory.parent.name == "datasets"
    is_transaction = (
        directory.name == "snapshot"
        and directory.parent.name == "plcs_dataset"
        and directory.parent.parent.name == ".transactions"
    )
    if (
        not (is_owner or is_transaction)
        or not directory.is_dir()
        or directory.is_symlink()
    ):
        raise ValueError(
            "PLCS validation requires its canonical owner or transaction snapshot."
        )
    top_level = {path.name for path in directory.iterdir()}
    if top_level != {"dataset.json", "backgrounds", "scenes", "diagnostics"}:
        raise ValueError(
            "PLCS dataset contains stale or non-canonical top-level paths."
        )
    manifest = _object(_load_json(directory / "dataset.json"), name="dataset.json")
    _keys(
        manifest,
        {
            "schema",
            "scene_id",
            "domain",
            "frame_inventory",
            "target_courts",
            "metadata",
            "diagnostics",
            "storage",
        },
        name="dataset.json",
    )
    if manifest["schema"] != PLCS_DATASET_SCHEMA or manifest["domain"] != "plcs":
        raise ValueError("Unsupported PLCS dataset schema or domain.")
    return manifest


def _logical_scene_map(
    metadata: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for value in _array(metadata["logical_scenes"], name="logical_scenes"):
        record = _object(value, name="logical scene")
        scene_id = _text(record.get("scene_id"), name="logical scene_id")
        if scene_id in result:
            raise ValueError("Duplicate PLCS logical scene ID.")
        result[scene_id] = record
    if not result:
        raise ValueError("PLCS logical scene inventory must not be empty.")
    return result


def _storage_scene_map(
    storage: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for value in _array(storage["scenes"], name="storage scenes"):
        record = _object(value, name="storage scene")
        _keys(
            record,
            {
                "scene_id",
                "chunks",
                "attempt_token",
                "sample_order",
                "supervision",
                "camera_ids",
                "object_ids",
            },
            name="storage scene",
        )
        scene_id = _text(record["scene_id"], name="storage scene_id")
        if scene_id in result:
            raise ValueError("Duplicate PLCS storage scene ID.")
        result[scene_id] = record
    if not result:
        raise ValueError("PLCS storage scene inventory must not be empty.")
    return result


def _scene_index(
    scene_id: str,
    scene: Mapping[str, object],
    storage: Mapping[str, object],
) -> PLCSSceneIndex:
    split = _text(scene["split"], name="split")
    if split not in {"train", "validation", "test"}:
        raise ValueError("PLCS logical scene split is unsupported.")
    inventory = FrameInventory.from_dict(scene["frame_inventory"])
    tracks_raw = tuple(
        _track(value) for value in _array(scene["tracks"], name="tracks")
    )
    object_ids = tuple(
        _text(track["object_id"], name="object_id") for track in tracks_raw
    )
    camera_ids = tuple(
        _text(value, name="camera_id")
        for value in _array(storage["camera_ids"], name="camera_ids")
    )
    persisted_object_ids = tuple(
        _text(value, name="object_id")
        for value in _array(storage["object_ids"], name="object_ids")
    )
    if persisted_object_ids != object_ids:
        raise ValueError("PLCS supervision object order differs from manifest tracks.")
    tracks = tuple(
        PLCSTrackIndex(
            scene_id=scene_id,
            split=split,
            object_id=object_id,
            object_index=object_index,
            start_frame=_nonnegative_integer(track["start_frame"], name="start_frame"),
            stop_frame=_positive_integer(track["stop_frame"], name="stop_frame"),
        )
        for object_index, (object_id, track) in enumerate(
            zip(object_ids, tracks_raw, strict=True)
        )
    )
    return PLCSSceneIndex(
        scene_id=scene_id,
        split=split,
        frame_count=inventory.source_count,
        camera_ids=camera_ids,
        object_ids=object_ids,
        tracks=tracks,
    )


def _supervision_arrays(
    directory: Path,
    record: Mapping[str, object],
    *,
    frame_count: int,
    camera_count: int,
    object_count: int,
) -> PLCSSupervisionArrays:
    relative = _relative_path(record["supervision"], name="supervision")
    path = directory / relative
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"PLCS supervision store is missing: {path}")
    expected_names = tuple(PLCSSupervisionArrays.__dataclass_fields__)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_names):
            raise ValueError("PLCS supervision array inventory is invalid.")
        values = {name: np.asarray(archive[name]) for name in expected_names}
    result = PLCSSupervisionArrays(**values)
    result.validate(
        frame_count=frame_count,
        camera_count=camera_count,
        object_count=object_count,
    )
    return result


def _scene_cameras(scene: Mapping[str, object]) -> tuple[SceneCamera, ...]:
    result: list[SceneCamera] = []
    for index, value in enumerate(_array(scene["cameras"], name="cameras")):
        camera_record = _object(value, name=f"cameras[{index}]")
        _keys(
            camera_record,
            {
                "slot_id",
                "court_local_center_m",
                "court_local_look_at_m",
                "hfov_degrees",
                "camera",
            },
            name=f"cameras[{index}]",
        )
        result.append(SceneCamera.from_dict(camera_record["camera"]))
    camera_ids = [camera.camera_id for camera in result]
    if not result or len(camera_ids) != len(set(camera_ids)):
        raise ValueError("PLCS logical-scene camera IDs must be non-empty and unique.")
    return tuple(result)


def _validate_camera_binding(
    scene: Mapping[str, object],
    *,
    cameras: tuple[SceneCamera, ...],
    binding: TargetCourtBinding,
) -> None:
    records = _array(scene["cameras"], name="cameras")
    for index, (value, camera) in enumerate(zip(records, cameras, strict=True)):
        record = _object(value, name=f"cameras[{index}]")
        center_raw = _array(record["court_local_center_m"], name="court_local_center_m")
        look_at_raw = _array(
            record["court_local_look_at_m"], name="court_local_look_at_m"
        )
        if len(center_raw) != 3 or len(look_at_raw) != 3:
            raise ValueError("PLCS court-local camera vectors must have length three.")
        center = tuple(_number(item, name="camera center") for item in center_raw)
        look_at = tuple(_number(item, name="camera look-at") for item in look_at_raw)
        local = make_look_at_camera(
            center,
            look_at=look_at,
            image_size=(camera.width, camera.height),
            hfov_deg=_positive_number(record["hfov_degrees"], name="hfov_degrees"),
        )
        camera_to_court = np.eye(4, dtype=np.float64)
        camera_to_court[:3, :3] = local.R.detach().cpu().numpy().astype(np.float64).T
        camera_to_court[:3, 3] = local.C.detach().cpu().numpy().astype(np.float64)
        expected = binding.scene_from_court.matrix() @ camera_to_court
        if not np.allclose(
            camera.camera_to_scene.matrix(),
            expected,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError(
                "PLCS logical-scene camera and target-court transform disagree."
            )


def _motion_source(value: object) -> Mapping[str, object]:
    result = _object(value, name="motion source")
    _keys(
        result,
        {
            "source_path",
            "category",
            "gender",
            "native_fps",
            "frame_count",
            "pose_dtype",
            "beta_count",
        },
        name="motion source",
    )
    _text(result["source_path"], name="motion source_path")
    _text(result["category"], name="motion category")
    _text(result["gender"], name="motion gender")
    _positive_number(result["native_fps"], name="native_fps")
    _positive_integer(result["frame_count"], name="motion frame_count")
    _text(result["pose_dtype"], name="pose_dtype")
    _positive_integer(result["beta_count"], name="beta_count")
    return result


def _track(value: object) -> Mapping[str, object]:
    result = _object(value, name="track")
    _keys(
        result,
        {
            "object_id",
            "instance_id",
            "asset_id",
            "start_frame",
            "stop_frame",
            "anchor_position_court_m",
            "yaw_radians",
        },
        name="track",
    )
    _text(result["object_id"], name="object_id")
    _positive_integer(result["instance_id"], name="instance_id")
    _text(result["asset_id"], name="asset_id")
    start = _nonnegative_integer(result["start_frame"], name="start_frame")
    stop = _positive_integer(result["stop_frame"], name="stop_frame")
    if stop <= start:
        raise ValueError("PLCS track stop_frame must exceed start_frame.")
    anchor = _array(result["anchor_position_court_m"], name="anchor")
    if len(anchor) != 3:
        raise ValueError("PLCS track anchor must contain three values.")
    for item in anchor:
        _number(item, name="anchor value")
    _number(result["yaw_radians"], name="yaw_radians")
    return result


def _validate_label(
    label: Mapping[str, object],
    *,
    scene_id: str,
    key: RenderSampleKey,
    visible_counts: Mapping[int, int],
    binding: TargetCourtBinding,
    sources: tuple[Mapping[str, object], ...],
    tracks: tuple[Mapping[str, object], ...],
    seed: int,
    camera_profile: str,
    camera_metadata: object,
) -> None:
    _keys(
        label,
        {
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
        name="PLCS compact label",
    )
    if (
        label["schema"] != PLCS_FRAME_LABEL_SCHEMA
        or label["scene_id"] != scene_id
        or label["frame_index"] != key.frame_index
        or label["camera_id"] != key.camera_id
        or label["target_court"] != binding.to_dict()
        or label["seed"] != seed
        or label["camera_profile"] != camera_profile
        or label["camera_parameters"] != camera_metadata
    ):
        raise ValueError(
            "PLCS compact label identity/binding disagrees with its delta."
        )
    objects = _array(label["objects"], name="objects")
    if len(objects) != len(tracks):
        raise ValueError("PLCS label must retain every declared track.")
    declared_ids: set[int] = set()
    expected_visible: dict[int, int] = {}
    for index, (item, source, track) in enumerate(
        zip(objects, sources, tracks, strict=True)
    ):
        record = _object(item, name=f"objects[{index}]")
        _keys(
            record,
            {
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
            name=f"objects[{index}]",
        )
        instance_id = _positive_integer(record["instance_id"], name="instance_id")
        if instance_id in declared_ids:
            raise ValueError("PLCS label contains duplicate instance IDs.")
        declared_ids.add(instance_id)
        if (
            record["object_id"] != track["object_id"]
            or instance_id != track["instance_id"]
            or record["motion_source"] != source["source_path"]
            or record["motion_category"] != source["category"]
            or record["gender"] != source["gender"]
            or record["native_fps"] != source["native_fps"]
        ):
            raise ValueError(
                "PLCS object label differs from its motion/track inventory."
            )
        start = cast(int, track["start_frame"])
        stop = cast(int, track["stop_frame"])
        expected_source = (
            key.frame_index - start if start <= key.frame_index < stop else None
        )
        present = record["present"]
        if not isinstance(present, bool):
            raise TypeError("PLCS object present must be boolean.")
        if (
            present != (expected_source is not None)
            or record["source_frame_index"] != expected_source
        ):
            raise ValueError("PLCS object presence/source mapping is incomplete.")
        count = _nonnegative_integer(
            record["visible_pixel_count"], name="visible_pixel_count"
        )
        if present != (record["scene_from_asset"] is not None):
            raise ValueError("PLCS object presence and transform disagree.")
        if not present and count != 0:
            raise ValueError("Absent PLCS object has renderer-visible pixels.")
        if count > 0:
            expected_visible[instance_id] = count
    if expected_visible != dict(visible_counts):
        raise ValueError("PLCS compact label visibility disagrees with its delta.")


def _load_json(path: Path) -> object:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Required PLCS JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _array(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} keys differ; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}."
        )


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _relative_path(value: object, *, name: str) -> Path:
    result = Path(_text(value, name=name))
    if result.is_absolute() or ".." in result.parts or result == Path("."):
        raise ValueError(f"{name} must be a contained relative path.")
    return result


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_number(value: object, *, name: str) -> float:
    result = _number(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


__all__ = [
    "PLCSAllViewScene",
    "PLCSCompactDatasetReader",
    "PLCSSceneIndex",
    "PLCSTrackIndex",
    "validate_plcs_dataset",
]
