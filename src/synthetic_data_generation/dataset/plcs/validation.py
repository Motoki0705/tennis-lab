"""Strict reader/validation for the sole compact PLCS dataset schema."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from src.synthetic_data_generation.dataset.plcs.assembler import (
    PLCS_DATASET_SCHEMA,
    PLCS_FRAME_LABEL_SCHEMA,
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


class PLCSCompactDatasetReader:
    """Cache-aware logical reader over shared backgrounds and delta chunks."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        manifest = _manifest(directory)
        metadata = _object(manifest["metadata"], name="metadata")
        cameras_raw = _array(metadata["cameras"], name="cameras")
        self.cameras = tuple(
            SceneCamera.from_dict(_object(item, name="camera record")["camera"])
            for item in cameras_raw
        )
        storage = _object(manifest["storage"], name="storage")
        background_relative = _relative_path(
            storage["background_store"], name="background_store"
        )
        self.backgrounds = SharedBackgroundStore(directory / background_relative)
        chunk_relatives = _array(storage["chunks"], name="chunks")
        self._readers = tuple(
            ChunkReader(directory / _relative_path(value, name="chunk"))
            for value in chunk_relatives
        )
        self._index: dict[RenderSampleKey, tuple[ChunkReader, int]] = {}
        for reader in self._readers:
            validated = reader.validate(
                expected_attempt_token=_text(
                    storage["attempt_token"], name="attempt_token"
                )
            )
            for ordinal, key in enumerate(validated.keys):
                if key in self._index:
                    raise ValueError(f"Duplicate compact PLCS sample: {key}.")
                self._index[key] = (reader, ordinal)

    def logical_sample(self, frame_index: int, camera_id: str) -> LogicalRenderSample:
        """Materialize one logical RGB/alpha/depth/instance sample on demand."""
        key = RenderSampleKey(frame_index, camera_id)
        try:
            reader, ordinal = self._index[key]
        except KeyError as error:
            raise KeyError(f"Unknown compact PLCS sample: {key}.") from error
        delta = reader.deltas()[ordinal]
        return materialize_logical_sample(self.backgrounds.load(camera_id), delta)


def validate_plcs_dataset(directory: Path) -> dict[str, int | float | str]:
    """Validate exact global frame-camera chunks, labels, and measured budgets."""
    manifest = _manifest(directory)
    scene_id = _text(manifest["scene_id"], name="scene_id")
    inventory = _object(manifest["frame_inventory"], name="frame_inventory")
    _keys(
        inventory,
        {"source", "planned", "rendered", "labelled", "first_frame", "last_frame"},
        name="frame_inventory",
    )
    frame_count = _positive_integer(inventory["source"], name="source")
    if (
        any(
            _positive_integer(inventory[name], name=name) != frame_count
            for name in ("planned", "rendered", "labelled")
        )
        or inventory["first_frame"] != 0
        or inventory["last_frame"] != frame_count - 1
    ):
        raise ValueError(
            "PLCS frame inventory is not exact source==planned==rendered==labelled."
        )
    metadata = _object(manifest["metadata"], name="metadata")
    _keys(
        metadata,
        {
            "mode",
            "seed",
            "camera_profile",
            "cameras",
            "motion_sources",
            "tracks",
            "continuity",
        },
        name="metadata",
    )
    cameras_raw = _array(metadata["cameras"], name="cameras")
    cameras: dict[str, SceneCamera] = {}
    for index, item in enumerate(cameras_raw):
        camera_record = _object(item, name=f"cameras[{index}]")
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
        camera = SceneCamera.from_dict(camera_record["camera"])
        if camera.camera_id in cameras:
            raise ValueError("PLCS camera IDs must be unique.")
        cameras[camera.camera_id] = camera
    expected_camera_count = 6 if metadata["camera_profile"] == "default" else 2
    if (
        metadata["camera_profile"] not in {"default", "broadcast"}
        or len(cameras) != expected_camera_count
    ):
        raise ValueError(
            "PLCS camera profile/count is not the canonical 6/2 distribution."
        )

    storage = _object(manifest["storage"], name="storage")
    _keys(
        storage,
        {
            "layout",
            "background_store",
            "chunks",
            "attempt_token",
            "sample_order",
        },
        name="storage",
    )
    if storage["layout"] != "shared-background-plus-foreground-delta":
        raise ValueError("PLCS dataset does not use the sole compact storage layout.")
    if storage["sample_order"] != "global-frame-then-configured-camera":
        raise ValueError("PLCS compact sample order is unsupported.")
    background_path = directory / _relative_path(
        storage["background_store"], name="background_store"
    )
    backgrounds = SharedBackgroundStore(background_path)
    backgrounds.validate_all()
    if backgrounds.camera_ids != tuple(cameras):
        raise ValueError("PLCS shared backgrounds differ from configured camera order.")
    chunk_paths = tuple(
        directory / _relative_path(value, name="chunk")
        for value in _array(storage["chunks"], name="chunks")
    )
    attempt_token = _text(storage["attempt_token"], name="attempt_token")
    readers = tuple(ChunkReader(path) for path in chunk_paths)
    validated = FinalDatasetAssembler(
        frame_count=frame_count,
        camera_ids=tuple(cameras),
        attempt_token=attempt_token,
    ).validate(readers)
    sample_count = 0
    for _chunk, reader in zip(validated, readers, strict=True):
        for delta, label in zip(reader.deltas(), reader.metadata(), strict=True):
            _validate_label(
                label,
                scene_id=scene_id,
                key=delta.key,
                visible_counts=delta.visible_instance_counts,
            )
            sample_count += 1

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
        machine.get("schema") != "tennis_plcs_diagnostics_v2"
        or machine.get("scene_id") != scene_id
        or machine.get("global_frame_count") != frame_count
        or machine.get("amass_compatible") is not True
    ):
        raise ValueError("PLCS machine diagnostics disagree with the dataset.")
    summary = (directory / "diagnostics" / "summary.txt").read_text(encoding="utf-8")
    if (
        "PLCS production diagnostics" not in summary
        or f"global frames: {frame_count}" not in summary
    ):
        raise ValueError("PLCS human diagnostics disagree with the dataset.")
    performance = DatasetPerformanceMetrics.from_dict(
        _load_json(directory / "diagnostics" / "performance.json")
    )
    if (
        performance.domain != "plcs"
        or performance.frame_count != frame_count
        or performance.camera_count != len(cameras)
        or performance.sample_count != sample_count
        or performance.execution_device.split(":", maxsplit=1)[0] != "cuda"
        or performance.nht_invocations != 1
        or performance.background_cache_misses != len(cameras)
        or performance.dense_reference_bytes <= 0
        or performance.generated_bytes < performance.published_bytes
        or performance.published_bytes
        != sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    ):
        raise ValueError(
            "PLCS measured performance evidence violates the compact schema."
        )
    return {
        "scene_id": scene_id,
        "frame_count": frame_count,
        "camera_count": len(cameras),
        "sample_count": sample_count,
        "published_bytes": performance.published_bytes,
        "dense_reference_bytes": performance.dense_reference_bytes,
    }


def _manifest(directory: Path) -> Mapping[str, object]:
    if (
        directory.name not in {"plcs", "staging"}
        or not directory.is_dir()
        or directory.is_symlink()
    ):
        raise ValueError(
            "PLCS validation requires a canonical plcs or staging directory."
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


def _validate_label(
    label: Mapping[str, object],
    *,
    scene_id: str,
    key: RenderSampleKey,
    visible_counts: Mapping[int, int],
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
    ):
        raise ValueError("PLCS compact label identity disagrees with its delta.")
    objects = _array(label["objects"], name="objects")
    if not objects:
        raise ValueError("PLCS label must retain every declared track.")
    declared_ids: set[int] = set()
    expected_visible: dict[int, int] = {}
    for index, item in enumerate(objects):
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
        present = record["present"]
        if not isinstance(present, bool):
            raise TypeError("PLCS object present must be boolean.")
        count = _nonnegative_integer(
            record["visible_pixel_count"], name="visible_pixel_count"
        )
        if present != (record["source_frame_index"] is not None):
            raise ValueError(
                "PLCS object presence disagrees with source frame mapping."
            )
        if present != (record["scene_from_asset"] is not None):
            raise ValueError(
                "PLCS object presence and transform disagree."
            )
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


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


__all__ = ["PLCSCompactDatasetReader", "validate_plcs_dataset"]
