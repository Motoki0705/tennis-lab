"""Replay event_detection against an adopted, immutable SceneResult v2."""

from __future__ import annotations

import json
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any, cast
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.ball_event_schema import BALL_EVENT_FIELDS
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.event_detection import (
    EVENT_DETECTION,
    EventDetectionConfig,
    EventDetectionInput,
    EventDetectionModule,
    EventDetectionResult,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.tennis_scene.pipeline.storage.scene_export import (
    export_scene,
    validate_scene_export,
)
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
from src.tennis_scene.schema import SceneResult, validate_scene_result_arrays


def _save_extension(scene: SceneResult, path: Path, *, base: Path) -> None:
    """Append event members to a copy of the validated v2 ZIP archive."""
    validate_scene_result_arrays(scene)
    shutil.copyfile(base, path)
    with ZipFile(path, "a", compression=ZIP_DEFLATED) as archive:
        if len(set(archive.namelist())) != len(archive.namelist()) or any(
            name.startswith("ball_event_") for name in archive.namelist()
        ):
            raise ValueError("Base v2 archive already has event members or duplicate members")
        for name in BALL_EVENT_FIELDS:
            value = getattr(scene, name)
            if value is not None:
                with archive.open(f"{name}.npy", "w") as member:
                    np.lib.format.write_array(member, value, allow_pickle=False)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(scene.metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def verify_archive_extension(base: Path, derived: Path) -> int:
    """Check that every original NPY member has identical bytes."""
    with ZipFile(base) as old, ZipFile(derived) as new:
        original = set(old.namelist())
        added = set(new.namelist()) - original
        if not original <= set(new.namelist()) or not added or not added <= {
            f"{name}.npy" for name in BALL_EVENT_FIELDS
        }:
            raise ValueError("Derived NPZ has unexpected member names")
        for name in original:
            with old.open(name) as before, new.open(name) as after:
                while block := before.read(4 * 1024**2):
                    if block != after.read(len(block)):
                        raise ValueError(f"Derived NPZ changed v2 array {name}: {derived}")
                if after.read(1):
                    raise ValueError(f"Derived NPZ changed the size of v2 array {name}: {derived}")
    return len(original)


def _base_v2_archive(store_root: Path, index: dict[str, Any], assembly: Any) -> Path:
    """Resolve the immutable v2 scene archive from either a v2 or v3 adopted store."""
    index_path = store_root / "scene.json"
    current_archive = Path(indexed_scene_path(index_path))
    export_inputs = index["exports"]["scene"]["inputs"]
    if export_inputs == {"scene_assembly": json_value(assembly)}:
        return current_archive
    event_ref = index["artifacts"].get(EVENT_DETECTION)
    if (
        isinstance(event_ref, dict)
        and (event_ref.get("schema"), event_ref.get("version")) == ("scene_result", 3)
        and export_inputs == {EVENT_DETECTION: event_ref}
    ):
        base_archive = Path(store_root / "exports" / str(assembly.artifact_id) / "scene.npz")
        if not base_archive.is_file():
            raise ValueError(f"Missing immutable v2 scene export for replay: {base_archive}")
        verify_archive_extension(base_archive, current_archive)
        return base_archive
    raise ValueError("Event replay requires an adopted SceneResult v2 assembly or v3 event export")


def replay_scene_events(
    store_root: Path, config: EventDetectionConfig, *,
    code_commit: str, code_identity: str,
) -> tuple[EventDetectionResult, Path, dict[str, Any]]:
    """Adopt v3 or v4 in a writable copy of a v2/v3 store, reusing its assembly arrays.

    Its immutable v2 assembly export and component must agree. The saved lineage
    continues to refer to the original upstream artifacts and their execution identities.
    """
    index_path = store_root / "scene.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    store = ClipStore(store_root, index["source"], memory_entries=0)
    source = cast(ClipSource, restore_type(index["source"], ClipSource))
    assembly = store.references()["scene_assembly"]
    if (assembly.schema, assembly.version) != ("scene_result", 2):
        raise ValueError("Event replay requires the adopted SceneResult v2 assembly export")
    base_archive = _base_v2_archive(store_root, index, assembly)
    base = store.load(assembly, ArtifactCodec(SceneResult))
    validate_scene_export(base, base_archive)
    module = EventDetectionModule(config, source.clip_id, code_commit=code_commit)
    result = module.process(EventDetectionInput(source, base))
    module.require_inputs()

    paths = {}
    for item in fields(SceneResult):
        value = getattr(base, item.name)
        if isinstance(value, np.memmap) and value.filename is not None:
            paths[id(value)] = Path(value.filename)
    from src.utils.checksum import dual_sha256

    identity = {
        "component": EVENT_DETECTION,
        "schema": module.io.output_schema,
        "version": module.io.version,
        "implementation": code_identity,
        "settings": module.cache_identity(),
        "source_sha256": store.source_key,
        "dependencies": {"scene": json_value(assembly)},
        "base_export_sha256": dual_sha256(base_archive),
    }
    reference = store.publish(
        EVENT_DETECTION, result, ArtifactCodec(EventDetectionResult, linked_arrays=paths),
        schema=module.io.output_schema, version=module.io.version, identity=identity,
        dependencies={"scene": assembly}, provenance={"origin": "component", "execution": "fixed_scene_replay"},
    )

    def save_verified(scene: SceneResult, path: Path) -> None:
        _save_extension(scene, path, base=base_archive)
        verify_archive_extension(base_archive, path)
        validate_scene_export(scene, path)

    derived = export_scene(store, result, reference, save=save_verified)
    adopted = json.loads(index_path.read_text(encoding="utf-8"))
    return result, derived, adopted


def verify_replayed_scene(
    base: Path,
    derived: Path,
    *,
    expected_version: int | None = None,
) -> SceneResult:
    """Recheck a published archive against its pinned predecessor."""
    path = indexed_scene_path(derived)
    base_index = json.loads(base.read_text(encoding="utf-8"))
    assembly = base_index["artifacts"]["scene_assembly"]
    if base_index["exports"]["scene"]["inputs"] == {"scene_assembly": assembly}:
        base_archive = indexed_scene_path(base)
    else:
        base_archive = base.parent / "exports" / assembly["artifact_id"] / "scene.npz"
        verify_archive_extension(base_archive, indexed_scene_path(base))
    verify_archive_extension(base_archive, path)
    scene = load_scene_result(path)
    allowed = (3, 4) if expected_version is None else (expected_version,)
    if scene.schema_version not in allowed:
        raise ValueError(f"Expected an adopted SceneResult v{allowed} archive, got v{scene.schema_version}")
    return scene
