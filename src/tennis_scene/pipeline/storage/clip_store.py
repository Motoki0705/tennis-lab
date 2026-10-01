"""Immutable component artifacts; scene.json is the clip's published index."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)
from src.utils.checksum import dual_sha256

OutputT = TypeVar("OutputT")
OutputCo = TypeVar("OutputCo", covariant=True)


class ArtifactReader(Protocol[OutputCo]):
    @property
    def output_type(self) -> type[OutputCo]: ...
    def load(self, payload: Any, directory: Path, arrays: Mapping[str, Any]) -> OutputCo: ...


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    path: str
    sha256: str
    schema: str
    version: int
    execution_key: str


def _component_name(node: str) -> str:
    parts = node.split("/")
    if not parts or any(not p or p in (".", "..") or not all(c.isalnum() or c in "_-" for c in p) for p in parts):
        raise ValueError(f"Invalid component node: {node}")
    return parts[0]


class ClipStore:
    """Disk is authoritative; bounded in-memory entries never change semantics."""

    def __init__(self, root: Path, source: Mapping[str, Any], *, memory_entries: int = 8) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = json_value(source)
        self.source_key = document_digest(self.source)
        self.memory_entries = memory_entries
        self._memory: OrderedDict[str, Any] = OrderedDict()
        with self._lock():
            if not self.index_path.exists():
                write_json_atomic(self.index_path, {"schema": "tennis_scene_index_v1", "source": self.source,
                    "source_sha256": self.source_key, "revision": 0, "artifacts": {}, "exports": {}})
            self._index()

    @property
    def index_path(self) -> Path:
        return self.root / "scene.json"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with (self.root / ".scene.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _index(self) -> dict[str, Any]:
        value = json.loads(self.index_path.read_text())
        if value.get("schema") != "tennis_scene_index_v1" or value.get("source_sha256") != self.source_key:
            raise ValueError("Clip store source/schema mismatch; use a separate clip store")
        return cast(dict[str, Any], value)

    def references(self) -> dict[str, ArtifactRef]:
        return {key: ArtifactRef(**value) for key, value in self._index()["artifacts"].items()}

    def active(self, node: str) -> ArtifactRef | None:
        _component_name(node)
        return self.references().get(node)

    def _path(self, relative: str) -> Path:
        result = (self.root / relative).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError("Artifact path escapes clip store")
        return result

    def descriptor(self, reference: ArtifactRef) -> dict[str, Any]:
        return read_component_descriptor(self.root, json_value(reference), node=None, source_sha256=self.source_key)

    def load(self, reference: ArtifactRef, codec: ArtifactReader[OutputT]) -> OutputT:
        descriptor = self.descriptor(reference)
        # Validate bytes on every disk read; a memory hit refers to the same immutable ID.
        if reference.artifact_id in self._memory:
            value = self._memory.pop(reference.artifact_id)
            self._memory[reference.artifact_id] = value
            if not isinstance(value, codec.output_type):
                raise TypeError("Cached artifact output contract mismatch")
            return value
        value = codec.load(descriptor["payload"], self._path(reference.path).parent, descriptor["arrays"])
        self._remember(reference.artifact_id, value)
        return value

    def _remember(self, key: str, value: Any) -> None:
        self._memory[key] = value
        while len(self._memory) > max(0, self.memory_entries):
            self._memory.popitem(last=False)

    def publish(self, node: str, value: OutputT, codec: ArtifactCodec[OutputT], *, schema: str,
                version: int, identity: Mapping[str, Any], dependencies: Mapping[str, ArtifactRef],
                provenance: Mapping[str, Any]) -> ArtifactRef:
        component = _component_name(node)
        execution_key = document_digest(identity)
        parent = self.root / "components" / component
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=parent))
        try:
            payload, arrays = codec.dump(value, temporary)
            document = {"schema": "component_artifact_v1", "node": node, "output_schema": schema,
                "output_version": version, "source_sha256": self.source_key, "execution_key": execution_key,
                "identity": json_value(identity), "dependencies": json_value(dependencies),
                "provenance": json_value(provenance), "payload": payload, "arrays": arrays, "status": "complete"}
            artifact_id = document_digest(document)
            document["artifact_id"] = artifact_id
            filename = f"{component}.json"
            write_json_atomic(temporary / filename, document)
            destination = parent / artifact_id
            with self._lock():
                if destination.exists():
                    if (destination / filename).read_bytes() != (temporary / filename).read_bytes():
                        raise ValueError("Artifact identity collision")
                    shutil.rmtree(temporary)
                else:
                    os.replace(temporary, destination)
                descriptor_path = destination / filename
                reference = ArtifactRef(artifact_id, str(descriptor_path.relative_to(self.root)),
                    dual_sha256(descriptor_path), schema, version, execution_key)
                index = self._index()
                encoded = json_value(reference)
                if index["artifacts"].get(node) != encoded:
                    # The published scene is a snapshot of one complete input
                    # lineage. Keep old export bytes immutable, but withdraw its
                    # active pointer as soon as any adopted component changes.
                    index["exports"].clear()
                index["artifacts"][node] = encoded
                index["revision"] += 1
                write_json_atomic(self.index_path, index)
            # The first consumer uses the same decoded representation as a restarted run.
            self._memory.pop(artifact_id, None)
            return reference
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def record_export(self, name: str, files: Mapping[str, Path], inputs: Mapping[str, ArtifactRef]) -> None:
        with self._lock():
            index = self._index()
            for node, reference in inputs.items():
                if index["artifacts"].get(node) != json_value(reference):
                    raise ValueError(f"Cannot export a superseded component artifact: {node}")
            assert_current_component_lineage(index, self.root, json_value(inputs))
            index["exports"][name] = {"files": {key: {"path": str(path.resolve().relative_to(self.root)), "sha256": dual_sha256(path)} for key, path in files.items()},
                "inputs": json_value(inputs)}
            index["revision"] += 1
            write_json_atomic(self.index_path, index)
