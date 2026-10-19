"""Read only the component artifacts adopted by one scene index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor


class StoreReader:
    """Read a saved ``scene.json`` without inference, output directories or writes.

    Payloads retain their stored schema. Consumers select supported schema/versions
    from the returned descriptor; this reader never converts historical artifacts.
    """

    def __init__(self, index: Path) -> None:
        self.index_path = index.resolve()
        document = json.loads(self.index_path.read_text())
        self.document = document
        if document.get("schema") != "tennis_scene_index_v1":
            raise ValueError("Expected tennis_scene_index_v1 scene.json")
        self.root = self.index_path.parent
        self.source = document["source"]
        self.source_sha256 = str(document["source_sha256"])
        self.camera_ids = tuple(video["camera_id"] for video in self.source["videos"])
        self.videos = {video["camera_id"]: Path(video["path"]) for video in self.source["videos"]}
        self.frame_count = int(self.source["videos"][0]["num_frames"])
        self.size = (int(self.source["videos"][0]["width"]), int(self.source["videos"][0]["height"]))
        self.references: dict[str, dict[str, Any]] = document["artifacts"]
        self._active_by_artifact = {reference["artifact_id"]: node for node, reference in self.references.items()}
        self._staleness: dict[str, tuple[str, ...]] = {}

    def descriptor(self, node: str) -> dict[str, Any]:
        """Return the descriptor after verifying its path, checksum and index entry."""
        descriptor: dict[str, Any] = read_component_descriptor(self.root, self.references[node], node=node, source_sha256=self.source_sha256)
        return descriptor

    def payload(self, node: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return ``(payload, descriptor)`` after verifying stored array checksums."""
        reference = self.references[node]
        descriptor = self.descriptor(node)
        location = (self.root / reference["path"]).resolve().parent
        payload = unpack_value(descriptor["payload"], location, descriptor["arrays"])
        if not isinstance(payload, dict):
            raise TypeError(f"{node} must have a structured component payload")
        return payload, descriptor

    def stale_dependencies(self, node: str, ancestors: frozenset[str] = frozenset()) -> tuple[str, ...]:
        """Do not show descendants of an artifact superseded in scene.json."""
        if node in self._staleness:
            return self._staleness[node]
        if node in ancestors:
            raise ValueError(f"Cyclic component artifact dependencies at {node}")
        descriptor = self.descriptor(node)
        reasons: list[str] = []
        for port, dependency in descriptor["dependencies"].items():
            producer = self._active_by_artifact.get(dependency["artifact_id"])
            if producer is None:
                reasons.append(f"{port}: upstream artifact {dependency['artifact_id'][:12]} was superseded")
            elif self.references[producer] != dependency:
                reasons.append(f"{port}: upstream artifact reference disagrees with the adopted index")
            elif self.stale_dependencies(producer, ancestors | {node}):
                reasons.append(f"{port}: upstream component {producer} is stale")
        result = tuple(reasons)
        self._staleness[node] = result
        return result
