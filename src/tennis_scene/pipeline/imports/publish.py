"""The one publish path of every importer: declared node, bound inputs, explicit origin."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.tennis_scene.pipeline.runner import ComponentNode
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec

IMPORT_ORIGIN = "import"


@dataclass(frozen=True)
class BoundImport:
    """A load-only node with the store's adopted artifacts of its declared inputs.

    ``references`` become the published artifact's dependencies, so the runner
    rejects the import once any of these upstream artifacts is replaced.
    """

    node: ComponentNode
    references: Mapping[str, ArtifactRef]
    artifacts: Mapping[str, Any]


def bind_import(nodes: Sequence[ComponentNode], name: str, store: ClipStore) -> BoundImport:
    """Resolve ``name``'s declared bindings to the store's adopted artifacts.

    Only a node declared ``load`` accepts an import: an executed node would
    replace it on the next run. Every bound producer must already be adopted.
    """
    by_name = {node.name: node for node in nodes}
    if name not in by_name:
        raise ValueError(f"Unknown import target: {name}")
    node = by_name[name]
    if node.source != "load":
        raise ValueError(f"{name} must be declared load-only (execution.{name.split('/')[0]}=load) to receive an import")
    references: dict[str, ArtifactRef] = {}
    artifacts: dict[str, Any] = {}
    for port, producer in node.bindings.items():
        reference = store.active(producer)
        if reference is None:
            raise FileNotFoundError(f"Cannot import {name} before its input {producer} exists in the store")
        references[port] = reference
        artifacts[port] = store.load(reference, ArtifactCodec(by_name[producer].io.output_type))
    return BoundImport(node, references, artifacts)


def publish_import(bound: BoundImport, store: ClipStore, value: Any, *, importer: str, version: int,
                   identity: Mapping[str, Any], provenance: Mapping[str, Any]) -> ArtifactRef:
    """Publish ``value`` under the node's declared schema with an ``import`` origin.

    The origin tells the runner (and every reader) that no component produced
    this artifact; ``identity`` must name every source file by content digest.
    """
    node = bound.node
    if not isinstance(value, node.io.output_type):
        raise TypeError(f"{importer} produced {type(value).__name__}, not {node.io.output_type.__name__} for {node.name}")
    reserved = {"importer", "importer_version", "source_sha256"} & (set(identity) | set(provenance))
    if reserved or "origin" in provenance:
        raise ValueError(f"Importer documents cannot override reserved fields: {sorted(reserved | ({'origin'} & set(provenance)))}")
    return store.publish(node.name, value, ArtifactCodec(node.io.output_type), schema=node.io.output_schema,
        version=node.io.version, identity={"importer": importer, "importer_version": version,
                                           "source_sha256": store.source_key, **identity},
        dependencies=bound.references,
        provenance={"origin": IMPORT_ORIGIN, "importer": importer, "importer_version": version, **provenance})
