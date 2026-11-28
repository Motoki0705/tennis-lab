"""Publish and restore the two pointers of an external ball scene revision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

import numpy as np

from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.clip_inputs import DatasetClipInputs, InputSnapshot
from src.tennis_scene.pipeline.dataset_publication import (
    ANNOTATION_SCHEMA_VERSION,
    scene_array_manifest,
)
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, clip_store_lock
from src.tennis_scene.pipeline.storage.scene_export import (
    scene_export_node,
    write_scene_export,
)
from src.tennis_scene.pipeline.storage.scene_index import (
    read_component_descriptor,
    resolve_indexed_scene,
)
from src.tennis_scene.pipeline.storage.scene_pointers import (
    read_pending_scene_update,
    replace_scene_pointers,
)
from src.tennis_scene.pipeline.video_artifact import validate_digest
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import dual_sha256
from src.utils.io import utc_now_iso

PUBLICATION = "external_ball_publication"
SCHEMA = "external_ball_publication_v1"
_RECEIPT_DIRECTORY = Path("updates") / "external_ball"
_RECOVERY = (
    "re-run the original publish command at the same source revision, "
    "or run '.venv/bin/python -m src.tennis_scene.scripts.import_external_ball "
    "restore --dataset <dataset> --video <video> --clip <clip>'."
)


def _encode(document: dict[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(content: str | None) -> str | None:
    return None if content is None else sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScenePointers:
    scene: str
    annotation: str | None

    @classmethod
    def read(cls, root: Path) -> ScenePointers:
        index = (root / "scene.json").read_bytes().decode("utf-8")
        marker = root / "annotation.json"
        return cls(
            index, marker.read_bytes().decode("utf-8") if marker.is_file() else None
        )

    def index(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.scene))

    def marker(self) -> dict[str, Any] | None:
        return None if self.annotation is None else json.loads(self.annotation)

    def revision(self) -> str | None:
        marker = self.marker()
        names = {
            value
            for doc in (self.index(), marker or {})
            if (value := doc.get(PUBLICATION)) is not None
        }
        if len(names) > 1:
            raise ValueError(
                "Scene and annotation pointers name different external ball revisions"
            )
        return validate_digest(names.pop()) if names else None

    def verify(self, root: Path) -> Path | None:
        index, marker = self.index(), self.marker()
        if marker is None:
            if index["exports"].get("scene") is not None:
                raise ValueError("Saved scene export has no annotation marker")
            return None
        version = marker.get("scene_schema_version")
        if type(version) is not int or version not in (2, 3, 4):
            raise ValueError("Saved annotation must declare SceneResult v2, v3 or v4")
        artifacts = index["artifacts"]
        assembly = artifacts.get("scene_assembly")
        if not isinstance(assembly, dict) or (
            assembly.get("schema"),
            assembly.get("version"),
        ) != ("scene_result", 2):
            raise ValueError("Saved scene requires an adopted SceneResult v2 assembly")
        node = scene_export_node(version)
        reference = artifacts.get(node)
        export = index["exports"].get("scene")
        if (
            not isinstance(reference, dict)
            or (reference.get("schema"), reference.get("version"))
            != ("scene_result", version)
            or not isinstance(export, dict)
            or export.get("inputs") != {node: reference}
        ):
            raise ValueError(
                f"SceneResult v{version} export must reference the adopted {node}"
            )
        path: Path = resolve_indexed_scene(
            index, root, required_components={"scene_assembly": assembly}
        )
        if version in (3, 4):
            descriptor = read_component_descriptor(
                root, reference, node=node, source_sha256=index["source_sha256"]
            )
            if descriptor["dependencies"].get("scene") != assembly:
                raise ValueError(
                    "Saved event_detection does not use the adopted scene_assembly"
                )
        metadata = json.loads(
            path.with_suffix(".metadata.json").read_text(encoding="utf-8")
        )
        archive_version = (
            metadata.get("scene_schema_version") if isinstance(metadata, dict) else None
        )
        if (
            marker.get("scene_index") != "scene.json"
            or marker.get("scene_result") != str(path.relative_to(root))
            or marker.get("clip_id") != index["source"]["clip_id"]
            or type(archive_version) is not int
            or archive_version != version
        ):
            raise ValueError("Saved scene index and annotation marker disagree")
        return path


@dataclass(frozen=True)
class PublicationReceipt:
    before: ScenePointers
    after: ScenePointers
    inputs: dict[str, Any]
    statistics: dict[str, Any]

    @classmethod
    def read(cls, root: Path, scene_id: str) -> PublicationReceipt:
        document = json.loads(_receipt_path(root, scene_id).read_text())
        if document["schema"] != SCHEMA:
            raise ValueError("Unsupported external ball publication receipt")
        receipt = cls(
            ScenePointers(**document["before"]),
            ScenePointers(**document["after"]),
            document["inputs"],
            document["statistics"],
        )
        if (
            receipt.before.revision() is not None
            or receipt.after.revision() != scene_id
            or receipt.after.index()["artifacts"]["scene_assembly"]["artifact_id"]
            != scene_id
            or document["pointer_digests"] != receipt.pointer_digests()
        ):
            raise ValueError("Invalid external ball publication receipt")
        return receipt

    def pointer_digests(self) -> dict[str, Any]:
        return {
            label: {
                "scene_sha256": _digest(pointers.scene),
                "annotation_sha256": _digest(pointers.annotation),
            }
            for label, pointers in (("before", self.before), ("after", self.after))
        }

    def require_current(self, current: ScenePointers) -> None:
        # A process can stop after either rename. An unrelated pointer change
        # must never be overwritten as part of recovery.
        if current.scene not in (
            self.before.scene,
            self.after.scene,
        ) or current.annotation not in (self.before.annotation, self.after.annotation):
            raise ValueError(
                "Clip pointers changed independently of the recorded external ball update"
            )

    def activate(self, root: Path, current: ScenePointers) -> None:
        self.require_current(current)
        if self.after.verify(root) is None:
            raise ValueError(
                "External ball publication requires a complete scene and marker"
            )
        self._switch(root, self.after)

    def restore(self, root: Path, current: ScenePointers) -> None:
        self.require_current(current)
        self.before.verify(root)
        self._switch(root, self.before)

    def _switch(self, root: Path, desired: ScenePointers) -> None:
        revision = self.after.revision()
        if revision is None:
            raise ValueError(
                "External ball publication receipt must identify its revision"
            )
        replace_scene_pointers(
            root,
            scene=desired.scene,
            annotation=desired.annotation,
            receipt=_receipt_path(root, revision),
            recovery=_RECOVERY,
        )


def _receipt_path(root: Path, scene_id: str) -> Path:
    return root / _RECEIPT_DIRECTORY / f"{validate_digest(scene_id)}.json"


def _current_receipt(root: Path, current: ScenePointers) -> PublicationReceipt | None:
    pending = read_pending_scene_update(root)
    if pending is None:
        revision = current.revision()
    else:
        if (
            pending.receipt.parent != root.resolve() / _RECEIPT_DIRECTORY
            or pending.receipt.suffix != ".json"
        ):
            raise pending.error()
        revision = pending.receipt.stem
    receipt = None if revision is None else PublicationReceipt.read(root, revision)
    if receipt is not None:
        receipt.require_current(current)
    return receipt


def current_publication(root: Path) -> tuple[ScenePointers, PublicationReceipt | None]:
    with clip_store_lock(root):
        current = ScenePointers.read(root)
        return current, _current_receipt(root, current)


def restore_publication(root: Path) -> bool:
    with clip_store_lock(root):
        current = ScenePointers.read(root)
        receipt = _current_receipt(root, current)
        if receipt is None:
            return False
        receipt.restore(root, current)
        return True


def publish_scene_revision(
    root: Path,
    current: ScenePointers,
    before: ScenePointers,
    nodes: dict[str, ArtifactRef],
    scene: SceneResult,
    clip: DatasetClipInputs,
    input_files: InputSnapshot,
    identity: dict[str, Any],
    statistics: dict[str, Any],
) -> PublicationReceipt:
    assembly_id = nodes["scene_assembly"].artifact_id
    journal = _receipt_path(root, assembly_id)
    with clip_store_lock(root):
        if ScenePointers.read(root) != current:
            raise ValueError(
                "Clip pointers changed during external ball reconstruction"
            )
        pending = read_pending_scene_update(root)
        if pending is not None and pending.receipt != journal.resolve():
            raise pending.error()
        input_files.assert_unchanged()
        if journal.exists():
            receipt = PublicationReceipt.read(root, assembly_id)
            if receipt.inputs != identity or receipt.before != before:
                raise ValueError(
                    "External ball receipt conflicts with the requested inputs"
                )
        else:
            receipt = _stage_publication(
                root, before, nodes, scene, clip, identity, statistics
            )
            write_json_atomic(
                journal,
                {
                    "schema": SCHEMA,
                    **json_value(receipt),
                    "pointer_digests": receipt.pointer_digests(),
                    "previous_scene_assembly": before.index()["artifacts"].get(
                        "scene_assembly"
                    ),
                    "previous_scene_export": before.index()["exports"].get("scene"),
                },
            )
        receipt.activate(root, current)
        return receipt


def _stage_publication(
    root: Path,
    before: ScenePointers,
    nodes: dict[str, ArtifactRef],
    scene: SceneResult,
    clip: DatasetClipInputs,
    identity: dict[str, Any],
    statistics: dict[str, Any],
) -> PublicationReceipt:
    node = scene_export_node(scene.schema_version)
    reference = nodes[node]
    assembly_id = nodes["scene_assembly"].artifact_id
    destination = root / "exports" / reference.artifact_id
    if not destination.exists():
        write_scene_export(scene, destination)
    archive = destination / "scene.npz"
    doc = before.index()
    descriptor = read_component_descriptor(
        root,
        json_value(reference),
        node=node,
        source_sha256=doc["source_sha256"],
    )
    _verify_archive(archive, scene, descriptor)
    doc["artifacts"].update(json_value(nodes))
    doc["revision"] += 1
    doc[PUBLICATION] = assembly_id
    doc["exports"]["scene"] = {
        "inputs": {node: json_value(reference)},
        "files": {
            role: {"path": str(path.relative_to(root)), "sha256": dual_sha256(path)}
            for role, path in (
                ("scene", archive),
                ("metadata", archive.with_suffix(".metadata.json")),
            )
        },
    }
    identity_digest = document_digest(identity)
    config = root / "configs" / f"{identity_digest}.yaml"
    # JSON is a subset of YAML. Keep the exact pinned import recipe beside the
    # previous reconstruction's immutable config.
    if config.exists():
        if json.loads(config.read_text()) != identity:
            raise ValueError("Saved external ball publication recipe changed")
    else:
        write_json_atomic(config, identity)
    annotation = {
        "version": ANNOTATION_SCHEMA_VERSION,
        "clip_id": clip.record.clip_id,
        "generator": "src.tennis_scene.scripts.import_external_ball",
        "generated_at": utc_now_iso(),
        "scene_result": str(archive.relative_to(root)),
        "scene_index": "scene.json",
        "pipeline_config": str(config.relative_to(root)),
        "clip_manifest_sha256": clip.manifest_sha256,
        "arrays": scene_array_manifest(scene),
        "scene_schema_version": scene.schema_version,
        "result_status": scene.metadata["status"],
        "validity_statistics": scene.metadata["validity_statistics"],
        "publication_identity_sha256": identity_digest,
        "media_sha256": dict(clip.media_sha256),
        PUBLICATION: assembly_id,
    }
    return PublicationReceipt(
        before, ScenePointers(_encode(doc), _encode(annotation)), identity, statistics
    )


def _verify_archive(path: Path, scene: SceneResult, descriptor: dict[str, Any]) -> None:
    """Read back the full archive in chunks against the staged NPY checksums."""
    arrays = {
        f"{name}.npy": descriptor["arrays"][entry["array"]]["sha256"]
        for name, entry in descriptor["payload"].items()
        if isinstance(entry, dict) and set(entry) == {"array"}
    }
    scalars = {
        f"{name}.npy": getattr(scene, name)
        for name in ("num_frames", "width", "height", "fps")
    }
    with ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(arrays.keys() | scalars.keys()):
            raise ValueError(
                "Staged scene archive member names disagree with the component"
            )
        for member, expected in arrays.items():
            digest = sha256()
            with archive.open(member) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Staged scene archive disagrees with {member}")
        for member, expected in scalars.items():
            if (
                np.load(BytesIO(archive.read(member)), allow_pickle=False).item()
                != expected
            ):
                raise ValueError(f"Staged scene archive disagrees with {member}")
    if json.loads(path.with_suffix(".metadata.json").read_text()) != json_value(
        scene.metadata
    ):
        raise ValueError("Staged scene archive metadata disagrees with the component")
