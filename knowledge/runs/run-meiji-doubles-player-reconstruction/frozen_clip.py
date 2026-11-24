"""Bind artifacts to the published Meiji external-ball revision."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np

from src.tennis_scene.generate_dataset.manifest import DatasetClipRecord
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.clip_inputs import (
    DatasetClipInputs,
    resolve_dataset_clip,
)
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO
from src.tennis_scene.pipeline.dataset_publication import ANNOTATION_RELATIVE_DIR
from src.tennis_scene.pipeline.imports.external_ball_publication import (
    PublicationKind,
    PublicationReceipt,
    ScenePointers,
    current_publication,
)
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import (
    ArtifactCodec,
    restore_type,
    unpack_value,
)
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
)
from src.tennis_scene.schema import SceneResult

OutputT = TypeVar("OutputT")
PLAYERS = PublicationKind(
    "doubles_players",
    "knowledge/runs/run-meiji-doubles-player-reconstruction/reconstruct.py",
)
BALL_FIELDS = ("ball_uv", "ball_vis", "ball_3d", "ball_3d_valid", "ball_rejection_code")
SCENE_FIELDS = (
    "player_track_ids",
    "player_kp_3d_vis",
    "player_valid",
    "player_smpl_valid",
)
UPSTREAM = (
    "person_detection",
    "person_pose_features",
    "person_appearance_features",
    "person_tracking",
)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def player_counts(arrays: dict[str, Any]) -> dict[str, int]:
    return {
        "players": len(arrays["player_track_ids"]),
        "joint_player_frames": int(arrays["player_kp_3d_vis"].any(axis=-1).sum()),
        "root_player_frames": int(arrays["player_valid"].sum()),
        "mesh_player_frames": int(arrays["player_smpl_valid"].sum()),
    }


@dataclass
class FrozenClip:
    clip: DatasetClipInputs
    store: ClipStore
    current: ScenePointers
    previous: ScenePointers
    player_receipt: PublicationReceipt | None
    source: ClipSource
    refs: dict[str, ArtifactRef]
    pinned: dict[str, Any]
    staged: dict[str, ArtifactRef] = field(default_factory=dict)

    @classmethod
    def bind(
        cls, dataset: Path, record: DatasetClipRecord, recipe: dict[str, Any]
    ) -> FrozenClip:
        clip = resolve_dataset_clip(dataset, record)
        root = (clip.clip_dir / ANNOTATION_RELATIVE_DIR).resolve(strict=True)
        current, player_receipt = current_publication(root, PLAYERS)
        previous = current if player_receipt is None else player_receipt.before
        external_id = previous.revision()
        if (
            external_id is None
            or PublicationReceipt.read(root, external_id).after != previous
        ):
            raise ValueError(
                f"{record.clip_id}: expected the completed #2458 pointer pair"
            )
        previous.verify(root)
        index = previous.index()
        source: ClipSource = restore_type(index["source"], ClipSource)
        marker = previous.marker()
        if (
            marker is None
            or marker["clip_manifest_sha256"] != clip.manifest_sha256
            or marker["media_sha256"] != clip.media_sha256
            or document_digest(source) != index["source_sha256"]
            or source.clip_id != record.clip_id
            or source.camera_ids != clip.camera_ids
            or (source.num_frames, *source.size, source.fps)
            != (record.num_frames, record.width, record.height, record.fps)
            or {v.camera_id: v.sha256 for v in source.videos} != clip.media_sha256
            or tuple(path.resolve() for path in source.paths)
            != tuple(path.resolve() for path in clip.video_paths)
        ):
            raise ValueError(
                f"{record.clip_id}: saved source and dataset media disagree"
            )
        refs = {name: ArtifactRef(**ref) for name, ref in index["artifacts"].items()}
        upstream = {
            f"{stage}/{camera}": index["artifacts"][f"{stage}/{camera}"]
            for stage in UPSTREAM
            for camera in source.camera_ids
        }
        assert_current_component_lineage(index, root, upstream)
        alignment = (
            "camera_alignment"
            if "camera_alignment" in refs
            else "external_ball_camera_geometry"
        )
        pinned = {
            **recipe,
            "source_sha256": index["source_sha256"],
            "clip_manifest_sha256": clip.manifest_sha256,
            "previous_pointer_sha256": document_digest(previous),
            "previous_scene_assembly": json_value(refs["scene_assembly"]),
            "previous_scene_export": index["exports"]["scene"],
            "alignment_node": alignment,
            "alignment": json_value(refs[alignment]),
            "ball": json_value(refs["ball_triangulation"]),
            "upstream": upstream,
        }
        return cls(
            clip,
            ClipStore(root, json_value(source), memory_entries=0),
            current,
            previous,
            player_receipt,
            source,
            refs,
            pinned,
        )

    def ref(self, name: str) -> ArtifactRef:
        return self.staged[name] if name in self.staged else self.refs[name]

    def descriptor(self, name: str, schema: str, version: int) -> dict[str, Any]:
        ref = self.ref(name)
        descriptor = self.store.descriptor(ref)
        if (
            (ref.schema, ref.version) != (schema, version)
            or descriptor["status"] != "complete"
            or descriptor["node"] != name
            or descriptor["execution_key"] != ref.execution_key
            or document_digest(descriptor["identity"]) != ref.execution_key
        ):
            raise ValueError(f"{self.source.clip_id}: unexpected artifact for {name}")
        return descriptor

    def load(
        self, name: str, output: type[OutputT], schema: str, version: int = 1
    ) -> OutputT:
        self.descriptor(name, schema, version)
        return self.store.load(self.ref(name), ArtifactCodec(output))

    def fields(
        self, name: str, fields: tuple[str, ...], schema: str, version: int
    ) -> dict[str, Any]:
        doc = self.descriptor(name, schema, version)
        directory = (self.store.root / self.ref(name).path).parent
        return {
            field_name: unpack_value(
                doc["payload"][field_name], directory, doc["arrays"]
            )
            for field_name in fields
        }

    def load_output(self, name: str, io: ComponentIO[Any, OutputT]) -> OutputT:
        return self.load(name, io.output_type, io.output_schema, io.version)

    def stage(
        self,
        name: str,
        value: OutputT,
        io: ComponentIO[Any, OutputT],
        bindings: dict[str, str],
        *,
        settings: dict[str, Any] | None = None,
    ) -> str:
        deps = {port: self.ref(producer) for port, producer in bindings.items()}
        self.staged[name] = self.store.publish(
            name,
            value,
            ArtifactCodec(io.output_type),
            schema=io.output_schema,
            version=io.version,
            identity={
                **self.pinned,
                "node": name,
                "dependencies": json_value(deps),
                "settings": settings,
            },
            dependencies=deps,
            provenance={"origin": "component", "execution": "doubles_player_replay"},
            adopt=False,
        )
        return name

    def check_staged(self, staged: dict[str, ArtifactRef]) -> None:
        # Validate restart data against the frozen index and immutable descriptors.
        index = self.previous.index()
        index["artifacts"].update(json_value(staged))
        if staged:
            assert_current_component_lineage(index, self.store.root, json_value(staged))
        for name, ref in staged.items():
            doc = self.store.descriptor(ref)
            if (
                any(
                    doc["identity"].get(key) != value
                    for key, value in self.pinned.items()
                )
                or doc["identity"].get("node") != name
                or doc["identity"].get("dependencies") != doc["dependencies"]
                or document_digest(doc["identity"]) != ref.execution_key
            ):
                raise ValueError(
                    f"{self.source.clip_id}: staged recipe changed for {name}"
                )

    def before_counts(self) -> dict[str, int]:
        return player_counts(
            self.fields("scene_assembly", SCENE_FIELDS, "scene_result", 2)
        )

    def check_ball(self, scene: SceneResult) -> None:
        old = self.fields("scene_assembly", BALL_FIELDS, "scene_result", 2)
        for name, value in old.items():
            revised = getattr(scene, name)
            if (
                revised is None
                or value.dtype != revised.dtype
                or not np.array_equal(value, revised)
            ):
                raise ValueError(
                    f"{self.source.clip_id}: preserved ball array changed: {name}"
                )
        metadata = self.descriptor("scene_assembly", "scene_result", 2)["payload"][
            "metadata"
        ]
        for key in ("court_reference", "external_ball"):
            if metadata[key] != json_value(scene.metadata[key]):
                raise ValueError(
                    f"{self.source.clip_id}: preserved {key} metadata changed"
                )
