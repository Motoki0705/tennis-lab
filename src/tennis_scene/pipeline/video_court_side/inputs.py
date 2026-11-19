"""Freeze the relevant component revisions of a clip without pinning downstream publications."""
from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.tasks.court_side.hypothesis import EvidenceKind
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.ball_points import (
    BALL_POINTS_PORT,
    BallPointsOutput,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import (
    CourtSideInput,
    PlayerBoxObservations,
    PlayerCourtSideInput,
)
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionModule
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    CourtSideInputAssembler,
)
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef
from src.tennis_scene.pipeline.storage.codec import (
    ArtifactCodec,
    restore_type,
    verified_array_path,
)
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)
from src.tennis_scene.pipeline.video_artifact import (
    VideoArtifactConfig,
    validate_digest,
)
from src.utils.configuration import PathResolver, PathRole


def side_input_stage(evidence: EvidenceKind) -> str:
    if evidence == "ball":
        return "ball_points"
    if evidence == "player":
        return "player_selection"
    raise ValueError(f"Unknown court side evidence: {evidence}")


@dataclass(frozen=True)
class PinnedSideInputs:
    directory: Path
    source: ClipSource
    source_digest: str
    references: dict[str, Any]
    evidence: EvidenceKind = "ball"

    def __post_init__(self) -> None:
        if document_digest(self.source) != self.source_digest:
            raise ValueError("Court side clip/source membership disagrees with its pinned revision")
        expected = {"court_calibration", *(f"{side_input_stage(self.evidence)}/{c}" for c in self.source.camera_ids)}
        if self.references.keys() != expected:
            raise ValueError("Court side input nodes disagree with the selected evidence")

    @property
    def digest(self) -> str:
        identity = {"source_sha256": self.source_digest, "artifacts": self.references}
        if self.evidence == "player":
            identity["evidence"] = self.evidence
        digest: str = document_digest(identity)
        return digest

    @classmethod
    def from_store(cls, directory: Path, *, evidence: EvidenceKind = "ball") -> PinnedSideInputs:
        directory = directory.resolve()
        index = json.loads((directory / "scene.json").read_text())
        if index.get("schema") != "tennis_scene_index_v1":
            raise ValueError("Invalid court side input store schema")
        source: ClipSource = restore_type(index["source"], ClipSource)
        nodes = ["court_calibration", *(f"{side_input_stage(evidence)}/{c}" for c in source.camera_ids)]
        refs = {node: index["artifacts"][node] for node in nodes}
        assert_current_component_lineage(index, directory, refs)
        return cls(directory, source, index["source_sha256"], refs, evidence)

    def assert_current(self) -> None:
        self.verify_arrays()
        if self.from_store(self.directory, evidence=self.evidence).digest != self.digest:
            raise ValueError(f"Court side inputs changed: {self.source.clip_id}")

    def verify_arrays(self) -> None:
        for node, record in self.references.items():
            descriptor = read_component_descriptor(self.directory, record, node=node, source_sha256=self.source_digest)
            directory = (self.directory / record["path"]).parent
            for name in descriptor["arrays"]:
                verified_array_path(directory, name, descriptor["arrays"])

    def assemble(self) -> CourtSideInput | PlayerCourtSideInput:
        artifacts = {}
        player_boxes = {}
        for node, record in self.references.items():
            ref = ArtifactRef(**record)
            descriptor = read_component_descriptor(self.directory, record, node=node, source_sha256=self.source_digest)
            directory = (self.directory / ref.path).parent
            payload, arrays = descriptor["payload"], descriptor["arrays"]
            is_calibration = node == "court_calibration"
            if not is_calibration and self.evidence == "player":
                selection = PlayerSelectionModule.io
                # Both selection versions preserve the same observed box fields.
                # Their tracking feature contracts are independent of this geometry.
                if ref.schema != selection.output_schema or ref.version not in {2, 3}:
                    raise ValueError(f"Invalid court side input schema: {node}")
                tracks = {field.name: payload["tracks"][field.name] for field in fields(PlayerBoxObservations)}
                player_boxes[node.split("/")[1]] = ArtifactCodec(PlayerBoxObservations).load(tracks, directory, arrays)
                continue
            port = ("local_court_calibration", 1) if is_calibration else (BALL_POINTS_PORT.schema, BALL_POINTS_PORT.version)
            if (ref.schema, ref.version) != port:
                raise ValueError(f"Invalid court side input schema: {node}")
            codec: ArtifactCodec[Any] = ArtifactCodec(CourtCalibrationOutput if is_calibration else BallPointsOutput)
            key = "calibration" if is_calibration else f"ball_{node.split('/')[1]}"
            artifacts[key] = codec.load(payload, directory, arrays)
        if self.evidence == "player":
            return PlayerCourtSideInput(self.source, artifacts["calibration"], player_boxes)
        return CourtSideInputAssembler(0.0).assemble(AssemblyContext(self.source), artifacts)


def selected_evidence_clips(members: Collection[str], requested: Sequence[str] | None) -> tuple[str, ...]:
    if requested is not None and (isinstance(requested, str) or not isinstance(requested, Sequence)
                                  or any(not isinstance(clip, str) for clip in requested)):
        raise ValueError("Court side evidence clip IDs must be a sequence of IDs")
    selected = tuple(sorted(members if requested is None else requested))
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(members):
        raise ValueError("Empty, repeated or unknown court side evidence clip")
    return selected


@dataclass(frozen=True)
class CourtSideBuildRequest:
    """The configured selection and revisions, independent of the current state of the clip stores."""

    stores: dict[str, tuple[Path, str]]
    evidence_clip_ids: tuple[str, ...]
    evidence: EvidenceKind

    @classmethod
    def read(cls, config: VideoArtifactConfig, resolver: PathResolver, *, evidence: EvidenceKind = "ball") -> CourtSideBuildRequest:
        assert config.request is not None
        raw = config.request.read_bytes()
        if sha256(raw).hexdigest() != config.request_sha256:
            raise ValueError("Shared court side build request changed")
        request = json.loads(raw)
        if (not isinstance(request, dict) or set(request) != {"schema", "clips", "evidence_clip_ids"}
                or request["schema"] != "video_court_side_inputs_v1" or not isinstance(request["clips"], dict)):
            raise ValueError("Invalid shared court side build request")
        stores = {}
        for clip, record in request["clips"].items():
            if (not isinstance(record, dict) or set(record) != {"role", "directory", "inputs_digest"}
                    or record["role"] not in {"data", "artifact"} or not isinstance(record["directory"], str)):
                raise ValueError("Court side build requires pinned data/artifact component stores")
            directory = resolver.resolve(PathRole(record["role"]), record["directory"])
            stores[clip] = (directory, validate_digest(record["inputs_digest"]))
        return cls(stores, selected_evidence_clips(stores, request["evidence_clip_ids"]), evidence)

    @property
    def input_revisions(self) -> dict[str, str]:
        return {clip: digest for clip, (_, digest) in self.stores.items()}

    def pin_inputs(self) -> dict[str, PinnedSideInputs]:
        pins = {}
        for clip, (directory, digest) in self.stores.items():
            pin = PinnedSideInputs.from_store(directory, evidence=self.evidence)
            if pin.source.clip_id != clip or pin.digest != digest:
                raise ValueError(f"Pinned court side inputs changed: {clip}")
            pins[clip] = pin
        return pins
