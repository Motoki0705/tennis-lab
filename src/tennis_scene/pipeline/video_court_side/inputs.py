"""Freeze the relevant component revisions of a clip without pinning downstream publications."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.ball_points import (
    BALL_POINTS_PORT,
    BallPointsOutput,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideInput
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    CourtSideInputAssembler,
)
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)


@dataclass(frozen=True)
class PinnedSideInputs:
    directory: Path
    source: ClipSource
    source_digest: str
    references: dict[str, Any]

    @property
    def digest(self) -> str:
        digest: str = document_digest({"source_sha256": self.source_digest, "artifacts": self.references})
        return digest

    @classmethod
    def from_store(cls, directory: Path) -> PinnedSideInputs:
        index = json.loads((directory / "scene.json").read_text())
        if index.get("schema") != "tennis_scene_index_v1" or document_digest(index["source"]) != index["source_sha256"]:
            raise ValueError("Invalid court side input store/source")
        source: ClipSource = restore_type(index["source"], ClipSource)
        nodes = ["court_calibration", *(f"ball_points/{c}" for c in source.camera_ids)]
        refs = {node: index["artifacts"][node] for node in nodes}
        assert_current_component_lineage(index, directory, refs)
        return cls(directory, source, index["source_sha256"], refs)

    def assert_current(self) -> None:
        if self.from_store(self.directory).digest != self.digest:
            raise ValueError(f"Court side inputs changed: {self.source.clip_id}")

    def assemble(self) -> CourtSideInput:
        artifacts = {}
        for node, record in self.references.items():
            ref = ArtifactRef(**record)
            descriptor = read_component_descriptor(self.directory, record, node=node, source_sha256=self.source_digest)
            is_calibration = node == "court_calibration"
            port = ("local_court_calibration", 1) if is_calibration else (BALL_POINTS_PORT.schema, BALL_POINTS_PORT.version)
            if (ref.schema, ref.version) != port:
                raise ValueError(f"Invalid court side input schema: {node}")
            codec: ArtifactCodec[Any] = ArtifactCodec(CourtCalibrationOutput if is_calibration else BallPointsOutput)
            key = "calibration" if is_calibration else f"ball_{node.split('/')[1]}"
            artifacts[key] = codec.load(descriptor["payload"], (self.directory / ref.path).parent, descriptor["arrays"])
        inputs: CourtSideInput = CourtSideInputAssembler(0.0).assemble(AssemblyContext(self.source), artifacts)
        return inputs
