"""Declared component contracts, independent of execution and persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
InputContra = TypeVar("InputContra", contravariant=True)
OutputCo = TypeVar("OutputCo", covariant=True)

STANDARD_COMPONENTS = (
    "court_detection", "court_calibration", "person_detection", "person_tracking", "pose_estimation",
    "ball_detection", "person_reid", "court_side", "camera_alignment", "player_triangulation",
    "ball_triangulation", "ball_smoothing", "body_view_selection", "gvhmr", "body_placement", "scene_assembly",
)


@dataclass(frozen=True)
class SourceVideo:
    camera_id: str
    path: Path
    sha256: str
    num_frames: int
    fps: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.camera_id or self.num_frames < 1 or self.fps <= 0 or min(self.width, self.height) < 1:
            raise ValueError("Invalid source video timeline")


@dataclass(frozen=True)
class ClipSource:
    clip_id: str
    videos: tuple[SourceVideo, ...]

    def __post_init__(self) -> None:
        if not self.videos or len(set(self.camera_ids)) != len(self.videos):
            raise ValueError("Clip requires unique source cameras")
        first = self.videos[0]
        if any((v.num_frames, v.width, v.height) != (first.num_frames, first.width, first.height) or abs(v.fps - first.fps) > 1e-6 for v in self.videos):
            raise ValueError("Clip videos must share frame count, FPS and image size")

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(v.camera_id for v in self.videos)

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(v.path for v in self.videos)

    @property
    def num_frames(self) -> int:
        return self.videos[0].num_frames

    @property
    def size(self) -> tuple[int, int]:
        return self.videos[0].width, self.videos[0].height

    @property
    def fps(self) -> float:
        return self.videos[0].fps

    def video(self, camera_id: str) -> SourceVideo:
        return self.videos[self.camera_ids.index(camera_id)]


@dataclass(frozen=True)
class InputPort:
    """One explicitly bound dependency; names never select an implicit producer."""

    schema: str
    version: int = 1


@dataclass(frozen=True)
class ComponentIO(Generic[InputT, OutputT]):
    name: str
    input_type: type[InputT]
    output_type: type[OutputT]
    inputs: Mapping[str, InputPort]
    output_schema: str
    version: int = 1


class Component(Protocol[InputContra, OutputCo]):
    def process(self, inputs: InputContra) -> OutputCo: ...


@dataclass(frozen=True)
class AssemblyContext:
    source: ClipSource
    camera_id: str | None = None


class InputAssembler(Protocol[OutputCo]):
    """Only declared dependencies are supplied; no unrestricted store access."""

    @property
    def version(self) -> int: ...

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> OutputCo: ...
