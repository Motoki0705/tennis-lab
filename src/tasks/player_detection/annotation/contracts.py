"""Native-pixel player proposals remain unreviewed, including empty predictions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from src.tasks.base.annotation.contracts import Digest, Document, after_validator

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class DinoCheckpointSpec(Document):
    schema_version: Literal["player.annotation-checkpoint.v1"] = (
        "player.annotation-checkpoint.v1"
    )
    architecture: Literal["dino_4scale_swin_l_player_class1"] = (
        "dino_4scale_swin_l_player_class1"
    )
    export_format: Literal["tennis_lab_player_dino.v1"] = "tennis_lab_player_dino.v1"
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    epoch: int = Field(ge=0)
    global_step: int = Field(ge=0)


class DinoInferenceConfig(Document):
    schema_version: Literal["player.annotation-inference.v1"] = (
        "player.annotation-inference.v1"
    )
    device: Literal["cuda"] = "cuda"
    confidence_threshold: float = Field(gt=0, lt=1)
    short_side: int = Field(gt=0)
    max_long_side: int = Field(gt=0)
    decode: Literal["pyav_bgr24_native"] = "pyav_bgr24_native"
    coordinates: Literal["native_pixel_xyxy_edges"] = "native_pixel_xyxy_edges"

    @_after
    def size_bounds(self) -> Self:
        if self.max_long_side < self.short_side:
            raise ValueError("DINO max long side must be at least its short side")
        return self


class Box(Document):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(gt=0)
    y2: float = Field(gt=0)

    @_after
    def positive_area(self) -> Self:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("player box must have positive area")
        return self


class PlayerProposal(Document):
    observation_id: Digest
    candidate_index: int = Field(ge=0)
    box: Box
    confidence: float = Field(ge=0, le=1)


class PlayerFrameProposals(Document):
    frame_index: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    source_pts: int
    clip_pts: int
    observations: tuple[PlayerProposal, ...]


class PlayerPredictions(Document):
    schema_version: Literal["player.annotation-prediction.v1"] = (
        "player.annotation-prediction.v1"
    )
    context_digest: Digest
    output_kind: Literal["bbox_proposals_unreviewed"] = "bbox_proposals_unreviewed"
    rows: tuple[PlayerFrameProposals, ...] = Field(min_length=1)
