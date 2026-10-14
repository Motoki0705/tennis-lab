"""MDD coordinate proposals do not assert confidence, visibility, or ball presence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from src.tasks.base.annotation.contracts import Digest, Document


class MDDCheckpointSpec(Document):
    schema_version: Literal["ball.annotation-checkpoint.v1"] = (
        "ball.annotation-checkpoint.v1"
    )
    checkpoint_schema: Literal["mdd_query_posttraining.v1", "mdd_coordinates.v4"]
    architecture: dict[str, Any]
    input_contract: dict[str, Any]
    training_manifest_digest: Digest


class MDDInferenceConfig(Document):
    schema_version: Literal["ball.annotation-inference.v1"] = (
        "ball.annotation-inference.v1"
    )
    window_frames: Literal[32] = 32
    stride: int = Field(ge=1, le=32)
    overlap_policy: Literal["nearest_window_center_then_earliest_start"] = (
        "nearest_window_center_then_earliest_start"
    )
    coordinates: Literal["native_pixel_endpoint"] = "native_pixel_endpoint"
    decode: Literal["pyav_rgb24_native"] = "pyav_rgb24_native"
    device: Literal["cuda", "cpu"]
    precision: Literal["float32", "bfloat16"]


class BallProposal(Document):
    frame_index: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    source_pts: int
    clip_pts: int
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    window_start: int = Field(ge=0)


class BallPredictions(Document):
    schema_version: Literal["ball.annotation-prediction.v1"] = (
        "ball.annotation-prediction.v1"
    )
    context_digest: Digest
    output_kind: Literal["coordinates_only_unreviewed"] = "coordinates_only_unreviewed"
    rows: tuple[BallProposal, ...] = Field(min_length=32)
