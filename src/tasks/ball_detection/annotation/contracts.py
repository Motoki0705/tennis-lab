"""Explicit coordinate and hybrid proposals, always separate from reviewed labels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol, Self, TypeVar, cast

from pydantic import Field, TypeAdapter

from src.tasks.base.annotation.artifacts import verify_reference
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    Digest,
    Document,
    after_validator,
)
from src.tasks.base.annotation.paths import AnnotationLayout

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class MDDCheckpointSpec(Document):
    schema_version: Literal["ball.annotation-checkpoint.v1"] = (
        "ball.annotation-checkpoint.v1"
    )
    checkpoint_schema: Literal["mdd_query_posttraining.v1", "mdd_coordinates.v4"]
    architecture: dict[str, Any]
    input_contract: dict[str, Any]
    training_manifest_digest: Digest


class ClipInferenceConfig(Document):
    window_frames: Literal[32] = 32
    stride: int = Field(ge=1, le=32)
    overlap_policy: Literal["nearest_window_center_then_earliest_start"] = (
        "nearest_window_center_then_earliest_start"
    )
    coordinates: Literal["native_pixel_endpoint"] = "native_pixel_endpoint"
    decode: Literal["pyav_rgb24_native", "opencv_jpeg_rgb24_native"] = (
        "pyav_rgb24_native"
    )
    device: Literal["cuda", "cpu"]
    precision: Literal["float32", "bfloat16"]

    def validate_media(self, manifest: ClipManifest) -> None:
        expected = (
            "opencv_jpeg_rgb24_native"
            if manifest.jpeg is not None
            else "pyav_rgb24_native"
        )
        if self.decode != expected:
            raise ValueError(
                "frozen inference decoder differs from the prepared media format"
            )


class MDDInferenceConfig(ClipInferenceConfig):
    schema_version: Literal["ball.annotation-inference.v1"] = (
        "ball.annotation-inference.v1"
    )


class HybridCheckpointSpec(Document):
    schema_version: Literal["ball.hybrid-annotation-checkpoint.v1"] = (
        "ball.hybrid-annotation-checkpoint.v1"
    )
    checkpoint_schema: Literal["mdd_hybrid_training.v1"] = "mdd_hybrid_training.v1"
    architecture: dict[str, Any]
    input_contract: dict[str, Any]
    training_manifest_digest: Digest


class HybridInferenceConfig(ClipInferenceConfig):
    schema_version: Literal["ball.hybrid-annotation-inference.v1"] = (
        "ball.hybrid-annotation-inference.v1"
    )
    presence_threshold: float = Field(gt=0, lt=1)
    subpixel: bool


class ProposalFrame(Document):
    frame_index: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    source_pts: int
    clip_pts: int
    window_start: int = Field(ge=0)


class BallProposal(ProposalFrame):
    x: float = Field(ge=0)
    y: float = Field(ge=0)

    @property
    def center(self) -> tuple[float, float]:
        return self.x, self.y

    def summary(self) -> str:
        return f"prediction=coordinates_only_unreviewed x={self.x:.2f} y={self.y:.2f}"


class HybridProposal(ProposalFrame):
    presence: Literal["present", "absent"]
    presence_logit: float
    presence_score: float = Field(ge=0, le=1)
    x: float | None = Field(ge=0)
    y: float | None = Field(ge=0)
    heatmap_peak: float = Field(ge=0, le=1)

    @_after
    def position(self) -> Self:
        if self.presence == "absent":
            if self.x is not None or self.y is not None:
                raise ValueError("predicted absence must have null coordinates")
        elif self.x is None or self.y is None:
            raise ValueError("predicted presence requires heatmap coordinates")
        return self

    @property
    def center(self) -> tuple[float, float] | None:
        return None if self.x is None or self.y is None else (self.x, self.y)

    def summary(self) -> str:
        return (
            f"prediction={self.presence}_unreviewed "
            f"presence_logit={self.presence_logit:.5g} heatmap_peak={self.heatmap_peak:.5g}"
        )


class BallPredictions(Document):
    schema_version: Literal["ball.annotation-prediction.v1"] = (
        "ball.annotation-prediction.v1"
    )
    context_digest: Digest
    output_kind: Literal["coordinates_only_unreviewed"] = "coordinates_only_unreviewed"
    rows: tuple[BallProposal, ...] = Field(min_length=32)


class HybridPredictions(Document):
    schema_version: Literal["ball.hybrid-annotation-prediction.v1"] = (
        "ball.hybrid-annotation-prediction.v1"
    )
    context_digest: Digest
    output_kind: Literal["heatmap_presence_unreviewed"] = "heatmap_presence_unreviewed"
    heatmap_size: tuple[int, int]  # (height, width), native /4 grid
    rows: tuple[HybridProposal, ...] = Field(min_length=32)


InferenceConfig = MDDInferenceConfig | HybridInferenceConfig
PredictionDocument = BallPredictions | HybridPredictions
_inference = TypeAdapter[InferenceConfig](
    Annotated[InferenceConfig, Field(discriminator="schema_version")]
)
_predictions = TypeAdapter[PredictionDocument](
    Annotated[PredictionDocument, Field(discriminator="schema_version")]
)


def read_inference(layout: AnnotationLayout, reference: ArtifactRef) -> InferenceConfig:
    result: InferenceConfig = _inference.validate_json(verify_reference(layout, reference).read_bytes())
    return result


def read_predictions(layout: AnnotationLayout, reference: ArtifactRef) -> PredictionDocument:
    result: PredictionDocument = _predictions.validate_json(verify_reference(layout, reference).read_bytes())
    return result
