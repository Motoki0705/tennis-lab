"""Immutable identities; task adapters own the meaning of labels and metrics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Annotated, Literal, Protocol, Self, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")]
PositiveInt = Annotated[int, Field(gt=0)]
_Model = TypeVar("_Model")


class _AfterValidator(Protocol):
    def __call__(
        self, function: Callable[[_Model], _Model]
    ) -> Callable[[_Model], _Model]: ...


class _ManifestSerializer(Protocol):
    def __call__(
        self,
        function: Callable[[_Model, SerializerFunctionWrapHandler], dict[str, object]],
    ) -> Callable[[_Model, SerializerFunctionWrapHandler], dict[str, object]]: ...


after_validator = cast(_AfterValidator, model_validator(mode="after"))
manifest_serializer = cast(_ManifestSerializer, model_serializer(mode="wrap"))


def canonical_bytes(value: object) -> bytes:
    """Encode identity inputs without whitespace, NaN, or key-order variation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or str(path) != value
        or path.is_absolute()
        or any(part in {".", ".."} for part in path.parts)
        or any(c in value for c in ("\\", "\x00", "\n", "\r"))
        or value == "."
    ):
        raise ValueError("artifact path must be a normalized relative POSIX child path")
    return value


class Document(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, frozen=True
    )

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.model_dump(mode="json"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class ArtifactRef(Document):
    """Main-repository role-relative input, pinned to exact bytes."""

    role: Literal["annotation", "data", "checkpoint", "artifact"]
    path: str
    digest: Digest
    size_bytes: int = Field(ge=0)

    @after_validator
    def valid_path(self) -> Self:
        relative_path(self.path)
        return self


class FrameRange(Document):
    """Half-open interval in presentation-frame order."""

    start: int = Field(ge=0)
    stop: PositiveInt

    @after_validator
    def ordered(self) -> Self:
        if self.start >= self.stop:
            raise ValueError("frame range must be nonempty and increasing")
        return self


class SourceIdentity(Document):
    provider: Identifier
    channel_id: Identifier
    video_id: Identifier
    media: ArtifactRef
    origin_url: str = Field(min_length=1)

    @property
    def canonical_id(self) -> str:
        # Re-encoding cannot make the same source into a new video.
        return content_digest([self.provider, self.video_id])


class FrameMap(Document):
    frame_index: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    source_pts: int
    clip_pts: int = Field(ge=0)
    duration_pts: PositiveInt


class MediaFingerprint(Document):
    """Historical bytes, deliberately not a dependency on an existing file."""

    digest: Digest
    size_bytes: PositiveInt


class SourceProvenance(Document):
    provider: Identifier
    channel_id: Identifier
    video_id: Identifier
    media: MediaFingerprint
    origin_url: str = Field(min_length=1)

    @property
    def canonical_id(self) -> str:
        return content_digest([self.provider, self.video_id])

    @classmethod
    def from_source(cls, source: SourceIdentity) -> SourceProvenance:
        return cls(
            provider=source.provider,
            channel_id=source.channel_id,
            video_id=source.video_id,
            origin_url=source.origin_url,
            media=MediaFingerprint(
                digest=source.media.digest, size_bytes=source.media.size_bytes
            ),
        )


class JpegFrame(Document):
    offset: int = Field(ge=0)
    length: PositiveInt
    digest: Digest


class JpegStorage(Document):
    encoding: Literal["opencv_bgr_q90"] = "opencv_bgr_q90"
    frames: tuple[JpegFrame, ...] = Field(min_length=1)


class ClipManifest(Document):
    schema_version: Literal["annotation.clip.v1", "annotation.clip.v2"] = (
        "annotation.clip.v1"
    )
    source: SourceIdentity | SourceProvenance
    media: ArtifactRef
    jpeg: JpegStorage | None = None
    preparation_digest: Digest
    width: PositiveInt
    height: PositiveInt
    time_base: str
    nominal_fps: str
    source_range: FrameRange
    target_range: FrameRange
    frames: tuple[FrameMap, ...] = Field(min_length=1)

    @manifest_serializer
    def serialize_manifest(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        result: dict[str, object] = handler(self)
        if self.schema_version == "annotation.clip.v1":
            # Preserve the identity of existing immutable prediction/review inputs.
            result.pop("jpeg", None)
        return result

    @after_validator
    def consistent(self) -> Self:
        if self.schema_version == "annotation.clip.v1":
            if self.jpeg is not None or not isinstance(self.source, SourceIdentity):
                raise ValueError("v1 clips require video and a live source identity")
        else:
            if self.jpeg is None or not isinstance(self.source, SourceProvenance):
                raise ValueError(
                    "v2 clips require JPEG storage and historical source provenance"
                )
            if len(self.jpeg.frames) != len(self.frames):
                raise ValueError("JPEG/frame mapping count mismatch")
            offset = 0
            for jpeg in self.jpeg.frames:
                if jpeg.offset != offset:
                    raise ValueError(
                        "JPEG ranges must be contiguous in presentation order"
                    )
                offset += jpeg.length
            if offset != self.media.size_bytes:
                raise ValueError("JPEG ranges must cover the complete media archive")
        try:
            time_base, fps = Fraction(self.time_base), Fraction(self.nominal_fps)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(
                "time base and nominal fps must be rational numbers"
            ) from error
        if time_base <= 0 or fps <= 0:
            raise ValueError("time base and nominal fps must be positive")
        if self.source_range.stop - self.source_range.start != len(self.frames):
            raise ValueError("source range must cover all presentation frames")
        if self.target_range.stop > len(self.frames):
            raise ValueError("target range must lie inside the clip")
        first_pts = self.frames[0].source_pts
        previous: FrameMap | None = None
        for index, frame in enumerate(self.frames):
            if frame.frame_index != index:
                raise ValueError("clip frames must be dense from zero")
            if frame.source_frame_index != self.source_range.start + index:
                raise ValueError("source frames must match the declared range")
            if frame.clip_pts != frame.source_pts - first_pts:
                raise ValueError("clip/source PTS offset mismatch")
            if previous and frame.clip_pts != previous.clip_pts + previous.duration_pts:
                raise ValueError(
                    "presentation frames must have contiguous increasing PTS"
                )
            previous = frame
        return self

    @property
    def clip_id(self) -> str:
        """Canonical acquisition identity, independent of encoding bytes and model."""
        return content_digest(
            [
                self.source.canonical_id,
                self.source_range.model_dump(),
            ]
        )


class ModelRef(Document):
    task: Identifier
    version: Identifier
    training_weights: ArtifactRef
    inference_weights: ArtifactRef
    configuration: ArtifactRef


class PredictionRef(Document):
    task: Identifier
    clip_id: Digest
    manifest_digest: Digest
    model_digest: Digest
    inference_digest: Digest
    artifact: ArtifactRef

    @property
    def request_id(self) -> str:
        return content_digest(
            [
                self.task,
                self.clip_id,
                self.manifest_digest,
                self.model_digest,
                self.inference_digest,
            ]
        )


class ReviewEvidence(Document):
    """Presentation is not visual review; adoption is recorded separately."""

    clip_id: Digest
    prediction: ArtifactRef
    revision: ArtifactRef
    presented_ranges: tuple[FrameRange, ...]
    reviewed_ranges: tuple[FrameRange, ...]
    unresolved_ranges: tuple[FrameRange, ...]
    evidence: tuple[ArtifactRef, ...]

    @after_validator
    def ordered_ranges(self) -> Self:
        for ranges in (
            self.presented_ranges,
            self.reviewed_ranges,
            self.unresolved_ranges,
        ):
            if any(a.stop > b.start for a, b in zip(ranges, ranges[1:], strict=False)):
                raise ValueError("coverage ranges must be sorted and disjoint")
        # Task validation checks frame bounds and image evidence against the manifest.
        return self


class AdoptionReceipt(Document):
    task: Identifier
    clip_id: Digest
    revision: ArtifactRef
    review_evidence: ArtifactRef
    policy: ArtifactRef
    teacher_mask: ArtifactRef
    reason: str = Field(min_length=1)


class PublicationReceipt(Document):
    task: Identifier
    product: Identifier
    request_id: Digest
    base_snapshot: ArtifactRef | None
    snapshot: ArtifactRef
    adoptions: tuple[ArtifactRef, ...]
    added_clips: tuple[Digest, ...]
    reused_clips: tuple[Digest, ...]

    @after_validator
    def unique_clips(self) -> Self:
        combined = self.added_clips + self.reused_clips
        if len(combined) != len(set(combined)):
            raise ValueError("published clip identities must be unique")
        if self.added_clips and not self.adoptions:
            raise ValueError("new teachers require adoption receipts")
        return self


class ProductOutcome(Document):
    """A publication and a reasoned exclusion are different terminal outcomes."""

    product: Identifier
    status: Literal["pending", "blocked", "failed", "published", "not_applicable"]
    publication: ArtifactRef | None
    reason: str | None

    @after_validator
    def terminal_evidence(self) -> Self:
        if (self.status == "published") != (self.publication is not None):
            raise ValueError("only published products carry a publication receipt")
        if self.status in {"blocked", "failed", "not_applicable"} and not (
            self.reason and self.reason.strip()
        ):
            raise ValueError("blocked, failed and excluded products require a reason")
        return self

    @property
    def annotation_complete(self) -> bool:
        return self.status in {"published", "not_applicable"}
