"""One immutable execution policy shared by all three concurrency conditions."""

from __future__ import annotations

from pydantic import Field

from src.tasks.base.annotation.contracts import ArtifactRef, Document

from ..contracts import RuntimeSettings
from ..reviews import ClipReview


class BenchmarkRuntime(Document):
    comparison: ArtifactRef
    settings: RuntimeSettings
    observation_interval_seconds: float = Field(gt=0)


class ConditionCompletion(Document):
    condition: ArtifactRef
    runtime: ArtifactRef
    reviews: tuple[ClipReview, ...] = Field(min_length=1)
    final_observation: ArtifactRef
