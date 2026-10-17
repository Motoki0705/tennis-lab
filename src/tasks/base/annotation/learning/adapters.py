"""Task-owned learning entry points; base never imports concrete task modules."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from ..plan import LearningPlan
from .contracts import (
    DatasetPartition,
    EvaluationReceipt,
    LearningIntent,
    SelectionVerdict,
    TrainingReceipt,
)


@runtime_checkable
class EvaluationResolver(Protocol):
    def resolve_evaluation(self, layout: AnnotationLayout, plan: LearningPlan,
                           publication: ArtifactRef) -> ArtifactRef:
        """Freeze holdouts idempotently for the pinned publication and plan recipe.

        Recovery before intent persistence may call this again; return the same
        reference and verify existing artifacts rather than replacing them.
        """
        ...


@dataclass(frozen=True)
class NoAdditionalTeachers:
    """Valid publication with no new teachers usable by this learning recipe."""

    reason: str


class LearningAdapter(Protocol):
    task: str
    product: str

    def training_partition(
        self, layout: AnnotationLayout, snapshot: ArtifactRef
    ) -> DatasetPartition: ...

    def validate_partition(
        self, layout: AnnotationLayout, partition: DatasetPartition
    ) -> None:
        """Verify exact manifest membership and nested identities, using metadata."""
        ...

    def validate_intent(
        self, layout: AnnotationLayout, intent: LearningIntent
    ) -> NoAdditionalTeachers | None:
        """Check inputs; explicitly report publications with no usable new teachers."""
        ...

    def input_references(
        self, layout: AnnotationLayout, intent: LearningIntent
    ) -> tuple[ArtifactRef, ...]:
        """Enumerate task-owned dataset files and prepared inputs after validation."""
        ...

    def result_references(
        self,
        layout: AnnotationLayout,
        result: TrainingReceipt | EvaluationReceipt,
    ) -> tuple[ArtifactRef, ...]:
        """Enumerate task-owned checkpoint and evaluation evidence after validation."""
        ...

    def command(
        self, layout: AnnotationLayout, request: ArtifactRef, result: Path
    ) -> tuple[str, ...]:
        """Return the task CLI command; it must verify the stage-owned input digest."""
        ...

    def validate_training(
        self, layout: AnnotationLayout, intent: LearningIntent, receipt: TrainingReceipt
    ) -> None:
        """Verify full checkpoint/export correspondence and run provenance."""
        ...

    def validate_evaluation(
        self,
        layout: AnnotationLayout,
        intent: LearningIntent,
        receipt: EvaluationReceipt,
    ) -> None:
        """Require all configured validation metrics; never substitute test results."""
        ...

    def resume_references(
        self, layout: AnnotationLayout, intent: LearningIntent, checkpoint: ArtifactRef
    ) -> tuple[ArtifactRef, ...]:
        """Validate this intent's resume checkpoint and enumerate its dependencies."""
        ...

    def select(
        self, layout: AnnotationLayout, intent: LearningIntent, evaluation: ArtifactRef
    ) -> SelectionVerdict:
        """Return a repeatable judgment from pinned evidence, without running inference.

        The registry recomputes it on adoption and replay to verify the persisted
        verdict; base never interprets the task's labels or numerical metrics.
        """
        ...

    def retained_references(
        self,
        layout: AnnotationLayout,
        intent: LearningIntent,
        evaluation: EvaluationReceipt,
    ) -> tuple[ArtifactRef, ...]:
        """Enumerate task-owned fixed inputs and completed-stage dependencies."""
        ...
