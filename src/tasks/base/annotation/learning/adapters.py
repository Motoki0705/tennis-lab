"""Task-owned learning entry points; base never imports concrete task modules."""

from pathlib import Path
from typing import Protocol

from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from .contracts import (
    DatasetPartition,
    EvaluationReceipt,
    LearningIntent,
    SelectionVerdict,
    TrainingReceipt,
)


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

    def validate_intent(self, layout: AnnotationLayout, intent: LearningIntent) -> None:
        """Check recipe, initialization weights and task-owned input contracts."""
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

    def validate_resume(
        self, layout: AnnotationLayout, intent: LearningIntent, checkpoint: ArtifactRef
    ) -> None:
        """Only this intent's training checkpoint can resume its optimizer/schedule."""
        ...

    def select(
        self, layout: AnnotationLayout, intent: LearningIntent, evaluation: ArtifactRef
    ) -> SelectionVerdict: ...
