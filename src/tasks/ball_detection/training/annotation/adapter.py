"""Concrete ball implementation of the shared LearningAdapter protocol."""

import sys
from pathlib import Path

from src.tasks.ball_detection.data.windows.learning.holdout import validate_holdout
from src.tasks.ball_detection.data.windows.learning.preparation import (
    training_partition,
)
from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.learning.adapters import NoAdditionalTeachers
from src.tasks.base.annotation.learning.contracts import (
    DatasetPartition,
    EvaluationReceipt,
    LearningIntent,
    SelectionVerdict,
    TrainingReceipt,
)
from src.tasks.base.annotation.learning.validation import check_evaluation
from src.tasks.base.annotation.paths import AnnotationLayout

from . import evidence, storage
from .contracts import BallSelectionPolicy
from .inputs import resolve_inputs
from .selection import rejection_reasons


class BallLearningAdapter:
    task = "ball"
    product = "frames"

    def retained_references(
        self,
        layout: AnnotationLayout,
        intent: LearningIntent,
        evaluation: EvaluationReceipt,
    ) -> tuple[ArtifactRef, ...]:
        return storage.retained_references(layout, intent, evaluation)

    def training_partition(
        self, layout: AnnotationLayout, snapshot: ArtifactRef
    ) -> DatasetPartition:
        result: DatasetPartition = training_partition(layout, snapshot)
        return result

    def validate_partition(
        self, layout: AnnotationLayout, partition: DatasetPartition
    ) -> None:
        if partition.split == "train":
            if training_partition(layout, partition.manifest) != partition:
                raise ValueError("ball training membership differs from its snapshot")
        else:
            validate_holdout(layout, partition)

    def validate_intent(
        self, layout: AnnotationLayout, intent: LearningIntent
    ) -> NoAdditionalTeachers | None:
        inputs = resolve_inputs(layout, intent, require_teachers=False)
        if inputs.prepared.status == "no_additional_coordinate_teachers":
            return NoAdditionalTeachers(
                "new clips contain no eligible MDD coordinate window; publication retained, old-only training is forbidden"
            )
        inputs.expected_recipe(layout, intent)
        return None

    def input_references(
        self, layout: AnnotationLayout, intent: LearningIntent
    ) -> tuple[ArtifactRef, ...]:
        dependencies: tuple[ArtifactRef, ...] = storage.input_references(layout, intent)
        return dependencies

    def result_references(
        self, layout: AnnotationLayout, result: TrainingReceipt | EvaluationReceipt
    ) -> tuple[ArtifactRef, ...]:
        dependencies: tuple[ArtifactRef, ...] = storage.result_references(layout, result)
        return dependencies

    def command(
        self, layout: AnnotationLayout, request: ArtifactRef, result: Path
    ) -> tuple[str, ...]:
        return (
            sys.executable,
            "-m",
            "src.tasks.ball_detection.training.annotation",
            "--input",
            str(verify_reference(layout, request)),
            "--input-sha256",
            request.digest,
            "--output",
            str(result),
        )

    def validate_training(
        self, layout: AnnotationLayout, intent: LearningIntent, receipt: TrainingReceipt
    ) -> None:
        evidence.validate_training(layout, intent, receipt)

    def validate_evaluation(
        self,
        layout: AnnotationLayout,
        intent: LearningIntent,
        receipt: EvaluationReceipt,
    ) -> None:
        evidence.validate_evaluation(layout, intent, receipt)

    def resume_references(
        self, layout: AnnotationLayout, intent: LearningIntent, checkpoint: ArtifactRef
    ) -> tuple[ArtifactRef, ...]:
        evidence.validate_resume(layout, intent, checkpoint)
        return ()

    def select(
        self, layout: AnnotationLayout, intent: LearningIntent, evaluation: ArtifactRef
    ) -> SelectionVerdict:
        actual, receipt = check_evaluation(layout, evaluation)
        if actual != intent:
            raise ValueError("ball selection belongs to a different intent")
        if intent.selection_policy is None:
            raise ValueError("ball selection requires an explicitly agreed policy")
        comparison = evidence.validate_evaluation(layout, intent, receipt)
        policy = read_document(layout, intent.selection_policy, BallSelectionPolicy)
        reasons = rejection_reasons(comparison, policy)
        return SelectionVerdict(
            evaluation=evaluation,
            policy=intent.selection_policy,
            status="rejected" if reasons else "passed",
            reason="; ".join(reasons)
            if reasons
            else "All agreed fixed-validation conditions passed",
        )
