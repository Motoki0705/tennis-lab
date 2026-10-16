"""Player implementation of the shared publication/learning lifecycle protocol."""

import sys
from pathlib import Path

from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.learning.contracts import (
    DatasetPartition,
    EvaluationReceipt,
    LearningIntent,
    SelectionVerdict,
    TrainingReceipt,
)
from src.tasks.base.annotation.learning.validation import check_evaluation
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.player_detection.data.snapshots.partitions import validate_partition

from . import checkpoints, evaluation, storage
from .contracts import PlayerSelectionPolicy
from .inputs import resolve_inputs, run_root, training_partition


class PlayerLearningAdapter:
    task = "player"
    product = "boxes"

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
                raise ValueError("player training membership differs from its snapshot")
        else:
            validate_partition(layout, partition)

    def validate_intent(self, layout: AnnotationLayout, intent: LearningIntent) -> None:
        resolve_inputs(layout, intent).configuration(
            layout, intent, run_root(layout, intent) / "attempt-001", None
        )

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
            "src.tasks.player_detection.training.annotation",
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
        checkpoints.validate_training(layout, intent, receipt)

    def validate_evaluation(
        self,
        layout: AnnotationLayout,
        intent: LearningIntent,
        receipt: EvaluationReceipt,
    ) -> None:
        evaluation.validate_evaluation(layout, intent, receipt)

    def resume_references(
        self, layout: AnnotationLayout, intent: LearningIntent, checkpoint: ArtifactRef
    ) -> tuple[ArtifactRef, ...]:
        binding = checkpoints.validate_resume(layout, intent, checkpoint)
        return (binding.initial_configuration,)

    def select(
        self, layout: AnnotationLayout, intent: LearningIntent, reference: ArtifactRef
    ) -> SelectionVerdict:
        actual, receipt = check_evaluation(layout, reference)
        if actual != intent or intent.selection_policy is None:
            raise ValueError(
                "player model selection requires this intent's completed evaluation and an explicit policy"
            )
        before, after, support = evaluation.validate_evaluation(layout, intent, receipt)
        policy = read_document(layout, intent.selection_policy, PlayerSelectionPolicy)
        reasons = evaluation.rejection_reasons(before, after, support, policy)
        return SelectionVerdict(
            evaluation=reference,
            policy=intent.selection_policy,
            status="rejected" if reasons else "passed",
            reason="; ".join(reasons)
            if reasons
            else "All agreed fixed-validation conditions passed",
        )
