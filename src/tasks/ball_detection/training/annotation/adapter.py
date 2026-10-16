"""Concrete ball implementation of the shared LearningAdapter protocol."""

import sys
from pathlib import Path

from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.data.windows.learning.evaluation import resolve_evaluation
from src.tasks.ball_detection.data.windows.learning.holdout import validate_holdout
from src.tasks.ball_detection.data.windows.learning.preparation import (
    require_additional_teachers,
    training_partition,
)
from src.tasks.ball_detection.training.replay import ReplayFPSMixSampler
from src.tasks.base.annotation.artifacts import verify_reference
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
from src.tasks.base.annotation.plan import LearningPlan

from . import evidence, storage
from .inputs import resolve_inputs
from .selection import read_policy, rejection_reasons


class BallLearningAdapter:
    task = "ball"
    product = "frames"

    def resolve_evaluation(self, layout: AnnotationLayout, plan: LearningPlan,
                           publication: ArtifactRef) -> ArtifactRef:
        return resolve_evaluation(layout, plan, publication)

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
        if inputs.recipe.replay is not None:
            data = FrozenBallWindowDataset(verify_reference(layout, require_additional_teachers(inputs.prepared)),
                                            split="train", requires_pose=False)
            ReplayFPSMixSampler(data, windows_per_epoch=inputs.recipe.windows_per_epoch, seed=inputs.recipe.seed,
                previously_trained_manifest=verify_reference(layout, inputs.recipe.replay.previously_trained_manifest),
                untrained_fraction=inputs.recipe.replay.untrained_fraction)
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
        policy = read_policy(layout, intent.selection_policy)
        reasons = rejection_reasons(comparison, policy)
        return SelectionVerdict(
            evaluation=evaluation,
            policy=intent.selection_policy,
            status="rejected" if reasons else "passed",
            reason="; ".join(reasons)
            if reasons
            else "All agreed model-selection conditions passed",
        )
