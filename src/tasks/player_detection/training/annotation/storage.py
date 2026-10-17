"""Player implementation of the required model-decision reference contract."""

from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.learning.contracts import (
    CompletedStage,
    EvaluationReceipt,
    FixedEvaluation,
    LearningIntent,
    TrainingReceipt,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import references
from src.tasks.base.annotation.storage.references import distinct
from src.tasks.player_detection.data.snapshots.contracts import PlayerPartition

from .contracts import (
    PlayerComparison,
    PlayerEvaluationMetrics,
    PlayerLearningRecipe,
    PlayerTrainingProvenance,
    RunBinding,
    TrainingComplete,
)


def retained_references(
    layout: AnnotationLayout, intent: LearningIntent, evaluation: EvaluationReceipt
) -> tuple[ArtifactRef, ...]:
    fixed = read_document(layout, intent.fixed_evaluation, FixedEvaluation)
    trained = read_document(layout, evaluation.training, CompletedStage)
    training = read_document(layout, trained.result, TrainingReceipt)
    provenance = read_document(layout, training.provenance, PlayerTrainingProvenance)
    binding = read_document(layout, provenance.run_binding, RunBinding)
    metrics = read_document(layout, evaluation.metrics, PlayerEvaluationMetrics)
    documents = (
        read_document(layout, intent.recipe, PlayerLearningRecipe),
        provenance,
        binding,
        read_document(layout, provenance.completed, TrainingComplete),
        metrics,
        read_document(layout, metrics.comparison, PlayerComparison),
        read_document(layout, binding.train_partition, PlayerPartition),
        *(
            read_document(layout, partition.manifest, PlayerPartition)
            for partition in (fixed.validation, fixed.test)
            if partition is not None
        ),
    )
    result: tuple[ArtifactRef, ...] = distinct(
        layout, (ref for doc in documents for ref in references(doc))
    )
    return result
