"""Task-owned dependencies retained atomically with every ball model decision."""

from src.tasks.ball_detection.data.windows.learning.contracts import (
    HoldoutWindows,
    TrainingWindows,
)
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

from .contracts import BallEvaluationMetrics, BallLearningRecipe, BallTrainingProvenance


def retained_references(
    layout: AnnotationLayout, intent: LearningIntent, evaluation: EvaluationReceipt
) -> tuple[ArtifactRef, ...]:
    fixed = read_document(layout, intent.fixed_evaluation, FixedEvaluation)
    trained = read_document(layout, evaluation.training, CompletedStage)
    training = read_document(layout, trained.result, TrainingReceipt)
    provenance = read_document(layout, training.provenance, BallTrainingProvenance)
    documents = (
        read_document(layout, intent.recipe, BallLearningRecipe),
        provenance,
        read_document(layout, provenance.prepared, TrainingWindows),
        read_document(layout, evaluation.metrics, BallEvaluationMetrics),
        *(
            read_document(layout, partition.manifest, HoldoutWindows)
            for partition in (fixed.validation, fixed.test)
            if partition is not None
        ),
    )
    result: tuple[ArtifactRef, ...] = distinct(
        layout, (ref for doc in documents for ref in references(doc))
    )
    return result
