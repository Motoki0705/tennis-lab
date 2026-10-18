"""Task-owned dependencies for learning inputs, results, and model decisions."""

from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Document,
    PublicationReceipt,
)
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
from src.tasks.player_detection.data.snapshots.contracts import (
    ExistingPlayerSnapshot,
    PlayerPartition,
)

from .contracts import (
    PlayerComparison,
    PlayerEvaluationMetrics,
    PlayerLearningRecipe,
    PlayerTrainingProvenance,
    RunBinding,
    TrainingComplete,
)
from .inputs import resolve_inputs


def result_references(
    layout: AnnotationLayout, result: TrainingReceipt | EvaluationReceipt
) -> tuple[ArtifactRef, ...]:
    docs: tuple[Document, ...]
    if isinstance(result, TrainingReceipt):
        provenance = read_document(layout, result.provenance, PlayerTrainingProvenance)
        binding = read_document(layout, provenance.run_binding, RunBinding)
        completed = read_document(layout, provenance.completed, TrainingComplete)
        docs = (result, provenance, binding, completed)
    else:
        metrics = read_document(layout, result.metrics, PlayerEvaluationMetrics)
        comparison = read_document(layout, metrics.comparison, PlayerComparison)
        docs = (result, metrics, comparison)
    retained: tuple[ArtifactRef, ...] = distinct(
        layout, (ref for doc in docs for ref in references(doc))
    )
    return retained


def input_references(
    layout: AnnotationLayout, intent: LearningIntent
) -> tuple[ArtifactRef, ...]:
    inputs = resolve_inputs(layout, intent)
    publication = read_document(layout, intent.publication, PublicationReceipt)
    refs = [
        inputs.train.manifest,
        *references(inputs.recipe),
        *references(inputs.train_support),
    ]
    snapshots = {intent.train.manifest, inputs.train_support.snapshot}
    if publication.base_snapshot is not None:
        snapshots.add(publication.base_snapshot)
    for partition in (
        inputs.fixed.validation,
        *((inputs.fixed.test,) if inputs.fixed.test is not None else ()),
    ):
        support = read_document(layout, partition.manifest, PlayerPartition)
        refs.extend((partition.manifest, *references(support)))
        snapshots.add(support.snapshot)
    for reference in snapshots:
        snapshot = read_document(layout, reference, ExistingPlayerSnapshot)
        refs.extend((reference, *references(snapshot)))
    retained: tuple[ArtifactRef, ...] = distinct(layout, refs)
    return retained


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
