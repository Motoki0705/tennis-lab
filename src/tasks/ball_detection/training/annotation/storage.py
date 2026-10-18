"""Task-owned dependencies for learning inputs, results, and model decisions."""

from src.tasks.ball_detection.data.windows.learning.contracts import (
    HoldoutWindows,
    TrainingWindows,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallSnapshot,
    PublishRequest,
)
from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import ArtifactRef, PublicationReceipt
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
from .inputs import resolve_inputs


def result_references(
    layout: AnnotationLayout, result: TrainingReceipt | EvaluationReceipt
) -> tuple[ArtifactRef, ...]:
    document = (
        read_document(layout, result.provenance, BallTrainingProvenance)
        if isinstance(result, TrainingReceipt)
        else read_document(layout, result.metrics, BallEvaluationMetrics)
    )
    retained: tuple[ArtifactRef, ...] = distinct(
        layout, (*references(result), *references(document))
    )
    return retained


def input_references(
    layout: AnnotationLayout, intent: LearningIntent
) -> tuple[ArtifactRef, ...]:
    inputs = resolve_inputs(layout, intent)
    publication = read_document(layout, intent.publication, PublicationReceipt)
    refs = [
        inputs.prepared_ref,
        *references(inputs.prepared),
        *references(inputs.recipe),
        *references(inputs.fixed),
    ]
    snapshots = {intent.train.manifest, inputs.prepared.snapshot}
    if publication.base_snapshot is not None:
        snapshots.add(publication.base_snapshot)
    partitions = [
        inputs.fixed.validation,
        *((inputs.fixed.test,) if inputs.fixed.test is not None else ()),
    ]
    if inputs.fixed.original is not None:
        original = read_document(layout, inputs.fixed.original, FixedEvaluation)
        refs.extend(references(original))
        partitions.extend((original.validation, *((original.test,) if original.test else ())))
    for partition in partitions:
        holdout = read_document(layout, partition.manifest, HoldoutWindows)
        refs.extend((partition.manifest, *references(holdout)))
        snapshots.add(holdout.snapshot)
    for reference in snapshots:
        snapshot = read_document(layout, reference, BallSnapshot)
        refs.extend((reference, *references(snapshot)))
        if snapshot.request is not None:
            request = read_document(layout, snapshot.request, PublishRequest)
            refs.extend(references(request))
    retained: tuple[ArtifactRef, ...] = distinct(layout, refs)
    return retained


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
