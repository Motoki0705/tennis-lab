"""Structural evidence checks, independent of task labels and metric definitions."""

from __future__ import annotations

from ..artifacts import read_document, verify_reference
from ..contracts import ArtifactRef, Document, PublicationReceipt
from ..execution.contracts import StageExit, StageLaunch
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, load_plan, references
from .contracts import (
    CampaignCompletion,
    CompletedStage,
    DatasetPartition,
    EvaluationReceipt,
    FixedEvaluation,
    LearningIntent,
    StageInput,
    TrainingReceipt,
)


def verify_inputs(layout: AnnotationLayout, document: Document) -> None:
    for reference in set(references(document)):
        verify_reference(layout, reference)


def check_completion(
    layout: AnnotationLayout, completion: CampaignCompletion
) -> CampaignPlan:
    plan = load_plan(verify_reference(layout, completion.plan))
    if plan.annotation_root != str(layout.root) or not plan.allows_adoption:
        raise ValueError("learning requires a production plan at the fixed root")
    actual = [
        (task.task, p.product) for task in completion.tasks for p in task.products
    ]
    expected = [(task.task, p.product) for task in plan.tasks for p in task.products]
    if sorted(actual) != sorted(expected) or len(
        {t.task for t in completion.tasks}
    ) != len(completion.tasks):
        raise ValueError(
            "completion must cover every planned task/product exactly once"
        )
    for task in completion.tasks:
        planned = next(t for t in plan.tasks if t.task == task.task)
        for outcome in task.products:
            if outcome.publication is None:
                continue
            product = next(p for p in planned.products if p.product == outcome.product)
            receipt = read_document(layout, outcome.publication, PublicationReceipt)
            if (receipt.task, receipt.product, receipt.base_snapshot) != (
                task.task,
                outcome.product,
                product.base_snapshot,
            ):
                raise ValueError(
                    "publication differs from the planned task/product/base"
                )
            verify_inputs(layout, receipt)
    return plan


def check_partitions(*partitions: DatasetPartition) -> None:
    sources: set[str] = set()
    clips: set[str] = set()
    for partition in partitions:
        if sources.intersection(partition.sources) or clips.intersection(
            partition.clips
        ):
            raise ValueError("train/validation/test source or clip identities overlap")
        sources.update(partition.sources)
        clips.update(partition.clips)


def completed_request(
    layout: AnnotationLayout, completed: CompletedStage
) -> StageInput:
    request: StageInput = read_document(layout, completed.request, StageInput)
    execution = read_document(layout, completed.execution, StageExit)
    launch = read_document(layout, execution.launch, StageLaunch)
    if execution.status != "succeeded" or execution.exit_code != 0:
        raise ValueError("learning output has no successful execution receipt")
    if launch.input != completed.request:
        raise ValueError("learning output was produced by a different stage")
    intent = read_document(layout, request.intent, LearningIntent)
    completion = read_document(layout, intent.completion, CampaignCompletion)
    if launch.resource.plan != completion.plan:
        raise ValueError("learning execution uses a different campaign plan")
    verify_inputs(layout, completed)
    verify_inputs(layout, execution)
    return request


def check_training(
    layout: AnnotationLayout, completed_ref: ArtifactRef
) -> tuple[StageInput, TrainingReceipt]:
    completed = read_document(layout, completed_ref, CompletedStage)
    return check_training_value(layout, completed)


def check_training_value(
    layout: AnnotationLayout, completed: CompletedStage
) -> tuple[StageInput, TrainingReceipt]:
    request = completed_request(layout, completed)
    result = read_document(layout, completed.result, TrainingReceipt)
    intent = read_document(layout, request.intent, LearningIntent)
    if request.stage != "train" or result.request != completed.request:
        raise ValueError("training result/request mismatch")
    if (
        result.candidate.task != intent.task
        or result.candidate.version == intent.parent.model.version
    ):
        raise ValueError("training must produce a new version of the intended task")
    verify_inputs(layout, result)
    return request, result


def check_evaluation(
    layout: AnnotationLayout, completed_ref: ArtifactRef
) -> tuple[LearningIntent, EvaluationReceipt]:
    completed = read_document(layout, completed_ref, CompletedStage)
    return check_evaluation_value(layout, completed)


def check_evaluation_value(
    layout: AnnotationLayout, completed: CompletedStage
) -> tuple[LearningIntent, EvaluationReceipt]:
    request = completed_request(layout, completed)
    result = read_document(layout, completed.result, EvaluationReceipt)
    intent = read_document(layout, request.intent, LearningIntent)
    fixed = read_document(layout, intent.fixed_evaluation, FixedEvaluation)
    completion = read_document(layout, intent.completion, CampaignCompletion)
    plan = check_completion(layout, completion)
    task = next(t for t in plan.tasks if t.task == intent.task)
    if not completion.complete or task.learning is None:
        raise ValueError("annotation or learning plan is incomplete")
    if (
        task.model,
        task.learning.recipe,
        task.learning.validation,
        task.learning.selection_policy,
    ) != (
        intent.parent.model,
        intent.recipe,
        intent.fixed_evaluation,
        intent.selection_policy,
    ):
        raise ValueError("learning intent differs from its frozen plan")
    if (
        request.stage != "evaluate"
        or result.request != completed.request
        or request.training != result.training
    ):
        raise ValueError("evaluation result/request mismatch")
    train_request, training = check_training(layout, result.training)
    if train_request.intent != request.intent or result.candidate != training.candidate:
        raise ValueError("evaluation must use this intent's completed candidate")
    if (
        result.baseline != intent.parent
        or result.validation != fixed.validation
        or fixed.task != intent.task
    ):
        raise ValueError("evaluation baseline or fixed validation changed")
    check_partitions(
        intent.train, fixed.validation, *((fixed.test,) if fixed.test else ())
    )
    verify_inputs(layout, result)
    return intent, result
