"""Validate frozen controls without acquiring media or invoking a model."""

from __future__ import annotations

from typing import Any

from ..artifacts import read_document, verify_reference
from ..contracts import ArtifactRef, ClipManifest, content_digest
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, load_plan, references
from ..predictions import PredictionReceipt
from ..resources.contracts import BudgetLimits, BudgetScope
from .contracts import ComparisonSpec, ConditionSpec, WorkerBinding


def condition(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[ConditionSpec, CampaignPlan, dict[str, WorkerBinding]]:
    spec = read_document(layout, reference, ConditionSpec)
    plan = load_plan(verify_reference(layout, spec.plan))
    scope = read_document(layout, spec.scope, BudgetScope)
    if plan.annotation_root != str(layout.root):
        raise ValueError("benchmark plan belongs to another annotation root")
    if BudgetLimits.from_budget(plan.resources) != scope.limits:
        raise ValueError("benchmark budget must retain the shared scope limits")
    stages = {stage.stage: stage for stage in plan.resources.stages}
    names = {binding.stage for binding in spec.bindings}
    if any(name not in stages or stages[name].kind != "codex" for name in names):
        raise ValueError("measured stages must be explicitly allocated Codex stages")
    if sum(stages[name].parallel for name in names) != spec.target_workers:
        raise ValueError(
            "configured review stage capacity differs from the benchmark target"
        )
    tasks = {task.task: task for task in plan.tasks}
    subjects: dict[str, set[str]] = {}
    for ref in spec.predictions:
        receipt = read_document(layout, ref, PredictionReceipt)
        context, prediction = receipt.context, receipt.prediction
        task = tasks.get(context.task)
        if task is None or (context.model, context.configuration) != (
            task.model,
            task.inference,
        ):
            raise ValueError(
                "benchmark prediction model/configuration differs from the plan"
            )
        manifest = read_document(layout, context.manifest, ClipManifest)
        if (
            context.clip_id != manifest.clip_id
            or prediction.clip_id != manifest.clip_id
            or prediction.task != context.task
            or prediction.manifest_digest != context.manifest.digest
            or prediction.model_digest != context.model.sha256
            or prediction.inference_digest != context.configuration.digest
        ):
            raise ValueError("benchmark prediction provenance is inconsistent")
        verify_reference(layout, prediction.artifact)
        seen = subjects.setdefault(context.task, set())
        if context.clip_id in seen:
            raise ValueError("benchmark inputs duplicate a task/clip")
        seen.add(context.clip_id)
    expected: dict[str, WorkerBinding] = {}
    for binding in spec.bindings:
        if not subjects.get(binding.task):
            raise ValueError("each measured task requires frozen predictions")
        for subject in subjects[binding.task]:
            expected[content_digest([binding.adapter, subject])] = binding
    return spec, plan, expected


def _controls(plan: CampaignPlan) -> dict[str, Any]:
    # Dataset output versions and capacities vary; learning/input/review semantics do not.
    value: dict[str, Any] = plan.model_dump(mode="json")
    del value["campaign_id"]
    del value["resources"]["codex_workers"]
    for stage in value["resources"]["stages"]:
        if stage["kind"] == "codex":
            del stage["parallel"]
    for task in value["tasks"]:
        for product in task["products"]:
            del product["dataset_version"]
    return value


def comparison(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[ComparisonSpec, tuple[ConditionSpec, ...]]:
    spec = read_document(layout, reference, ComparisonSpec)
    for ref in references(spec):
        verify_reference(layout, ref)
    conditions: list[ConditionSpec] = []
    plans: list[CampaignPlan] = []
    input_files: set[ArtifactRef] = set()
    for ref in spec.conditions:
        entry, plan, _ = condition(layout, ref)
        conditions.append(entry)
        plans.append(plan)
        input_files.update(references(plan))
        for prediction_ref in entry.predictions:
            prediction = read_document(layout, prediction_ref, PredictionReceipt)
            input_files.update(references(prediction))
            manifest = read_document(layout, prediction.context.manifest, ClipManifest)
            input_files.update(references(manifest))
    for ref in input_files:
        verify_reference(layout, ref)
    first = conditions[0]
    if [entry.target_workers for entry in conditions] != [10, 25, 50]:
        raise ValueError("comparison conditions must be ordered 10, 25, 50")
    if (
        len({entry.condition_id for entry in conditions}) != 3
        or len({plan.campaign_id for plan in plans}) != 3
    ):
        raise ValueError("each condition requires its own immutable campaign identity")
    if any(
        entry.benchmark_id != spec.benchmark_id
        or entry.scope != first.scope
        or entry.predictions != first.predictions
        or entry.bindings != first.bindings
        for entry in conditions
    ):
        raise ValueError(
            "comparison must share budget, exact input receipts and worker bindings"
        )
    if any(_controls(plan) != _controls(plans[0]) for plan in plans[1:]):
        raise ValueError(
            "concurrency comparison changed model, effort, code or input/review policy"
        )
    return spec, tuple(conditions)
