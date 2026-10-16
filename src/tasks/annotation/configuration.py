"""Freeze campaign/runtime inputs and explicitly create or reuse a budget scope."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.collection.contracts import PreparationConfig
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Document,
    Identifier,
    PositiveInt,
)
from src.tasks.base.annotation.execution.queue import validate_queue
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    CampaignPlan,
    load_plan,
    references,
    save_plan,
    validate_inputs,
)
from src.tasks.base.annotation.reproducibility import require_frozen_code
from src.tasks.base.annotation.resources import BudgetLimits, ResourceManager
from src.tasks.base.annotation.resources.contracts import BudgetScope
from src.tasks.base.annotation.screening.contracts import ScreeningPolicy
from src.utils.paths import PROJECT_ROOT

from .contracts import Runtime, RuntimeSettings


class ExistingBudget(Document):
    action: Literal["reuse"] = "reuse"
    scope: ArtifactRef


class NewBudget(Document):
    action: Literal["create"] = "create"
    scope_id: Identifier
    roots: tuple[str, ...] = Field(min_length=1)
    previously_allocated_bytes: int = Field(ge=0)
    previously_used_gpu_seconds: float = Field(ge=0)
    max_pending_per_stage: PositiveInt


class Configuration(Document):
    schema_version: Literal["annotation.configuration.v1"] = (
        "annotation.configuration.v1"
    )
    plan: CampaignPlan
    budget: Annotated[ExistingBudget | NewBudget, Field(discriminator="action")]
    runtime: RuntimeSettings


class ConfigurationReceipt(Document):
    request: ArtifactRef
    plan: ArtifactRef
    scope: ArtifactRef
    runtime: ArtifactRef


def validate_runtime(
    layout: AnnotationLayout,
    settings: RuntimeSettings,
    plan: CampaignPlan,
    roots: tuple[str, ...],
) -> None:
    """Read-only validation shared by configuration and controller startup."""
    if len({cost.stage for cost in settings.costs}) != len(settings.costs):
        raise ValueError("stage costs must be unique")
    declared = {stage.stage: stage for stage in plan.resources.stages}
    for cost in settings.costs:
        if cost.stage not in declared or (cost.gpu_seconds > 0) != (
            declared[cost.stage].kind == "gpu"
        ):
            raise ValueError("runtime resource costs differ from declared stage kinds")
        if cost.gpu_seconds and cost.timeout_seconds + 10 > cost.gpu_seconds:
            raise ValueError("GPU timeout must leave the reserved teardown margin")
    codex_home = Path(settings.codex_home)
    if not codex_home.is_absolute() or codex_home.resolve() != codex_home:
        raise ValueError("Codex state must use an explicit normalized directory")
    if not any(codex_home.is_relative_to(Path(root)) for root in roots):
        raise ValueError("Codex state must be inside the resource inventory")
    validate_queue(layout, settings.queue)


def _scope(layout: AnnotationLayout, data: Configuration) -> BudgetScope:
    if isinstance(data.budget, ExistingBudget):
        scope = read_document(layout, data.budget.scope, BudgetScope)
        if layout.artifact_path(data.budget.scope) != layout.path(
            "resources", scope.scope_id, "scope.json"
        ):
            raise ValueError("reuse requires the canonical budget scope file")
        verify_reference(layout, scope.baseline)
    else:
        scope = BudgetScope(
            scope_id=data.budget.scope_id,
            limits=BudgetLimits.from_budget(data.plan.resources),
            roots=data.budget.roots,
            baseline=ArtifactRef(
                role="annotation", path="pending", digest="0" * 64, size_bytes=0
            ),
            previously_allocated_bytes=data.budget.previously_allocated_bytes,
            previously_used_gpu_seconds=data.budget.previously_used_gpu_seconds,
            max_pending_per_stage=data.budget.max_pending_per_stage,
        )
    if scope.limits != BudgetLimits.from_budget(data.plan.resources):
        raise ValueError("campaign limits differ from the reused budget scope")
    roots = tuple(Path(root) for root in scope.roots)
    if any(not root.is_dir() for root in roots):
        raise ValueError("every storage inventory root must already be a directory")
    if not any(layout.root.is_relative_to(root) for root in roots):
        raise ValueError("storage inventory must include the annotation root")
    if layout.mode == "production" and layout.repository not in roots:
        raise ValueError(
            "production inventory must include the complete main repository"
        )
    return scope


def configure(layout: AnnotationLayout, data: Configuration) -> ArtifactRef:
    data = Configuration.model_validate_json(data.canonical_bytes())
    plan = data.plan
    if plan.annotation_root != str(layout.root):
        raise ValueError("configuration plan belongs to a different annotation root")
    if layout.mode == "production":
        require_frozen_code(PROJECT_ROOT, plan)
    validate_inputs(layout, plan)
    read_document(layout, plan.selection, ScreeningPolicy)
    read_document(layout, plan.preparation, PreparationConfig)
    if any(task.task not in {"ball", "player"} for task in plan.tasks):
        raise ValueError("workflow has no adapter for a configured task")
    required = {
        "acquire",
        "screen",
        "render",
        "prepare",
        *("infer_" + task.task for task in plan.tasks),
    }
    if plan.phase == "production":
        from .reviews import require_supported

        require_supported(plan)
        required.update(("review", "publish_ball"))
        if plan.tasks[0].learning is not None:
            required.update(("train_ball", "evaluate_ball"))
    scope = _scope(layout, data)
    validate_runtime(layout, data.runtime, plan, scope.roots)
    declared = {stage.stage: stage for stage in plan.resources.stages}
    for stage in required:
        data.runtime.cost(stage)
        if (stage in {"screen", "review"}) != (declared[stage].kind == "codex"):
            raise ValueError(
                "screen/review must use Codex capacity; other operations cannot"
            )
        if stage in {"train_ball", "evaluate_ball"} and declared[stage].kind != "gpu":
            raise ValueError("learning stages require explicit shared GPU reservations")
    for cost in data.runtime.costs:
        if (
            cost.pss_bytes > scope.limits.total_pss_bytes
            or cost.storage_bytes > scope.limits.additional_storage_bytes
            or cost.gpu_seconds > scope.limits.gpu_seconds
        ):
            raise ValueError("a stage reservation exceeds the total campaign budget")
    directory = layout.path("campaigns", plan.campaign_id, "configuration")
    with artifact_lock(layout, "workflow-configuration-" + plan.campaign_id):
        plan_path = layout.path("campaigns", plan.campaign_id, "plan.json")
        if plan_path.exists() and load_plan(plan_path) != plan:
            raise ValueError("configuration differs from the existing immutable plan")
        # Resolve immutable scope inputs under the scope lock, then validate and
        # reserve the campaign binding in its registry transaction. A conflict
        # must not freeze a request/plan that cannot be configured successfully.
        if isinstance(data.budget, ExistingBudget):
            resources = ResourceManager(layout, data.budget.scope)
        else:
            resources = ResourceManager.create(
                layout,
                scope_id=scope.scope_id,
                limits=scope.limits,
                roots=tuple(Path(root) for root in scope.roots),
                previously_allocated_bytes=scope.previously_allocated_bytes,
                previously_used_gpu_seconds=scope.previously_used_gpu_seconds,
                max_pending_per_stage=scope.max_pending_per_stage,
            )
        resources.bind(plan)
        request_ref = write_document(layout, directory / "request.json", data)
        plan_ref = annotation_reference(layout, save_plan(layout, plan))
        runtime = Runtime(
            plan=plan_ref, scope=resources.reference, **data.runtime.model_dump()
        )
        runtime_ref = write_document(layout, directory / "runtime.json", runtime)
        receipt = ConfigurationReceipt(
            request=request_ref,
            plan=plan_ref,
            scope=resources.reference,
            runtime=runtime_ref,
        )
        for reference in references(receipt):
            verify_reference(layout, reference)
        result: ArtifactRef = write_document(layout, directory / "result.json", receipt)
        return result
