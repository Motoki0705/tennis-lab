from __future__ import annotations

from pathlib import Path

import pytest

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ModelRef,
    ProductOutcome,
    PublicationReceipt,
)
from src.tasks.base.annotation.learning.contracts import (
    CampaignCompletion,
    TaskProducts,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    CampaignPlan,
    CollectionCounts,
    ProductPlan,
    ResourceBudget,
    ReviewPolicyRef,
    StageCapacity,
    TaskPlan,
    save_plan,
)
from src.tasks.base.annotation.reports import publication_summary
from src.tasks.base.annotation.worker_decision import load_worker_decision


@pytest.fixture
def products(
    tmp_path: Path,
) -> tuple[AnnotationLayout, ArtifactRef, CampaignCompletion]:
    layout = AnnotationLayout.testing(tmp_path / "a")
    worker = load_worker_decision()
    blob = write_asset(layout, layout.path("models", "input"), b"synthetic input")
    tasks = tuple(
        TaskPlan(
            task=task,
            model=ModelRef(
                task=task,
                version="initial",
                training_weights=blob,
                inference_weights=blob,
                configuration=blob,
            ),
            inference=blob,
            review=ReviewPolicyRef(policy=blob, preview_evidence=blob, decision=blob),
            learning=None,
            products=tuple(
                ProductPlan(
                    product=product,
                    dataset_task=task + "_detection",
                    dataset_version="next",
                    component=product,
                    base_snapshot=None,
                    generation_policy=blob,
                )
                for product in names
            ),
        )
        for task, names in (("ball", ("frames",)), ("player", ("boxes", "poses")))
    )
    plan = CampaignPlan(
        campaign_id="report",
        phase="production",
        annotation_root=str(layout.root),
        counts=CollectionCounts(
            channels=2, videos_per_channel=3, max_clips_per_video=4
        ),
        selection=blob,
        preparation=blob,
        code_commit="a" * 40,
        code_diff_digest="b" * 64,
        agent_model=worker.agent_model,
        agent_effort=worker.agent_effort,
        resources=ResourceBudget(
            codex_workers=1,
            additional_storage_bytes=1024,
            minimum_free_bytes=1,
            total_pss_bytes=1024,
            gpu_seconds=1,
            stages=(
                StageCapacity(
                    stage="review", kind="codex", parallel=1, threads_per_process=1
                ),
            ),
        ),
        tasks=tasks,
    )
    plan_ref = annotation_reference(layout, save_plan(layout, plan))
    outcomes = []
    for task in tasks:
        rows = []
        for product in task.products:
            ref = write_document(
                layout,
                layout.path(
                    "campaigns", "report", f"{task.task}-{product.product}.json"
                ),
                PublicationReceipt(
                    task=task.task,
                    product=product.product,
                    request_id="c" * 64,
                    base_snapshot=None,
                    snapshot=blob,
                    adoptions=(blob,),
                    added_clips=("d" * 64,),
                    reused_clips=("e" * 64,),
                ),
            )
            rows.append(
                ProductOutcome(
                    product=product.product,
                    status="published",
                    publication=ref,
                    reason=None,
                )
            )
        outcomes.append(TaskProducts(task=task.task, products=tuple(rows)))
    return layout, plan_ref, CampaignCompletion(plan=plan_ref, tasks=tuple(outcomes))


def test_report_keeps_every_product_pending_without_completion(products: tuple) -> None:
    layout, plan_ref, _ = products
    summary = publication_summary(layout, plan_ref, None)
    assert not summary["complete"]
    assert [(p["task"], p["product"]) for p in summary["products"]] == [
        ("ball", "frames"),
        ("player", "boxes"),
        ("player", "poses"),
    ]
    assert all(
        p["status"] == "pending" and p["added_clips"] is None
        for p in summary["products"]
    )


@pytest.mark.parametrize(
    "state", ["published", "not_applicable", "pending", "blocked", "failed"]
)
def test_report_distinguishes_publication_skip_and_incomplete_products(
    products: tuple, state: str
) -> None:
    layout, plan_ref, completion = products
    player = completion.tasks[1]
    pose = player.products[1]
    if state != "published":
        pose = pose.model_copy(
            update={
                "status": state,
                "publication": None,
                "reason": "explicit pose generation outcome",
            }
        )
    completion = completion.model_copy(
        update={
            "tasks": (
                completion.tasks[0],
                player.model_copy(update={"products": (player.products[0], pose)}),
            )
        }
    )
    ref = write_document(
        layout, layout.path("campaigns", "report", "products.json"), completion
    )
    summary = publication_summary(layout, plan_ref, ref)
    assert summary["complete"] == (state in {"published", "not_applicable"})
    assert summary["products"][0]["added_clips"] == 1
    assert summary["products"][0]["reused_clips"] == 1
    assert summary["products"][2]["status"] == state
    assert summary["products"][2]["publication"] == (
        pose.publication.model_dump(mode="json") if pose.publication else None
    )
    assert summary["products"][2]["reason"] == pose.reason


def test_report_rejects_missing_products_and_changed_snapshot(products: tuple) -> None:
    layout, plan_ref, completion = products
    missing = write_document(
        layout,
        layout.path("campaigns", "report", "missing.json"),
        completion.model_copy(update={"tasks": completion.tasks[:1]}),
    )
    with pytest.raises(ValueError, match="every planned task/product"):
        publication_summary(layout, plan_ref, missing)
    ref = write_document(
        layout, layout.path("campaigns", "report", "products.json"), completion
    )
    layout.path("models", "input").write_bytes(b"changed snapshot")
    with pytest.raises(ValueError, match="changed"):
        publication_summary(layout, plan_ref, ref)
