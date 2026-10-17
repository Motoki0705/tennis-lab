"""Receipt-backed product reporting, independent of task label semantics."""

from __future__ import annotations

from typing import Any

from .artifacts import read_document
from .contracts import ArtifactRef, ProductOutcome, PublicationReceipt
from .learning.contracts import CampaignCompletion, TaskProducts
from .learning.validation import check_completion
from .paths import AnnotationLayout
from .plan import CampaignPlan


def publication_summary(
    layout: AnnotationLayout,
    plan_ref: ArtifactRef,
    completion_ref: ArtifactRef | None,
) -> dict[str, Any]:
    """Keep missing, excluded and published products distinct, including receipts."""
    plan = read_document(layout, plan_ref, CampaignPlan)
    if completion_ref is None:
        completion = CampaignCompletion(
            plan=plan_ref,
            tasks=tuple(
                TaskProducts(
                    task=task.task,
                    products=tuple(
                        ProductOutcome(
                            product=product.product,
                            status="pending",
                            publication=None,
                            reason="preview does not publish teachers"
                            if plan.phase == "preview"
                            else "no verified campaign completion receipt",
                        )
                        for product in task.products
                    ),
                )
                for task in plan.tasks
            ),
        )
    else:
        completion = read_document(layout, completion_ref, CampaignCompletion)
        if completion.plan != plan_ref:
            raise ValueError("report completion belongs to a different plan")
        check_completion(layout, completion)
    products = []
    for task in completion.tasks:
        for outcome in task.products:
            receipt = (
                read_document(layout, outcome.publication, PublicationReceipt)
                if outcome.publication is not None
                else None
            )
            products.append(
                {
                    "task": task.task,
                    **outcome.model_dump(mode="json"),
                    "snapshot": receipt.snapshot.model_dump(mode="json")
                    if receipt is not None
                    else None,
                    "added_clips": len(receipt.added_clips)
                    if receipt is not None
                    else None,
                    "reused_clips": len(receipt.reused_clips)
                    if receipt is not None
                    else None,
                }
            )
    return {
        "complete": completion_ref is not None and completion.complete,
        "completion": completion_ref.model_dump(mode="json")
        if completion_ref is not None
        else None,
        "products": products,
    }
