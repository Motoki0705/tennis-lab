"""Read exploration policy from immutable registration records, never defaults."""

from __future__ import annotations

from src.tasks.ball_detection.annotation.video.agent.contracts import AnnotationPolicy
from src.tasks.base.annotation.artifacts import annotation_reference, read_document
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan

from .contracts import LocalAnnotationRuntime


def read_registered_annotation_policy(
    layout: AnnotationLayout, runtime: LocalAnnotationRuntime
) -> AnnotationPolicy:
    plan = read_document(layout, runtime.plan, LocalAnnotationPlan)
    runtime_ref = annotation_reference(
        layout,
        layout.path("campaigns", plan.campaign_id, "configuration", "runtime.json"),
    )
    if read_document(layout, runtime_ref, LocalAnnotationRuntime) != runtime:
        raise ValueError("annotation policy requires the registered runtime")
    if plan.phase == "benchmark":
        from .benchmark.blank import BlankCondition

        if layout.review_namespace != plan.campaign_id:
            raise ValueError("benchmark policy requires its registered namespace")
        directory = layout.path("campaigns", plan.campaign_id, "benchmark")
        condition = read_document(
            layout,
            annotation_reference(layout, directory / "blank-condition.json"),
            BlankCondition,
        )
        if (
            condition.plan != runtime.plan
            or condition.runtime != runtime_ref
            or plan.publish_dataset
        ):
            raise ValueError(
                "benchmark policy differs from its registered runtime/plan"
            )
        policy: AnnotationPolicy = read_document(
            layout,
            annotation_reference(layout, directory / "policy.json"),
            AnnotationPolicy,
        )
        if policy != condition.policy:
            raise ValueError("benchmark policy differs from its frozen condition")
    elif plan.phase == "production" and layout.review_namespace is None:
        from .configuration import LocalVideoConfiguration
        from .local_videos import LocalVideoConfigurationReceipt

        directory = layout.path("campaigns", plan.campaign_id, "configuration")
        receipt = read_document(
            layout,
            annotation_reference(layout, directory / "result.json"),
            LocalVideoConfigurationReceipt,
        )
        if (receipt.plan, receipt.runtime, receipt.scope) != (
            runtime.plan,
            runtime_ref,
            runtime.scope,
        ) or layout.artifact_path(
            receipt.annotation_policy
        ) != directory / "annotation-policy.json":
            raise ValueError("annotation policy differs from its registration receipt")
        policy = read_document(layout, receipt.annotation_policy, AnnotationPolicy)
        scene_request = layout.path(
            "campaigns", plan.campaign_id, "scene-workflow", "request.json"
        )
        if scene_request.exists():
            from .scene_workflow import SceneAnnotationRequest

            scene = read_document(
                layout,
                annotation_reference(layout, scene_request),
                SceneAnnotationRequest,
            )
            if scene.policy != policy or scene.configuration != read_document(
                layout, receipt.request, LocalVideoConfiguration
            ):
                raise ValueError(
                    "annotation policy differs from its frozen scene request"
                )
    else:
        raise ValueError(
            "annotation policy requires a registered production or benchmark run"
        )
    mirror = layout.path(
        "campaigns", plan.campaign_id, "workflow", "annotation-policy.json"
    )
    if (
        mirror.exists()
        and read_document(
            layout, annotation_reference(layout, mirror), AnnotationPolicy
        )
        != policy
    ):
        raise ValueError("annotation policy mirror differs from its registration")
    return policy
