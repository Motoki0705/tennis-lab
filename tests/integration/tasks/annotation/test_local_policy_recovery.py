"""CLI recovery must retain the policy pinned before a controller exists."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tasks.annotation import local_cli
from src.tasks.annotation.__main__ import main
from src.tasks.annotation.configuration import configure_local_videos
from src.tasks.annotation.contracts import LocalAnnotationRuntime
from src.tasks.annotation.local_controller import LocalAnnotationController
from src.tasks.annotation.local_videos import LocalVideoConfigurationReceipt
from src.tasks.annotation.scene_workflow import (
    SceneAnnotationRequest,
    SceneAnnotationWorkflow,
)
from src.tasks.ball_detection.annotation.video.agent.contracts import (
    DEFAULT_POLICY,
    AnnotationPolicy,
)
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.workers.contracts import WorkRequest
from tests.support.tasks.local_annotation import local_configuration


def custom_policy() -> AnnotationPolicy:
    return AnnotationPolicy(
        policy_id="sparse-context",
        version="recovery-v2",
        instructions="Choose sparse temporal anchors and revise uncertain positions.",
    )


def interrupted_workflow(
    tmp_path: Path,
) -> tuple[SceneAnnotationWorkflow, LocalVideoConfigurationReceipt]:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    cfg = local_configuration(layout, review_timeout=5400)
    cfg = cfg.model_copy(
        update={
            "videos": cfg.videos[:1],
            "campaign": cfg.campaign.model_copy(
                update={
                    "resources": cfg.campaign.resources.model_copy(
                        update={"total_pss_bytes": 1536 * 1024**2}
                    )
                }
            ),
        }
    )
    workflow = SceneAnnotationWorkflow(
        layout,
        SceneAnnotationRequest(
            configuration=cfg, policy=custom_policy(), publication=None
        ),
    )

    def interrupt(*_args, **_kwargs):
        raise InterruptedError("after registration, before controller initialization")

    with pytest.raises(InterruptedError, match="before controller"):
        workflow.annotate(controller_factory=interrupt)
    receipt = read_document(
        layout,
        annotation_reference(
            layout,
            layout.path(
                "campaigns", cfg.campaign.campaign_id, "configuration", "result.json"
            ),
        ),
        LocalVideoConfigurationReceipt,
    )
    assert not layout.path(
        "campaigns", cfg.campaign.campaign_id, "workflow", "annotation-policy.json"
    ).exists()
    return workflow, receipt


@pytest.mark.parametrize("command", ["status", "tick", "run"])
def test_cli_recovery_after_registration_uses_the_scene_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    workflow, receipt = interrupted_workflow(tmp_path)
    layout = workflow.layout
    controllers = []

    def factory(*args, **kwargs):
        controller = LocalAnnotationController(
            *args, **kwargs, test_command=("/usr/bin/true",)
        )
        controllers.append(controller)
        return controller

    monkeypatch.setattr(local_cli, "LocalAnnotationController", factory)
    clock = itertools.count()
    monkeypatch.setattr(
        local_cli, "time", SimpleNamespace(monotonic=lambda: float(next(clock)))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotation",
            command,
            "--runtime",
            str(layout.artifact_path(receipt.runtime)),
            "--sha256",
            receipt.runtime.digest,
            "--test-root",
            str(layout.root),
            *(["--max-seconds", "1"] if command == "run" else []),
        ],
    )
    try:
        main()
        policy = read_document(
            layout,
            annotation_reference(
                layout,
                layout.path(
                    "campaigns",
                    workflow.request.configuration.campaign.campaign_id,
                    "workflow",
                    "annotation-policy.json",
                ),
            ),
            AnnotationPolicy,
        )
        assert policy == workflow.request.policy, (
            "CLI recovery replaced the registered scene policy"
        )
        runtime = read_document(layout, receipt.runtime, LocalAnnotationRuntime)
        # The original workflow can build its controller with the same saved
        # policy after CLI recovery; a conflicting immutable mirror must not remain.
        LocalAnnotationController(
            layout,
            runtime,
            annotation_policy=workflow.request.policy,
            test_command=("/usr/bin/true",),
        )
    finally:
        for controller in controllers:
            for child in (
                *controller.pool.children.values(),
                *controller.jobs.stages.children.values(),
            ):
                child.wait(timeout=15)
            for ref in controller.production.requests:
                request = read_document(layout, ref, WorkRequest)
                controller.pool.reconcile(request)


@pytest.mark.parametrize("explicit", [False, True])
def test_registration_freezes_default_or_explicit_policy_and_rejects_replacement(
    tmp_path: Path, explicit: bool
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    cfg = local_configuration(layout, review_timeout=5400)
    selected = custom_policy() if explicit else DEFAULT_POLICY
    kwargs = {"annotation_policy": selected} if explicit else {}
    reference = configure_local_videos(layout, cfg, **kwargs)
    receipt = read_document(layout, reference, LocalVideoConfigurationReceipt)
    assert (
        read_document(layout, receipt.annotation_policy, AnnotationPolicy) == selected
    )
    before = layout.artifact_path(receipt.annotation_policy).read_bytes()
    assert configure_local_videos(layout, cfg, **kwargs) == reference
    other = DEFAULT_POLICY if explicit else custom_policy()
    with pytest.raises(ValueError, match="original registration"):
        configure_local_videos(layout, cfg, annotation_policy=other)
    assert layout.artifact_path(receipt.annotation_policy).read_bytes() == before
    runtime = read_document(layout, receipt.runtime, LocalAnnotationRuntime)
    with pytest.raises(ValueError, match="differs from its registration"):
        LocalAnnotationController(layout, runtime, annotation_policy=other)
    from src.tasks.ball_detection.annotation.video.agent.worker import BallVideoWorker
    from src.tasks.base.annotation.plan import LocalAnnotationPlan

    request = read_document(layout, receipt.registrations[0].request, WorkRequest)
    with pytest.raises(ValueError, match="registered annotation policy"):
        BallVideoWorker(other).prepare(
            layout,
            read_document(layout, receipt.plan, LocalAnnotationPlan),
            request,
            layout.path("campaigns", "rejected-worker"),
            timeout_seconds=5400,
        )


@pytest.mark.parametrize("violation", ["missing", "changed", "scene", "mirror"])
def test_cli_recovery_rejects_missing_or_conflicting_registered_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
) -> None:
    workflow, receipt = interrupted_workflow(tmp_path)
    layout = workflow.layout
    mirror = layout.path(
        "campaigns",
        workflow.request.configuration.campaign.campaign_id,
        "workflow",
        "annotation-policy.json",
    )
    if violation == "missing":
        layout.artifact_path(receipt.annotation_policy).unlink()
    elif violation == "changed":
        layout.artifact_path(receipt.annotation_policy).write_bytes(
            DEFAULT_POLICY.canonical_bytes() + b"\n"
        )
    elif violation == "scene":
        layout.artifact_path(workflow.reference).write_bytes(
            workflow.request.model_copy(
                update={"policy": DEFAULT_POLICY}
            ).canonical_bytes()
            + b"\n"
        )
    else:
        write_document(layout, mirror, DEFAULT_POLICY)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotation",
            "status",
            "--runtime",
            str(layout.artifact_path(receipt.runtime)),
            "--sha256",
            receipt.runtime.digest,
            "--test-root",
            str(layout.root),
        ],
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        main()
    assert not layout.path(
        "campaigns", workflow.request.configuration.campaign.campaign_id, "workers"
    ).exists()
    if violation == "mirror":
        assert (
            read_document(
                layout, annotation_reference(layout, mirror), AnnotationPolicy
            )
            == DEFAULT_POLICY
        )
    else:
        assert not mirror.exists()


@pytest.mark.parametrize("violation", [None, "missing", "changed"])
def test_benchmark_cli_uses_its_frozen_condition_before_controller_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    violation: str | None,
) -> None:
    import json

    from src.tasks.annotation.benchmark.blank import (
        BlankCondition,
        register_blank_condition,
    )

    workflow, receipt = interrupted_workflow(tmp_path)
    layout = workflow.layout
    trial_policy = custom_policy().model_copy(
        update={"policy_id": "trial-specific", "version": "trial-v3"}
    )
    condition_ref = register_blank_condition(
        layout, receipt.plan, "trial", trial_policy
    )
    condition = read_document(layout, condition_ref, BlankCondition)
    local = layout.for_benchmark("trial")
    policy_path = local.path("campaigns", "trial", "benchmark", "policy.json")
    mirror = local.path("campaigns", "trial", "workflow", "annotation-policy.json")
    if violation == "missing":
        policy_path.unlink()
    elif violation == "changed":
        policy_path.write_bytes(DEFAULT_POLICY.canonical_bytes() + b"\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotation",
            "status",
            "--runtime",
            str(layout.artifact_path(condition.runtime)),
            "--sha256",
            condition.runtime.digest,
            "--test-root",
            str(layout.root),
        ],
    )
    if violation is None:
        main()
        assert json.loads(capsys.readouterr().out)["state"] == "reviewing"
        assert (
            read_document(local, annotation_reference(local, mirror), AnnotationPolicy)
            == condition.policy
        )
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            main()
        assert not mirror.exists()
