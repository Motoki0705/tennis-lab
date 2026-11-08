from __future__ import annotations

from pathlib import Path

import pytest

from src.tasks.annotation.contracts import (
    AnnotationContinuation,
    LocalAnnotationRuntime,
)
from src.tasks.annotation.local_controller import LocalAnnotationController
from src.tasks.annotation.recovery import recover_annotation
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.editing.contracts import Revision
from tests.support.tasks.blank_agent import BlankSession


@pytest.fixture
def session(tmp_path: Path) -> BlankSession:
    return BlankSession(tmp_path / "annotation", timeout_seconds=5400)


def controller_for(session: BlankSession) -> LocalAnnotationController:
    runtime = read_document(
        session.layout, session.receipt.runtime, LocalAnnotationRuntime
    )
    return LocalAnnotationController(session.layout, runtime)


def save(session: BlankSession, count: int = 1) -> None:
    annotation = session.store.annotation(session.store.current()[0])
    session.edit(
        tuple(
            row.model_copy(update={"state": "unknown"})
            for row in annotation.frames[:count]
        )
    )


def ended(session: BlankSession, state: str = "timed_out") -> None:
    execution = session.execution().model_copy(
        update={
            "status": state,
            "exit_code": 124 if state == "timed_out" else 1,
            "reason": "synthetic " + state,
        }
    )
    write_document(session.layout, session.directory / "exit.json", execution)


@pytest.mark.parametrize(
    ("scenario", "state", "action", "detail"),
    [
        ("partial", "ready", "continue", "newly processed"),
        ("complete_unknown", "complete", "complete", "durably processed"),
        ("no_progress", "timed_out", "hold", "no new processed-frame"),
        ("missing", "invalid_saved_progress", "hold", "invalid saved progress"),
        ("corrupt", "invalid_saved_progress", "hold", "invalid saved progress"),
        ("quota", "quota", "hold", "explicitly resume"),
        ("failed", "failed", "hold", "external service failure"),
        ("cancelled", "cancelled", "hold", "explicit retry"),
    ],
)
def test_automatic_policy_distinguishes_completion_progress_and_abnormal_stops(
    session: BlankSession,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    state: str,
    action: str,
    detail: str,
) -> None:
    if scenario != "no_progress":
        save(session, 3 if scenario == "complete_unknown" else 1)
    if scenario in {"missing", "corrupt"}:
        payload = read_document(
            session.layout, session.store.current()[0], Revision
        ).payload
        path = session.layout.artifact_path(payload)
        if scenario == "missing":
            path.unlink()
        else:
            path.write_bytes(b"invalid annotation")
    ended(
        session,
        scenario if scenario in {"quota", "failed", "cancelled"} else "timed_out",
    )
    controller = controller_for(session)
    pool = controller.production.pool
    monkeypatch.setattr(
        pool, "start", lambda *_a, **_k: pytest.fail("authorization must not launch")
    )
    reference = session.receipt.registrations[0].request
    result = recover_annotation(
        session.layout, controller.production.policy, pool, reference
    )
    assert result["state"] == state
    assert detail in result["reason"]
    from src.tasks.annotation.__main__ import run_finished
    from src.tasks.annotation.production import LocalAnnotationWorkflow

    # One stopped camera must not end the run while another still needs services.
    live = LocalAnnotationWorkflow.summarize(
        [result, {"state": "running", "worker_state": "running"}]
    )
    assert live["live"] and not run_finished(live)
    settled = LocalAnnotationWorkflow.summarize(
        [result, {"state": "complete", "worker_state": "complete"}]
    )
    assert run_finished(settled) == (action != "continue")
    (decision_path,) = session.layout.path(
        "campaigns",
        session.plan.campaign_id,
        "workflow",
        "continuations",
        session.request.work_id,
    ).glob("*.json")
    decision = read_document(
        session.layout,
        annotation_reference(session.layout, decision_path),
        AnnotationContinuation,
    )
    assert decision.action == action
    assert decision.after_attempt == session.launch.attempt
    repeated = recover_annotation(
        session.layout, controller.production.policy, pool, reference
    )
    assert repeated["state"] == state
    assert len(pool.ledger.history(session.plan, session.request)) == 1
    if scenario == "partial":
        assert (
            decision.processed_before,
            decision.processed_after,
            decision.remaining,
        ) == (0, 1, 2)
        assert pool.ledger.task(session.request.work_id)["reason"] == decision.reason
    if scenario == "complete_unknown":
        from src.tasks.annotation.contracts import RecoveryInput, RetryWorker
        from src.tasks.annotation.recovery import recover

        assert result["summary"]["final_unknown"] == 3
        assert session.store.current()[1] is not None
        assert repeated["completion"] == result["completion"]
        assert pool.ledger.task(session.request.work_id)["state"] == "timed_out"
        with pytest.raises(ValueError, match="already complete"):
            recover(
                controller,
                RecoveryInput(
                    plan=controller.runtime.plan,
                    reason="must not restart final unknowns",
                    operation=RetryWorker(
                        request=reference, after_attempt=session.launch.attempt
                    ),
                ),
            )


@pytest.mark.parametrize("after_authorization", [False, True])
def test_controller_restart_recovers_both_sides_of_retry_authorization(
    session: BlankSession,
    monkeypatch: pytest.MonkeyPatch,
    after_authorization: bool,
) -> None:
    save(session)
    ended(session)
    controller = controller_for(session)
    pool = controller.production.pool
    original = pool.authorize_retry

    def interrupted(*args, **kwargs):
        if after_authorization:
            original(*args, **kwargs)
        raise RuntimeError("interrupted authorization boundary")

    monkeypatch.setattr(pool, "authorize_retry", interrupted)
    reference = session.receipt.registrations[0].request
    with pytest.raises(RuntimeError, match="interrupted authorization"):
        recover_annotation(
            session.layout, controller.production.policy, pool, reference
        )
    assert pool.ledger.task(session.request.work_id)["state"] == (
        "ready" if after_authorization else "timed_out"
    )
    restarted = controller_for(session)
    result = recover_annotation(
        session.layout,
        restarted.production.policy,
        restarted.production.pool,
        reference,
    )
    assert result["state"] == "ready"
    assert restarted.production.policy == controller.production.policy
    assert (
        pool.ledger.task(session.request.work_id)["current_attempt"]
        == session.launch.attempt
    )
    assert len(pool.ledger.history(session.plan, session.request)) == 1


def test_timeout_uses_committed_head_after_interruption_during_save(
    session: BlankSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.ball_detection.annotation.video.agent.store import VideoRevisionStore

    save(session)
    committed = session.store.current()[0]
    with monkeypatch.context() as patch:

        def interrupted(*_a, **_k):
            raise RuntimeError("stop before head transaction commits")

        patch.setattr(VideoRevisionStore, "_protect", interrupted)
        with pytest.raises(RuntimeError, match="head transaction"):
            save(session, 3)
    assert session.store.current()[0] == committed
    ended(session)
    controller = controller_for(session)
    result = recover_annotation(
        session.layout,
        controller.production.policy,
        controller.production.pool,
        session.receipt.registrations[0].request,
    )
    assert result["state"] == "ready"
    assert (result["processed_after"], result["summary"]["unprocessed"]) == (1, 2)


def test_concurrent_explicit_retry_supersedes_the_old_automatic_decision(
    session: BlankSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save(session)
    ended(session)
    controller = controller_for(session)

    def retry_first(*_a, **_k):
        session.prepare(2)
        raise ValueError("worker attempt changed")

    monkeypatch.setattr(controller.pool, "authorize_retry", retry_first)
    result = recover_annotation(
        session.layout,
        controller.production.policy,
        controller.pool,
        session.receipt.registrations[0].request,
    )
    assert (result["state"], result["current_attempt"]) == (
        "running",
        session.launch.attempt,
    )
    assert [
        h.state for h in controller.pool.ledger.history(session.plan, session.request)
    ] == ["timed_out", "running"]


def test_quota_requires_recorded_resume_and_explicit_retry(
    session: BlankSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.annotation.contracts import (
        RecoveryInput,
        RecoveryReceipt,
        ResumeWorkers,
        RetryWorker,
    )
    from src.tasks.annotation.recovery import recover

    save(session)
    ended(session, "quota")
    controller = controller_for(session)
    result = recover_annotation(
        session.layout,
        controller.production.policy,
        controller.pool,
        session.receipt.registrations[0].request,
    )
    assert result["state"] == "quota"
    resumed = recover(
        controller,
        RecoveryInput(
            plan=controller.runtime.plan,
            reason="service capacity restored",
            operation=ResumeWorkers(),
        ),
    )
    assert (
        read_document(session.layout, resumed, RecoveryReceipt).worker_attempt is None
    )
    launched = []

    def start(request, *, retry):
        launched.append((request, retry))
        return "explicit-next-attempt"

    monkeypatch.setattr(controller.pool, "start", start)
    retry = RecoveryInput(
        plan=controller.runtime.plan,
        reason="service capacity restored",
        operation=RetryWorker(
            request=session.receipt.registrations[0].request,
            after_attempt=session.launch.attempt,
        ),
    )
    reference = recover(controller, retry)
    assert (
        read_document(session.layout, reference, RecoveryReceipt).worker_attempt
        == "explicit-next-attempt"
    )
    assert recover(controller, retry) == reference
    assert launched == [(session.request, True)]


def test_local_controller_rejects_short_test_timeout_as_production_policy(
    tmp_path: Path,
) -> None:
    session = BlankSession(tmp_path / "annotation")
    with pytest.raises(ValueError, match="5400 seconds"):
        controller_for(session)


def test_completed_cameras_wait_for_the_last_service_stage_to_release_ownership(
    session: BlankSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.annotation.__main__ import run_finished

    controller = controller_for(session)
    path = session.layout.path(
        "campaigns", session.plan.campaign_id, "jobs", "stage", "launch.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    monkeypatch.setattr(
        controller.jobs.stages, "status", lambda ref: {"state": "draining"}
    )
    progress = controller._stage_status({"state": "complete", "live": False})
    assert progress["state"] == "reviewing" and progress["live"]
    assert not run_finished(progress)
    monkeypatch.setattr(
        controller.jobs.stages, "status", lambda ref: {"state": "succeeded"}
    )
    progress = controller._stage_status({"state": "complete", "live": False})
    assert progress["state"] == "complete" and not progress["live"]
    assert run_finished(progress)
