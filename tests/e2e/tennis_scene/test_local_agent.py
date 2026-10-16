"""Exercise the replacement ball workflow and reject retired CLI entrypoints."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("command", ["run", "qa", "intake"])
def test_retired_ball_commands_are_rejected(command: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.tennis_scene.chat_annotation.local_agent", command],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_local_agent_cli_annotation_and_adoption(tmp_path: Path) -> None:
    """The replacement CLI protocol retains the legacy E2E's 120-second allowance."""
    from src.tasks.base.annotation.artifacts import (
        annotation_reference,
        read_document,
        write_document,
    )
    from src.tasks.base.annotation.contracts import ArtifactRef
    from src.tasks.base.annotation.workers.contracts import AttemptResult
    from src.tasks.base.annotation.workers.pool import WorkerPool
    from src.tasks.base.annotation.workers.review.service import execute_service
    from src.tasks.base.annotation.workers.services import ServiceReply, ServiceRequest
    from src.utils.paths import PROJECT_ROOT
    from tests.support.tasks.ball_detection.review_worker import ReviewSession

    session = ReviewSession(tmp_path / "annotation")
    session.state("failed")
    env = session.env
    binary = env.layout.repository / "fake codex"
    binary.write_text(
        f"#!{sys.executable}\nimport sys, runpy\nsys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        f"module = runpy.run_path({str(PROJECT_ROOT / 'tests/support/tasks/ball_detection/review_worker.py')!r})\n"
        "module['run_synthetic_cli'](show_image=True)\n"
    )
    binary.chmod(0o700)
    pool = WorkerPool(
        env.layout,
        env.plan,
        (session.worker,),
        codex_binary=str(binary),
        codex_home=env.layout.repository / "fake home",
        timeout_seconds=120,
    )
    assert pool.start(session.request, retry=True)
    directory = session.directory.with_name("attempt-0002")

    def diagnostics() -> dict[str, str]:
        return {
            str(path.relative_to(directory)): path.read_text()
            for name in ("stderr.log", "supervisor.log", "events.jsonl", "exit.json")
            for path in directory.glob(name)
        }

    # Real image I/O and several Python CLIs can exceed 20 seconds on CI.
    deadline = time.monotonic() + 120
    try:
        while True:
            state = pool.reconcile(session.request)
            if state["state"] not in {"running", "exited"}:
                break
            for path in (directory / "work/service-requests").glob("*.json"):
                service_ref = annotation_reference(env.layout, path)
                service = read_document(env.layout, service_ref, ServiceRequest)
                reply = directory / "services" / service.sha256 / "reply.json"
                if not reply.exists():
                    result = execute_service(
                        env.layout, env.plan, service_ref, session.services
                    )
                    write_document(
                        env.layout,
                        reply,
                        ServiceReply(
                            request_digest=service.sha256,
                            status="succeeded",
                            result=result,
                            reason=None,
                        ),
                    )
            assert time.monotonic() < deadline, (state, diagnostics())
            time.sleep(0.1)
        assert state["state"] == "complete", (state, diagnostics())
        final = read_document(
            env.layout, ArtifactRef.model_validate_json(state["result"]), AttemptResult
        )
        assert final.telemetry is not None and final.outcome.adoption is not None
        assert env.store.current(env.context_ref, env.adapter)[1] == final.outcome.adoption
        assert pool.reconcile(session.request)["state"] == "complete"
    finally:
        if pool.ledger.task(session.request.work_id)["state"] in {"running", "exited"}:
            pool.stop(session.request, reason="synthetic E2E teardown")
