from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from src.tasks.annotation.__main__ import run_finished
from src.tasks.annotation.configuration import configure_local_videos
from src.tasks.annotation.contracts import (
    AnnotationContinuation,
    LocalAnnotationRuntime,
)
from src.tasks.annotation.local_controller import LocalAnnotationController
from src.tasks.annotation.local_videos import LocalVideoConfigurationReceipt
from src.tasks.ball_detection.annotation.video.agent.contracts import (
    AgentTask,
    VideoCompletion,
)
from src.tasks.ball_detection.annotation.video.agent.store import VideoRevisionStore
from src.tasks.ball_detection.annotation.video.io import read_annotation
from src.tasks.base.annotation.artifacts import annotation_reference, read_document
from src.tasks.base.annotation.editing.contracts import BlankAnnotationContext, Revision
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.workers.contracts import ExitReceipt, Launch, WorkRequest
from src.tasks.base.annotation.workers.processes import ProcessIdentity, owned_members
from src.utils.paths import PROJECT_ROOT
from tests.support.tasks.annotation_events import AnnotationEvents
from tests.support.tasks.blank_agent import fake_worker
from tests.support.tasks.local_annotation import local_configuration


def test_three_camera_workflow_continues_saved_timeout_without_restarting_completed_cameras(
    tmp_path: Path,
) -> None:
    # The real budget counts its controller once. A full-suite xdist process has
    # unrelated model/test assets resident; it must not become this controller's
    # baseline. Keep the same fixed 1 GiB budget in a clean CPU process, with all
    # resource accounting, service stages and supervisors still real.
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.integration.tasks.annotation.test_automatic_continuation",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, (
        _workflow_output(result) + _attempt_evidence(tmp_path / "annotation")
    )


def _workflow_output(result: subprocess.CompletedProcess[str]) -> str:
    limit = 16384
    return "".join(
        f"\n{name}:\n"
        + (
            data[: limit // 2]
            + f"\n[truncated {len(data) - limit} characters from middle]\n"
            + data[-limit // 2 :]
            if len(data) > limit
            else data
        )
        for name, data in (("stdout", result.stdout), ("stderr", result.stderr))
    )


def _attempt_evidence(root: Path) -> str:
    """Keep the native cause when the workflow reports a policy-level failure."""
    file_limit, total_limit = 16384, 262144
    evidence = [
        f"\nattempt evidence (last {file_limit} bytes/file; "
        f"at most {total_limit} characters including headers):\n"
    ]
    directories = sorted(root.glob("campaigns/*/workers/*/attempt-*")) + sorted(
        root.glob("campaigns/*/jobs/*")
    )
    paths = [
        directory / name
        for directory in directories
        for name in ("exit.json", "stderr.log", "supervisor.log", "events.jsonl")
        if (directory / name).is_file()
    ]
    remaining = total_limit - len(evidence[0])
    for index, path in enumerate(paths):
        section = f"\n{path.relative_to(root)}:\n"
        try:
            with path.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - file_limit))
                data = stream.read(file_limit)
            if size > file_limit:
                section += (
                    f"[truncated {size - file_limit} bytes; "
                    f"showing last {file_limit} bytes]\n"
                )
            section += data.decode("utf-8", errors="replace")
        except OSError as error:
            section += f"[could not read: {type(error).__name__}: {error}]\n"
        omitted = (
            f"\n[truncated diagnostic collection at {total_limit} characters; "
            f"{len(paths) - index} files omitted]\n"
        )
        if len(section) + len(omitted) > remaining:
            evidence.append(omitted)
            break
        evidence.append(section)
        remaining -= len(section)
    return "".join(evidence)


def _exercise_three_camera_workflow(tmp_path: Path) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    configuration = local_configuration(layout, review_timeout=5400, three_cameras=True)
    budget = configuration.campaign.resources.model_copy(
        update={
            "codex_workers": 3,
            "total_pss_bytes": 1024**3,
            "stages": tuple(
                s.model_copy(update={"parallel": 3 if s.stage == "review" else 1})
                for s in configuration.campaign.resources.stages
            ),
        }
    )
    configuration = configuration.model_copy(
        update={
            "campaign": configuration.campaign.model_copy(update={"resources": budget}),
            "runtime": configuration.runtime.model_copy(
                update={
                    "costs": tuple(
                        c.model_copy(update={"pss_bytes": 128 * 1024**2})
                        for c in configuration.runtime.costs
                    ),
                }
            ),
        }
    )
    receipt = read_document(
        layout,
        configure_local_videos(layout, configuration),
        LocalVideoConfigurationReceipt,
    )
    runtime = read_document(layout, receipt.runtime, LocalAnnotationRuntime)
    requests = {}
    for entry in receipt.registrations:
        request = read_document(layout, entry.request, WorkRequest)
        context = read_document(layout, request.payload, BlankAnnotationContext)
        from src.tasks.ball_detection.annotation.video import VideoAnnotationInput

        camera = read_document(
            layout, context.task_input, VideoAnnotationInput
        ).source.camera_id
        requests[camera] = request
    controllers: list[LocalAnnotationController] = []
    # Retain exit codes after reconciliation removes finished pool children.
    supervisors: list[subprocess.Popen[bytes]] = []

    def restarted() -> LocalAnnotationController:
        result = LocalAnnotationController(
            layout,
            runtime,
            test_clock=True,
            test_command=fake_worker(
                pause_after_save=True, save_limit=4, pause_camera="cam0"
            ),
        )
        controllers.append(result)
        return result

    controller = restarted()
    events = AnnotationEvents(layout.path("campaigns"))

    def drive(until: Callable[[], bool]) -> None:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            events.refresh()
            result = controller.cycle()
            if until():
                return
            assert result["state"] != "blocked", result
            events.refresh()
            children = [
                child
                for c in controllers
                for child in (
                    *c.production.pool.children.values(),
                    *c.jobs.stages.children.values(),
                )
            ]
            events.wait(children, max(0, deadline - time.monotonic()))
        raise AssertionError(controller.status())

    def directory(camera: str, number: int) -> Path:
        result: Path = layout.path(
            "campaigns",
            controller.plan.campaign_id,
            "workers",
            requests[camera].work_id,
            f"attempt-{number:04d}",
        )
        return result

    try:
        controller.cycle()
        first_pool = controller.production.pool
        supervisors.extend(first_pool.children.values())
        for control in first_pool.test_controls.values():
            control.settimeout(10)
            assert json.loads(control.recv(4096)) == {
                "state": "ready",
                "timeout_seconds": 5400,
            }
        first = directory("cam0", 1)
        drive(
            lambda: (
                (first / "work/saved.json").exists()
                and all(
                    first_pool.ledger.task(requests[c].work_id)["state"] == "complete"
                    for c in ("cam1", "cam2")
                )
            )
        )
        task = read_document(
            layout, annotation_reference(layout, first / "request.json"), AgentTask
        )
        store = VideoRevisionStore(layout, task.run)
        saved = store.current()[0]
        before = store.annotation(saved)
        assert [f.state for f in before.frames[:4]] == [
            "unknown",
            "in_frame",
            "in_frame",
            "in_frame",
        ]
        assert [f.state for f in before.frames[4:]] == ["unprocessed"] * 8
        assert not (first / "work/result.json").exists()
        complete_before = {
            r["camera_id"]: r["completion"]
            for r in controller.status()["reviews"]
            if r["state"] == "complete"
        }
        assert set(complete_before) == {"cam1", "cam2"}
        attempt = first_pool.ledger.task(requests["cam0"].work_id)["current_attempt"]
        control = first_pool.test_controls[attempt]
        control.send(json.dumps({"elapsed_seconds": 5399.999}).encode())
        assert json.loads(control.recv(4096))["state"] == "running"
        controller.cycle()
        assert first_pool.ledger.task(requests["cam0"].work_id)["state"] == "running"
        assert not (first / "exit.json").exists()
        control.send(json.dumps({"elapsed_seconds": 5400}).encode())
        assert json.loads(control.recv(4096))["state"] == "timed_out"
        drive(
            lambda: first_pool.ledger.task(requests["cam0"].work_id)["state"] == "ready"
        )
        execution = read_document(
            layout, annotation_reference(layout, first / "exit.json"), ExitReceipt
        )
        assert execution.status == "timed_out" and execution.exit_code == 124
        owner = ProcessIdentity.from_document(
            json.loads((first / "owner.json").read_text())
        )
        assert not owned_members(owner)
        history = first_pool.ledger.history(controller.plan, requests["cam0"])
        assert [item.state for item in history] == ["timed_out"]
        decisions = list(
            layout.path(
                "campaigns",
                controller.plan.campaign_id,
                "workflow",
                "continuations",
                requests["cam0"].work_id,
            ).glob("*.json")
        )
        assert len(decisions) == 1
        decision = read_document(
            layout, annotation_reference(layout, decisions[0]), AnnotationContinuation
        )
        assert (
            decision.action,
            decision.processed_before,
            decision.processed_after,
            decision.remaining,
        ) == ("continue", 0, 4, 8)
        assert not directory("cam0", 2).exists()

        # Discard controller state after authorization, then again after launch.
        controller = restarted()
        controller.cycle()
        second_pool = controller.production.pool
        supervisors.extend(second_pool.children.values())
        second_attempt = second_pool.ledger.task(requests["cam0"].work_id)[
            "current_attempt"
        ]
        assert second_attempt != attempt
        second_control = second_pool.test_controls[second_attempt]
        second_control.settimeout(10)
        assert json.loads(second_control.recv(4096))["timeout_seconds"] == 5400
        second = directory("cam0", 2)
        next_task = read_document(
            layout, annotation_reference(layout, second / "request.json"), AgentTask
        )
        assert next_task.initial_revision == saved and next_task.run == task.run
        for path in (first, second):
            launch = read_document(
                layout, annotation_reference(layout, path / "launch.json"), Launch
            )
            assert (
                launch.timeout_seconds,
                launch.plan,
                launch.resource_scope,
                launch.request,
            ) == (5400, receipt.plan, receipt.scope, requests["cam0"])
        second_control.send(json.dumps({"elapsed_seconds": 5399.999}).encode())
        assert json.loads(second_control.recv(4096))["state"] == "running"
        controller = restarted()
        controller.cycle()
        assert not directory("cam0", 3).exists()
        drive(lambda: controller.status()["state"] == "complete")
        final = controller.status()
        assert run_finished(final) and not final["live"]
        for _ in range(2):
            assert controller.cycle()["state"] == "complete"
        for row in final["reviews"]:
            completion = read_document(
                layout,
                type(receipt.plan).model_validate(row["completion"]),
                VideoCompletion,
            )
            annotation = read_annotation(
                layout.artifact_path(completion.annotation), require_complete=True
            )
            assert (
                len(annotation.frames) == 12
                and annotation.source.camera_id == row["camera_id"]
            )
            if row["camera_id"] == "cam0":
                assert annotation.frames[:4] == before.frames[:4]
                assert [f.state for f in annotation.frames[4:]] == ["in_frame"] * 8
                assert (
                    read_document(layout, completion.revision, Revision).sequence == 2
                )
            else:
                assert row["completion"] == complete_before[row["camera_id"]]
        assert [
            h.state
            for h in first_pool.ledger.history(controller.plan, requests["cam0"])
        ] == ["timed_out", "complete"]
        for camera in ("cam1", "cam2"):
            assert (
                len(first_pool.ledger.history(controller.plan, requests[camera])) == 1
            )
        with controller.resources.registry.transaction() as db:
            assert (
                db.execute(
                    "SELECT count(*) FROM resource_requests WHERE stage='review'"
                ).fetchone()[0]
                == 4
            )
        handoff = next(
            json.loads(line)["envelope"]["view"]
            for line in (second / "events.jsonl").read_text().splitlines()
            if json.loads(line).get("type") == "tool.synthetic"
        )
        assert handoff["remaining"] == list(range(4, 12))
        assert handoff["summary"]["final_unknown"] == 1
        assert store.annotation(saved) == before
    finally:
        if sys.exc_info()[0] is not None:
            for c in controllers:
                supervisors.extend(c.production.pool.children.values())
            print(
                "supervisor process snapshot before cleanup: "
                + json.dumps(
                    [
                        {"pid": child.pid, "returncode": child.poll(), "args": child.args}
                        for child in {c.pid: c for c in supervisors}.values()
                    ]
                ),
                file=sys.stderr,
                flush=True,
            )
        for request in requests.values():
            pool = controller.production.pool
            if pool.ledger.task(request.work_id)["state"] in {"running", "exited"}:
                pool.stop(request, reason="integration cleanup")
        for c in controllers:
            for control in c.production.pool.test_controls.values():
                control.close()
            for child in (
                *c.production.pool.children.values(),
                *c.jobs.stages.children.values(),
            ):
                child.wait(timeout=15)
        events.close()


if __name__ == "__main__":
    _exercise_three_camera_workflow(Path(sys.argv[1]))
