"""Exercise the replacement ball workflow and reject retired CLI entrypoints."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "entrypoint",
    [
        "local_agent",
        "local_agent.qa",
        "scripts.prepare",
        "scripts.annotate",
        "scripts.review_ui",
        "scripts.serve_artifacts",
        "scripts.sync_done",
    ],
)
def test_retired_annotation_entrypoints_are_removed(entrypoint: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"src.tennis_scene.chat_annotation.{entrypoint}",
            "--help",
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "No module named 'src.tennis_scene.chat_annotation'" in result.stderr


def test_review_cli_annotation_and_adoption(tmp_path: Path) -> None:
    """Exercise the review CLI protocol with a synthetic worker and real parent services."""
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
        assert (
            env.store.current(env.context_ref, env.adapter)[1] == final.outcome.adoption
        )
        assert pool.reconcile(session.request)["state"] == "complete"
    finally:
        if pool.ledger.task(session.request.work_id)["state"] in {"running", "exited"}:
            pool.stop(session.request, reason="synthetic E2E teardown")


def test_scene_entrypoint_annotates_continues_reloads_triangulates_and_publishes(tmp_path: Path) -> None:
    """A clean CPU process keeps unrelated pytest model memory out of the budget."""
    from src.utils.paths import PROJECT_ROOT
    result = subprocess.run([sys.executable, "-m", "tests.e2e.tasks.annotation.test_workflow", str(tmp_path)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr


def _exercise_direct_scene(tmp_path: Path) -> None:
    import json
    from inspect import unwrap

    import numpy as np

    from src.tasks.annotation.__main__ import main as annotation_main
    from src.tasks.annotation.benchmark.blank import BlankRunMeasurements
    from src.tasks.annotation.local_controller import LocalAnnotationController
    from src.tasks.annotation.local_videos import LocalVideoConfigurationReceipt
    from src.tasks.annotation.scene_workflow import (
        CompletedSceneAnnotations,
        SceneAnnotationWorkflow,
    )
    from src.tasks.ball_detection.annotation.video.agent.contracts import (
        AgentTask,
        AnnotationPolicy,
        AnnotationRun,
    )
    from src.tasks.ball_detection.data.windows.active_ball import (
        ActiveBallWindowDataset,
    )
    from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
        load_snapshot,
    )
    from src.tasks.base.annotation.artifacts import annotation_reference, read_document
    from src.tasks.base.annotation.contracts import ArtifactRef, PublicationReceipt
    from src.tennis_scene.archive import load_scene_result
    from src.tennis_scene.pipeline import TennisSceneOrchestrator
    from src.tennis_scene.pipeline.components.annotation_triangulation import (
        BallAnnotationTriangulationOutput,
    )
    from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
    from src.tennis_scene.scripts import run_pipeline
    from tests.support.tasks.annotation_events import AnnotationEvents
    from tests.support.tasks.blank_agent import fake_worker
    from tests.support.tennis_scene.direct_workflow import direct_fixture

    fixture = direct_fixture(tmp_path)
    layout = fixture.layout
    workflow = SceneAnnotationWorkflow(layout, fixture.request)
    def interrupted(*_args, **_kwargs):
        raise InterruptedError("registration complete; controller has not started")
    with pytest.raises(InterruptedError, match="controller has not started"):
        workflow.annotate(controller_factory=interrupted)
    registration = read_document(layout, annotation_reference(layout,
        layout.path("campaigns", fixture.request.configuration.campaign.campaign_id, "configuration", "result.json")), LocalVideoConfigurationReceipt)
    mirror = layout.path("campaigns", fixture.request.configuration.campaign.campaign_id, "workflow", "annotation-policy.json")
    assert not mirror.exists()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", ["annotation", "status", "--runtime", str(layout.artifact_path(registration.runtime)),
            "--sha256", registration.runtime.digest, "--test-root", str(layout.root)])
        annotation_main()
    assert read_document(layout, annotation_reference(layout, mirror), AnnotationPolicy) == fixture.request.policy
    controllers = []
    events = AnnotationEvents(layout.path("campaigns"))
    clocks, timed_out = set(), set()
    def factory(*args, **kwargs):
        result = LocalAnnotationController(*args, **kwargs, test_clock=True,
            test_command=fake_worker(pause_after_save=True, pause_camera="right", save_limit=1, decisions=fixture.decisions))
        controllers.append(result)
        return result

    def wait(controller):
        events.refresh()
        for attempt, control in controller.pool.test_controls.items():
            if attempt not in clocks:
                control.settimeout(10)
                assert json.loads(control.recv(4096))["timeout_seconds"] == 5400
                clocks.add(attempt)
        for path in layout.path("campaigns").glob("*/workers/*/attempt-0001/work/saved.json"):
            actor = read_document(layout, annotation_reference(layout, path.parent.parent / "request.json"), AgentTask)
            camera = read_document(layout, actor.run, AnnotationRun).context
            from src.tasks.ball_detection.annotation.video import VideoAnnotationInput
            from src.tasks.base.annotation.editing.contracts import (
                BlankAnnotationContext,
            )
            context = read_document(layout, camera, BlankAnnotationContext)
            if read_document(layout, context.task_input, VideoAnnotationInput).source.camera_id != "right":
                continue
            from src.tasks.base.annotation.workers.contracts import Launch
            launch = read_document(layout, annotation_reference(layout, path.parent.parent / "launch.json"), Launch)
            if launch.attempt in timed_out:
                continue
            control = controller.pool.test_controls[launch.attempt]
            control.send(json.dumps({"elapsed_seconds": 5400}).encode())
            assert json.loads(control.recv(4096))["state"] == "timed_out"
            timed_out.add(launch.attempt)
        events.refresh()
        events.wait([child for c in controllers for child in (*c.pool.children.values(), *c.jobs.stages.children.values())], 30)

    orchestrator = TennisSceneOrchestrator(fixture.config, annotation_layout=layout,
        annotation_controller_factory=factory, annotation_wait=wait)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(TennisSceneOrchestrator, "from_runtime_config", lambda cfg: orchestrator)
            assert unwrap(run_pipeline.main)(fixture.hydra_config) == 0
        scene = load_scene_result(fixture.config.output_path)
        np.testing.assert_array_equal(scene.ball_3d_valid, [True, True, True, False])
        np.testing.assert_allclose(scene.ball_3d[:3], fixture.fixed.truth[:3], atol=2e-5)
        assert len(timed_out) == 1 and len(clocks) == 4
        receipt = orchestrator.last_receipt
        assert receipt["annotation"]["status"] == receipt["scene"]["status"] == receipt["dataset"]["status"] == "complete"
        assert receipt["learning"]["status"] == "not_requested"
        assert orchestrator.last_store is not None
        references = orchestrator.last_store.references()
        ball = orchestrator.last_store.load(references["ball_triangulation"], ArtifactCodec(BallAnnotationTriangulationOutput))
        assert ball.estimated[1, 0] and ball.used_views[1, 0] and not ball.observed[1, 0]
        assert ball.missing_reasons[-1] == "insufficient_views"
        publication = read_document(layout, ArtifactRef.model_validate(receipt["dataset"]["receipt"]), PublicationReceipt)
        _, store = load_snapshot(layout, publication.snapshot)
        assert {c.split for c in store.clips} == {"val"} and len({c.group_id for c in store.clips}) == 1
        dataset = ActiveBallWindowDataset(store, split="val", window_length=2)
        assert any(sample["estimated"].any() for sample in dataset)
        completed = read_document(layout, annotation_reference(layout, Path(receipt["annotation"]["receipt"])), CompletedSceneAnnotations)
        measurements = read_document(layout, completed.measurements, BlankRunMeasurements)
        assert {r.worker_kind for r in measurements.results} == {"synthetic"}
        assert measurements.acceptance == "unverified"
        for ref in completed.completions:
            from src.tasks.ball_detection.annotation.video.agent.contracts import (
                VideoCompletion,
            )
            completion = read_document(layout, ref, VideoCompletion)
            assert read_document(layout, completion.run, AnnotationRun).policy == fixture.request.policy
    finally:
        for controller in controllers:
            for ref in controller.production.requests:
                from src.tasks.base.annotation.workers.contracts import WorkRequest
                request = read_document(layout, ref, WorkRequest)
                if controller.pool.ledger.task(request.work_id)["state"] in {"running", "exited"}:
                    controller.pool.stop(request, reason="E2E teardown")
            for child in (*controller.pool.children.values(), *controller.jobs.stages.children.values()):
                child.wait(timeout=15)
            for control in controller.pool.test_controls.values():
                control.close()
        events.close()


if __name__ == "__main__":
    _exercise_direct_scene(Path(sys.argv[1]))
