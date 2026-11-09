from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.tasks.annotation import scene_workflow
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    load_snapshot,
)
from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import ArtifactRef, PublicationReceipt
from src.tennis_scene.pipeline import TennisSceneOrchestrator
from src.utils.configuration import PathRole
from src.utils.paths import PROJECT_ROOT
from tests.support.tennis_scene.direct_workflow import (
    complete_annotations,
    direct_fixture,
)


@pytest.mark.parametrize("publish", [False, True])
def test_optional_publication_and_receipt_interruption_resume_only_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publish: bool
) -> None:
    fixture = direct_fixture(tmp_path, publish=publish)
    workflow = complete_annotations(fixture)
    before = {
        ref: fixture.layout.artifact_path(ref).read_bytes()
        for ref in workflow.saved().completions
    }
    # These integration cases begin with saved model judgments. The E2E owns
    # actual workers/services and timeout continuation; no need to repeat them.
    forbidden = Mock(
        side_effect=AssertionError("Completed annotation restarted a worker")
    )
    orchestrator = TennisSceneOrchestrator(
        fixture.config,
        annotation_layout=fixture.layout,
        annotation_controller_factory=forbidden,
    )
    args = dict(
        video_paths=fixture.config.video_paths,
        camera_ids=fixture.config.camera_ids,
        video_role=PathRole.DATA,
        store_root=fixture.store,
    )
    writer = scene_workflow.write_document
    if publish:

        def interrupted(layout, path, document):
            if path.name == "publication.json":
                raise InterruptedError(
                    "After publisher commit, before workflow receipt"
                )
            return writer(layout, path, document)

        with monkeypatch.context() as patch:
            patch.setattr(scene_workflow, "write_document", interrupted)
            with pytest.raises(InterruptedError, match="publisher commit"):
                orchestrator.run(**args)
        status = json.loads((fixture.store / "run.json").read_text())
        assert status["annotation"]["status"] == status["scene"]["status"] == "complete"
        assert status["dataset"]["status"] == "failed"
        assert not (workflow.directory / "publication.json").exists()
        assert (
            fixture.layout.repository
            / "data/ball_detection/tiny-active/frames/snapshot.json"
        ).exists()
        assert fixture.config.direct_annotation is not None
        resumed = replace(
            fixture.config,
            direct_annotation=replace(
                fixture.config.direct_annotation, resume="publication"
            ),
        )
        orchestrator = TennisSceneOrchestrator(
            resumed,
            annotation_layout=fixture.layout,
            annotation_controller_factory=forbidden,
        )
        monkeypatch.setattr(
            "src.tennis_scene.pipeline.orchestrator.standard_definition",
            Mock(side_effect=AssertionError("Publication recovery rebuilt scene")),
        )
    scene = orchestrator.run(**args)
    assert scene.ball_3d_valid.tolist() == [True, True, True, False]
    status = orchestrator.last_receipt
    assert status["dataset"]["status"] == ("complete" if publish else "not_requested")
    assert status["learning"]["status"] == "not_requested"
    if publish:
        receipt = ArtifactRef.model_validate(status["dataset"]["receipt"])
        publication = read_document(fixture.layout, receipt, PublicationReceipt)
        _, store = load_snapshot(fixture.layout, publication.snapshot)
        assert len(store.clips) == 3 and len(store) == 12
        orchestrator.run(**args)
        assert orchestrator.last_receipt["dataset"]["receipt"] == receipt.model_dump(
            mode="json"
        )
    else:
        assert not (
            fixture.layout.repository / "data/ball_detection/tiny-active"
        ).exists()
    forbidden.assert_not_called()
    for ref, payload in before.items():
        assert fixture.layout.artifact_path(ref).read_bytes() == payload


def test_blank_cli_pins_policy_runs_isolated_condition_and_reports_without_publication(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # ResourceManager observes its controller PID. Full-suite xdist workers
    # retain unrelated model/rendering memory, which is not this campaign's
    # baseline. Match the clean CPU process in test_automatic_continuation,
    # retaining the real scheduler and the fixed 1536 MiB budget.
    parent_pss = _process_pss()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.integration.tasks.annotation.test_scene_workflow",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        timeout=180,
        start_new_session=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    request.node.user_properties.append(
        (
            "resource_controller",
            json.dumps(
                {"pytest_pid": os.getpid(), "pytest_pss_bytes": parent_pss, **report}
            ),
        )
    )
    assert report["controller_pid"] != os.getpid()
    assert os.getpid() not in report["measured_pids"]
    assert report["controller_pss_bytes"] < report["pss_limit_bytes"]


def _process_pss() -> int:
    return next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines()
        if line.startswith("Pss:")
    )


def _exercise_blank_cli(tmp_path: Path) -> dict[str, object]:
    import time

    from src.tasks.annotation.__main__ import main
    from src.tasks.annotation.benchmark.blank import (
        BlankComparison,
        BlankComparisonRequest,
        BlankCondition,
        BlankConditionRegistration,
    )
    from src.tasks.annotation.configuration import configure_local_videos
    from src.tasks.annotation.contracts import LocalAnnotationRuntime
    from src.tasks.annotation.local_controller import LocalAnnotationController
    from src.tasks.annotation.local_videos import LocalVideoConfigurationReceipt
    from src.tasks.ball_detection.annotation.video.agent.contracts import (
        AnnotationPolicy,
    )
    from src.tasks.base.annotation.artifacts import write_document
    from src.tasks.base.annotation.paths import AnnotationLayout
    from tests.support.tasks.annotation_events import AnnotationEvents
    from tests.support.tasks.blank_agent import fake_worker
    from tests.support.tasks.local_annotation import local_configuration

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
    registration = read_document(
        layout, configure_local_videos(layout, cfg), LocalVideoConfigurationReceipt
    )

    def cli(arguments: list[str]) -> str:
        output = StringIO()
        with (
            patch.object(sys, "argv", ["annotation", *arguments]),
            redirect_stdout(output),
        ):
            main()
        return output.getvalue()

    def call(command, document, name):
        ref = write_document(
            layout, layout.path("campaigns", "cli-inputs", name + ".json"), document
        )
        return ArtifactRef.model_validate_json(
            cli(
                [
                    command,
                    "--operation",
                    str(layout.artifact_path(ref)),
                    "--sha256",
                    ref.digest,
                    "--test-root",
                    str(layout.root),
                ]
            ).strip()
        )

    conditions = tuple(
        call(
            "blank-register",
            BlankConditionRegistration(
                source_plan=registration.plan,
                condition_id=name,
                policy=AnnotationPolicy(
                    policy_id=name,
                    version="1",
                    instructions="Choose camera anchors and revise judgments.",
                ),
            ),
            name,
        )
        for name in ("chronological", "adaptive")
    )
    condition = read_document(layout, conditions[0], BlankCondition)
    local = layout.for_benchmark("chronological")
    runtime = read_document(local, condition.runtime, LocalAnnotationRuntime)
    controller = LocalAnnotationController(
        local, runtime, annotation_policy=condition.policy, test_command=fake_worker()
    )
    baseline = controller.resources.sample()
    proc_pss = _process_pss()
    events = AnnotationEvents(layout.path("campaigns"))
    deadline = time.monotonic() + 90
    try:
        while True:
            status = controller.cycle()
            if status["state"] == "complete":
                break
            assert status["state"] != "blocked" and time.monotonic() < deadline, status
            events.refresh()
            events.wait(
                [
                    *controller.pool.children.values(),
                    *controller.jobs.stages.children.values(),
                ],
                30,
            )
        result = call(
            "blank-compare", BlankComparisonRequest(conditions=conditions), "comparison"
        )
        comparison = read_document(layout, result, BlankComparison)
        assert comparison.results[0][0].worker_kind == "synthetic"
        assert comparison.results[1][0].worker_kind == "not_started"
        assert comparison.acceptance == "unverified"
        status_output = cli(
            [
                "status",
                "--runtime",
                str(layout.artifact_path(condition.runtime)),
                "--sha256",
                condition.runtime.digest,
                "--test-root",
                str(layout.root),
            ],
        )
        assert json.loads(status_output)["state"] == "complete"
        assert not (layout.repository / "data/ball_detection").exists()
        with pytest.raises(ValueError, match="comparison runs"):
            scene_workflow.SceneAnnotationWorkflow(local, Mock())
    finally:
        from src.tasks.base.annotation.workers.contracts import WorkRequest

        for ref in controller.production.requests:
            request = read_document(local, ref, WorkRequest)
            if controller.pool.ledger.task(request.work_id)["state"] in {
                "running",
                "exited",
            }:
                controller.pool.stop(request, reason="comparison test teardown")
        for child in (
            *controller.pool.children.values(),
            *controller.jobs.stages.children.values(),
        ):
            child.wait(timeout=15)
        events.close()
    return {
        "controller_pid": os.getpid(),
        "controller_pss_bytes": baseline.pss_bytes,
        "controller_proc_pss_bytes": proc_pss,
        "measured_pids": [process.pid for process in baseline.processes],
        "pss_limit_bytes": cfg.campaign.resources.total_pss_bytes,
    }


if __name__ == "__main__":
    print(json.dumps(_exercise_blank_cli(Path(sys.argv[1]))))
