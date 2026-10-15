from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.execution.contracts import (
    QueueConfiguration,
    StageExit,
    StageLaunch,
)
from src.tasks.base.annotation.execution.queue import (
    enqueue,
    queue_owner,
    queue_state,
    validate_queue,
)
from src.tasks.base.annotation.execution.runner import StageRunner, stage_directory
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    CampaignPlan,
    file_digest,
    references,
    save_plan,
)
from src.tasks.base.annotation.reproducibility import require_frozen_code
from src.tasks.base.annotation.resources import (
    BudgetLimits,
    CapacityUnavailable,
    ResourceManager,
    ResourceRequest,
)
from src.tasks.base.annotation.workers.processes import owned_members, process_snapshot
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def setup(tmp_path: Path) -> tuple[AnnotationLayout, ResourceManager, StageLaunch]:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    ref = write_asset(layout, layout.path("campaigns", "input.txt"), b"fixture")
    plan = CampaignPlan.model_validate_json(
        json.dumps(
            {
                "campaign_id": "execution",
                "phase": "preview",
                "annotation_root": str(layout.root),
                "counts": {
                    "channels": 1,
                    "videos_per_channel": 1,
                    "max_clips_per_video": 1,
                },
                "selection": ref.model_dump(),
                "preparation": ref.model_dump(),
                "code_commit": "a" * 40,
                "code_diff_digest": "b" * 64,
                "agent_model": "gpt-6.1-sol",
                "agent_effort": "high",
                "resources": {
                    "codex_workers": 1,
                    "additional_storage_bytes": 100_000_000,
                    "minimum_free_bytes": 1,
                    "total_pss_bytes": 2 * 1024**3,
                    "gpu_seconds": 60,
                    "stages": [
                        {
                            "stage": name,
                            "kind": name,
                            "parallel": 1,
                            "threads_per_process": 1,
                        }
                        for name in ("cpu", "gpu")
                    ],
                },
                "tasks": [
                    {
                        "task": "fixture",
                        "model": {
                            "task": "fixture",
                            "version": "v1",
                            "training_weights": ref.model_dump(),
                            "inference_weights": ref.model_dump(),
                            "configuration": ref.model_dump(),
                        },
                        "inference": ref.model_dump(),
                        "review": None,
                        "learning": None,
                        "products": [
                            {
                                "product": "frames",
                                "dataset_task": "fixture",
                                "dataset_version": "v1",
                                "component": "frames",
                                "base_snapshot": None,
                                "generation_policy": ref.model_dump(),
                            }
                        ],
                    }
                ],
            }
        )
    )
    manager = ResourceManager.create(
        layout,
        scope_id="execution",
        limits=BudgetLimits.from_budget(plan.resources),
        roots=(layout.repository,),
        previously_allocated_bytes=0,
        previously_used_gpu_seconds=0.0,
        max_pending_per_stage=5,
    )
    launch = StageLaunch(
        scope=manager.reference,
        resource=ResourceRequest(
            request_id="operation",
            plan=annotation_reference(layout, save_plan(layout, plan)),
            stage="cpu",
            pss_bytes=100_000_000,
            storage_bytes=100_000,
            gpu_seconds=0.0,
        ),
        command=(sys.executable, "-c", "print('completed')"),
        code_root=str(PROJECT_ROOT),
        timeout_seconds=10.0,
        test_root=str(layout.root),
    )
    return layout, manager, launch


def wait_exit(runner: StageRunner, ref: ArtifactRef) -> dict[str, object]:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state: dict[str, object] = runner.status(ref)
        if state["state"] not in {"running", "unfinished", "queued", "draining"}:
            return state
        time.sleep(0.02)
    raise AssertionError("stage supervisor did not exit")


def test_cpu_receipt_resume_and_accounted_runtime_caches(setup: tuple) -> None:
    layout, manager, launch = setup
    input_ref = write_asset(
        layout, layout.path("campaigns", "operation-input"), b"input"
    )
    dependency = write_asset(layout, layout.path("campaigns", "input-media"), b"media")
    code = "import json,os;print(json.dumps({k:os.environ[k] for k in ('TMPDIR','TORCH_HOME','CUDA_CACHE_PATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS')}))"
    launch = launch.model_copy(
        update={
            "command": (sys.executable, "-c", code),
            "input": input_ref,
            "dependencies": (dependency,),
        }
    )
    runner = StageRunner(layout)
    result = runner.submit(launch)
    ref = ArtifactRef.model_validate(result["launch"])
    assert wait_exit(runner, ref)["state"] == "succeeded"
    directory = stage_directory(layout, launch)
    receipt = read_document(
        layout, annotation_reference(layout, directory / "exit.json"), StageExit
    )
    environment = json.loads(layout.artifact_path(receipt.stdout).read_text())
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["OMP_NUM_THREADS"] == "1"
    assert all(
        environment[k].startswith(str(layout.root / "cache/runtime"))
        for k in ("TMPDIR", "TORCH_HOME", "CUDA_CACHE_PATH")
    )
    before = (directory / "exit.json").read_bytes()
    assert StageRunner(layout).submit(launch)["state"] == "succeeded"
    assert (directory / "exit.json").read_bytes() == before
    assert manager.sample().gpu_charged_seconds == 0
    retained = {
        reference
        for record in manager.storage.catalog.snapshot().protections
        for reference in record.references
    }
    assert {ref, input_ref, dependency} <= retained


def test_queue_wait_protects_launch_and_explicit_input_dependencies(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manager, launch = setup
    input_ref = write_asset(
        layout, layout.path("campaigns", "operation-input"), b"input"
    )
    dependency = write_asset(layout, layout.path("campaigns", "input-media"), b"media")
    launch = launch.model_copy(
        update={
            "queue": queue_config(layout),
            "resource": launch.resource.model_copy(
                update={"stage": "gpu", "gpu_seconds": 30.0}
            ),
            "input": input_ref,
            "dependencies": (dependency,),
        }
    )

    def queued(
        layout: AnnotationLayout,
        config: QueueConfiguration,
        reference: ArtifactRef,
        code: Path,
    ) -> str:
        assert manager._row(launch.resource)["state"] == "pending"
        protections = manager.storage.catalog.snapshot().protections
        retained = {ref for record in protections for ref in record.references}
        assert {reference, input_ref, dependency, manager.scope.baseline} <= retained
        assert not any(
            record.protection_id.startswith("resource-attempt-v1-")
            for record in protections
        )
        return "fixture.job"

    monkeypatch.setattr("src.tasks.base.annotation.execution.runner.enqueue", queued)
    result = StageRunner(layout).submit(launch)
    assert result["state"] == "queued"
    before = manager.storage.catalog.snapshot().digest
    assert StageRunner(layout).submit(launch) == result
    assert manager.storage.catalog.snapshot().digest == before


def test_missing_explicit_dependency_rejects_stage_before_admission(
    setup: tuple,
) -> None:
    layout, manager, launch = setup
    dependency = write_asset(layout, layout.path("campaigns", "input-media"), b"media")
    layout.artifact_path(dependency).unlink()
    launch = launch.model_copy(update={"dependencies": (dependency,)})
    with pytest.raises(ValueError, match="artifact missing"):
        StageRunner(layout).submit(launch)
    with pytest.raises(ValueError, match="missing"):
        manager._row(launch.resource)


def test_timeout_reaps_all_operation_descendants(setup: tuple) -> None:
    layout, manager, launch = setup
    script = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);time.sleep(60)"
    launch = launch.model_copy(
        update={"command": (sys.executable, "-c", script), "timeout_seconds": 0.3}
    )
    runner = StageRunner(layout)
    ref = ArtifactRef.model_validate(runner.submit(launch)["launch"])
    assert wait_exit(runner, ref)["state"] == "timed_out"
    assert manager.sample().active_by_stage == {}
    assert runner.submit(launch)["state"] == "timed_out"


def test_stage_capacity_and_missing_receipt_are_explicit(setup: tuple) -> None:
    layout, manager, launch = setup
    marker = layout.repository / "operation.pid"
    code = "import os,sys,time;from pathlib import Path;p=Path(sys.argv[1]);t=p.with_suffix('.pending');t.write_text(str(os.getpid()));t.replace(p);time.sleep(60)"
    launch = launch.model_copy(
        update={"command": (sys.executable, "-c", code, str(marker))}
    )
    runner = StageRunner(layout)
    first = ArtifactRef.model_validate(runner.submit(launch)["launch"])
    second = launch.model_copy(
        update={"resource": launch.resource.model_copy(update={"request_id": "second"})}
    )
    with pytest.raises(CapacityUnavailable):
        runner.submit(second)
    process = runner.children[first.digest]
    snap = process_snapshot(process.pid)
    assert snap is not None
    from src.tasks.base.annotation.workers.processes import signal_owned

    try:
        # owner.json precedes Popen: wait for the operation itself to acknowledge
        # startup so the kill snapshot cannot miss a concurrently spawned child.
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        operation = process_snapshot(int(marker.read_text()))
        assert operation is not None and operation[0] in owned_members(snap[0])
    finally:
        signal_owned(snap[0], signal.SIGKILL)
        process.wait(timeout=5)
    # wait() reaps only the supervisor, not its separately killed child.
    deadline = time.monotonic() + 5
    while owned_members(snap[0]) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not owned_members(snap[0])
    assert wait_exit(runner, first)["state"] == "failed"
    assert StageRunner(layout).status(first)["state"] == "interrupted"
    assert manager.sample().gpu_charged_seconds == 0


def queue_config(layout: AnnotationLayout) -> QueueConfiguration:
    script = layout.repository / "queue.sh"
    script.write_text("# fixture queue protocol --idempotency-key\n")
    return QueueConfiguration(
        script=str(script),
        script_digest=file_digest(script),
        directory=str(layout.repository / ".training_queue"),
        session="fixture-session",
        issue=1088,
    )


def test_queue_registration_is_pinned_and_idempotent(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _manager, launch = setup
    config = queue_config(layout)
    ref = write_document(
        layout, stage_directory(layout, launch) / "launch.json", launch
    )
    calls: list[list[str]] = []

    def register(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        assert command[command.index("--resource") + 1] == "all"
        assert command[command.index("--session") + 1] == config.session
        assert kwargs["cwd"] == PROJECT_ROOT
        key = command[command.index("--idempotency-key") + 1]
        receipt = (
            Path(config.directory)
            / "control/registrations"
            / (hashlib.sha256(key.encode()).hexdigest() + ".receipt")
        )
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("a" * 64 + " synthetic.job\n")
        pending = Path(config.directory) / "jobs/synthetic.job"
        pending.parent.mkdir(exist_ok=True)
        pending.write_text("fixture")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "src.tasks.base.annotation.execution.queue.subprocess.run", register
    )
    assert enqueue(layout, config, ref, PROJECT_ROOT) == "synthetic.job"
    assert enqueue(layout, config, ref, PROJECT_ROOT) == "synthetic.job"
    assert calls[0] == calls[1]
    assert queue_state(config, ref)["state"] == "queued"
    Path(config.script).write_text("changed --idempotency-key")
    with pytest.raises(ValueError, match="changed"):
        validate_queue(layout, config)


def test_gpu_stage_requires_queue_and_keeps_queue_process_group(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A synthetic GPU reservation: no CUDA import, inference or training occurs here.
    layout, manager, launch = setup
    config = queue_config(layout)
    monkeypatch.delenv("TENNIS_RUN_ID", raising=False)
    with pytest.raises(ValueError, match="job identity"):
        queue_owner(layout, config)
    launch = launch.model_copy(
        update={
            "queue": config,
            "resource": launch.resource.model_copy(
                update={"stage": "gpu", "gpu_seconds": 30.0}
            ),
            "command": (
                sys.executable,
                "-c",
                "import os,json;print(json.dumps({'pgid':os.getpgrp(),'sid':os.getsid(0)}))",
            ),
        }
    )
    ref = write_document(
        layout, stage_directory(layout, launch) / "launch.json", launch
    )
    manager.submit(launch.resource, inputs=(ref, *references(launch)))
    repro = Path(config.directory) / "repro/synthetic"
    repro.mkdir(parents=True)
    (repro / "run.json").write_text(
        json.dumps({"provider": "codex", "session": config.session, "resource": "all"})
    )
    environment = {
        **os.environ,
        "TENNIS_RUN_ID": "synthetic",
        "TENNIS_REPRO_DIR": str(repro),
        "TENNIS_GPU_RESOURCE": "all",
    }
    command = [
        sys.executable,
        "-m",
        "src.tasks.base.annotation.execution",
        "--launch",
        str(layout.artifact_path(ref)),
        "--sha256",
        ref.digest,
        "--queued",
        "--test-root",
        str(layout.root),
    ]
    wrapper = "import subprocess,sys;sys.exit(subprocess.call(sys.argv[1:]))"
    owner = subprocess.Popen(
        [sys.executable, "-c", wrapper, *command],
        env=environment,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = owner.communicate(timeout=15)
    assert owner.returncode == 0, (stdout, stderr)
    assert StageRunner(layout).status(ref)["state"] == "succeeded"
    receipt = read_document(
        layout,
        annotation_reference(layout, stage_directory(layout, launch) / "exit.json"),
        StageExit,
    )
    identity = json.loads(layout.artifact_path(receipt.stdout).read_text())
    assert identity == {"pgid": owner.pid, "sid": owner.pid}
    assert 0 < manager.sample().gpu_charged_seconds < 30


def test_frozen_code_rejects_dirty_commit_and_untracked_sources(
    setup: tuple, tmp_path: Path
) -> None:
    layout, _manager, launch = setup
    plan = read_document(layout, launch.resource.plan, CampaignPlan)
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "source.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    plan = plan.model_copy(
        update={
            "code_commit": commit,
            "code_diff_digest": hashlib.sha256(b"").hexdigest(),
        }
    )
    require_frozen_code(root, plan)
    (root / "extra.py").write_text("unexpected = True\n")
    with pytest.raises(ValueError, match="source checkout"):
        require_frozen_code(root, plan)
    (root / "extra.py").unlink()
    (root / "source.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="source checkout"):
        require_frozen_code(root, plan)
