from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Event
from typing import Any

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
from src.tasks.base.annotation.execution.runner import (
    StageRunner,
    stage_directory,
    stage_lock,
)
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
from src.tasks.base.annotation.resources.measurements import process_usage
from src.tasks.base.annotation.workers.processes import owned_members, process_snapshot
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def setup(tmp_path: Path) -> tuple[AnnotationLayout, ResourceManager, StageLaunch]:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    controller = process_snapshot(os.getpid())
    assert controller is not None
    controller_pss = sum(
        item.pss_bytes for item in process_usage((), (controller[0],))
    )
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
                    # StageRunner observes this pytest process. Keep the same
                    # reservation headroom after imports from the full suite.
                    "total_pss_bytes": controller_pss + 2 * 1024**3,
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
        if state["state"] not in {
            "running",
            "unfinished",
            "queued",
            "draining",
            "waiting",
        }:
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


def test_watchdog_emergency_terminates_the_stage_child(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manager, launch = setup
    original = subprocess.Popen

    def with_emergency(command: list[str], **kwargs: Any) -> Any:
        if command[1:3] == ["-m", "src.tasks.base.annotation.execution"]:
            injected = (
                "import runpy; from src.tasks.base.annotation.resources import ResourceManager, BudgetStopped\n"
                "def emergency(self): raise BudgetStopped('synthetic observed host space emergency')\n"
                "ResourceManager.watch = emergency\n"
                "runpy.run_module('src.tasks.base.annotation.execution', run_name='__main__')\n"
            )
            command = [command[0], "-c", injected, *command[3:]]
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", with_emergency)
    launch = launch.model_copy(
        update={
            "command": (sys.executable, "-c", "import time; time.sleep(60)"),
            "timeout_seconds": 90.0,
        }
    )
    runner = StageRunner(layout)
    result = runner.submit(launch)
    ref = ArtifactRef.model_validate(result["launch"])
    state = wait_exit(runner, ref)
    assert state["state"] == "failed"
    directory = stage_directory(layout, launch)
    receipt = read_document(
        layout, annotation_reference(layout, directory / "exit.json"), StageExit
    )
    assert "synthetic observed host space emergency" in receipt.reason
    assert receipt.elapsed_seconds < 15
    from src.tasks.base.annotation.workers.processes import ProcessIdentity

    owner = ProcessIdentity.from_document(
        json.loads((directory / "owner.json").read_text())
    )
    assert not owned_members(owner)
    assert manager.sample().active_by_stage == {}


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
        wait_for(marker.exists)
        operation = process_snapshot(int(marker.read_text()))
        assert operation is not None and operation[0] in owned_members(snap[0])
    finally:
        signal_owned(snap[0], signal.SIGKILL)
        process.wait(timeout=5)
    # wait() reaps only the supervisor, not its separately killed child.
    wait_for(lambda: not owned_members(snap[0]))
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


def live_queue_config(
    layout: AnnotationLayout, monkeypatch: pytest.MonkeyPatch
) -> QueueConfiguration:
    config = queue_config(layout)
    script = Path(config.script)
    script.write_bytes(
        (
            PROJECT_ROOT / ".agents/skills/training-queue/scripts/training_queue.sh"
        ).read_bytes()
    )
    # Every namespace, including the system-lock probe, belongs to this test.
    for name in tuple(os.environ):
        if name.startswith("TRAINING_QUEUE_TEST_"):
            monkeypatch.delenv(name)
    for name, value in {
        "TRAINING_QUEUE_DIR": config.directory,
        "TRAINING_QUEUE_LOCK_FILE": str(layout.repository / "gpu.lock"),
        "TRAINING_QUEUE_SYSTEM_LOCK_FILE": str(layout.repository / "system.lock"),
        "TRAINING_QUEUE_TEST_TERM_GRACE_SECONDS": "1",
        "GIT_CEILING_DIRECTORIES": str(PROJECT_ROOT),
        "CUDA_VISIBLE_DEVICES": "",
    }.items():
        monkeypatch.setenv(name, value)
    binaries = layout.repository / "bin"
    binaries.mkdir()
    telemetry = binaries / "nvidia-smi"
    telemetry.write_text(
        "#!/bin/sh\necho 'synthetic test: no GPU telemetry' >&2\nexit 1\n"
    )
    telemetry.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    return config.model_copy(update={"script_digest": file_digest(script)})


def wait_for(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("queue transition did not complete")


@contextmanager
def running_queue(
    layout: AnnotationLayout, config: QueueConfiguration
) -> Iterator[None]:
    with (layout.repository / "worker.log").open("wb") as log:
        worker = subprocess.Popen(
            ["bash", config.script, "serve", "--idle-timeout", "30"],
            cwd=layout.repository,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            wait_for(lambda: (Path(config.directory) / "worker.pid").exists())
            yield
        finally:
            worker.terminate()
            worker.wait(timeout=15)


def gpu_launch(launch: StageLaunch, config: QueueConfiguration) -> StageLaunch:
    return launch.model_copy(
        update={
            "queue": config,
            "resource": launch.resource.model_copy(
                update={"stage": "gpu", "gpu_seconds": 30.0}
            ),
        }
    )


@pytest.mark.parametrize("capacity", ["pss", "storage"])
def test_queued_capacity_wait_reuses_unexecuted_request(
    setup: tuple, monkeypatch: pytest.MonkeyPatch, capacity: str
) -> None:
    layout, manager, launch = setup
    config = live_queue_config(layout, monkeypatch)
    reservation = launch.resource.model_copy(
        update={
            "request_id": "reservation",
            "pss_bytes": 1_500_000_000 if capacity == "pss" else 100_000_000,
            "storage_bytes": 70_000_000 if capacity == "storage" else 100_000,
        }
    )
    launch = gpu_launch(launch, config)
    launch = launch.model_copy(
        update={
            "resource": launch.resource.model_copy(
                update={
                    "pss_bytes": 1_000_000_000 if capacity == "pss" else 100_000_000,
                    "storage_bytes": 40_000_000 if capacity == "storage" else 100_000,
                }
            )
        }
    )
    runner = StageRunner(layout)
    with ExitStack() as reservations:
        reservations.enter_context(manager.lease(reservation))
        result = runner.submit(launch)
        ref = ArtifactRef.model_validate(result["launch"])
        job = result["queue_job"]
        with running_queue(layout, config):
            wait_for(
                lambda: (
                    runner.status(ref)["state"] == "failed"
                    or "reservations do not fit"
                    in (runner.status(ref).get("reason") or "")
                )
            )
            state = runner.status(ref)
            assert state["state"] == "waiting", state
            assert "reservations do not fit" in state["reason"]
            assert not (stage_directory(layout, launch) / "owner.json").exists()
            assert not (stage_directory(layout, launch) / "exit.json").exists()
            assert (Path(config.directory) / "running" / job).exists()
            reservations.close()
            assert wait_exit(runner, ref)["state"] == "succeeded"
            wait_for(lambda: (Path(config.directory) / "done" / job).exists())
            assert runner.submit(launch)["state"] == "succeeded"
    assert (
        len(list((Path(config.directory) / "control/registrations").glob("*.receipt")))
        == 1
    )
    assert not list((Path(config.directory) / "failed").glob("*.job"))
    assert (stage_directory(layout, launch) / "stdout.log").read_text() == "completed\n"


@pytest.mark.parametrize("action", ["budget_stop", "resource_cancel", "queue_cancel"])
def test_queued_capacity_wait_does_not_retry_stop_or_cancel(
    setup: tuple, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    layout, manager, launch = setup
    config = live_queue_config(layout, monkeypatch)
    reservation = launch.resource.model_copy(
        update={"request_id": "reservation", "storage_bytes": 70_000_000}
    )
    launch = gpu_launch(launch, config)
    launch = launch.model_copy(
        update={
            "resource": launch.resource.model_copy(update={"storage_bytes": 40_000_000})
        }
    )
    runner = StageRunner(layout)
    with manager.lease(reservation):
        result = runner.submit(launch)
        ref = ArtifactRef.model_validate(result["launch"])
        job = result["queue_job"]
        with running_queue(layout, config):
            wait_for(
                lambda: (
                    manager.request_status(launch.resource)
                    == ("pending", "storage reservations do not fit")
                )
            )
            assert runner.status(ref)["state"] == "waiting"
            if action == "budget_stop":
                # Simulate an already-persisted watchdog stop, not the separate
                # admission/watchdog transaction race covered by another issue.
                with manager.registry.transaction() as db:
                    db.execute(
                        "UPDATE resource_scopes SET stop_reason=? WHERE id=?",
                        ("synthetic budget stop", manager.scope.scope_id),
                    )
            elif action == "resource_cancel":
                manager.cancel_pending(launch.resource, reason="synthetic cancellation")
            else:
                subprocess.run(
                    ["bash", config.script, "cancel", job],
                    cwd=layout.repository,
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
            expected = "cancelled" if action == "queue_cancel" else "failed"
            assert wait_exit(runner, ref)["state"] == expected
            assert (Path(config.directory) / expected / job).exists()
            directory = stage_directory(layout, launch)
            assert not (directory / "owner.json").exists()
            assert not (directory / "stdout.log").exists()
            assert not (directory / "exit.json").exists()
            state, reason = manager.request_status(launch.resource)
            if action == "budget_stop":
                assert (state, reason) == ("pending", "synthetic budget stop")
                log = (
                    Path(config.directory) / "logs" / (Path(job).stem + ".log")
                ).read_text()
                assert "BudgetStopped: synthetic budget stop" in log
            elif action == "resource_cancel":
                assert (state, reason) == ("cancelled", "synthetic cancellation")
    assert manager.sample().gpu_charged_seconds == 0


def test_live_queue_waits_for_registration_lock_handoff(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _manager, launch = setup
    config = live_queue_config(layout, monkeypatch)
    launch = gpu_launch(launch, config)
    runner = StageRunner(layout)
    registered, release = Event(), Event()

    def hold_registration(
        layout: AnnotationLayout,
        config: QueueConfiguration,
        ref: ArtifactRef,
        code_root: Path,
    ) -> str:
        job: str = enqueue(layout, config, ref, code_root)
        registered.set()
        assert release.wait(timeout=15), "registration was not released"
        return job

    monkeypatch.setattr(
        "src.tasks.base.annotation.execution.runner.enqueue", hold_registration
    )
    with running_queue(layout, config), ThreadPoolExecutor(max_workers=1) as pool:
        submitted = pool.submit(runner.submit, launch)
        try:
            assert registered.wait(timeout=10)
            ref = annotation_reference(
                layout, stage_directory(layout, launch) / "launch.json"
            )
            lock = layout.path("cache", "locks", stage_lock(launch) + ".lock")
            inode = lock.stat().st_ino

            def blocked_on_registration() -> bool:
                return any(
                    "-> FLOCK" in line and f":{inode} " in line
                    for line in Path("/proc/locks").read_text().splitlines()
                )

            wait_for(
                lambda: (
                    blocked_on_registration()
                    or queue_state(config, ref)["state"] == "failed"
                )
            )
            assert blocked_on_registration(), queue_state(config, ref)
            assert not (stage_directory(layout, launch) / "owner.json").exists()
        finally:
            release.set()
        result = submitted.result(timeout=10)
        assert wait_exit(runner, ref)["state"] == "succeeded"
        job = result["queue_job"]
        wait_for(lambda: (Path(config.directory) / "done" / job).exists())
        assert runner.submit(launch)["state"] == "succeeded"
    assert (stage_directory(layout, launch) / "stdout.log").read_text() == "completed\n"


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


def registered_queue(
    layout: AnnotationLayout, launch: StageLaunch
) -> tuple[QueueConfiguration, ArtifactRef]:
    config = queue_config(layout)
    ref = write_document(
        layout, stage_directory(layout, launch) / "launch.json", launch
    )
    queue = Path(config.directory)
    registration = (
        queue
        / "control/registrations"
        / (
            hashlib.sha256(("annotation-stage:" + ref.digest).encode()).hexdigest()
            + ".receipt"
        )
    )
    registration.parent.mkdir(parents=True)
    registration.write_text("a" * 64 + " moving.job\n")
    for state in ("jobs", "running", "done", "failed", "cancelled"):
        (queue / state).mkdir()
    return config, ref


@pytest.mark.parametrize(
    ("transitions", "expected"),
    [
        ([("jobs", "running")], "running"),
        ([("running", "jobs")], "running"),
        ([("jobs", "running"), ("running", "done")], "failed"),
    ],
)
def test_queue_state_reconciles_job_moves_during_observation(
    setup: tuple,
    monkeypatch: pytest.MonkeyPatch,
    transitions: list[tuple[str, str]],
    expected: str,
) -> None:
    layout, _manager, launch = setup
    config, ref = registered_queue(layout, launch)
    queue = Path(config.directory)
    (queue / transitions[0][0] / "moving.job").write_text("fixture; never executed")
    real_exists = Path.exists
    moves: list[str] = []

    def move_after_observation(path: Path) -> bool:
        exists = real_exists(path)
        # Model the worker's atomic renames immediately after a positive stat.
        if exists and len(moves) < len(transitions):
            source, target = transitions[len(moves)]
            if path == queue / source / "moving.job":
                path.rename(queue / target / "moving.job")
                moves.append(target)
        return exists

    monkeypatch.setattr(Path, "exists", move_after_observation)
    result = queue_state(config, ref)
    assert moves == [target for _source, target in transitions]
    assert result["state"] == expected
    if expected == "failed":
        assert "queue is done" in result["reason"]
    else:
        # A single observation may precede the rename; it remains nonterminal.
        current = "queued" if transitions[-1][1] == "jobs" else "running"
        assert queue_state(config, ref)["state"] == current


def test_queue_state_reports_continuous_moves_as_unfinished(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _manager, launch = setup
    config, ref = registered_queue(layout, launch)
    queue = Path(config.directory)
    (queue / "jobs/moving.job").write_text("fixture; never executed")
    real_exists = Path.exists

    def move_on_every_observation(path: Path) -> bool:
        exists = real_exists(path)
        if exists and path in (queue / "jobs/moving.job", queue / "running/moving.job"):
            target = "running" if path.parent.name == "jobs" else "jobs"
            path.rename(queue / target / path.name)
        return exists

    monkeypatch.setattr(Path, "exists", move_on_every_observation)
    result = queue_state(config, ref)
    assert result["state"] == "unfinished"
    assert "no unique lifecycle state" in result["reason"]


@pytest.mark.parametrize("states", [(), ("jobs", "running")])
def test_queue_state_does_not_hide_stable_inconsistency(
    setup: tuple, states: tuple[str, ...]
) -> None:
    layout, _manager, launch = setup
    config, ref = registered_queue(layout, launch)
    for state in states:
        (Path(config.directory) / state / "moving.job").write_text("fixture")
    result = queue_state(config, ref)
    assert result["state"] == "unfinished"
    assert "inspect queue if this persists" in result["reason"]


def test_stage_status_collects_receipt_published_after_initial_check(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _manager, launch = setup
    directory = stage_directory(layout, launch)
    ref = write_document(layout, directory / "launch.json", launch)
    stdout = write_asset(layout, directory / "stdout.log", b"completed")
    stderr = write_asset(layout, directory / "stderr.log", b"")
    receipt = StageExit(
        launch=ref,
        status="succeeded",
        exit_code=0,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
        elapsed_seconds=1.0,
        stdout=stdout,
        stderr=stderr,
        reason=None,
    )
    from src.tasks.base.annotation.execution import runner

    real_lock = runner.artifact_lock
    published = False

    @contextmanager
    def finish_before_lock(
        root: AnnotationLayout, name: str, *, blocking: bool = True
    ) -> Iterator[int]:
        nonlocal published
        if name == runner.stage_lock(launch) and not published:
            published = True
            write_document(layout, directory / "exit.json", receipt)
        with real_lock(root, name, blocking=blocking) as descriptor:
            yield descriptor

    monkeypatch.setattr(runner, "artifact_lock", finish_before_lock)
    state = StageRunner(layout).status(ref)
    assert published
    assert state["state"] == "succeeded"
    assert (
        read_document(layout, ArtifactRef.model_validate(state["exit"]), StageExit)
        == receipt
    )
    with real_lock(layout, runner.stage_lock(launch)):
        assert StageRunner(layout).status(ref)["state"] == "draining"
    assert StageRunner(layout).status(ref)["state"] == "succeeded"


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
