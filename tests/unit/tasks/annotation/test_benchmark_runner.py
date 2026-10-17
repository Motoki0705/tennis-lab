from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tasks.annotation.benchmark import runner as runner_module
from src.tasks.annotation.benchmark.contracts import (
    BenchmarkRuntime,
    ConditionCompletion,
)
from src.tasks.annotation.benchmark.operations import execute
from src.tasks.annotation.benchmark.recovery import recover
from src.tasks.annotation.benchmark.runner import BenchmarkRunner
from src.tasks.annotation.contracts import (
    OperationInput,
    PublishOperation,
    RecoveryInput,
    ResumeWorkers,
    RetryLearning,
    RuntimeSettings,
    StageCost,
)
from src.tasks.ball_detection.annotation.inference import BallPredictionAdapter
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
)
from src.tasks.base.annotation.benchmarks.capture import directory
from src.tasks.base.annotation.benchmarks.contracts import (
    ComparisonReport,
    ComparisonSpec,
    ConditionSpec,
    Observation,
    WorkerBinding,
)
from src.tasks.base.annotation.benchmarks.report import build_report
from src.tasks.base.annotation.execution.contracts import QueueConfiguration
from src.tasks.base.annotation.plan import (
    CampaignPlan,
    StageCapacity,
    file_digest,
    save_plan,
)
from src.tasks.base.annotation.resources import (
    BudgetLimits,
    CapacityUnavailable,
    ResourceManager,
    ResourceRequest,
)
from src.tasks.base.annotation.resources.contracts import GPUUsage
from src.tasks.base.annotation.workers.contracts import Launch, WorkRequest
from src.tasks.base.annotation.workers.ledger import WorkerBusy
from src.tasks.base.annotation.workers.pool import WorkerPool
from src.tasks.base.annotation.workers.processes import (
    ProcessIdentity,
    owned_members,
    signal_owned,
)
from src.utils.paths import PROJECT_ROOT
from tests.support.tasks.ball_detection.annotation import Environment


class Setup:
    def __init__(
        self, root: Path, observation_limit: int = 1000, *, pending_limit: int = 50
    ) -> None:
        def transform(plan: CampaignPlan) -> CampaignPlan:
            return plan.model_copy(
                update={
                    "resources": plan.resources.model_copy(
                        update={
                            "total_pss_bytes": 8 * 1024**3,
                            "stages": (
                                StageCapacity(
                                    stage="review",
                                    kind="codex",
                                    parallel=1,
                                    threads_per_process=1,
                                ),
                                StageCapacity(
                                    stage="render",
                                    kind="cpu",
                                    parallel=2,
                                    threads_per_process=1,
                                ),
                            ),
                        }
                    )
                }
            )

        self.env = Environment(root, plan_transform=transform)
        env = self.env
        self.manager = ResourceManager.create(
            env.layout,
            scope_id="shared-budget",
            limits=BudgetLimits.from_budget(env.plan.resources),
            roots=(env.layout.repository,),
            previously_allocated_bytes=0,
            previously_used_gpu_seconds=3,
            max_pending_per_stage=pending_limit,
        )
        self.conditions = []
        for number in (10, 25, 50):
            plan = env.plan.model_copy(
                update={
                    "phase": "benchmark",
                    "campaign_id": f"condition-{number}",
                    "resources": env.plan.resources.model_copy(
                        update={
                            "codex_workers": number,
                            "stages": tuple(
                                stage.model_copy(update={"parallel": number})
                                if stage.kind == "codex"
                                else stage
                                for stage in env.plan.resources.stages
                            ),
                        }
                    ),
                }
            )
            ref = env.save(
                ConditionSpec(
                    benchmark_id="comparison",
                    condition_id=f"workers-{number}",
                    plan=annotation_reference(env.layout, save_plan(env.layout, plan)),
                    scope=self.manager.reference,
                    predictions=(env.context.prediction,),
                    bindings=(
                        WorkerBinding(
                            stage="review",
                            adapter="ball_review",
                            adapter_version="v1",
                            task="ball",
                        ),
                    ),
                    target_workers=number,
                    maximum_observations=observation_limit,
                )
            )
            self.conditions.append(ref)
        comparison = env.save(
            ComparisonSpec(
                benchmark_id="comparison",
                conditions=tuple(self.conditions),
                quality_policy=None,
            )
        )
        binary = env.layout.repository / "synthetic-codex"
        binary.write_text(
            f"#!{sys.executable}\nimport sys, runpy\nsys.path.insert(0, {str(PROJECT_ROOT)!r})\nmodule=runpy.run_path({str(PROJECT_ROOT / 'tests/support/tasks/ball_detection/review_worker.py')!r})\nmodule['run_synthetic_cli'](show_image=True)\n"
        )
        binary.chmod(0o700)
        script = env.layout.repository / "queue.sh"
        script.write_text("# fixture --idempotency-key\n")
        self.runtime = BenchmarkRuntime(
            comparison=comparison,
            settings=RuntimeSettings(
                queue=QueueConfiguration(
                    script=str(script),
                    script_digest=file_digest(script),
                    directory=str(env.layout.repository / ".training_queue"),
                    session="fixture",
                    issue=1088,
                ),
                codex_binary=str(binary),
                codex_home=str(env.layout.repository / "codex-home"),
                costs=tuple(
                    StageCost(
                        stage=stage,
                        pss_bytes=512 * 1024**2,
                        storage_bytes=1024**2,
                        timeout_seconds=60.0,
                        gpu_seconds=0,
                    )
                    for stage in ("review", "render")
                ),
                maximum_services_per_attempt=8,
            ),
            observation_interval_seconds=0.2,
        )
        self.reference = env.save(self.runtime)

    def runner(self, number: int = 10) -> BenchmarkRunner:
        return BenchmarkRunner(self.env.layout, self.reference, f"workers-{number}")

    def add_prediction(self) -> None:
        extra = Environment(
            self.env.layout.root, video_id="second", campaign_id="second"
        )
        self.conditions = [
            self.env.save(
                read_document(self.env.layout, ref, ConditionSpec).model_copy(
                    update={
                        "predictions": (
                            self.env.context.prediction,
                            extra.context.prediction,
                        )
                    }
                )
            )
            for ref in self.conditions
        ]
        comparison = read_document(
            self.env.layout, self.runtime.comparison, ComparisonSpec
        )
        self.runtime = self.runtime.model_copy(
            update={
                "comparison": self.env.save(
                    comparison.model_copy(update={"conditions": tuple(self.conditions)})
                )
            }
        )
        self.reference = self.env.save(self.runtime)


@pytest.fixture(autouse=True)
def no_hardware_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.tasks.base.annotation.resources.scheduler.gpu_usage",
        lambda _pids: GPUUsage(
            status="unavailable",
            owned_vram_bytes=None,
            device_vram_bytes=None,
            maximum_utilization_percent=None,
            reason="CPU fixture",
        ),
    )


@pytest.fixture
def setup(tmp_path: Path, request: pytest.FixtureRequest) -> Setup:
    return Setup(tmp_path / "annotation", getattr(request, "param", 1000))


def test_status_does_not_register_inputs_or_launch_workers(setup: Setup) -> None:
    runner = setup.runner()
    assert runner.status()["state"] == "not_started"
    assert not runner.layout.registry.exists()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.annotation",
            "benchmark-status",
            "--runtime",
            str(setup.env.layout.artifact_path(setup.reference)),
            "--sha256",
            setup.reference.digest,
            "--condition",
            "workers-10",
            "--test-root",
            str(setup.env.layout.root),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["task_states"] == {"unregistered": 1}
    assert not runner.layout.registry.exists()


def test_conditions_are_ordered_and_runtime_cannot_change(setup: Setup) -> None:
    with pytest.raises(ValueError, match="in order"):
        setup.runner(25).tick()
    first = setup.runner()
    first._initialize()
    changed = setup.runtime.model_copy(update={"observation_interval_seconds": 0.5})
    other = BenchmarkRunner(setup.env.layout, setup.env.save(changed), "workers-10")
    with pytest.raises(
        (ValueError, FileExistsError), match="immutable|different|overwrite"
    ):
        other._initialize()
    assert first.pool is not None
    assert all(
        row["state"] == "ready"
        for row in [first.pool.ledger.task(work_id) for work_id in first.expected]
    )


def test_stopped_budget_and_recovery_do_not_start_workers(setup: Setup) -> None:
    with setup.manager.registry.transaction() as db:
        db.execute("UPDATE resource_scopes SET stop_reason='disk reserve'")
    runner = setup.runner()
    assert runner.tick()["state"] == "budget_stopped"
    assert runner.pool is not None
    assert all(
        runner.pool.ledger.task(work_id)["state"] == "ready"
        for work_id in runner.expected
    )
    request = RecoveryInput(
        plan=runner.spec.plan,
        reason="resume worker admission only",
        operation=ResumeWorkers(),
    )
    receipt = recover(runner, request)
    assert recover(runner, request) == receipt
    assert runner.tick()["state"] == "budget_stopped"
    assert setup.manager.scope.previously_used_gpu_seconds == 3


def test_stage_rejects_unowned_or_non_review_operation(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = setup.runner()
    ref = setup.env.save(
        OperationInput(
            plan=runner.spec.plan, operation=PublishOperation(review=setup.env.blob)
        )
    )
    with pytest.raises(ValueError, match="resource-accounted"):
        execute(setup.env.layout, ref)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", ref.digest)
    with pytest.raises(ValueError, match="review services only"):
        execute(setup.env.layout, ref)
    with pytest.raises(ValueError, match="worker operation"):
        recover(
            runner,
            RecoveryInput(
                plan=runner.spec.plan,
                reason="invalid learning",
                operation=RetryLearning(
                    intent=setup.env.blob, stage="train", resume_checkpoint=None
                ),
            ),
        )


def test_cpu_children_review_three_conditions_without_production_side_effects(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        BallPredictionAdapter,
        "produce",
        lambda *args: pytest.fail("benchmark must not infer"),
    )
    env = setup.env
    original = env.store.current(env.context_ref, env.adapter)[0]
    completions = []
    runners = []
    try:
        for number in (10, 25, 50):
            runner = setup.runner(number)
            runners.append(runner)
            until = time.monotonic() + 100
            while True:
                state = runner.tick()
                if (
                    state["state"] in {"workers_complete", "blocked", "budget_stopped"}
                    or time.monotonic() >= until
                ):
                    break
                time.sleep(0.05)
            assert state["state"] == "workers_complete", state
            complete = runner.completion(runner.index)
            assert complete is not None
            completions.append(complete)
            assert setup.runner(number).tick()["completion"] == complete.model_dump(
                mode="json"
            )
        assert len(set(completions)) == 3
        assert env.store.current(env.context_ref, env.adapter)[0] == original
        assert not (env.layout.repository / "data/datasets/ball_detection").exists()
        assert not env.layout.path("models").exists()
        with setup.manager.registry.transaction() as db:
            requests = [
                json.loads(row[0])
                for row in db.execute("SELECT request FROM resource_requests")
            ]
        assert {item["stage"] for item in requests} == {"review", "render"}
        assert all(item["gpu_seconds"] == 0 for item in requests)
        assert (
            len(requests) == 9
        )  # One worker and two accounted services per condition.
        report = read_document(
            env.layout,
            build_report(env.layout, setup.runtime.comparison),
            ComparisonReport,
        )
        assert report.acceptance == "unverified"
        assert all(entry.execution == "workers_complete" for entry in report.conditions)
        assert all(entry.observed_real_workers_max == 0 for entry in report.conditions)
    finally:
        for runner in runners:
            runner.stop("fixture teardown")


def test_capacity_wait_restarts_same_unlaunched_attempt(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reserve capacity without counting the pytest process as benchmark workload.
    monkeypatch.setattr(ResourceManager, "observe", lambda self, pid: None)
    runner = setup.runner()
    runner._initialize()
    assert runner.pool is not None
    request = read_document(runner.layout, runner.requests[0], WorkRequest)
    blocker = ResourceRequest(
        request_id="fixture-capacity-blocker",
        plan=runner.spec.plan,
        stage="render",
        pss_bytes=runner.plan.resources.total_pss_bytes
        - runner.runtime.cost("review").pss_bytes // 2,
        storage_bytes=0,
        gpu_seconds=0,
    )
    path = runner.layout.path(
        "campaigns",
        runner.plan.campaign_id,
        "workers",
        request.work_id,
        "attempt-0001",
        "launch.json",
    )
    try:
        with runner.resources.lease(blocker):
            assert runner.tick()["task_states"] == {"ready": 1}
            launch_ref = annotation_reference(runner.layout, path)
            launch = read_document(runner.layout, launch_ref, Launch)
            assert launch.resource_request is not None
            resource_id = runner.resources.key(launch.resource_request)
            # Reopening the runner must preserve the pending request and launch.
            runner = setup.runner()
            assert runner.tick()["task_states"] == {"ready": 1}
            assert runner.pool is not None and not runner.pool.children
            with runner.pool.ledger.registry.transaction() as db:
                assert (
                    db.execute("SELECT count(*) FROM worker_attempts").fetchone()[0]
                    == 0
                )
                assert (
                    db.execute(
                        "SELECT name FROM sqlite_master WHERE name='resource_requests'"
                    ).fetchone()
                    is None
                )
            with runner.resources.registry.transaction() as db:
                rows = db.execute(
                    "SELECT id,state,reason FROM resource_requests WHERE stage='review'"
                ).fetchall()
            assert [tuple(row) for row in rows] == [
                (resource_id, "pending", "PSS reservations do not fit")
            ]
            assert annotation_reference(runner.layout, path) == launch_ref

        runner = setup.runner()
        state = runner.tick()
        assert runner.pool is not None
        row = runner.pool.ledger.task(request.work_id)
        assert row["current_attempt"] == launch.attempt
        assert row["state"] in {"running", "exited", "complete"}
        until = time.monotonic() + 100
        while state["state"] == "reviewing" and time.monotonic() < until:
            time.sleep(0.05)
            state = runner.tick()
        assert state["state"] == "workers_complete", state
        with runner.pool.ledger.registry.transaction() as db:
            attempts = db.execute("SELECT id,state FROM worker_attempts").fetchall()
        assert [tuple(row) for row in attempts] == [(launch.attempt, "complete")]
        # Worker reconciliation joins its supervisor, but cached resource polls
        # can leave its lease draining. Recover it before inspecting the ledger.
        runner.resources.sample()
        with runner.resources.registry.transaction() as db:
            rows = db.execute(
                "SELECT id,state,request FROM resource_requests WHERE stage='review'"
            ).fetchall()
        assert [tuple(row) for row in rows] == [
            (
                resource_id,
                "complete",
                launch.resource_request.canonical_bytes().decode(),
            )
        ]
        assert annotation_reference(runner.layout, path) == launch_ref
    finally:
        until = time.monotonic() + 15
        while runner.stop("fixture teardown") and time.monotonic() < until:
            time.sleep(0.05)


def test_missing_scope_ledger_is_not_recreated(setup: Setup) -> None:
    with setup.manager.registry.transaction() as db:
        db.execute("DELETE FROM resource_scopes")
    with pytest.raises(ValueError, match="existing registered"):
        setup.runner()
    with setup.manager.registry.transaction() as db:
        assert db.execute("SELECT count(*) FROM resource_scopes").fetchone()[0] == 0


def test_failed_worker_requires_explicit_retry_and_recovery_is_idempotent(
    setup: Setup,
) -> None:
    from src.tasks.annotation.contracts import RetryWorker
    from src.tasks.base.annotation.workers.contracts import WorkRequest

    runner = setup.runner()
    runner._initialize()
    assert runner.pool is not None
    runner.pool.test_command = (sys.executable, "-c", "raise SystemExit(1)")
    work_ref = runner.requests[0]
    work = read_document(runner.layout, work_ref, WorkRequest)
    runner.pool.start(work)

    pool = runner.pool

    def settle() -> dict:
        until = time.monotonic() + 15
        while time.monotonic() < until:
            row: dict = pool.reconcile(work)
            if row["state"] not in {"running", "exited"}:
                return row
            time.sleep(0.05)
        raise AssertionError("synthetic child did not exit")

    first = settle()
    assert first["state"] == "failed"
    assert runner.tick()["state"] == "blocked"
    assert (
        runner.pool.ledger.task(work.work_id)["current_attempt"]
        == first["current_attempt"]
    )
    request = RecoveryInput(
        plan=runner.spec.plan,
        reason="explicit fixture retry",
        operation=RetryWorker(request=work_ref, after_attempt=first["current_attempt"]),
    )
    receipt = recover(runner, request)
    second = settle()
    assert second["state"] == "failed"
    assert second["current_attempt"] != first["current_attempt"]
    assert recover(runner, request) == receipt
    assert (
        runner.pool.ledger.task(work.work_id)["current_attempt"]
        == second["current_attempt"]
    )
    changed = request.model_copy(update={"reason": "new stale retry"})
    with pytest.raises(ValueError, match="attempt changed"):
        recover(runner, changed)


@pytest.mark.parametrize("setup", [1], indirect=True)
def test_observation_limit_stops_active_worker_and_never_completes(
    setup: Setup,
) -> None:
    runner = setup.runner()
    runner._initialize()
    assert runner.pool is not None
    runner.pool.test_command = (sys.executable, "-c", "import time; time.sleep(30)")
    try:
        assert runner.tick()["state"] == "reviewing"
        result = runner.tick()
        until = time.monotonic() + 15
        while result["state"] == "stopping" and time.monotonic() < until:
            time.sleep(0.05)
            result = runner.tick()
        assert result["state"] == "blocked"
        assert result["reason"] == "benchmark observation limit reached"
        assert result["completion"] is None
        assert all(
            runner.pool.ledger.task(work_id)["state"] == "cancelled"
            for work_id in runner.expected
        )
    finally:
        until = time.monotonic() + 15
        while runner.stop("fixture teardown") and time.monotonic() < until:
            time.sleep(0.05)


def test_observation_stop_frees_pending_capacity_for_other_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ResourceManager, "observe", lambda self, pid: None)
    setup = Setup(tmp_path / "annotation", observation_limit=1, pending_limit=2)
    runner = setup.runner()
    runner._initialize()
    assert runner.pool is not None
    work = read_document(runner.layout, runner.requests[0], WorkRequest)
    other = ResourceRequest(
        request_id="other-campaign-waiting",
        plan=setup.env.plan_ref,
        stage="review",
        pss_bytes=runner.runtime.cost("review").pss_bytes,
        storage_bytes=0,
        gpu_seconds=0,
    )
    runner.resources.submit(other)
    next_request = other.model_copy(update={"request_id": "other-campaign-next"})
    blocker = other.model_copy(
        update={
            "request_id": "capacity-blocker",
            "stage": "render",
            "pss_bytes": runner.plan.resources.total_pss_bytes - other.pss_bytes // 2,
        }
    )
    with runner.resources.lease(blocker):
        assert runner.tick()["task_states"] == {"ready": 1}
        launch_ref = annotation_reference(
            runner.layout,
            runner.layout.path(
                "campaigns",
                runner.plan.campaign_id,
                "workers",
                work.work_id,
                "attempt-0001",
                "launch.json",
            ),
        )
        launch = read_document(runner.layout, launch_ref, Launch)
        assert launch.resource_request is not None
        assert runner.resources._row(launch.resource_request)["reason"] == (
            "PSS reservations do not fit"
        )
        with pytest.raises(CapacityUnavailable, match="backlog is full"):
            runner.resources.submit(next_request)

    # Reopening must recover the condition's pending request from the shared DB.
    runner = setup.runner()
    stopped = runner.tick()
    assert stopped["state"] == "blocked"
    assert stopped["reason"] == "benchmark observation limit reached"
    with runner.resources.lease(next_request):
        assert runner.resources._row(next_request)["state"] == "running"
    cancelled = runner.resources._row(launch.resource_request)
    assert (cancelled["state"], cancelled["reason"]) == ("cancelled", stopped["reason"])
    assert runner.resources._row(other)["state"] == "pending"
    assert runner.pool is not None and not runner.pool.children
    assert runner.pool.ledger.task(work.work_id)["state"] == "cancelled"
    assert runner.stop("repeat stop") == ()
    assert runner.resources._row(launch.resource_request) == cancelled


def test_observation_interval_survives_repeated_runner_restarts(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup.runtime = setup.runtime.model_copy(
        update={"observation_interval_seconds": 30}
    )
    setup.reference = setup.env.save(setup.runtime)
    monkeypatch.setattr(WorkerPool, "start", lambda *args, **kwargs: None)
    clock = [100.0]
    timer = SimpleNamespace(
        monotonic=lambda: clock[0], monotonic_ns=lambda: int(clock[0] * 1e9)
    )
    monkeypatch.setattr(runner_module, "time", timer)
    monkeypatch.setattr("src.tasks.base.annotation.benchmarks.capture.time", timer)
    runner = setup.runner()
    assert runner.tick()["state"] == "reviewing"
    observations = directory(runner.shared, runner.spec) / "observations"
    assert len(list(observations.glob("*.json"))) == 1
    for seconds in (1, 5, 29.9):
        clock[0] = 100.0 + seconds
        runner = setup.runner()
        assert runner.tick()["state"] == "reviewing"
        assert len(list(observations.glob("*.json"))) == 1
    clock[0] = 130.0
    assert runner.tick()["state"] == "reviewing"
    assert len(list(observations.glob("*.json"))) == 2


def test_observation_interval_does_not_compare_different_boot_clocks(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(WorkerPool, "start", lambda *args, **kwargs: None)
    runner = setup.runner()
    runner.tick()
    path = directory(runner.shared, runner.spec) / "observations" / "000001.json"
    previous = read_document(
        runner.shared, annotation_reference(runner.shared, path), Observation
    )
    path.write_bytes(
        previous.model_copy(
            update={"boot_id": "earlier-boot", "cut_monotonic_ns": 10**30}
        ).canonical_bytes()
    )
    restarted = setup.runner()
    assert restarted.tick()["state"] == "reviewing"
    assert (path.parent / "000002.json").exists()


@pytest.mark.parametrize("setup", [1], indirect=True)
def test_completion_recovers_after_last_observation_before_receipt(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = setup.runner()
    original_observe = runner._observe
    monkeypatch.setattr(
        runner,
        "_observe",
        lambda *, force=False: original_observe(force=True) if force else None,
    )
    original_write = runner_module.write_document

    def interrupted_write(layout, path, document):
        if isinstance(document, ConditionCompletion):
            raise RuntimeError("interrupted after final observation")
        return original_write(layout, path, document)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(runner_module, "write_document", interrupted_write)
            until = time.monotonic() + 60
            with pytest.raises(RuntimeError, match="interrupted after final"):
                while time.monotonic() < until:
                    state = runner.tick()
                    assert state["state"] == "reviewing", state
                    time.sleep(0.05)
        assert runner.completion(0) is None
        observations = directory(runner.shared, runner.spec) / "observations"
        assert len(list(observations.glob("*.json"))) == 1
        final = annotation_reference(runner.shared, observations / "000001.json")
        assert read_document(runner.shared, final, Observation).task_states == {
            "complete": 1
        }
        restarted = setup.runner()
        monkeypatch.setattr(
            WorkerPool,
            "start",
            lambda *args, **kwargs: pytest.fail("completed work must not restart"),
        )
        monkeypatch.setattr(
            restarted,
            "_observe",
            lambda **kwargs: pytest.fail("must reuse final observation"),
        )
        state = restarted.tick()
        assert state["state"] == "workers_complete", state
        completion = restarted.completion(0)
        assert completion is not None
        receipt = read_document(restarted.shared, completion, ConditionCompletion)
        assert receipt.final_observation == final
        assert setup.runner().tick()["completion"] == completion.model_dump(mode="json")
        setup.runner(25)._initialize()
    finally:
        runner.stop("fixture teardown")


@pytest.mark.parametrize("pending_first", [False, True])
def test_stop_reaches_orphan_and_other_workers_and_retains_pending(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, pending_first: bool
) -> None:
    setup.add_prediction()
    runner = setup.runner()
    runner._initialize()
    assert runner.pool is not None
    pool = runner.pool
    pool.test_command = (sys.executable, "-c", "import time; time.sleep(30)")
    requests = [
        read_document(runner.layout, ref, WorkRequest) for ref in runner.requests
    ]
    owners = []
    try:
        for work in requests:
            pool.start(work)
            _, launch = pool._launch(pool.ledger.task(work.work_id))
            until = time.monotonic() + 10
            while (
                not (Path(launch.directory) / "child.json").exists()
                and time.monotonic() < until
            ):
                time.sleep(0.02)
            owner = ProcessIdentity.from_document(
                json.loads((Path(launch.directory) / "owner.json").read_text())
            )
            owners.append(owner)
            assert (Path(launch.directory) / "child.json").exists()
        os.kill(owners[0].pid, signal.SIGKILL)
        _, orphan = pool._launch(pool.ledger.task(requests[0].work_id))
        pool.children[orphan.attempt].wait(timeout=5)
        with pytest.raises(WorkerBusy, match="owned processes"):
            pool.reconcile(requests[0])
        original_stop = pool.stop

        def stop(work, *, reason):
            if pending_first and work == requests[0]:
                raise WorkerBusy("owned processes have not exited")
            return original_stop(work, reason=reason)

        with monkeypatch.context() as patch:
            patch.setattr(pool, "stop", stop)
            result = runner._stopped("test stop", "blocked")
        assert result["state"] == ("stopping" if pending_first else "blocked")
        assert result["pending_workers"] == (
            (requests[0].work_id,) if pending_first else ()
        )
        assert pool.ledger.task(requests[1].work_id)["state"] == "cancelled"
        assert not owned_members(owners[1])
        if pending_first:
            assert owned_members(owners[0])
            assert runner._stopped("test stop", "blocked")["pending_workers"] == ()
        assert all(not owned_members(owner) for owner in owners)
    finally:
        for owner in owners:
            signal_owned(owner, signal.SIGKILL)
        for child in pool.children.values():
            child.wait(timeout=5)


def test_benchmark_run_services_healthy_worker_after_another_fails(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.tasks.annotation.__main__ import main

    setup.add_prediction()
    runner = setup.runner()
    runner._initialize()
    assert runner.pool is not None
    pool = runner.pool
    pool.test_command = (sys.executable, "-c", "raise SystemExit(1)")
    failed = read_document(runner.layout, runner.requests[0], WorkRequest)
    try:
        pool.start(failed)
        until = time.monotonic() + 10
        row = pool.reconcile(failed)
        while row["state"] == "running" and time.monotonic() < until:
            time.sleep(0.05)
            row = pool.reconcile(failed)
        assert row["state"] == "failed", row
        pool.test_command = None
        monkeypatch.setattr(runner_module, "BenchmarkRunner", lambda *args: runner)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "annotation",
                "benchmark-run",
                "--runtime",
                str(setup.env.layout.artifact_path(setup.reference)),
                "--sha256",
                setup.reference.digest,
                "--condition",
                "workers-10",
                "--max-seconds",
                "60",
                "--test-root",
                str(setup.env.layout.root),
            ],
        )
        main()
        result = json.loads(capsys.readouterr().out)
        assert result["state"] == "blocked"
        assert result["task_states"] == {"complete": 1, "failed": 1}
        # Service receipts can precede supervisor exit. Join remaining children,
        # then bypass poll()'s cache to recover their finished resource leases.
        for child in runner.jobs.stages.children.values():
            assert child.wait(timeout=10) == 0
        runner.resources.sample()
        with setup.manager.registry.transaction() as db:
            assert (
                db.execute(
                    "SELECT count(*) FROM resource_requests WHERE stage='render' AND state='complete'"
                ).fetchone()[0]
                == 2
            )
    finally:
        runner.stop("fixture teardown")
