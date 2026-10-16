from __future__ import annotations

import json
import multiprocessing
import os
import signal
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.tasks.base.annotation.artifacts import (
    artifact_lock,
    read_document,
    write_asset,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan
from src.tasks.base.annotation.resources import (
    BudgetLimits,
    CapacityUnavailable,
    ResourceManager,
)
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.command import codex_command, worker_environment
from src.tasks.base.annotation.workers.contracts import (
    AttemptResult,
    ChildReceipt,
    ExitReceipt,
    Launch,
    PreparedWorker,
    WorkOutcome,
    WorkRequest,
)
from src.tasks.base.annotation.workers.ledger import WorkerBusy, WorkerStopped
from src.tasks.base.annotation.workers.pool import WorkerPool
from src.tasks.base.annotation.workers.processes import (
    ProcessIdentity,
    owned_members,
    signal_owned,
)


@dataclass
class Adapter:
    name: str = "fixture"
    version: str = "v1"
    validations: int = 0
    outcome: str = "complete"
    adopt: bool = False
    expects_telemetry: bool = False

    def prepare(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
    ) -> PreparedWorker:
        del plan, request
        return PreparedWorker(
            prompt=write_asset(layout, directory / "prompt.md", b"fixture only")
        )

    def validate(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
        result: ArtifactRef,
        execution: ExitReceipt,
        telemetry: ArtifactRef | None,
    ) -> WorkOutcome:
        del plan, request
        assert (telemetry is not None) == self.expects_telemetry
        self.validations += 1
        assert execution.status == "succeeded"
        if json.loads(layout.artifact_path(result).read_text()) != {"fixture": True}:
            raise ValueError("fixture labels failed their mechanical validation")
        receipt = write_asset(
            layout, directory / "validated.json", b'{"fixture_validated":true}\n'
        )
        outcome: WorkOutcome = WorkOutcome.model_validate_json(
            json.dumps(
                {
                    "status": self.outcome,
                    "receipt": receipt.model_dump(),
                    "adoption": receipt.model_dump() if self.adopt else None,
                    "reason": None
                    if self.outcome == "complete"
                    else "fixture remains unresolved",
                }
            )
        )
        return outcome

    def telemetry_images(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
    ) -> tuple[RenderedImage, ...]:
        del layout, plan, request, directory
        return ()

    def classify_tool(self, name: str | None, raw_input: str) -> str:
        del name, raw_input
        return "fixture"


@dataclass
class Setup:
    layout: AnnotationLayout
    plan: CampaignPlan
    adapter: Adapter
    request: WorkRequest
    script: Path

    def pool(
        self, *, timeout: float = 30, plan: CampaignPlan | None = None
    ) -> WorkerPool:
        return WorkerPool(
            self.layout,
            plan or self.plan,
            (self.adapter,),
            timeout_seconds=timeout,
            test_command=(sys.executable, str(self.script)),
        )

    def program(
        self, *, delay: float = 0.0, quota: bool = False, invalid: bool = False
    ) -> None:
        body = f"""import json, pathlib, sys, time
sys.stdin.read()
with pathlib.Path({str(self.script.parent / "launches.txt")!r}).open('a') as handle:
    handle.write('launched\\n')
print(json.dumps({{"type":"thread.started","thread_id":"fixture-thread"}}), flush=True)
time.sleep({delay!r})
pathlib.Path('result.json').write_text(json.dumps({{"fixture":{not invalid!r}}}))
print(json.dumps({{"type":"error","message":"usage limit reached"}}) if {quota!r} else json.dumps({{"type":"turn.completed","usage":{{}}}}), flush=True)
"""
        self.script.write_text(body)

    def launches(self) -> int:
        path = self.script.parent / "launches.txt"
        return len(path.read_text().splitlines()) if path.exists() else 0


def plan_for(
    layout: AnnotationLayout, reference: ArtifactRef, name: str = "fixture"
) -> CampaignPlan:
    ref = reference.model_dump()
    return CampaignPlan.model_validate_json(
        json.dumps(
            {
                "campaign_id": name,
                "phase": "preview",
                "annotation_root": str(layout.root),
                "counts": {
                    "channels": 1,
                    "videos_per_channel": 1,
                    "max_clips_per_video": 1,
                },
                "selection": ref,
                "preparation": ref,
                "code_commit": "a" * 40,
                "code_diff_digest": "b" * 64,
                "agent_model": "gpt-6.1-sol",
                "agent_effort": "high",
                "resources": {
                    "codex_workers": 1,
                    "additional_storage_bytes": 30 * 1024**3,
                    "minimum_free_bytes": 30 * 1024**3,
                    "total_pss_bytes": 22 * 1024**3,
                    "gpu_seconds": 43200,
                    "stages": [
                        {
                            "stage": "review",
                            "kind": "codex",
                            "parallel": 1,
                            "threads_per_process": 1,
                        }
                    ],
                },
                "tasks": [
                    {
                        "task": "fixture",
                        "model": {
                            "task": "fixture",
                            "version": "v1",
                            "training_weights": ref,
                            "inference_weights": ref,
                            "configuration": ref,
                        },
                        "inference": ref,
                        "review": None,
                        "learning": None,
                        "products": [
                            {
                                "product": "fixture",
                                "dataset_task": "fixture",
                                "dataset_version": "v1",
                                "component": "frames",
                                "base_snapshot": None,
                                "generation_policy": ref,
                            }
                        ],
                    }
                ],
            }
        )
    )


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    reference = write_asset(
        layout, layout.path("campaigns", "inputs", "payload.txt"), b"fixture"
    )
    request = WorkRequest(
        adapter="fixture",
        adapter_version="v1",
        stage="review",
        subject_id="c" * 64,
        payload=reference,
        inputs=(reference,),
    )
    result = Setup(
        layout,
        plan_for(layout, reference),
        Adapter(),
        request,
        layout.path("campaigns", "inputs", "fake_worker.py"),
    )
    result.program()
    return result


def settled(
    pool: WorkerPool, request: WorkRequest, timeout: float = 10
) -> dict[str, object]:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        state: dict[str, object] = pool.reconcile(request)
        if state["state"] not in {"running", "exited"}:
            return state
        time.sleep(0.02)
    raise AssertionError(f"worker did not settle: {pool.ledger.task(request.work_id)}")


def directory(pool: WorkerPool, request: WorkRequest) -> Path:
    _, launch = pool._launch(pool.ledger.task(request.work_id))
    return Path(launch.directory)


def owner_ready(pool: WorkerPool, request: WorkRequest) -> ProcessIdentity:
    path = directory(pool, request) / "owner.json"
    until = time.monotonic() + 5
    while not path.exists() and time.monotonic() < until:
        time.sleep(0.02)
    return ProcessIdentity.from_document(json.loads(path.read_text()))


def test_success_is_receipt_based_and_never_relaunches(setup: Setup) -> None:
    pool = setup.pool()
    assert pool.start(setup.request) is not None
    state = settled(pool, setup.request)
    assert state["state"] == "complete", state
    assert setup.adapter.validations == 1
    assert setup.pool().start(setup.request) is None
    assert setup.launches() == 1
    child = ChildReceipt.model_validate_json(
        (directory(pool, setup.request) / "child.json").read_bytes()
    )
    owner = ProcessIdentity.from_document(
        json.loads((directory(pool, setup.request) / "owner.json").read_text())
    )
    assert child.process is not None
    assert child.process.pid != owner.pid
    assert child.process.group == owner.group and child.process.session == owner.session
    final = read_document(
        setup.layout, ArtifactRef.model_validate_json(state["result"]), AttemptResult
    )
    assert final.outcome.adoption is None
    assert child.launch_digest == final.launch.digest
    assert all(word not in str(state) for word in ("qa_wait", "qa_approved"))
    setup.layout.artifact_path(final.result).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        pool.reconcile(setup.request)


def test_concurrent_start_and_cross_campaign_identity(setup: Setup) -> None:
    setup.program(delay=0.5)
    pools = [setup.pool() for _ in range(4)]

    def start(pool: WorkerPool) -> str | None:
        try:
            result: str | None = pool.start(setup.request)
            return result
        except WorkerBusy:
            return "busy"

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(start, pools))
    assert sum(value not in {"busy", None} for value in results) == 1
    assert settled(pools[0], setup.request)["state"] == "complete"
    assert setup.launches() == 1
    other = setup.pool(plan=plan_for(setup.layout, setup.request.payload, "another"))
    with pytest.raises(ValueError, match="already owned"):
        other.register(setup.request)


def test_shared_worker_capacity_spans_campaigns(setup: Setup) -> None:
    setup.program(delay=0.6)
    pool = setup.pool()
    other = setup.pool(plan=plan_for(setup.layout, setup.request.payload, "another"))
    second = setup.request.model_copy(update={"subject_id": "d" * 64})
    pool.start(setup.request)
    with pytest.raises(WorkerBusy, match="capacity"):
        other.start(second)
    assert settled(pool, setup.request)["state"] == "complete"
    other.start(second)
    assert settled(other, second)["state"] == "complete"
    assert setup.launches() == 2


def test_final_receipt_recovers_after_database_gap(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = setup.pool()
    pool.start(setup.request)
    with monkeypatch.context() as patch:
        original = pool._mark

        def fail(
            work_id: str,
            state: str,
            reason: str | None,
            *,
            attempt: str,
            from_states: tuple[str, ...] = ("running", "exited"),
            result: ArtifactRef | None = None,
        ) -> bool:
            if state == "complete":
                raise RuntimeError("database gap")
            return original(
                work_id,
                state,
                reason,
                attempt=attempt,
                from_states=from_states,
                result=result,
            )

        patch.setattr(pool, "_mark", fail)
        with pytest.raises(RuntimeError, match="database gap"):
            settled(pool, setup.request)
    assert setup.adapter.validations == 1
    assert settled(pool, setup.request)["state"] == "complete"
    assert setup.adapter.validations == 1 and setup.launches() == 1


def test_parent_exit_does_not_release_running_worker(setup: Setup) -> None:
    setup.program(delay=0.8)

    def dispatcher() -> None:
        setup.pool().start(setup.request)
        os._exit(0)

    process = multiprocessing.get_context("fork").Process(target=dispatcher)
    process.start()
    process.join(5)
    assert process.exitcode == 0
    pool = setup.pool()
    with pytest.raises(WorkerBusy):
        pool.start(setup.request)
    assert settled(pool, setup.request)["state"] == "complete"
    assert setup.launches() == 1


def test_supervisor_death_preserves_orphan_ownership(setup: Setup) -> None:
    setup.program(delay=30)
    pool = setup.pool()
    pool.start(setup.request)
    owner = owner_ready(pool, setup.request)
    until = time.monotonic() + 3
    while setup.launches() == 0 and time.monotonic() < until:
        time.sleep(0.02)
    try:
        os.kill(owner.pid, signal.SIGKILL)
        pool.children[next(iter(pool.children))].wait(timeout=5)
        with pytest.raises(WorkerBusy, match="owned processes"):
            pool.reconcile(setup.request)
        with pytest.raises(WorkerBusy):
            pool.start(setup.request, retry=True)
    finally:
        signal_owned(owner, signal.SIGKILL)
    until = time.monotonic() + 3
    while owned_members(owner) and time.monotonic() < until:
        time.sleep(0.02)
    assert settled(pool, setup.request)["state"] == "failed"
    assert setup.launches() == 1


@pytest.mark.parametrize(
    "mode,status",
    [
        ("invalid", "validation_failed"),
        ("deferred", "deferred"),
        ("adopt", "validation_failed"),
    ],
)
def test_unreviewed_or_invalid_results_do_not_complete(
    setup: Setup, mode: str, status: str
) -> None:
    setup.program(invalid=mode == "invalid")
    setup.adapter.outcome = "deferred" if mode == "deferred" else "complete"
    setup.adapter.adopt = mode == "adopt"
    pool = setup.pool()
    pool.start(setup.request)
    assert settled(pool, setup.request)["state"] == status
    with pytest.raises(WorkerStopped, match="explicit retry"):
        pool.start(setup.request)
    assert setup.launches() == 1


def test_quota_stops_new_work_until_explicit_resume(setup: Setup) -> None:
    setup.program(quota=True)
    pool = setup.pool()
    pool.start(setup.request)
    assert settled(pool, setup.request)["state"] == "quota"
    second = setup.request.model_copy(update={"subject_id": "d" * 64})
    with pytest.raises(WorkerStopped, match="usage limit"):
        pool.start(second)
    setup.program()
    pool.ledger.resume_campaign(setup.plan.campaign_id)
    pool.start(setup.request, retry=True)
    assert settled(pool, setup.request)["state"] == "complete"
    assert setup.launches() == 2


def test_unreconciled_quota_blocks_admission(setup: Setup) -> None:
    setup.program(quota=True)
    pool = setup.pool()
    attempt = pool.start(setup.request)
    assert attempt is not None
    pool.children[attempt].wait(timeout=10)
    second = setup.request.model_copy(update={"subject_id": "d" * 64})
    with pytest.raises(WorkerStopped, match="usage limit"):
        setup.pool().start(second)
    assert pool.ledger.task(setup.request.work_id)["state"] == "quota"
    assert setup.launches() == 1


@pytest.mark.parametrize("admission", [False, True])
def test_quota_and_campaign_stop_roll_back_together(
    setup: Setup, admission: bool
) -> None:
    setup.program(quota=True)
    pool = setup.pool()
    attempt = pool.start(setup.request)
    assert attempt is not None
    pool.children[attempt].wait(timeout=10)
    with pool.ledger.registry.transaction() as database:
        database.execute(
            "CREATE TRIGGER interrupt_stop BEFORE INSERT ON worker_campaign_stops "
            "BEGIN SELECT RAISE(ABORT, 'interrupted campaign stop'); END"
        )
    second = setup.request.model_copy(update={"subject_id": "d" * 64})
    with pytest.raises(sqlite3.IntegrityError, match="interrupted campaign stop"):
        if admission:
            pool.start(second)
        else:
            pool.reconcile(setup.request)
    assert pool.ledger.task(setup.request.work_id)["state"] == "running"
    with pool.ledger.registry.transaction() as database:
        assert (
            database.execute(
                "SELECT state FROM worker_attempts WHERE id=?", (attempt,)
            ).fetchone()[0]
            == "running"
        )
        assert database.execute("SELECT * FROM worker_campaign_stops").fetchall() == []
        database.execute("DROP TRIGGER interrupt_stop")
    with pytest.raises(WorkerStopped, match="usage limit"):
        setup.pool().start(second)
    assert pool.ledger.task(setup.request.work_id)["state"] == "quota"
    assert setup.launches() == 1


def test_reconcile_reads_attempt_after_acquiring_lock(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool, other = setup.pool(), setup.pool()
    first = pool.start(setup.request)
    assert first is not None
    pool.children[first].wait(timeout=10)
    second = None

    @contextmanager
    def interleave(layout, name, *, blocking=True):
        nonlocal second
        if name == "worker-" + setup.request.work_id and second is None:
            with monkeypatch.context() as patch:
                patch.setattr(
                    "src.tasks.base.annotation.workers.pool.artifact_lock",
                    artifact_lock,
                )
                setup.adapter.outcome = "deferred"
                assert other.reconcile(setup.request)["state"] == "deferred"
                setup.adapter.outcome = "complete"
                second = other.start(setup.request, retry=True)
                assert second is not None
                other.children[second].wait(timeout=10)
        with artifact_lock(layout, name, blocking=blocking) as descriptor:
            yield descriptor

    monkeypatch.setattr(
        "src.tasks.base.annotation.workers.pool.artifact_lock", interleave
    )
    state = pool.reconcile(setup.request)
    assert state["current_attempt"] == second
    assert state["state"] == "complete"
    final = read_document(
        setup.layout, ArtifactRef.model_validate_json(state["result"]), AttemptResult
    )
    launch = read_document(setup.layout, final.launch, Launch)
    assert launch.attempt == second
    assert setup.adapter.validations == 2


@pytest.mark.parametrize("finish_retry", [False, True])
def test_stop_does_not_cancel_concurrent_retry(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, finish_retry: bool
) -> None:
    pool, other = setup.pool(), setup.pool()
    first = pool.start(setup.request)
    assert first is not None
    pool.children[first].wait(timeout=10)
    original_reconcile = pool.reconcile
    second = None

    def retry_after_reconcile(request):
        nonlocal second
        setup.adapter.outcome = "deferred"
        current = original_reconcile(request)
        assert current["state"] == "deferred"
        setup.adapter.outcome = "complete"
        second = other.start(request, retry=True)
        assert second is not None
        if finish_retry:
            other.children[second].wait(timeout=10)
            assert other.reconcile(request)["state"] == "complete"
        return current

    monkeypatch.setattr(pool, "reconcile", retry_after_reconcile)
    try:
        state = pool.stop(setup.request, reason="stop only the original attempt")
        assert state["current_attempt"] == second
        assert state["state"] == ("complete" if finish_retry else "running")
    finally:
        if second in other.children:
            other.children[second].wait(timeout=10)
    final = other.reconcile(setup.request)
    assert final["state"] == "complete" and final["result"] is not None


def test_stop_does_not_overwrite_completion_after_reconcile(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool, other = setup.pool(), setup.pool()
    attempt = pool.start(setup.request)
    assert attempt is not None
    pool.children[attempt].wait(timeout=10)

    def complete_after_snapshot(request):
        snapshot = pool.ledger.task(request.work_id)
        assert other.reconcile(request)["state"] == "complete"
        return snapshot

    monkeypatch.setattr(pool, "reconcile", complete_after_snapshot)
    state = pool.stop(setup.request, reason="completion won the transaction race")
    assert state["state"] == "complete" and state["result"] is not None


def test_settle_dead_slots_preserves_concurrent_completion(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool, other = setup.pool(), setup.pool()
    attempt = pool.start(setup.request)
    assert attempt is not None
    pool.children[attempt].wait(timeout=10)

    @contextmanager
    def complete_before_lock(layout, name, *, blocking=True):
        with monkeypatch.context() as patch:
            patch.setattr(
                "src.tasks.base.annotation.workers.pool.artifact_lock", artifact_lock
            )
            assert other.reconcile(setup.request)["state"] == "complete"
        with artifact_lock(layout, name, blocking=blocking) as descriptor:
            yield descriptor

    monkeypatch.setattr(
        "src.tasks.base.annotation.workers.pool.artifact_lock", complete_before_lock
    )
    pool._settle_dead_slots()
    state = pool.ledger.task(setup.request.work_id)
    assert state["state"] == "complete" and state["result"] is not None


def test_timeout_and_owned_stop_terminate_descendants(setup: Setup) -> None:
    setup.program(delay=30)
    pool = setup.pool(timeout=0.1)
    pool.start(setup.request)
    assert settled(pool, setup.request)["state"] == "timed_out"
    owner = owner_ready(pool, setup.request)
    assert not owned_members(owner)
    second = setup.request.model_copy(update={"subject_id": "d" * 64})
    stopping_pool = setup.pool()
    stopping_pool.start(second)
    owner = owner_ready(stopping_pool, second)
    state = stopping_pool.stop(second, reason="operator requested fixture stop")
    assert state["state"] == "cancelled"
    assert not owned_members(owner)


def test_missing_exit_receipt_is_explicit_failure(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = setup.pool()
    with monkeypatch.context() as patch:
        patch.setattr(
            "src.tasks.base.annotation.workers.pool.subprocess.Popen",
            lambda *_args, **_kw: (_ for _ in ()).throw(OSError("cannot start")),
        )
        with pytest.raises(OSError, match="cannot start"):
            pool.start(setup.request)
    assert pool.ledger.task(setup.request.work_id)["state"] == "failed"
    with pytest.raises(WorkerStopped, match="explicit retry"):
        pool.start(setup.request)
    pool.start(setup.request, retry=True)
    assert settled(pool, setup.request)["state"] == "complete"


def test_codex_model_environment_and_module_boundaries(setup: Setup) -> None:
    pool = setup.pool()
    pool.start(setup.request)
    assert settled(pool, setup.request)["state"] == "complete"
    ref, launch = pool._launch(pool.ledger.task(setup.request.work_id))
    assert read_document(setup.layout, ref, Launch) == launch
    command = codex_command(launch, setup.plan)
    assert command[command.index("-m") + 1] == "gpt-6.1-sol"
    assert "--ignore-user-config" in command and "--ignore-rules" in command
    assert "workspace-write" in command and 'approval_policy="never"' in command
    assert command[command.index("-C") + 1].endswith("/work")
    environment = worker_environment(launch, setup.plan)
    assert (
        environment["CUDA_VISIBLE_DEVICES"] == ""
        and environment["OMP_NUM_THREADS"] == "1"
    )
    assert "CODEX_THREAD_ID" not in environment
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.tasks.base.annotation.workers.pool; assert 'torch' not in sys.modules; assert not any(k.startswith(('src.tasks.ball_detection', 'src.tasks.player_detection')) for k in sys.modules)",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.tasks.base as b; from src.tasks.base.inference.predictor import BasePredictor; assert b.BasePredictor is BasePredictor; assert all(name in dir(b) for name in b.__all__)",
        ],
        check=True,
    )


def test_completion_winning_stop_race_keeps_its_receipt(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup.program(delay=0.2)
    pool = setup.pool()
    pool.start(setup.request)
    owner_ready(pool, setup.request)

    def completed_first(*_args: object, **_kwargs: object) -> int:
        assert settled(pool, setup.request)["state"] == "complete"
        return 0

    monkeypatch.setattr(
        "src.tasks.base.annotation.workers.pool.signal_owned", completed_first
    )
    state = pool.stop(setup.request, reason="raced with completion")
    assert state["state"] == "complete" and state["result"] is not None


def test_worker_inherits_resource_lease_and_records_real_pss(setup: Setup) -> None:
    raw = setup.plan.model_dump(mode="json")
    raw["resources"]["minimum_free_bytes"] = 1
    plan = CampaignPlan.model_validate_json(json.dumps(raw))
    resources = ResourceManager.create(
        setup.layout,
        scope_id="workers",
        limits=BudgetLimits.from_budget(plan.resources),
        roots=(setup.layout.repository,),
        previously_allocated_bytes=0,
        previously_used_gpu_seconds=0.0,
        max_pending_per_stage=2,
    )
    pool = WorkerPool(
        setup.layout,
        plan,
        (setup.adapter,),
        test_command=(sys.executable, str(setup.script)),
        resources=resources,
        worker_pss_bytes=128 * 1024**2,
        worker_storage_bytes=1024**2,
    )
    setup.program(delay=0.5)
    pool.start(setup.request)
    assert settled(pool, setup.request)["state"] == "complete"
    sample = resources.sample()
    assert sample.active_by_stage == {}
    assert any(
        item.pid == os.getpid() and item.pss_bytes > 0 for item in sample.processes
    )
    assert resources.status()["requests"][0]["state"] == "complete"
    assert setup.launches() == 1


def test_worker_budget_admission_precedes_process_launch(setup: Setup) -> None:
    raw = setup.plan.model_dump(mode="json")
    raw["resources"]["minimum_free_bytes"] = 1
    plan = CampaignPlan.model_validate_json(json.dumps(raw))
    resources = ResourceManager.create(
        setup.layout,
        scope_id="workers",
        limits=BudgetLimits.from_budget(plan.resources),
        roots=(setup.layout.repository,),
        previously_allocated_bytes=0,
        previously_used_gpu_seconds=0.0,
        max_pending_per_stage=2,
    )
    pool = WorkerPool(
        setup.layout,
        plan,
        (setup.adapter,),
        test_command=(sys.executable, str(setup.script)),
        resources=resources,
        worker_pss_bytes=30 * 1024**3,
        worker_storage_bytes=0,
    )
    with pytest.raises(CapacityUnavailable, match="PSS"):
        pool.start(setup.request)
    assert setup.launches() == 0
    assert resources.status()["requests"][0]["reason"] == "PSS reservations do not fit"


@pytest.mark.parametrize("bad_log", [None, "missing", "wrong_model"])
def test_real_command_protocol_requires_exact_rollout_before_adapter_validation(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
    bad_log: str | None,
) -> None:
    """Local fake CLI exercises log ingestion, without an API or a test-command bypass."""
    codex_home = setup.layout.repository / "fake-codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    binary = setup.layout.repository / "fake-codex"
    binary.write_text(f"""#!{sys.executable}
import json, os, pathlib, sys
sys.stdin.read()
thread = "11111111-1111-1111-1111-111111111111"
print(json.dumps({{"type":"thread.started","thread_id":thread}}), flush=True)
pathlib.Path("result.json").write_text('{{"fixture":true}}')
if {bad_log!r} != "missing":
    root = pathlib.Path(os.environ["CODEX_HOME"]) / "sessions/2026/10/14"
    root.mkdir(parents=True, exist_ok=True)
    records = [
        {{"type":"session_meta","payload":{{"id":thread,"cwd":str(pathlib.Path.cwd()),"cli_version":"synthetic-protocol"}}}},
        {{"type":"turn_context","payload":{{"model":"wrong" if {bad_log!r} == "wrong_model" else "gpt-6.1-sol"}}}},
        {{"type":"token_usage_record","payload":{{"response_id":"synthetic-response","usage":{{"input_tokens":10,"output_tokens":2}}}}}},
    ]
    (root / ("rollout-fixture-" + thread + ".jsonl")).write_text("".join(json.dumps(r)+"\\n" for r in records))
print(json.dumps({{"type":"turn.completed","usage":{{}}}}), flush=True)
""")
    binary.chmod(0o700)
    setup.adapter.expects_telemetry = True
    pool = WorkerPool(
        setup.layout, setup.plan, (setup.adapter,), codex_binary=str(binary)
    )
    pool.register(setup.request)
    pool.start(setup.request)
    state = settled(pool, setup.request)
    if bad_log is None:
        assert state["state"] == "complete" and setup.adapter.validations == 1
        final = read_document(
            setup.layout,
            ArtifactRef.model_validate_json(state["result"]),
            AttemptResult,
        )
        assert final.telemetry is not None
        assert pool.reconcile(setup.request)["state"] == "complete"
    else:
        assert state["state"] == "validation_failed" and setup.adapter.validations == 0
        assert isinstance(state["reason"], str)
        assert (
            "rollout" in state["reason"]
            if bad_log == "missing"
            else "model" in state["reason"]
        )


@pytest.fixture
def resource_pool(setup: Setup) -> WorkerPool:
    raw = setup.plan.model_dump(mode="json")
    raw["resources"]["minimum_free_bytes"] = 1
    plan = CampaignPlan.model_validate_json(json.dumps(raw))
    resources = ResourceManager.create(
        setup.layout,
        scope_id="interrupted-workers",
        limits=BudgetLimits.from_budget(plan.resources),
        roots=(setup.layout.repository,),
        previously_allocated_bytes=0,
        previously_used_gpu_seconds=0.0,
        max_pending_per_stage=1,
    )
    return WorkerPool(
        setup.layout,
        plan,
        (setup.adapter,),
        test_command=(sys.executable, str(setup.script)),
        resources=resources,
        worker_pss_bytes=128 * 1024**2,
        worker_storage_bytes=1024**2,
    )


@pytest.mark.parametrize("recover_resources_first", [False, True])
@pytest.mark.parametrize("retry_immediately", [False, True])
def test_retry_recovers_resource_admission_before_worker_registration(
    setup: Setup,
    resource_pool: WorkerPool,
    monkeypatch: pytest.MonkeyPatch,
    recover_resources_first: bool,
    retry_immediately: bool,
) -> None:
    resources = resource_pool.resources
    assert resources is not None
    original_admit = resources._admit

    def interrupt(request):
        original_admit(request)
        raise RuntimeError("dispatcher interrupted after resource admission")

    with monkeypatch.context() as patch:
        patch.setattr(resources, "_admit", interrupt)
        with pytest.raises(RuntimeError, match="dispatcher interrupted"):
            resource_pool.start(setup.request)
    assert setup.launches() == 0
    resumed_resources = ResourceManager(setup.layout, resources.reference)
    if recover_resources_first:
        assert resumed_resources.status()["requests"][0]["state"] == "abandoned"
    resumed = WorkerPool(
        setup.layout,
        resource_pool.plan,
        (setup.adapter,),
        test_command=(sys.executable, str(setup.script)),
        resources=resumed_resources,
        worker_pss_bytes=resource_pool.worker_pss_bytes,
        worker_storage_bytes=resource_pool.worker_storage_bytes,
    )
    if not retry_immediately:
        with pytest.raises(WorkerStopped, match="explicit retry"):
            resumed.start(setup.request)
    attempt = resumed.start(setup.request, retry=True)
    assert attempt == setup.request.work_id + "-0002"
    assert settled(resumed, setup.request)["state"] == "complete"
    with resumed.ledger.registry.transaction() as database:
        attempts = database.execute(
            "SELECT number,state FROM worker_attempts WHERE work_id=? ORDER BY number",
            (setup.request.work_id,),
        ).fetchall()
    assert [tuple(row) for row in attempts] == [(1, "failed"), (2, "complete")]
    assert sorted(row["state"] for row in resumed_resources.status()["requests"]) == [
        "abandoned",
        "complete",
    ]
    assert setup.launches() == 1


def test_pending_resource_admission_reuses_unlaunched_attempt(
    setup: Setup, resource_pool: WorkerPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = resource_pool.resources
    assert resources is not None

    def temporarily_full(_request):
        raise CapacityUnavailable("fixture capacity is temporarily full")

    with monkeypatch.context() as patch:
        patch.setattr(resources, "_admit", temporarily_full)
        with pytest.raises(CapacityUnavailable, match="temporarily full"):
            resource_pool.start(setup.request)
    requests = resources.status()["requests"]
    assert len(requests) == 1 and requests[0]["state"] == "pending"
    assert resource_pool.start(setup.request) == setup.request.work_id + "-0001"
    assert settled(resource_pool, setup.request)["state"] == "complete"
    requests = resources.status()["requests"]
    assert len(requests) == 1 and requests[0]["state"] == "complete"
    assert setup.launches() == 1
