from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event

import pytest

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    write_asset,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, save_plan
from src.tasks.base.annotation.resources import (
    BudgetLimits,
    BudgetStopped,
    CapacityUnavailable,
    ResourceManager,
    ResourceRequest,
)
from src.tasks.base.annotation.resources.contracts import GPUUsage
from src.tasks.base.annotation.resources.measurements import (
    gpu_usage,
    process_usage,
    storage_usage,
    write_baseline,
)
from src.tasks.base.annotation.workers.processes import process_snapshot


@pytest.fixture(autouse=True)
def no_hardware_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.tasks.base.annotation.resources.scheduler.gpu_usage",
        lambda _pids: GPUUsage(
            status="unavailable",
            owned_vram_bytes=None,
            device_vram_bytes=None,
            maximum_utilization_percent=None,
            reason="synthetic resource tests do not query a GPU",
        ),
    )


def make_plan(
    layout: AnnotationLayout, name: str = "fixture", /, **limits: int
) -> CampaignPlan:
    ref = write_asset(
        layout, layout.path("campaigns", "inputs", "fixture.txt"), b"resource fixture"
    ).model_dump()
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
                    "codex_workers": 2,
                    "additional_storage_bytes": 100_000_000,
                    "minimum_free_bytes": 1,
                    "total_pss_bytes": 1024**3,
                    "gpu_seconds": 100,
                    "stages": [
                        {
                            "stage": stage,
                            "kind": kind,
                            "parallel": 1,
                            "threads_per_process": 1,
                        }
                        for stage, kind in [
                            ("review", "codex"),
                            ("screen", "codex"),
                            ("decode", "cpu"),
                            ("gpu", "gpu"),
                        ]
                    ],
                    **limits,
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


def manager(
    layout: AnnotationLayout, plan: CampaignPlan, *, pending: int = 2, prepaid: int = 0
) -> ResourceManager:
    return ResourceManager.create(
        layout,
        scope_id="all-cycles",
        limits=BudgetLimits.from_budget(plan.resources),
        roots=(layout.repository,),
        previously_allocated_bytes=prepaid,
        previously_used_gpu_seconds=0.0,
        max_pending_per_stage=pending,
    )


def request(
    layout: AnnotationLayout,
    plan: CampaignPlan,
    name: str,
    *,
    stage: str = "review",
    pss: int = 1024,
    storage: int = 0,
    gpu: float = 0.0,
) -> ResourceRequest:
    return ResourceRequest(
        request_id=name,
        plan=annotation_reference(layout, save_plan(layout, plan)),
        stage=stage,
        pss_bytes=pss,
        storage_bytes=storage,
        gpu_seconds=gpu,
    )


def test_allocated_storage_counts_hardlinks_once_and_never_credits_old_deletion(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    old, removable = data / "old", data / "removable"
    old.write_bytes(b"a" * 1024**2)
    removable.write_bytes(b"b" * (2 * 1024**2))
    baseline = tmp_path / "baseline.jsonl"
    write_baseline(baseline, (str(data),))
    os.link(old, data / "linked")
    before = storage_usage((str(data),), baseline, 17)
    assert before.logical_bytes > before.allocated_bytes
    assert before.additional_bytes == 17
    (data / "new").write_bytes(b"c" * 1024**2)
    removable.unlink()
    after = storage_usage((str(data),), baseline, 17)
    assert after.additional_bytes >= 1024**2 + 17
    outside = tmp_path / "outside"
    outside.write_bytes(b"d" * 1024**2)
    (data / "link").symlink_to(outside)
    assert storage_usage((str(data),), baseline, 17).additional_bytes < 2 * 1024**2


def test_budget_is_immutable_and_campaign_cannot_reset_it(tmp_path: Path) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout)
    first = manager(layout, plan)
    first.bind(plan)
    assert manager(layout, plan).reference == first.reference
    with pytest.raises(ValueError, match="immutable|existing|different"):
        manager(layout, plan, prepaid=1)
    other = ResourceManager.create(
        layout,
        scope_id="other",
        limits=first.scope.limits,
        roots=(layout.repository,),
        previously_allocated_bytes=0,
        previously_used_gpu_seconds=0.0,
        max_pending_per_stage=2,
    )
    with pytest.raises(ValueError, match="different budget scope"):
        other.bind(plan)
    with pytest.raises(ValueError, match="differs"):
        first.bind(make_plan(layout, "changed", gpu_seconds=200))


def test_pending_bound_and_immutable_request_identity(tmp_path: Path) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout)
    guard = manager(layout, plan, pending=1)
    a, b = request(layout, plan, "a"), request(layout, plan, "b")
    guard.submit(a)
    with pytest.raises(CapacityUnavailable, match="backlog"):
        guard.submit(b)
    with pytest.raises(ValueError, match="different"):
        guard.submit(a.model_copy(update={"pss_bytes": 2048}))
    guard.cancel_pending(a, reason="explicit fixture cancellation")
    guard.submit(b)
    assert guard.sample().pending_by_stage == {"review": 1}


def test_competing_campaigns_share_the_stage_capacity(tmp_path: Path) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    a, b = make_plan(layout, "a"), make_plan(layout, "b")
    guard = manager(layout, a)
    requests = [request(layout, plan, plan.campaign_id) for plan in (a, b)]
    for item in requests:
        guard.submit(item)
    barrier, release = Barrier(2), Event()

    def admit(item: ResourceRequest) -> bool:
        barrier.wait(timeout=5)
        try:
            with guard.lease(item):
                release.wait(timeout=1)
                return True
        except CapacityUnavailable:
            release.set()
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(admit, requests))
    assert sorted(result) == [False, True]
    assert guard.sample().active_by_stage == {}


@pytest.mark.parametrize(
    "costs,match",
    [
        ({"pss": 1024**3 + 1}, "PSS"),
        ({"storage": 100_000_001}, "storage"),
        ({"stage": "gpu", "gpu": 101.0}, "GPU"),
    ],
)
def test_reservation_over_budget_never_starts(
    tmp_path: Path, costs: dict[str, object], match: str
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout)
    guard = manager(layout, plan)
    item = request(layout, plan, "too-large", **costs)  # type: ignore[arg-type]
    with pytest.raises(CapacityUnavailable, match=match), guard.lease(item):
        pytest.fail("over-budget attempt entered")
    assert guard.sample().active_by_stage == {}


def test_measured_limit_latches_until_explicit_resume(tmp_path: Path) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout, additional_storage_bytes=1_000_000)
    guard = manager(layout, plan)
    added = layout.root / "large"
    added.write_bytes(b"x" * 1_100_000)
    assert guard.sample().stopped_reason == "additional storage budget reached"
    with pytest.raises(BudgetStopped, match="storage"):
        guard.resume(reason="still too large")
    added.unlink()
    item = request(layout, plan, "new")
    with pytest.raises(BudgetStopped), guard.lease(item):
        pytest.fail("latched stop ignored")
    guard.resume(reason="removed the fixture's unreferenced output")
    with guard.lease(item):
        pass


def test_real_pss_counts_controller_once_and_rejects_recycled_observer() -> None:
    snap = process_snapshot(os.getpid())
    assert snap is not None
    records = process_usage((), (snap[0], snap[0]))
    assert len(records) == 1 and records[0].pid == os.getpid()
    assert 0 < records[0].pss_bytes <= records[0].rss_bytes
    changed = type(snap[0])(
        **{**snap[0].document(), "start_ticks": snap[0].start_ticks + 1}
    )
    assert process_usage((), (changed,)) == ()


_CHILD = r"""
import json, os, pathlib, subprocess, sys, time
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.resources import ResourceManager, ResourceRequest
from src.tasks.base.annotation.workers.processes import capture_supervisor
root, ref, req, fd, mode = sys.argv[1:]
manager = ResourceManager(AnnotationLayout.testing(pathlib.Path(root)), ArtifactRef.model_validate_json(ref))
request = ResourceRequest.model_validate_json(req)
owner = capture_supervisor(os.getpid())
manager.attach(request, int(fd), owner)
pathlib.Path(root, "attached").write_text(str(os.getpid()))
if mode == "orphan":
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    os._exit(0)
if mode == "coordinated":
    assert sys.stdin.read(1) == "f"
    completed_at = manager._row(request)["started"] + 2.5
    time.monotonic = lambda: completed_at
else:
    time.sleep(.1)
manager.finish(request, reason="fixture completed")
"""


@pytest.mark.parametrize("mode", ["normal", "orphan"])
def test_inherited_lease_and_orphan_recovery_charge_gpu_conservatively(
    tmp_path: Path, mode: str
) -> None:
    # Synthetic logical GPU lease; the child only sleeps, and never touches CUDA.
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout)
    guard = manager(layout, plan)
    item = request(layout, plan, "gpu-op", stage="gpu", gpu=5.0)
    with guard.lease(item) as fd:
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CHILD,
                str(layout.root),
                guard.reference.model_dump_json(),
                item.model_dump_json(),
                str(fd),
                mode,
            ],
            pass_fds=(fd,),
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 5
        while not (layout.root / "attached").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (layout.root / "attached").exists()
        assert guard.sample().active_by_stage == {"gpu": 1}
        child.wait(timeout=5)
        if mode == "orphan":
            assert guard.sample().active_by_stage == {"gpu": 1}
        deadline = time.monotonic() + 5
        while guard.sample().active_by_stage and time.monotonic() < deadline:
            time.sleep(0.02)
        sample = guard.sample()
        assert sample.active_by_stage == {}
        assert 0 < sample.gpu_charged_seconds <= 5
        if mode == "orphan":
            assert sample.gpu_charged_seconds == 5
        # A new campaign inherits consumed time from this same budget scope.
        next_plan = make_plan(layout, "next")
        too_much = request(layout, next_plan, "next", stage="gpu", gpu=100.0)
        with pytest.raises(CapacityUnavailable, match="GPU"), guard.lease(too_much):
            pytest.fail("a new campaign reset cumulative GPU usage")
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGKILL)
        child.wait(timeout=5)


def test_lost_dispatch_before_attach_is_terminal_and_charged_once(
    tmp_path: Path,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout)
    guard = manager(layout, plan)
    item = request(layout, plan, "abandoned", stage="gpu", gpu=4.0)
    with guard.lease(item):
        pass
    assert guard.sample().gpu_charged_seconds == 4
    assert guard.sample().gpu_charged_seconds == 4
    with pytest.raises(CapacityUnavailable, match="terminal"), guard.lease(item):
        pytest.fail("same attempt executed again")


@pytest.mark.parametrize("stage", ["gpu", "decode"])
def test_recovery_preserves_finish_record_written_before_lease_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout)
    guard = manager(layout, plan)
    item = request(
        layout, plan, "finishing", stage=stage, gpu=100.0 if stage == "gpu" else 0.0
    )
    with guard.lease(item) as fd:
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CHILD,
                str(layout.root),
                guard.reference.model_dump_json(),
                item.model_dump_json(),
                str(fd),
                "coordinated",
            ],
            stdin=subprocess.PIPE,
            pass_fds=(fd,),
            start_new_session=True,
        )
    expected_charge = 2.5 if stage == "gpu" else 0.0

    @contextmanager
    def finish_before_lock(layout, name, *, blocking=True):
        if name == guard.lock_name(item):
            assert guard._row(item)["state"] == "running"
            assert child.stdin is not None
            child.stdin.write(b"f")
            child.stdin.flush()
            assert child.wait(timeout=5) == 0
            finished = guard._row(item)
            assert finished["state"] == "draining"
            assert finished["charged"] == expected_charge
        with artifact_lock(layout, name, blocking=blocking) as descriptor:
            yield descriptor

    try:
        deadline = time.monotonic() + 5
        while not (layout.root / "attached").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (layout.root / "attached").exists()
        assert guard._row(item)["state"] == "running"
        with monkeypatch.context() as patch:
            patch.setattr(
                "src.tasks.base.annotation.resources.scheduler.artifact_lock",
                finish_before_lock,
            )
            sample = guard.sample()
        row = guard._row(item)
        assert row["state"] == "complete"
        assert row["charged"] == expected_charge
        assert row["charge_kind"] == "measured_wall_seconds"
        assert row["reason"] == "fixture completed"
        assert sample.active_by_stage == {}
        assert sample.gpu_charged_seconds == expected_charge
        assert sample.stopped_reason is None
        assert guard.sample().gpu_charged_seconds == expected_charge
        next_item = request(layout, plan, "next-gpu", stage="gpu", gpu=50.0)
        with guard.lease(next_item):
            assert guard._row(next_item)["state"] == "running"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        if child.stdin is not None:
            child.stdin.close()


@pytest.mark.parametrize(
    "limits,reason",
    [({"total_pss_bytes": 1}, "PSS"), ({"minimum_free_bytes": 10**20}, "free storage")],
)
def test_memory_and_free_disk_actual_limits_latch(
    tmp_path: Path, limits: dict[str, int], reason: str
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout, **limits)
    guard = manager(layout, plan)
    guard.observe(os.getpid())
    assert reason in (guard.sample().stopped_reason or "")
    with (
        pytest.raises(BudgetStopped, match=reason),
        guard.lease(request(layout, plan, "blocked")),
    ):
        pytest.fail("measured budget limit ignored")


def test_gpu_telemetry_keeps_owned_and_device_memory_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def query(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        output = (
            "1000, 65\n" if args[1].startswith("--query-gpu=") else "12, 250\n13, 450\n"
        )
        return subprocess.CompletedProcess(args, 0, stdout=output)

    monkeypatch.setattr(subprocess, "run", query)
    result = gpu_usage({12})
    assert result.owned_vram_bytes == 250 * 1024**2
    assert result.device_vram_bytes == 1000 * 1024**2
    assert result.maximum_utilization_percent == 65.0
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="N/A, N/A\n"
        ),
    )
    unavailable = gpu_usage({12})
    assert unavailable.status == "unavailable" and unavailable.owned_vram_bytes is None


def test_completed_allocation_invalidates_cached_storage_before_admission(
    tmp_path: Path,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout, additional_storage_bytes=1_000_000)
    guard = manager(layout, plan)
    first = request(layout, plan, "writer", stage="decode", storage=500_000)
    next_item = request(layout, plan, "reader", stage="review")
    with guard.lease(first):
        (layout.root / "fixture-oversized-output").write_bytes(b"x" * 1_100_000)
    with pytest.raises(BudgetStopped, match="storage"), guard.lease(next_item):
        pytest.fail("completed writes were omitted from cached storage")
