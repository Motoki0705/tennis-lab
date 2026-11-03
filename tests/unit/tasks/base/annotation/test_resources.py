from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any

import pytest

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, save_plan
from src.tasks.base.annotation.resources import (
    BudgetLimits,
    BudgetStopped,
    CapacityUnavailable,
    ResourceManager,
    ResourceRequest,
)
from src.tasks.base.annotation.resources.contracts import BudgetResumption, GPUUsage
from src.tasks.base.annotation.resources.measurements import (
    free_bytes,
    gpu_usage,
    process_usage,
    storage_usage,
    write_baseline,
)
from src.tasks.base.annotation.storage import (
    InventoryScope,
    ManagedArtifact,
    Retention,
    StorageCatalog,
    inventory,
    plan_gc,
)
from src.tasks.base.annotation.workers.processes import (
    ProcessIdentity,
    process_snapshot,
)


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
    layout: AnnotationLayout, name: str = "fixture", /, **limits: int | None
) -> CampaignPlan:
    ref = write_asset(
        layout, layout.path("campaigns", "inputs", "fixture.txt"), b"resource fixture"
    ).model_dump()
    result: CampaignPlan = CampaignPlan.model_validate_json(
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
                "agent_effort": "max",
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

    return result


def manager(
    layout: AnnotationLayout,
    plan: CampaignPlan,
    *,
    pending: int = 2,
    prepaid: int = 0,
    free_space_mounts: tuple[Path, ...] = (),
) -> ResourceManager:
    return ResourceManager.create(
        layout,
        scope_id="all-cycles",
        limits=BudgetLimits.from_budget(plan.resources),
        roots=(layout.repository,),
        previously_allocated_bytes=prepaid,
        previously_used_gpu_seconds=0.0,
        max_pending_per_stage=pending,
        free_space_mounts=free_space_mounts,
    )


def test_host_mount_free_space_does_not_inventory_host_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, host = tmp_path / "data", tmp_path / "host"
    data.mkdir()
    host.mkdir()
    (data / "data.bin").write_bytes(b"data")
    baseline = tmp_path / "baseline.jsonl"
    write_baseline(baseline, (str(data),))
    statvfs = os.statvfs
    monkeypatch.setattr(Path, "is_mount", lambda path: path == host)
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: (
            SimpleNamespace(f_bavail=123, f_frsize=4096)
            if Path(path) == host
            else statvfs(path)
        ),
    )
    before = storage_usage((str(data),), baseline, 0, free_space_mounts=(str(host),))
    (host / "virtual-disk.bin").write_bytes(b"x" * 1024**2)
    after = storage_usage((str(data),), baseline, 0, free_space_mounts=(str(host),))
    assert before == after
    assert after.additional_bytes == 0
    assert after.free_bytes == 123 * 4096
    assert str(host) not in baseline.read_text()


@pytest.mark.parametrize("sample_method", ["sample", "watch"])
def test_host_mount_reserve_stops_full_and_cached_inventory_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_method: str
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    host = tmp_path / "host"
    host.mkdir()
    available = [1000]
    statvfs = os.statvfs
    monkeypatch.setattr(Path, "is_mount", lambda path: path == host)
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: (
            SimpleNamespace(f_bavail=available[0], f_frsize=1)
            if Path(path) == host
            else statvfs(path)
        ),
    )
    plan = make_plan(layout, minimum_free_bytes=100)
    guard = manager(layout, plan, free_space_mounts=(host,))
    guard.bind(plan)
    assert guard.sample().stopped_reason is None
    available[0] = 99
    sample = getattr(guard, sample_method)()
    assert sample is not None
    assert sample.storage.free_bytes == 99
    assert sample.stopped_reason == "free storage is below the required reserve"
    with (
        pytest.raises(BudgetStopped),
        guard.lease(request(layout, plan, "after-host-disk-filled")),
    ):
        pytest.fail("an exhausted host disk must prevent admission")


def test_missing_host_mount_is_not_measured_as_its_parent(tmp_path: Path) -> None:
    ordinary = tmp_path / "unmounted-host"
    ordinary.mkdir()
    with pytest.raises(ValueError, match="required free-space mount is unavailable"):
        free_bytes((str(tmp_path),), mounts=(str(ordinary),))
    ordinary.rmdir()
    with pytest.raises(ValueError, match="required free-space mount is unavailable"):
        free_bytes((str(tmp_path),), mounts=(str(ordinary),))


@pytest.mark.parametrize(
    "outcome", ["host_stop", "ordinary_stop", "cumulative_cap", "evidence_changed"]
)
def test_authorized_host_observation_preserves_other_limits_and_stops_live_watch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from src.tasks.base.annotation.resources.contracts import ObservedHostSpacePolicy
    from src.tasks.base.annotation.resources.host_space import activate

    layout = AnnotationLayout.testing(tmp_path / "annotation")
    host = tmp_path / "host"
    host.mkdir()
    available = {"host": 110, "ordinary": 10**12}
    monkeypatch.setattr(Path, "is_mount", lambda path: path == host)
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: SimpleNamespace(
            f_bavail=available["host" if Path(path) == host else "ordinary"], f_frsize=1
        ),
    )
    plan = make_plan(layout, minimum_free_bytes=100)
    guard = manager(layout, plan, free_space_mounts=(host,))
    item = request(layout, plan, "observed", storage=20)
    with (
        pytest.raises(CapacityUnavailable, match="storage reservations"),
        guard.lease(item),
    ):
        pytest.fail("default host reservation must stay strict")
    baseline = guard.scope.baseline
    authorization = write_asset(
        layout,
        layout.path("campaigns", "authorization.txt"),
        b"synthetic observed host-space authorization",
    )
    policy = write_document(
        layout,
        layout.path("resources", "host-policy.json"),
        ObservedHostSpacePolicy(
            scope=guard.reference,
            plan=item.plan,
            mount=str(host),
            minimum_free_bytes=10,
            authorization=authorization,
            reason="synthetic host observation",
        ),
    )
    activate(guard, policy)
    activate(guard, policy)
    available["host"] = 15
    guard = ResourceManager(layout, guard.reference)
    assert guard.scope.baseline == baseline
    sample = guard.sample()
    assert (
        sample.stopped_reason is None and sample.free_space_mount_bytes[str(host)] == 15
    )
    if outcome == "cumulative_cap":
        oversized = request(
            layout, plan, "over-budget", storage=plan.resources.additional_storage_bytes
        )
        with (
            pytest.raises(CapacityUnavailable, match="storage reservations"),
            guard.lease(oversized),
        ):
            pytest.fail("additional budget must still apply")
    elif outcome == "evidence_changed":
        layout.artifact_path(authorization).write_bytes(b"tampered authorization")
        with pytest.raises(ValueError, match="size changed|hash changed"):
            guard.watch()
    else:
        with guard.lease(item):
            available["host" if outcome == "host_stop" else "ordinary"] = 9
            with pytest.raises(BudgetStopped, match="reserve"):
                guard.watch()
        with (
            pytest.raises(BudgetStopped),
            guard.lease(request(layout, plan, "after-stop")),
        ):
            pytest.fail("an observed disk stop must persist")


@pytest.mark.parametrize("legacy_policy", [False, True])
@pytest.mark.parametrize("stop_budget", [False, True])
def test_observed_host_policy_reauthorization_preserves_scope_across_campaigns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_policy: bool,
    stop_budget: bool,
) -> None:
    from src.tasks.base.annotation.resources.contracts import ObservedHostSpacePolicy
    from src.tasks.base.annotation.resources.host_space import activate, current

    layout = AnnotationLayout.testing(tmp_path / "annotation")
    host = tmp_path / "host"
    host.mkdir()
    available = {"host": 15}
    monkeypatch.setattr(Path, "is_mount", lambda path: path == host)
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: SimpleNamespace(
            f_bavail=available["host"] if Path(path) == host else 10**12,
            f_frsize=1,
        ),
    )
    first_plan = make_plan(layout, "first", minimum_free_bytes=100)
    second_plan = make_plan(layout, "second", minimum_free_bytes=100)
    guard = manager(layout, first_plan, free_space_mounts=(host,))
    original_scope = layout.artifact_path(guard.reference).read_bytes()
    baseline = layout.artifact_path(guard.scope.baseline).read_bytes()
    first = request(layout, first_plan, "first-job", stage="gpu", gpu=10, storage=20)
    second = request(layout, second_plan, "second-job", stage="gpu", gpu=20, storage=20)
    authorization = write_asset(
        layout, layout.path("resources", "approval.txt"), b"synthetic approval"
    )
    policies = tuple(
        write_document(
            layout,
            layout.path("resources", name + "-policy.json"),
            ObservedHostSpacePolicy(
                scope=guard.reference,
                plan=item.plan,
                mount=str(host),
                minimum_free_bytes=10,
                authorization=authorization,
                reason="authorize " + name,
            ),
        )
        for name, item in (("first", first), ("second", second))
    )
    if legacy_policy:
        # The pre-fix database has only the current-policy pointer, no history rows.
        guard.bind(first_plan)
        with guard.registry.transaction() as db:
            db.execute(
                "INSERT INTO observed_host_space_policies VALUES (?,?)",
                (
                    guard.scope.scope_id,
                    policies[0].canonical_bytes().decode(),
                ),
            )
    else:
        activate(guard, policies[0])
    changed_policy = write_document(
        layout,
        layout.path("resources", "changed-policy.json"),
        read_document(layout, policies[0], ObservedHostSpacePolicy).model_copy(
            update={"minimum_free_bytes": 1}
        ),
    )
    with pytest.raises(ValueError, match="immutable"):
        activate(guard, changed_policy)
    with guard.lease(first):
        if stop_budget:
            available["host"] = 9
            with pytest.raises(BudgetStopped):
                guard.watch()
        with pytest.raises(ValueError, match="stop and drain"):
            activate(guard, policies[1])
    assert guard.sample().gpu_charged_seconds == 10
    activate(guard, policies[1])
    activate(guard, policies[1])
    active_policy = current(guard)
    assert active_policy is not None and active_policy.plan == second.plan
    assert (guard.sample().stopped_reason is not None) == stop_budget
    available["host"] = 15
    if stop_budget:
        guard.resume(reason="explicit resume for second campaign")
    with guard.lease(second):
        pass
    assert guard.sample().gpu_charged_seconds == 30
    assert layout.artifact_path(guard.reference).read_bytes() == original_scope
    assert layout.artifact_path(guard.scope.baseline).read_bytes() == baseline
    with guard.registry.transaction() as db:
        history = db.execute(
            "SELECT reference FROM observed_host_space_history WHERE scope=?",
            (guard.scope.scope_id,),
        ).fetchall()
    assert {ArtifactRef.model_validate_json(row[0]) for row in history} == set(policies)
    with pytest.raises(ValueError, match="superseded"):
        activate(guard, policies[0])


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


def retained(guard: ResourceManager) -> set[ArtifactRef]:
    return {
        reference
        for record in guard.storage.catalog.snapshot().protections
        for reference in record.references
    }


def attempt_is_protected(guard: ResourceManager, item: ResourceRequest) -> bool:
    return any(
        record.protection_id == "resource-attempt-v1-" + guard.key(item)
        for record in guard.storage.catalog.snapshot().protections
    )


def test_pending_and_cancelled_requests_retain_inputs_for_handoff(
    tmp_path: Path,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout)
    guard = manager(layout, plan)
    item = request(layout, plan, "pending")
    ref = write_asset(layout, layout.path("campaigns", "inputs", "media"), b"media")
    catalog = guard.storage.catalog
    catalog.register(
        ManagedArtifact(
            reference=ref,
            owner="fixture",
            stage="decode",
            kind="source",
            regenerable=False,
            retention=Retention(mode="expire", expires_at_ns=0, reason="fixture TTL"),
        )
    )

    def eligible() -> bool:
        observed = inventory(
            layout,
            (InventoryScope(role="annotation", path="campaigns/inputs"),),
            max_entries=100,
        )
        result: bool = next(
            row.eligible
            for row in plan_gc(catalog, observed).decisions
            if row.path == catalog.key(ref)
        )
        return result

    assert eligible()
    guard.submit(item, inputs=(ref,))
    assert not eligible()
    assert {
        item.plan,
        guard.reference,
        guard.scope.baseline,
        plan.selection,
        ref,
    } <= retained(guard)
    guard.cancel_pending(item, reason="deferred until explicit retry")
    assert not eligible()
    assert not attempt_is_protected(guard, item)


def test_request_input_registration_is_immutable_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout)
    guard = manager(layout, plan)
    item = request(layout, plan, "pending")
    ref = write_asset(layout, layout.path("campaigns", "inputs", "media"), b"media")
    original = StorageCatalog.protect

    def fail(self: StorageCatalog, record: object, **kwargs: object) -> None:
        original(self, record, **kwargs)  # type: ignore[arg-type]
        raise RuntimeError("storage commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(StorageCatalog, "protect", fail)
        with pytest.raises(RuntimeError, match="storage commit failure"):
            guard.submit(item, inputs=(ref,))
    with pytest.raises(ValueError, match="missing"):
        guard._row(item)
    assert not retained(guard)
    guard.submit(item, inputs=(ref,))
    before = guard.storage.catalog.snapshot().digest
    guard.submit(item)  # lease/attach may reuse the fixed input set
    guard.submit(item, inputs=(ref, ref))
    assert guard.storage.catalog.snapshot().digest == before
    with pytest.raises(ValueError, match="input references changed"):
        guard.submit(item, inputs=())
    assert guard.storage.catalog.snapshot().digest == before


def test_failed_admission_and_recovery_keep_state_and_pins_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout)
    guard = manager(layout, plan)
    item = request(layout, plan, "attempt")
    ref = write_asset(layout, layout.path("campaigns", "inputs", "media"), b"media")
    guard.submit(item, inputs=(ref,))
    original_admit = guard.storage.admit

    def fail_admit(*args: object, **kwargs: object) -> None:
        original_admit(*args, **kwargs)  # type: ignore[arg-type]
        raise RuntimeError("admission commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(guard.storage, "admit", fail_admit)
        with (
            pytest.raises(RuntimeError, match="admission commit failure"),
            guard.lease(item),
        ):
            pytest.fail("uncommitted admission launched work")
    assert guard._row(item)["state"] == "pending"
    assert not attempt_is_protected(guard, item)
    with guard.lease(item):
        assert attempt_is_protected(guard, item)
        guard._recover()
        assert guard._row(item)["state"] == "running"
    original_settle = guard.storage.settle

    def fail_settle(*args: object, **kwargs: object) -> None:
        original_settle(*args, **kwargs)  # type: ignore[arg-type]
        raise RuntimeError("settlement commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(guard.storage, "settle", fail_settle)
        with pytest.raises(RuntimeError, match="settlement commit failure"):
            guard._recover()
    assert guard._row(item)["state"] == "running"
    assert attempt_is_protected(guard, item)
    guard._recover()
    assert guard._row(item)["state"] == "abandoned"
    assert not attempt_is_protected(guard, item)
    assert ref in retained(guard)


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
    with pytest.raises(ValueError, match="immutable|existing|different"):
        manager(layout, plan, free_space_mounts=(Path("/"),))
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


def test_uncapped_gpu_keeps_accounting_and_other_reservations(tmp_path: Path) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout, gpu_seconds=None)
    guard = manager(layout, plan)
    assert guard.scope.limits.gpu_seconds is None
    item = request(layout, plan, "uncapped", stage="gpu", gpu=200_000.0)
    with guard.lease(item):
        sample = guard.sample()
        assert sample.gpu_reserved_seconds == 200_000.0
        assert sample.stopped_reason is None
    # An unattached orphan is still charged conservatively; uncapped does not
    # discard the usage record or relax PSS/storage admission.
    sample = guard.sample()
    assert sample.gpu_charged_seconds == 200_000.0
    assert sample.stopped_reason is None
    too_large = request(
        layout,
        plan,
        "large",
        stage="gpu",
        gpu=200_000.0,
        pss=plan.resources.total_pss_bytes + 1,
    )
    with pytest.raises(CapacityUnavailable, match="PSS"), guard.lease(too_large):
        pytest.fail("uncapped GPU bypassed PSS reservations")


def test_gpu_limit_is_required_even_when_explicitly_uncapped(tmp_path: Path) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout, gpu_seconds=None)
    values = plan.resources.model_dump(mode="json")
    del values["gpu_seconds"]
    with pytest.raises(ValueError, match="gpu_seconds"):
        type(plan.resources).model_validate_json(json.dumps(values))


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
    reference = guard.resume(reason="removed the fixture's unreferenced output")
    resumption = read_document(layout, reference, BudgetResumption)
    assert resumption.scope == guard.reference
    assert resumption.reason == "removed the fixture's unreferenced output"
    assert (
        resumption.checked_sample.stopped_reason == "additional storage budget reached"
    )
    assert resumption.checked_sample.storage.additional_bytes < 1_000_000
    with guard.registry.transaction() as db:
        row = db.execute("SELECT * FROM resource_resumptions").fetchone()
    assert ArtifactRef.model_validate_json(row["reference"]) == reference
    assert row["scope"] == guard.scope.scope_id
    with guard.lease(item):
        pass


@pytest.mark.parametrize("boundary", ["evidence", "commit", "sample_cache"])
def test_resume_reason_and_stop_clear_survive_interruption_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    guard = manager(layout, make_plan(layout))
    original_scope = guard.scope.canonical_bytes()
    with guard.registry.transaction() as db:
        db.execute(
            "UPDATE resource_scopes SET stop_reason=? WHERE id=?",
            ("fixture maintenance", guard.scope.scope_id),
        )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected interruption")

    if boundary == "evidence":
        monkeypatch.setattr(
            "src.tasks.base.annotation.resources.scheduler.write_document", fail
        )
    elif boundary == "commit":
        with guard.registry.transaction() as db:
            db.execute(
                "CREATE TRIGGER fail_resume BEFORE UPDATE ON resource_scopes "
                "WHEN NEW.stop_reason IS NULL BEGIN "
                "SELECT RAISE(ABORT, 'injected interruption'); END"
            )
    else:
        monkeypatch.setattr(guard, "_record", fail)
    with pytest.raises((OSError, sqlite3.IntegrityError), match="injected"):
        guard.resume(reason="maintenance finished")

    with guard.registry.transaction() as db:
        stop = db.execute(
            "SELECT stop_reason FROM resource_scopes WHERE id=?",
            (guard.scope.scope_id,),
        ).fetchone()[0]
        rows = db.execute("SELECT reference FROM resource_resumptions").fetchall()
    if boundary == "sample_cache":
        assert stop is None and len(rows) == 1
        reference = ArtifactRef.model_validate_json(rows[0][0])
        resumption = read_document(layout, reference, BudgetResumption)
        assert resumption.reason == "maintenance finished"
        assert resumption.scope == guard.reference
        assert resumption.checked_sample.stopped_reason == "fixture maintenance"
    else:
        assert stop == "fixture maintenance" and rows == []
    assert guard.scope.canonical_bytes() == original_scope


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


@pytest.mark.parametrize("denied_file", ["smaps_rollup", "io"])
@pytest.mark.parametrize("after", ["missing", "Z", "X", "recycled", "live"])
def test_procfs_denial_rechecks_liveness_without_assuming_zero(
    monkeypatch: pytest.MonkeyPatch,
    denied_file: str,
    after: str,
) -> None:
    snapshot = process_snapshot(os.getpid())
    assert snapshot is not None
    identity, initial_state = snapshot
    original_read = Path.read_text
    calls = 0

    def current(pid: int) -> tuple[ProcessIdentity, str] | None:
        nonlocal calls
        if pid != identity.pid:
            other: tuple[ProcessIdentity, str] | None = process_snapshot(pid)
            return other
        calls += 1
        if calls == 1 or after == "live":
            return identity, initial_state
        if after == "missing":
            return None
        if after == "recycled":
            return replace(identity, start_ticks=identity.start_ticks + 1), "R"
        return identity, after

    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == Path("/proc") / str(identity.pid) / denied_file:
            raise PermissionError("injected procfs access refusal")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(
        "src.tasks.base.annotation.resources.measurements.process_snapshot", current
    )
    monkeypatch.setattr(Path, "read_text", read)
    if after == "live":
        with pytest.raises(PermissionError, match="injected procfs"):
            process_usage((), (identity,))
    else:
        assert process_usage((), (identity,)) == ()
    assert calls == 2


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
assert sys.stdin.read(1) == "f"
completed_at = manager._row(request)["started"] + 2.5
time.monotonic = lambda: completed_at
manager.finish(request, reason="fixture completed")
if mode == "draining":
    os.close(int(fd))
    pathlib.Path(root, "finished").write_text("lease released; process remains")
    while not pathlib.Path(root, "may_exit").exists():
        time.sleep(.02)
"""


@pytest.mark.parametrize("mode", ["normal", "orphan", "draining"])
def test_inherited_lease_and_orphan_recovery_charge_gpu_conservatively(
    tmp_path: Path, mode: str
) -> None:
    # Synthetic GPU lease with pipe synchronization and an injected clock.
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout)
    guard = manager(layout, plan)
    item = request(layout, plan, "gpu-op", stage="gpu", gpu=5.0)
    with guard.lease(item) as fd:
        assert attempt_is_protected(guard, item)
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
            stdin=subprocess.PIPE,
            pass_fds=(fd,),
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 5
        while not (layout.root / "attached").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (layout.root / "attached").exists()
        assert guard.sample().active_by_stage == {"gpu": 1}
        if mode != "orphan":
            assert child.stdin is not None
            child.stdin.write(b"f")
            child.stdin.flush()
        if mode == "draining":
            deadline = time.monotonic() + 5
            while (
                not (layout.root / "finished").exists() and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert (layout.root / "finished").exists()
            assert guard.sample().active_by_stage == {"gpu": 1}
            assert guard._row(item)["state"] == "draining"
            assert attempt_is_protected(guard, item)
            (layout.root / "may_exit").touch()
        child.wait(timeout=5)
        if mode == "orphan":
            assert guard.sample().active_by_stage == {"gpu": 1}
            assert attempt_is_protected(guard, item)
        deadline = time.monotonic() + 5
        while guard.sample().active_by_stage and time.monotonic() < deadline:
            time.sleep(0.02)
        sample = guard.sample()
        assert sample.active_by_stage == {}
        assert not attempt_is_protected(guard, item)
        assert item.plan in retained(guard)
        assert sample.gpu_charged_seconds == (5 if mode == "orphan" else 2.5)
        # A new campaign inherits consumed time from this same budget scope.
        next_plan = make_plan(layout, "next")
        too_much = request(layout, next_plan, "next", stage="gpu", gpu=100.0)
        with pytest.raises(CapacityUnavailable, match="GPU"), guard.lease(too_much):
            pytest.fail("a new campaign reset cumulative GPU usage")
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGKILL)
        child.wait(timeout=5)
        if child.stdin is not None:
            child.stdin.close()


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


def test_deleted_baseline_anchor_cannot_credit_a_surviving_hardlink(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    anchor = data / "anchor"
    anchor.write_bytes(b"a" * 1024**2)
    baseline = tmp_path / "baseline.jsonl"
    write_baseline(baseline, (str(data),))
    os.link(anchor, data / "survivor")
    anchor.unlink()
    assert storage_usage((str(data),), baseline, 0).additional_bytes >= 1024**2


def test_inventory_cache_detects_same_size_tampering_even_with_restored_mtime(
    tmp_path: Path,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout)
    guard = manager(layout, plan)
    path = layout.artifact_path(guard.scope.baseline)
    prior = path.stat()
    original = path.read_bytes()
    path.write_bytes(b"!" + original[1:])
    os.utime(path, ns=(prior.st_atime_ns, prior.st_mtime_ns))
    with pytest.raises(ValueError, match="hash changed"):
        ResourceManager(layout, guard.reference)


def test_slow_inventory_does_not_block_watchdog_or_release_unmeasured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = make_plan(layout, additional_storage_bytes=1_000_000)
    guard = manager(layout, plan)
    first = request(layout, plan, "writer", stage="decode", storage=500_000)
    with guard.lease(first):
        (layout.root / "oversized").write_bytes(b"x" * 1_100_000)
    entered, release = Event(), Event()

    def slow(*args: object, **kwargs: object):
        entered.set()
        assert release.wait(timeout=5)
        return storage_usage(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "src.tasks.base.annotation.resources.scheduler.storage_usage", slow
    )
    later = request(layout, plan, "reader", stage="review")

    def admit() -> None:
        with guard.lease(later):
            pytest.fail("unmeasured oversized output released capacity")

    with ThreadPoolExecutor(max_workers=2) as pool:
        full = pool.submit(guard.sample)
        assert entered.wait(timeout=2)
        try:
            started = time.monotonic()
            watch = guard.watch()
            assert watch is not None
            assert time.monotonic() - started < 1
            assert watch.gpu.status == "unavailable"
            waiting = pool.submit(admit)
            assert not waiting.done()
        finally:
            release.set()
        assert (
            full.result(timeout=5).stopped_reason == "additional storage budget reached"
        )
        with pytest.raises(BudgetStopped):
            waiting.result(timeout=5)


def test_samples_retain_monotonic_storage_time_across_wall_clock_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    layout = AnnotationLayout.testing(tmp_path / "annotation")
    guard = manager(layout, make_plan(layout))
    first = guard.sample()

    class Clock:
        @staticmethod
        def now(_timezone: object) -> datetime:
            return datetime(2000, 1, 1, tzinfo=UTC)

    monkeypatch.setattr("src.tasks.base.annotation.resources.scheduler.datetime", Clock)
    second = guard._sample(refresh_storage=False)
    assert second.measured_at < first.measured_at
    assert first.measured_monotonic_ns is not None
    assert second.measured_monotonic_ns >= first.measured_monotonic_ns
    assert second.storage_measured_monotonic_ns == first.storage_measured_monotonic_ns
    assert second.clock_boot_id == first.clock_boot_id
