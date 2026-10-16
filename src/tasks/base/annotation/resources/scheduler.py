"""Admission is serialized across campaigns. Kernel leases outlive the dispatcher."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    sync_directory,
    verify_reference,
    write_document,
)
from ..contracts import ArtifactRef, content_digest
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, load_plan
from ..registry import Registry
from ..workers.processes import ProcessIdentity, owned_members, process_snapshot
from .contracts import (
    BudgetLimits,
    BudgetScope,
    GPUUsage,
    ResourceRequest,
    ResourceSample,
)
from .measurements import (
    free_bytes,
    gpu_usage,
    process_usage,
    storage_usage,
    write_baseline,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS resource_scopes (
 id TEXT PRIMARY KEY, reference TEXT NOT NULL, stop_reason TEXT
);
CREATE TABLE IF NOT EXISTS resource_campaigns (
 campaign TEXT PRIMARY KEY, scope TEXT NOT NULL, plan TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resource_requests (
 id TEXT PRIMARY KEY, scope TEXT NOT NULL, campaign TEXT NOT NULL,
 request TEXT NOT NULL, stage TEXT NOT NULL, kind TEXT NOT NULL,
 state TEXT NOT NULL, owner TEXT, started REAL, boot TEXT,
 charged REAL NOT NULL DEFAULT 0, charge_kind TEXT, reason TEXT
);
CREATE TABLE IF NOT EXISTS resource_observers (
 identity TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS resource_samples (
 scope TEXT PRIMARY KEY, sample TEXT NOT NULL, storage_at REAL NOT NULL,
 measured_at REAL NOT NULL, boot TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resource_baselines (
 digest TEXT PRIMARY KEY, fingerprint TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resource_watchdogs (
 scope TEXT PRIMARY KEY, sample TEXT NOT NULL, measured_at REAL NOT NULL, boot TEXT NOT NULL
);
"""


class CapacityUnavailable(RuntimeError):
    """Bounded backlog or reservations are full; no process was started."""


class BudgetStopped(RuntimeError):
    """Persistent stop, requiring an explicit resume after the cause is resolved."""


class ResourceManager:
    def __init__(self, layout: AnnotationLayout, reference: ArtifactRef) -> None:
        benchmark = layout.review_namespace is not None
        layout = layout.shared()
        self.layout, self.reference = layout, reference
        self.scope = read_document(layout, reference, BudgetScope)
        self.registry = Registry(layout)
        if benchmark:
            self._require_registered_scope()
            self._baseline_path()
            return
        self.registry.initialize("annotation.resources.v1", _SCHEMA)
        self._baseline_path()
        with self.registry.transaction() as db:
            row = db.execute(
                "SELECT reference FROM resource_scopes WHERE id=?",
                (self.scope.scope_id,),
            ).fetchone()
            encoded = reference.canonical_bytes().decode()
            if row is not None and row[0] != encoded:
                raise ValueError("budget scope is immutable")
            db.execute(
                "INSERT OR IGNORE INTO resource_scopes(id,reference) VALUES (?,?)",
                (self.scope.scope_id, encoded),
            )

    def _require_registered_scope(self) -> None:
        error = "benchmark must reuse an exact registered shared scope"
        if not self.layout.registry.exists():
            raise ValueError(error)
        with sqlite3.connect(
            self.layout.registry.as_uri() + "?mode=ro", uri=True
        ) as db:
            table = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='resource_scopes'"
            ).fetchone()
            if table is None:
                raise ValueError(error)
            row = db.execute(
                "SELECT reference FROM resource_scopes WHERE id=?",
                (self.scope.scope_id,),
            ).fetchone()
            if row is None or row[0] != self.reference.canonical_bytes().decode():
                raise ValueError(error)

    def _baseline_path(self) -> Path:
        """Cache verified immutable inventory bytes, invalidated by any file change."""
        ref = self.scope.baseline
        path: Path = self.layout.artifact_path(ref)

        def fingerprint() -> str:
            stat = path.stat()
            result: str = content_digest(
                [
                    str(path),
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                    stat.st_mode,
                ]
            )
            return result

        with artifact_lock(self.layout, "resource-baseline-" + ref.digest):
            before = fingerprint()
            with self.registry.transaction() as db:
                cached = db.execute(
                    "SELECT fingerprint FROM resource_baselines WHERE digest=?",
                    (ref.digest,),
                ).fetchone()
            if cached is None or cached[0] != before:
                verify_reference(self.layout, ref)
                if fingerprint() != before:
                    raise ValueError("resource baseline changed during verification")
                with self.registry.transaction() as db:
                    db.execute(
                        "INSERT OR REPLACE INTO resource_baselines VALUES (?,?)",
                        (ref.digest, before),
                    )
        return path

    @classmethod
    def create(
        cls,
        layout: AnnotationLayout,
        *,
        scope_id: str,
        limits: BudgetLimits,
        roots: tuple[Path, ...],
        previously_allocated_bytes: int,
        previously_used_gpu_seconds: float,
        max_pending_per_stage: int,
    ) -> ResourceManager:
        if layout.review_namespace is not None:
            raise ValueError(
                "benchmark review must reuse an existing shared resource scope"
            )
        values = dict(
            scope_id=scope_id,
            limits=limits,
            roots=tuple(str(p) for p in roots),
            previously_allocated_bytes=previously_allocated_bytes,
            previously_used_gpu_seconds=previously_used_gpu_seconds,
            max_pending_per_stage=max_pending_per_stage,
        )
        # Validate every boundary before making the baseline; retries never replace it.
        BudgetScope(
            **values,
            baseline=ArtifactRef(
                role="annotation", path="pending", digest="0" * 64, size_bytes=0
            ),
        )
        if not any(layout.root.is_relative_to(root) for root in roots):
            raise ValueError("storage inventory must include the annotation root")
        if layout.mode == "production" and not any(
            layout.repository == root for root in roots
        ):
            raise ValueError(
                "production inventory must include the complete main repository"
            )
        directory = layout.path("resources", scope_id)
        with artifact_lock(layout, "resource-scope-" + scope_id):
            directory.mkdir(parents=True, exist_ok=True)
            write_document(
                layout,
                directory / "inputs.json",
                BudgetScope(
                    **values,
                    baseline=ArtifactRef(
                        role="annotation", path="pending", digest="0" * 64, size_bytes=0
                    ),
                ),
            )
            path = directory / "scope.json"
            if path.exists():
                reference = annotation_reference(layout, path)
                prior = read_document(layout, reference, BudgetScope)
                if any(getattr(prior, key) != value for key, value in values.items()):
                    raise ValueError(
                        "cannot reset or change an existing resource budget"
                    )
            else:
                baseline = directory / "baseline.jsonl"
                if not baseline.exists():
                    temporary = directory / "baseline.partial"
                    # Only an incomplete, unregistered inventory from this scope may be retried.
                    temporary.unlink(missing_ok=True)
                    write_baseline(temporary, tuple(str(root) for root in roots))
                    os.replace(temporary, baseline)
                    sync_directory(directory)
                reference = write_document(
                    layout,
                    path,
                    BudgetScope(
                        **values, baseline=annotation_reference(layout, baseline)
                    ),
                )
        return cls(layout, reference)

    def bind(self, plan: CampaignPlan) -> None:
        if (
            plan.annotation_root != str(self.layout.root)
            or BudgetLimits.from_budget(plan.resources) != self.scope.limits
        ):
            raise ValueError(
                "campaign budget/root differs from its persistent resource scope"
            )
        with self.registry.transaction() as db:
            row = db.execute(
                "SELECT scope,plan FROM resource_campaigns WHERE campaign=?",
                (plan.campaign_id,),
            ).fetchone()
            if row is not None and tuple(row) != (self.scope.scope_id, plan.sha256):
                raise ValueError(
                    "campaign already belongs to a different budget scope or plan"
                )
            db.execute(
                "INSERT OR IGNORE INTO resource_campaigns VALUES (?,?,?)",
                (plan.campaign_id, self.scope.scope_id, plan.sha256),
            )

    def observe(self, pid: int) -> None:
        """Count the controller itself; workers and GPU children are counted by group."""
        snapshot = process_snapshot(pid)
        if snapshot is None:
            raise ProcessLookupError(pid)
        with self.registry.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO resource_observers VALUES (?)",
                (json.dumps(snapshot[0].document(), sort_keys=True),),
            )

    def key(self, request: ResourceRequest) -> str:
        result: str = content_digest([self.scope.scope_id, request.request_id])
        return result

    def lock_name(self, request: ResourceRequest) -> str:
        return "resource-lease-" + self.key(request)

    def _row(self, request: ResourceRequest) -> dict[str, Any]:
        with self.registry.transaction() as db:
            row = db.execute(
                "SELECT * FROM resource_requests WHERE id=?", (self.key(request),)
            ).fetchone()
        if row is None or row["request"] != request.canonical_bytes().decode():
            raise ValueError("resource request is missing or was changed")
        return dict(row)

    def submit(self, request: ResourceRequest) -> None:
        plan = load_plan(verify_reference(self.layout, request.plan))
        self.bind(plan)
        stages = [
            stage for stage in plan.resources.stages if stage.stage == request.stage
        ]
        if len(stages) != 1:
            raise ValueError("resource stage is not declared by the plan")
        stage = stages[0]
        if (stage.kind == "gpu") != (request.gpu_seconds > 0):
            raise ValueError(
                "GPU stages require a positive time reservation; other stages require zero"
            )
        with self.registry.transaction() as db:
            row = db.execute(
                "SELECT request FROM resource_requests WHERE id=?", (self.key(request),)
            ).fetchone()
            if row is not None:
                if row[0] != request.canonical_bytes().decode():
                    raise ValueError(
                        "resource request identity was reused with different costs/inputs"
                    )
                return
            pending = db.execute(
                "SELECT count(*) FROM resource_requests WHERE stage=? AND state='pending'",
                (request.stage,),
            ).fetchone()[0]
            if pending >= self.scope.max_pending_per_stage:
                raise CapacityUnavailable("shared stage backlog is full")
            db.execute(
                "INSERT INTO resource_requests(id,scope,campaign,request,stage,kind,state) VALUES (?,?,?,?,?,?,'pending')",
                (
                    self.key(request),
                    self.scope.scope_id,
                    plan.campaign_id,
                    request.canonical_bytes().decode(),
                    request.stage,
                    stage.kind,
                ),
            )

    def _active(self) -> list[dict[str, Any]]:
        with self.registry.transaction() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM resource_requests WHERE state IN ('running','draining')"
                )
            ]

    def _recover(self) -> None:
        for active in self._active():
            try:
                with (
                    artifact_lock(
                        self.layout, "resource-lease-" + active["id"], blocking=False
                    ),
                    self.registry.transaction() as db,
                ):
                    # finish() can commit after the active snapshot and before
                    # lease acquisition. Read and settle its latest record atomically.
                    row = db.execute(
                        "SELECT * FROM resource_requests WHERE id=?",
                        (active["id"],),
                    ).fetchone()
                    if row is None:
                        raise ValueError("resource request disappeared during recovery")
                    if row["state"] not in {"running", "draining"}:
                        continue
                    if row["owner"] is not None and owned_members(
                        ProcessIdentity.from_document(json.loads(row["owner"]))
                    ):
                        continue
                    request = ResourceRequest.model_validate_json(row["request"])
                    missing = row["state"] == "running"
                    db.execute(
                        "UPDATE resource_requests SET state=?,charged=?,charge_kind=?,reason=? WHERE id=? AND state=?",
                        (
                            "abandoned" if missing else "complete",
                            request.gpu_seconds if missing else row["charged"],
                            "reserved_upper_bound" if missing else row["charge_kind"],
                            "owner exited without a finish receipt"
                            if missing
                            else row["reason"],
                            row["id"],
                            row["state"],
                        ),
                    )
                    # Finished output must be measured before releasing its reservation.
                    db.execute("UPDATE resource_samples SET storage_at=-1")
            except BlockingIOError:
                continue

    def _sample(
        self, *, refresh_storage: bool = True, watchdog: bool = False
    ) -> ResourceSample:
        active = self._active()
        with self.registry.transaction() as db:
            observers = tuple(
                ProcessIdentity(**json.loads(row[0]))
                for row in db.execute("SELECT identity FROM resource_observers")
            )
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM resource_requests WHERE scope=?",
                    (self.scope.scope_id,),
                )
            ]
            stopped = db.execute(
                "SELECT stop_reason FROM resource_scopes WHERE id=?",
                (self.scope.scope_id,),
            ).fetchone()[0]
            cached = db.execute(
                "SELECT * FROM resource_samples WHERE scope=?", (self.scope.scope_id,)
            ).fetchone()
        owners = tuple(
            ProcessIdentity.from_document(json.loads(row["owner"]))
            for row in active
            if row["owner"] is not None
        )
        processes = process_usage(owners, observers)
        boot, now = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            time.monotonic(),
        )
        measured_at = datetime.now(UTC).isoformat()
        if (
            not refresh_storage
            and cached is not None
            and cached["boot"] == boot
            and (watchdog or now - cached["storage_at"] < 30)
        ):
            previous = ResourceSample.model_validate_json(cached["sample"])
            storage, storage_measured_at = (
                previous.storage.model_copy(
                    update={"free_bytes": free_bytes(self.scope.roots)}
                ),
                previous.storage_measured_at,
            )
        else:
            if watchdog:
                raise RuntimeError("watchdog requires an admission-time storage sample")
            storage = storage_usage(
                self.scope.roots,
                self._baseline_path(),
                self.scope.previously_allocated_bytes,
            )
            storage_measured_at = measured_at
        active_seconds = sum(
            max(0.0, now - row["started"])
            for row in rows
            if row["kind"] == "gpu"
            and row["state"] == "running"
            and row["started"] is not None
            and row["boot"] == boot
        )
        return ResourceSample(
            measured_at=measured_at,
            storage_measured_at=storage_measured_at,
            scope_id=self.scope.scope_id,
            storage=storage,
            processes=processes,
            gpu=GPUUsage(
                status="unavailable",
                owned_vram_bytes=None,
                device_vram_bytes=None,
                maximum_utilization_percent=None,
                reason="watchdog measures PSS/free space; GPU telemetry is recorded by controller samples",
            )
            if watchdog
            else gpu_usage({item.pid for item in processes}),
            pss_bytes=sum(p.pss_bytes for p in processes),
            rss_bytes=sum(p.rss_bytes for p in processes),
            gpu_charged_seconds=self.scope.previously_used_gpu_seconds
            + sum(row["charged"] for row in rows),
            gpu_active_seconds=active_seconds,
            gpu_reserved_seconds=sum(
                ResourceRequest.model_validate_json(row["request"]).gpu_seconds
                for row in rows
                if row["state"] == "running"
            ),
            active_by_stage=dict(Counter(row["stage"] for row in active)),
            pending_by_stage=dict(
                Counter(row["stage"] for row in rows if row["state"] == "pending")
            ),
            stopped_reason=stopped,
        )

    def _problem(self, sample: ResourceSample) -> str | None:
        limit = self.scope.limits
        if sample.storage.additional_bytes >= limit.additional_storage_bytes:
            return "additional storage budget reached"
        if sample.storage.free_bytes < limit.minimum_free_bytes:
            return "free storage is below the required reserve"
        if sample.pss_bytes >= limit.total_pss_bytes:
            return "owned-process PSS budget reached"
        if sample.gpu_charged_seconds + sample.gpu_active_seconds >= limit.gpu_seconds:
            return "cumulative GPU time budget reached"
        return None

    def _record(self, sample: ResourceSample) -> ResourceSample:
        problem = self._problem(sample)
        if problem is not None and sample.stopped_reason is None:
            with self.registry.transaction() as db:
                db.execute(
                    "UPDATE resource_scopes SET stop_reason=? WHERE id=? AND stop_reason IS NULL",
                    (problem, self.scope.scope_id),
                )
            sample = sample.model_copy(update={"stopped_reason": problem})
        path = self.layout.path("resources", self.scope.scope_id, "samples.jsonl")
        with path.open("ab") as handle:
            handle.write(sample.canonical_bytes() + b"\n")
            handle.flush()
        with self.registry.transaction() as db:
            prior = db.execute(
                "SELECT storage_at FROM resource_samples WHERE scope=?",
                (self.scope.scope_id,),
            ).fetchone()
            now = time.monotonic()
            storage_at = (
                now
                if sample.storage_measured_at == sample.measured_at or prior is None
                else prior[0]
            )
            db.execute(
                "INSERT OR REPLACE INTO resource_samples VALUES (?,?,?,?,?)",
                (
                    self.scope.scope_id,
                    sample.canonical_bytes().decode(),
                    storage_at,
                    now,
                    Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                ),
            )
        return sample

    def sample(self) -> ResourceSample:
        with artifact_lock(self.layout, "resource-admission"):
            self._recover()
            return self._record(self._sample())

    def poll(self) -> ResourceSample:
        """One shared PSS sample per two seconds and disk scan per thirty seconds."""
        with artifact_lock(self.layout, "resource-admission"):
            with self.registry.transaction() as db:
                cached = db.execute(
                    "SELECT * FROM resource_samples WHERE scope=?",
                    (self.scope.scope_id,),
                ).fetchone()
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if (
                cached is not None
                and cached["boot"] == boot
                and time.monotonic() - cached["measured_at"] < 2
            ):
                result: ResourceSample = ResourceSample.model_validate_json(
                    cached["sample"]
                )
                return result
            self._recover()
            return self._record(self._sample(refresh_storage=False))

    def watch(self) -> ResourceSample | None:
        """Bounded supervisor check, independent of slow inventory/admission scans.

        It never releases reservations, runs a disk inventory, invokes nvidia-smi,
        or overwrites the admission cache. The controller owns full samples.
        """
        try:
            with artifact_lock(self.layout, "resource-watchdog", blocking=False):
                boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                with self.registry.transaction() as db:
                    cached = db.execute(
                        "SELECT * FROM resource_watchdogs WHERE scope=?",
                        (self.scope.scope_id,),
                    ).fetchone()
                if (
                    cached is not None
                    and cached["boot"] == boot
                    and time.monotonic() - cached["measured_at"] < 2
                ):
                    result: ResourceSample = ResourceSample.model_validate_json(
                        cached["sample"]
                    )
                    return result
                sample: ResourceSample = self._sample(
                    refresh_storage=False, watchdog=True
                )
                problem = self._problem(sample)
                with self.registry.transaction() as db:
                    if problem is not None:
                        db.execute(
                            "UPDATE resource_scopes SET stop_reason=? WHERE id=? AND stop_reason IS NULL",
                            (problem, self.scope.scope_id),
                        )
                    stopped = db.execute(
                        "SELECT stop_reason FROM resource_scopes WHERE id=?",
                        (self.scope.scope_id,),
                    ).fetchone()[0]
                    sample = sample.model_copy(update={"stopped_reason": stopped})
                    db.execute(
                        "INSERT OR REPLACE INTO resource_watchdogs VALUES (?,?,?,?)",
                        (
                            self.scope.scope_id,
                            sample.canonical_bytes().decode(),
                            time.monotonic(),
                            boot,
                        ),
                    )
                path = self.layout.path(
                    "resources", self.scope.scope_id, "watchdog.jsonl"
                )
                with path.open("ab") as handle:
                    handle.write(sample.canonical_bytes() + b"\n")
                    handle.flush()
                return sample
        except BlockingIOError:
            return None

    def resume(self, *, reason: str) -> None:
        if not reason.strip():
            raise ValueError("resuming requires an explicit reason")
        with artifact_lock(self.layout, "resource-admission"):
            self._recover()
            sample = self._sample()
            if problem := self._problem(sample):
                raise BudgetStopped(problem)
            with self.registry.transaction() as db:
                db.execute(
                    "UPDATE resource_scopes SET stop_reason=NULL WHERE id=?",
                    (self.scope.scope_id,),
                )
            write_document(
                self.layout,
                self.layout.path(
                    "resources",
                    self.scope.scope_id,
                    "resume-" + str(time.time_ns()) + ".json",
                ),
                sample,
            )
            self._record(sample.model_copy(update={"stopped_reason": None}))

    def _admit(self, request: ResourceRequest) -> None:
        with artifact_lock(self.layout, "resource-admission"):
            self._recover()
            row = self._row(request)
            if row["state"] != "pending":
                raise CapacityUnavailable(
                    "resource attempt is already running or terminal; never resubmit it"
                )
            sample = self._record(self._sample(refresh_storage=False))
            if sample.stopped_reason is not None:
                raise BudgetStopped(sample.stopped_reason)
            active = self._active()
            plan = load_plan(verify_reference(self.layout, request.plan))
            plans = [plan] + [
                load_plan(
                    verify_reference(
                        self.layout,
                        ResourceRequest.model_validate_json(item["request"]).plan,
                    )
                )
                for item in active
            ]
            stage_limit = min(
                stage.parallel
                for item in plans
                for stage in item.resources.stages
                if stage.stage == request.stage
            )
            if any(
                stage.kind != row["kind"]
                for item in plans
                for stage in item.resources.stages
                if stage.stage == request.stage
            ):
                raise ValueError("the same shared stage cannot change resource kind")
            if sum(item["stage"] == request.stage for item in active) >= stage_limit:
                raise CapacityUnavailable("shared stage capacity is full")
            if row["kind"] == "codex" and sum(
                item["kind"] == "codex" for item in active
            ) >= min(item.resources.codex_workers for item in plans):
                raise CapacityUnavailable("shared Codex capacity is full")
            requests = [
                ResourceRequest.model_validate_json(item["request"]) for item in active
            ]
            # Count measured PSS once, then reserve each group's unused peak.
            headroom = 0
            for active_row, active_request in zip(active, requests, strict=True):
                owner = (
                    None
                    if active_row["owner"] is None
                    else ProcessIdentity.from_document(json.loads(active_row["owner"]))
                )
                measured = (
                    0
                    if owner is None
                    else sum(
                        item.pss_bytes
                        for item in sample.processes
                        if (item.group, item.session) == (owner.group, owner.session)
                    )
                )
                headroom += max(0, active_request.pss_bytes - measured)
            if sample.pss_bytes + headroom + request.pss_bytes > min(
                item.resources.total_pss_bytes for item in plans
            ):
                raise CapacityUnavailable("PSS reservations do not fit")
            storage_reserved = sum(item.storage_bytes for item in requests)
            if (
                sample.storage.additional_bytes
                + storage_reserved
                + request.storage_bytes
                > self.scope.limits.additional_storage_bytes
                or sample.storage.free_bytes - storage_reserved - request.storage_bytes
                < self.scope.limits.minimum_free_bytes
            ):
                raise CapacityUnavailable("storage reservations do not fit")
            if (
                sample.gpu_charged_seconds
                + sample.gpu_reserved_seconds
                + request.gpu_seconds
                > self.scope.limits.gpu_seconds
            ):
                raise CapacityUnavailable(
                    "remaining cumulative GPU budget cannot cover this reservation"
                )
            with self.registry.transaction() as db:
                db.execute(
                    "UPDATE resource_requests SET state='running' WHERE id=? AND state='pending'",
                    (self.key(request),),
                )

    @contextmanager
    def lease(self, request: ResourceRequest) -> Iterator[int]:
        """Pass this FD to the supervisor before releasing the dispatcher's copy."""
        self.submit(request)
        try:
            with artifact_lock(
                self.layout, self.lock_name(request), blocking=False
            ) as descriptor:
                self._admit(request)
                yield descriptor
        except BlockingIOError as error:
            raise CapacityUnavailable(
                "resource lease is held by another process"
            ) from error
        except (CapacityUnavailable, BudgetStopped) as error:
            with self.registry.transaction() as db:
                db.execute(
                    "UPDATE resource_requests SET reason=? WHERE id=? AND state='pending'",
                    (str(error), self.key(request)),
                )
            raise

    def status(self) -> dict[str, Any]:
        sample = self.sample()
        with self.registry.transaction() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM resource_requests WHERE scope=? ORDER BY id",
                    (self.scope.scope_id,),
                )
            ]
        plans = {
            row["campaign"]: load_plan(
                verify_reference(
                    self.layout,
                    ResourceRequest.model_validate_json(row["request"]).plan,
                )
            )
            for row in rows
        }
        return {
            "scope": self.scope.model_dump(mode="json"),
            "sample": sample.model_dump(mode="json"),
            "configured": {
                name: plan.resources.model_dump(mode="json")
                for name, plan in plans.items()
            },
            "requests": rows,
        }

    def attach(
        self, request: ResourceRequest, descriptor: int, owner: ProcessIdentity
    ) -> None:
        lock_path = self.layout.path(
            "cache", "locks", self.lock_name(request) + ".lock"
        )
        if (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino) != (
            lock_path.stat().st_dev,
            lock_path.stat().st_ino,
        ):
            raise ValueError("resource supervisor did not inherit its exact lease")
        if owner not in owned_members(owner):
            raise ValueError("resource process owner is not alive")
        encoded = json.dumps(owner.document(), sort_keys=True)
        with artifact_lock(self.layout, "resource-admission"):
            row = self._row(request)
            if row["state"] != "running" or (
                row["owner"] is not None and row["owner"] != encoded
            ):
                raise ValueError("resource owner cannot change")
            if row["owner"] is None:
                with self.registry.transaction() as db:
                    db.execute(
                        "UPDATE resource_requests SET owner=?,started=?,boot=? WHERE id=?",
                        (encoded, time.monotonic(), owner.boot_id, self.key(request)),
                    )

    def finish(self, request: ResourceRequest, *, reason: str | None = None) -> None:
        row = self._row(request)
        if row["state"] == "draining":
            return
        if row["state"] != "running" or row["owner"] is None or row["started"] is None:
            raise ValueError("only an attached running resource lease can finish")
        owner = ProcessIdentity.from_document(json.loads(row["owner"]))
        current = process_snapshot(os.getpid())
        if (
            current is None
            or current[0].group != owner.group
            or current[0].session != owner.session
        ):
            raise ValueError("only the owning supervisor group can finish its lease")
        elapsed = max(0.0, time.monotonic() - row["started"])
        charged = elapsed if request.gpu_seconds else 0.0
        with self.registry.transaction() as db:
            db.execute(
                "UPDATE resource_requests SET state='draining',charged=?,charge_kind='measured_wall_seconds',reason=? WHERE id=?",
                (charged, reason, self.key(request)),
            )

    def cancel_pending(self, request: ResourceRequest, *, reason: str) -> None:
        if not reason.strip():
            raise ValueError("cancelling requires a reason")
        with artifact_lock(self.layout, "resource-admission"):
            if self._row(request)["state"] != "pending":
                raise ValueError("only pending resource requests can be cancelled here")
            with self.registry.transaction() as db:
                db.execute(
                    "UPDATE resource_requests SET state='cancelled',reason=? WHERE id=?",
                    (reason, self.key(request)),
                )
