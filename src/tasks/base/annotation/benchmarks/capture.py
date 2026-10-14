"""Observe two-sided child liveness; a reserved or draining slot is not a worker."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    write_document,
)
from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from ..resources import ResourceManager
from ..workers.contracts import ChildReceipt, Launch, WorkRequest
from ..workers.processes import process_snapshot
from .contracts import (
    AttemptSnapshot,
    ConditionSpec,
    LiveWorker,
    Observation,
    WorkerBinding,
)
from .validation import condition


def directory(layout: AnnotationLayout, spec: ConditionSpec) -> Path:
    path: Path = layout.path(
        "campaigns", spec.benchmark_id, "reports", "benchmarks", spec.condition_id
    )
    return path


def _optional(layout: AnnotationLayout, path: Path) -> ArtifactRef | None:
    return annotation_reference(layout, path) if path.exists() else None


def _rows(
    layout: AnnotationLayout,
    spec: ConditionSpec,
    campaign: str,
    expected: dict[str, WorkerBinding],
) -> tuple[dict[str, int], tuple[AttemptSnapshot, ...], tuple[str, ...]]:
    states: Counter[str] = Counter()
    attempts: list[AttemptSnapshot] = []
    foreign: set[str] = set()
    if not layout.registry.exists():
        return {"unregistered": len(expected)}, (), ()
    # Opening evidence never creates or repairs a domain table.
    with sqlite3.connect(layout.registry.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        found: set[str] = set()
        if "worker_tasks" in tables:
            rows = db.execute(
                "SELECT * FROM worker_tasks WHERE campaign=?", (campaign,)
            ).fetchall()
            for row in rows:
                request = WorkRequest.model_validate_json(row["request"])
                if request.stage not in {binding.stage for binding in spec.bindings}:
                    continue
                binding = expected.get(row["id"])
                if binding is None or (
                    request.work_id != row["id"]
                    or (request.adapter, request.adapter_version, request.stage)
                    != (binding.adapter, binding.adapter_version, binding.stage)
                ):
                    raise ValueError(
                        "measured campaign contains work outside its frozen cohort/bindings"
                    )
                found.add(row["id"])
                states[row["state"]] += 1
                for attempt in db.execute(
                    "SELECT * FROM worker_attempts WHERE work_id=? ORDER BY number",
                    (row["id"],),
                ):
                    ref = ArtifactRef.model_validate_json(attempt["launch"])
                    launch = read_document(layout, ref, Launch)
                    path = layout.path(
                        "campaigns",
                        campaign,
                        "workers",
                        request.work_id,
                        f"attempt-{attempt['number']:04d}",
                    )
                    if (
                        launch.request != request
                        or launch.plan != spec.plan
                        or launch.work_id != request.work_id
                        or launch.attempt != attempt["id"]
                        or Path(launch.directory) != path
                        or layout.artifact_path(ref) != path / "launch.json"
                        or launch.resource_scope != spec.scope
                    ):
                        raise ValueError(
                            "attempt launch differs from the measured campaign/scope"
                        )
                    pointer = _optional(layout, path / "telemetry.ref.json")
                    attempts.append(
                        AttemptSnapshot(
                            launch=ref,
                            state=attempt["state"],
                            execution=_optional(layout, path / "exit.json"),
                            telemetry=None
                            if pointer is None
                            else read_document(layout, pointer, ArtifactRef),
                            child=_optional(layout, path / "child.json"),
                        )
                    )
        states["unregistered"] += len(set(expected) - found)
        if "resource_requests" in tables:
            # These scope-wide resource values may include other campaigns.
            for row in db.execute(
                "SELECT request FROM resource_requests WHERE state IN ('running','draining')"
            ):
                value = json.loads(row["request"])
                if value["plan"] != spec.plan.model_dump(mode="json"):
                    foreign.add(value["plan"]["digest"])
    return (
        {key: value for key, value in sorted(states.items()) if value},
        tuple(attempts),
        tuple(sorted(foreign)),
    )


def _live(child: ChildReceipt) -> bool:
    if child.process is None:
        return False
    snapshot = process_snapshot(child.process.pid)
    return (
        snapshot is not None
        and snapshot[1] not in {"Z", "X"}
        and snapshot[0].document() == child.process.model_dump()
    )


def live_workers(
    layout: AnnotationLayout, attempts: tuple[AttemptSnapshot, ...]
) -> tuple[tuple[LiveWorker, ...], int, str, str]:
    first_at = datetime.now(UTC).isoformat()
    candidates: list[tuple[AttemptSnapshot, Launch, ChildReceipt]] = []
    identities: set[tuple[str, int, int]] = set()
    for attempt in attempts:
        if attempt.child is None or attempt.execution is not None:
            continue
        launch = read_document(layout, attempt.launch, Launch)
        child = read_document(layout, attempt.child, ChildReceipt)
        if child.launch_digest != attempt.launch.digest:
            raise ValueError("worker child evidence belongs to another launch")
        if _live(child):
            assert child.process is not None
            identity = (
                child.process.boot_id,
                child.process.pid,
                child.process.start_ticks,
            )
            if identity in identities:
                raise ValueError("two worker attempts claim the same CLI child")
            identities.add(identity)
            candidates.append((attempt, launch, child))
    # Every retained process is live on both sides of this common cut. This is a
    # conservative simultaneous count, not a sum of independently sampled PIDs.
    cut = time.monotonic_ns()
    workers: list[LiveWorker] = []
    for attempt, launch, child in candidates:
        if _live(child):
            assert attempt.child is not None
            workers.append(
                LiveWorker(
                    launch=attempt.launch,
                    child=attempt.child,
                    stage=launch.request.stage,
                    synthetic=layout.mode == "test"
                    or launch.test_command is not None
                    or launch.test_root is not None,
                )
            )
    return tuple(workers), cut, first_at, datetime.now(UTC).isoformat()


def observe(layout: AnnotationLayout, reference: ArtifactRef) -> ArtifactRef:
    spec, plan, expected = condition(layout, reference)
    root = directory(layout, spec)
    with artifact_lock(
        layout, "benchmark-" + spec.benchmark_id + "-" + spec.condition_id
    ):
        existing = sorted((root / "observations").glob("*.json"))
        if len(existing) >= spec.maximum_observations:
            raise ValueError(
                "benchmark observation limit reached; existing evidence is retained"
            )
        for number, path in enumerate(existing, 1):
            if path.name != f"{number:06d}.json":
                raise ValueError("benchmark observations are not a contiguous sequence")
        if existing:
            previous = read_document(
                layout, annotation_reference(layout, existing[-1]), Observation
            )
            if previous.specification != reference:
                raise ValueError("benchmark condition specification changed")
        resources = ResourceManager(layout, spec.scope)
        # No budget creation, reset, resume, media acquisition or worker submission.
        resources.bind(plan)
        resources.observe(os.getpid())
        sample = resources.poll()
        states, attempts, foreign = _rows(layout, spec, plan.campaign_id, expected)
        workers, cut, first, last = live_workers(layout, attempts)
        result = Observation(
            specification=reference,
            sequence=len(existing) + 1,
            observed_at=datetime.now(UTC).isoformat(),
            cut_monotonic_ns=cut,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            first_check_at=first,
            last_check_at=last,
            resources=sample,
            live=workers,
            attempts=attempts,
            expected_work=len(expected),
            task_states=states,
            other_active_plan_digests=foreign,
        )
        return write_document(
            layout, root / "observations" / f"{result.sequence:06d}.json", result
        )
