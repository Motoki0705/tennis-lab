"""Explicit recovery restricted to this condition's frozen review workers."""

from __future__ import annotations

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.plan import references
from src.tasks.base.annotation.workers.contracts import WorkRequest

from ..contracts import RecoveryInput, RecoveryReceipt, ResumeWorkers, RetryWorker
from .runner import BenchmarkRunner


def recover(runner: BenchmarkRunner, data: RecoveryInput) -> ArtifactRef:
    if (
        data.plan != runner.spec.plan
        or not data.reason.strip()
        or not isinstance(data.operation, (ResumeWorkers, RetryWorker))
    ):
        raise ValueError(
            "benchmark recovery requires its exact plan, a reason and a worker operation"
        )
    layout, op = runner.layout, data.operation
    for ref in references(data):
        verify_reference(layout, ref)
    with artifact_lock(
        layout, "benchmark-run-" + runner.comparison.benchmark_id, blocking=False
    ):
        runner._initialize()
        assert runner.pool is not None
        if isinstance(op, RetryWorker) and op.request not in runner.requests:
            raise ValueError("benchmark recovery worker is outside its frozen cohort")
        directory = layout.path(
            "campaigns", runner.plan.campaign_id, "benchmark", "recovery", data.sha256
        )
        request_ref = write_document(layout, directory / "request.json", data)
        output = directory / "result.json"
        if output.exists():
            ref = annotation_reference(layout, output)
            receipt = read_document(layout, ref, RecoveryReceipt)
            if receipt.request != request_ref:
                raise ValueError("benchmark recovery result belongs to another request")
            for dep in references(receipt):
                verify_reference(layout, dep)
            return ref
        receipt = RecoveryReceipt(request=request_ref)
        if isinstance(op, ResumeWorkers):
            runner.pool.ledger.resume_campaign(runner.plan.campaign_id)
        else:
            request = read_document(layout, op.request, WorkRequest)
            row = runner.pool.reconcile(request)
            if row["current_attempt"] != op.after_attempt:
                raise ValueError("worker attempt changed; inspect its current state")
            if row["state"] not in {
                "failed",
                "deferred",
                "cancelled",
                "quota",
                "timed_out",
                "validation_failed",
            }:
                raise ValueError(
                    "only a failed or stopped benchmark worker can be retried"
                )
            attempt = runner.pool.start(request, retry=True)
            if attempt is None:
                raise ValueError("benchmark worker already completed")
            receipt = RecoveryReceipt(request=request_ref, worker_attempt=attempt)
        result: ArtifactRef = write_document(layout, output, receipt)
        return result
