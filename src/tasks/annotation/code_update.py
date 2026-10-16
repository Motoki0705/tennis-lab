"""Record a tested code update without changing campaign data, policy, or budget."""

from __future__ import annotations

import hashlib

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, Document
from src.tasks.base.annotation.reproducibility import (
    CodeRevision,
    checkout_identity,
    require_frozen_code,
)
from src.tasks.base.annotation.workers.pool import WorkerPool
from src.utils.paths import PROJECT_ROOT

from .contracts import Runtime
from .controller import Controller


class CodeUpdateReceipt(Document):
    revision: ArtifactRef
    runtime: ArtifactRef


def update_code(
    controller: Controller,
    previous_runtime: ArtifactRef,
    verification: ArtifactRef,
    *,
    reason: str,
) -> ArtifactRef:
    layout, plan = controller.layout, controller.plan
    previous = read_document(layout, previous_runtime, Runtime)
    if previous != controller.runtime or not reason.strip():
        raise ValueError("code update requires the exact current runtime and a reason")
    verify_reference(layout, verification)
    commit, diff = checkout_identity(PROJECT_ROOT)
    if diff != hashlib.sha256(b"").hexdigest():
        raise ValueError("commit the verified code update before recording it")
    revision = CodeRevision(
        plan=previous.plan,
        scope=previous.scope,
        previous_runtime=previous_runtime,
        code_commit=commit,
        code_diff_digest=diff,
        verification=verification,
        reason=reason,
    )
    directory = layout.path(
        "campaigns", plan.campaign_id, "code-updates", revision.sha256
    )
    with artifact_lock(layout, "workflow-recovery-" + plan.campaign_id, blocking=False):
        result_path = directory / "result.json"
        if result_path.exists():
            result_ref: ArtifactRef = annotation_reference(layout, result_path)
            result = read_document(layout, result_ref, CodeUpdateReceipt)
            if read_document(layout, result.revision, CodeRevision) != revision:
                raise ValueError("code update receipt differs from its request")
            saved = read_document(layout, result.runtime, Runtime)
            if saved != previous.model_copy(update={"code_revision": result.revision}):
                raise ValueError("code update changed runtime settings")
            require_frozen_code(
                PROJECT_ROOT,
                plan,
                layout=layout,
                revision=result.revision,
                scope=previous.scope,
            )
            return result_ref
        controller.resources.poll()
        # The operator must stop the controller first. Serialize against admission
        # while checking both resource owners and worker supervisors.
        with (
            artifact_lock(layout, "worker-admission", blocking=False),
            artifact_lock(layout, "resource-admission", blocking=False),
        ):
            with controller.resources.registry.transaction() as db:
                active = db.execute(
                    "SELECT count(*) FROM resource_requests WHERE scope=? AND state IN ('running','draining')",
                    (controller.resources.scope.scope_id,),
                ).fetchone()[0]
                workers = db.execute(
                    "SELECT state,current_attempt FROM worker_tasks WHERE campaign=?",
                    (plan.campaign_id,),
                ).fetchall()
            if active or any(row["state"] in {"running", "exited"} for row in workers):
                raise ValueError("stop and drain campaign jobs before updating code")
            status = controller.status()
            if any(
                row["state"] in {"running", "draining", "queued"}
                for row in status["stages"]
            ):
                raise ValueError("queued or live stages prevent a code update")
            for path in layout.path("campaigns", plan.campaign_id, "workers").glob(
                "*/*/owner.json"
            ):
                WorkerPool._require_no_members(path.parent)
            revision_ref = write_document(layout, directory / "revision.json", revision)
            require_frozen_code(
                PROJECT_ROOT,
                plan,
                layout=layout,
                revision=revision_ref,
                scope=previous.scope,
            )
            runtime_ref = write_document(
                layout,
                directory / "runtime.json",
                previous.model_copy(update={"code_revision": revision_ref}),
            )
            completed: ArtifactRef = write_document(
                layout,
                result_path,
                CodeUpdateReceipt(revision=revision_ref, runtime=runtime_ref),
            )
            return completed
