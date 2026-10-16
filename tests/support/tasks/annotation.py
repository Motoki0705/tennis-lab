"""Synthetic stage receipts; never start a queue, GPU, or model."""

from src.tasks.annotation.contracts import OperationInput, OperationReceipt
from src.tasks.annotation.controller import Controller
from src.tasks.base.annotation.artifacts import write_asset, write_document
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.execution.contracts import StageExit, StageLaunch
from src.tasks.base.annotation.resources import ResourceRequest
from src.utils.paths import PROJECT_ROOT


def record_stage(
    controller: Controller,
    data: OperationInput,
    *,
    attempt: int = 1,
    results: tuple[ArtifactRef, ...] | None = None,
    exited: bool = False,
    failed: bool = False,
) -> ArtifactRef:
    """Publish synthetic supervisor evidence without starting a queue or model."""
    layout = controller.layout
    request_id = f"{data.sha256}-{attempt}"
    directory = layout.path(
        "campaigns", controller.plan.campaign_id, "jobs", request_id
    )
    operation = write_document(layout, directory / "operation.json", data)
    cost = controller.runtime.cost(data.stage)
    command: tuple[str, ...] = (
        str(PROJECT_ROOT / ".venv/bin/python"),
        "-m",
        "src.tasks.annotation",
        "stage",
        "--operation",
        str(layout.artifact_path(operation)),
        "--sha256",
        operation.digest,
        "--attempt",
        str(attempt),
    )
    if layout.mode == "test":
        command += ("--test-root", str(layout.root))
    launch = write_document(
        layout,
        directory / "launch.json",
        StageLaunch(
            scope=controller.runtime.scope,
            resource=ResourceRequest(
                request_id=request_id,
                plan=controller.runtime.plan,
                stage=data.stage,
                pss_bytes=cost.pss_bytes,
                storage_bytes=cost.storage_bytes,
                gpu_seconds=cost.gpu_seconds,
            ),
            command=command,
            input=operation,
            code_root=str(PROJECT_ROOT),
            timeout_seconds=cost.timeout_seconds,
            queue=controller.runtime.queue if cost.gpu_seconds else None,
            test_root=str(layout.root),
        ),
    )
    if results is not None:
        write_document(
            layout,
            directory / "result.json",
            OperationReceipt(
                operation=operation,
                status="complete",
                results=results,
                reason=None,
            ),
        )
    if exited:
        log = write_asset(layout, directory / "log.txt", b"synthetic supervisor\n")
        write_document(
            layout,
            directory / "exit.json",
            StageExit(
                launch=launch,
                status="failed" if failed else "succeeded",
                exit_code=1 if failed else 0,
                started_at="2026-10-16T00:00:00Z",
                ended_at="2026-10-16T00:00:01Z",
                elapsed_seconds=1.0,
                stdout=log,
                stderr=log,
                reason="synthetic failure" if failed else None,
            ),
        )
    return operation
