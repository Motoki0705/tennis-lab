"""CLI dispatch for pinned direct annotation runtimes, including isolated trials."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from src.tasks.base.annotation.artifacts import annotation_reference, read_document
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan
from src.tasks.base.annotation.resources import BudgetStopped

from .contracts import LocalAnnotationRuntime, RecoveryInput
from .local_controller import LocalAnnotationController


def execute_local(
    layout: AnnotationLayout,
    reference: ArtifactRef,
    args: argparse.Namespace,
    operation: Path | None,
) -> None:
    runtime = read_document(layout, reference, LocalAnnotationRuntime)
    plan = read_document(layout, runtime.plan, LocalAnnotationPlan)
    if plan.phase == "benchmark":
        layout = layout.for_benchmark(plan.campaign_id)
    if args.command not in {
        "tick",
        "run",
        "status",
        "recover",
        "resume-budget",
        "report",
    }:
        raise ValueError(
            "Local annotation supports run/tick/status/recover/resume-budget/report only"
        )
    if args.command == "run" and (
        not math.isfinite(args.max_seconds) or args.max_seconds <= 0
    ):
        raise ValueError("controller run duration must be positive")
    controller = LocalAnnotationController(layout, runtime)
    if args.command == "recover":
        from .recovery import recover

        if operation is None:
            raise ValueError("Recovery requires a pinned operation")
        print(
            recover(
                controller,
                read_document(
                    layout, annotation_reference(layout, operation), RecoveryInput
                ),
            ).model_dump_json()
        )
        return
    if args.command == "resume-budget":
        print(controller.resources.resume(reason=args.reason).model_dump_json())
        return
    if args.command == "status":
        print(json.dumps(controller.status(), ensure_ascii=False))
        return
    if args.command == "report":
        from .benchmark.blank import measure_blank_plan

        print(measure_blank_plan(layout, runtime.plan).model_dump_json())
        return
    from .__main__ import compact_status, run_finished

    deadline = time.monotonic() + (args.max_seconds if args.command == "run" else 0)
    last, printed = "", 0.0
    while True:
        try:
            result = controller.cycle()
        except BudgetStopped as error:
            result = {
                "state": "budget_stopped",
                "reason": str(error),
                "progress": controller.status(),
            }
        summary = json.dumps(compact_status(result), ensure_ascii=False)
        if summary != last or time.monotonic() - printed >= 30:
            print(summary, flush=True)
            last, printed = summary, time.monotonic()
        if run_finished(result) or time.monotonic() >= deadline:
            return
        time.sleep(0.2)
