"""Run the local annotation workflow from a pinned runtime in the shared root."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.tasks.base.annotation.artifacts import annotation_reference, read_document
from src.tasks.base.annotation.contracts import Document
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.resources import BudgetStopped
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from .contracts import CandidateManifest, OperationInput, RecoveryInput, Runtime

PATH_BOUNDARY = NonHydraPathBoundary(
    name="annotation.workflow.cli",
    fields=(
        BoundaryPathField(
            "runtime",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            required=False,
        ),
        BoundaryPathField(
            "candidates",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            required=False,
        ),
        BoundaryPathField(
            "operation",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            required=False,
        ),
        BoundaryPathField(
            "configuration",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            required=False,
        ),
    ),
)


def compact_status(status: dict[str, Any]) -> dict[str, Any]:
    if status["state"] == "budget_stopped":
        return {
            "state": status["state"],
            "reason": status["reason"],
            "progress": compact_status(status["progress"]),
        }
    return {
        key: value
        for key, value in {
            "state": status["state"],
            "collection": status.get("collection"),
            "task_states": status.get("task_states"),
            "workflow": None
            if status.get("workflow") is None
            else {
                "state": status["workflow"]["state"],
                "reason": status["workflow"].get("reason"),
                "learning_state": status["workflow"].get("learning", {}).get("state"),
            },
            "worker_states": dict(
                Counter(row["state"] for row in status.get("workers", []))
            ),
            "stage_states": dict(
                Counter(row["state"] for row in status.get("stages", []))
            ),
            "failures": [
                {"state": row["state"], "reason": row.get("reason")}
                for row in status.get("failures", [])
            ][:10],
        }.items()
        if value is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema")
    schema.add_argument(
        "--type",
        choices=(
            "configuration",
            "runtime",
            "candidates",
            "operation",
            "recovery",
            "benchmark_condition",
            "benchmark_comparison",
            "benchmark_runtime",
        ),
        required=True,
    )
    configuration = commands.add_parser("configure")
    configuration.add_argument("--configuration", type=Path, required=True)
    configuration.add_argument("--sha256", required=True)
    configuration.add_argument("--test-root", type=Path)
    for name in (
        "register",
        "submit",
        "recover",
        "status",
        "tick",
        "run",
        "resume-budget",
    ):
        command = commands.add_parser(name)
        command.add_argument("--runtime", type=Path, required=True)
        command.add_argument("--sha256", required=True)
        command.add_argument("--test-root", type=Path)
        if name == "register":
            command.add_argument("--candidates", type=Path, required=True)
        elif name in {"submit", "recover"}:
            command.add_argument("--operation", type=Path, required=True)
            if name == "submit":
                command.add_argument("--attempt", type=int)
        elif name == "run":
            command.add_argument("--max-seconds", type=float, required=True)
        elif name == "status":
            command.add_argument("--details", action="store_true")
        elif name == "resume-budget":
            command.add_argument("--reason", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--operation", type=Path, required=True)
    stage.add_argument("--sha256", required=True)
    stage.add_argument("--test-root", type=Path)
    for name in (
        "benchmark-run",
        "benchmark-tick",
        "benchmark-status",
        "benchmark-recover",
    ):
        benchmark = commands.add_parser(name)
        benchmark.add_argument("--runtime", type=Path, required=True)
        benchmark.add_argument("--sha256", required=True)
        benchmark.add_argument("--condition", required=True)
        benchmark.add_argument("--test-root", type=Path)
        if name == "benchmark-run":
            benchmark.add_argument("--max-seconds", type=float, required=True)
        elif name == "benchmark-recover":
            benchmark.add_argument("--operation", type=Path, required=True)
    for name in ("benchmark-observe", "benchmark-report", "benchmark-stage"):
        measurement = commands.add_parser(name)
        measurement.add_argument("--operation", type=Path, required=True)
        measurement.add_argument("--sha256", required=True)
        measurement.add_argument("--test-root", type=Path)
    args = parser.parse_args()
    if args.command == "schema":
        from src.tasks.base.annotation.benchmarks.contracts import (
            ComparisonSpec,
            ConditionSpec,
        )

        from .benchmark.contracts import BenchmarkRuntime
        from .configuration import Configuration

        models: dict[str, type[Document]] = {
            "configuration": Configuration,
            "runtime": Runtime,
            "candidates": CandidateManifest,
            "operation": OperationInput,
            "recovery": RecoveryInput,
            "benchmark_condition": ConditionSpec,
            "benchmark_comparison": ComparisonSpec,
            "benchmark_runtime": BenchmarkRuntime,
        }
        print(
            json.dumps(
                models[args.type].model_json_schema(), ensure_ascii=False, indent=2
            )
        )
        return
    layout = (
        AnnotationLayout.production()
        if args.test_root is None
        else AnnotationLayout.testing(args.test_root)
    )
    paths = PATH_BOUNDARY.validate(
        {
            name: getattr(args, name)
            for name in ("runtime", "candidates", "operation", "configuration")
            if getattr(args, name, None) is not None
        },
        resolver=layout.resolver,
    )
    primary = (
        "configuration"
        if args.command == "configure"
        else "operation"
        if args.command
        in {"stage", "benchmark-observe", "benchmark-report", "benchmark-stage"}
        else "runtime"
    )
    ref = annotation_reference(layout, paths.declared(primary).path)
    if ref.digest != args.sha256:
        raise ValueError("CLI input digest does not match its pinned bytes")
    if args.command == "configure":
        from .configuration import Configuration, configure

        receipt = configure(layout, read_document(layout, ref, Configuration))
        print(receipt.model_dump_json())
        return
    if args.command == "stage":
        from .operations import execute

        print(execute(layout, ref).model_dump_json())
        return
    if args.command == "benchmark-stage":
        from .benchmark.operations import execute as execute_benchmark

        print(execute_benchmark(layout, ref).model_dump_json())
        return
    if args.command in {
        "benchmark-run",
        "benchmark-tick",
        "benchmark-status",
        "benchmark-recover",
    }:
        from .benchmark.runner import BenchmarkRunner

        if args.command == "benchmark-run" and (
            not math.isfinite(args.max_seconds) or args.max_seconds <= 0
        ):
            raise ValueError("benchmark run duration must be positive")
        runner = BenchmarkRunner(layout, ref, args.condition)
        if args.command == "benchmark-recover":
            from .benchmark.recovery import recover as recover_benchmark

            recovery = read_document(
                layout,
                annotation_reference(layout, paths.declared("operation").path),
                RecoveryInput,
            )
            print(recover_benchmark(runner, recovery).model_dump_json())
            return
        if args.command == "benchmark-status":
            benchmark_result = runner.status()
        else:
            until = (
                time.monotonic() + args.max_seconds
                if args.command == "benchmark-run"
                else 0.0
            )
            while True:
                benchmark_result = runner.tick()
                if (
                    args.command != "benchmark-run"
                    or benchmark_result["state"]
                    in {"workers_complete", "blocked", "budget_stopped"}
                    or time.monotonic() >= until
                ):
                    break
                time.sleep(min(1.0, max(0.0, until - time.monotonic())))
        print(json.dumps(benchmark_result, ensure_ascii=False))
        return
    if args.command in {"benchmark-observe", "benchmark-report"}:
        from src.tasks.base.annotation.benchmarks.capture import observe
        from src.tasks.base.annotation.benchmarks.report import build_report

        measured = (
            observe(layout, ref)
            if args.command == "benchmark-observe"
            else build_report(layout, ref)
        )
        print(measured.model_dump_json())
        return
    from .controller import Controller

    controller = Controller(layout, read_document(layout, ref, Runtime))
    if args.command == "register":
        candidates = read_document(
            layout,
            annotation_reference(layout, paths.declared("candidates").path),
            CandidateManifest,
        )
        print(json.dumps(controller.register(candidates), ensure_ascii=False))
        return
    result: dict[str, Any]
    if args.command == "submit":
        operation = read_document(
            layout,
            annotation_reference(layout, paths.declared("operation").path),
            OperationInput,
        )
        result = controller.operation(operation, attempt=args.attempt)
    elif args.command == "recover":
        from .recovery import recover

        recovery = read_document(
            layout,
            annotation_reference(layout, paths.declared("operation").path),
            RecoveryInput,
        )
        result = recover(controller, recovery).model_dump(mode="json")
    elif args.command == "status":
        result = controller.status()
        if not args.details:
            result = compact_status(result)
    elif args.command == "resume-budget":
        controller.resources.resume(reason=args.reason)
        result = {"state": "budget_resumed", "reason": args.reason}
    else:
        if args.command == "run" and (
            not math.isfinite(args.max_seconds) or args.max_seconds <= 0
        ):
            raise ValueError("controller run duration must be positive")
        deadline = time.monotonic() + (args.max_seconds if args.command == "run" else 0)
        last, printed = "", 0.0
        while True:
            try:
                controller.cycle()
                result = controller.status()
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
            if (
                result["state"]
                in {"preview_ready", "complete", "blocked", "budget_stopped"}
                or time.monotonic() >= deadline
            ):
                return
            time.sleep(1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
