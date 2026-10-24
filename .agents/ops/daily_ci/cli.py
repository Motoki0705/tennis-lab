"""Production entrypoint: always reads the host clock; no date/now override."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import CycleAPI, CycleStore
from .controller import IO_ERRORS, Controller
from .github import GitHub
from .model import window
from .store import load


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--repo", required=True)
    commands = root.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--baseline-sha", required=True)
    initialize.add_argument("--workflow", action="append")
    initialize.add_argument("--integrations", type=Path)
    for name in ("status", "enable", "disable", "tick", "watchdog", "run"):
        commands.add_parser(name)
    show = commands.add_parser("cycle")
    show.add_argument("--cycle-id", required=True)
    repair = commands.add_parser("register-repair")
    repair.add_argument("--cycle-id", required=True)
    repair.add_argument("--pr", required=True, type=int)
    repair.add_argument("--head-sha", required=True)
    repair.add_argument("--signature", action="append", required=True)
    worker = commands.add_parser("register-worker")
    worker.add_argument("--cycle-id", required=True)
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--handle-file", required=True, type=Path)
    permit = commands.add_parser("permit-worker")
    permit.add_argument("--cycle-id", required=True)
    permit.add_argument("--worker-id")
    observation = commands.add_parser("record-result")
    observation.add_argument("--input", required=True, type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        api = CycleAPI(CycleStore(args.repo))
        controller = Controller(api, GitHub(api.store.repository))
        value: object = {"schema_version": 1, "acknowledged": True}
        errors = []
        if args.command == "init":
            value = api.initialize(
                args.baseline_sha,
                workflows=args.workflow,
                integrations=load(args.integrations.read_text())
                if args.integrations
                else None,
            )
        elif args.command == "status":
            value = api.snapshot()
        elif args.command == "cycle":
            value = api.snapshot()["cycles"][args.cycle_id]
        elif args.command == "enable":
            controller.enable()
        elif args.command == "disable":
            controller.disable()
            errors = controller.watchdog()
        elif args.command == "watchdog":
            errors = controller.watchdog()
            value = {"schema_version": 1, "errors": errors, "state": api.snapshot()}
        elif args.command in {"tick", "run"}:
            while True:
                try:
                    result = controller.tick()
                    errors = result["errors"]
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                except IO_ERRORS as exc:
                    errors = [f"{type(exc).__name__}: {exc}"]
                    print(json.dumps({"errors": errors}), file=sys.stderr, flush=True)
                now = api.clock()
                _, opens, deadline = window(now)
                if (
                    args.command == "tick"
                    or not opens <= now < deadline
                    or not api.snapshot()["enabled"]
                ):
                    return 1 if errors else 0
                time.sleep(min(10, max(0, (deadline - now).total_seconds())))
        elif args.command == "register-repair":
            api.register_repair(args.cycle_id, args.pr, args.head_sha, args.signature)
        elif args.command == "register-worker":
            api.register_worker(
                args.cycle_id, args.worker_id, load(args.handle_file.read_text())
            )
        elif args.command == "permit-worker":
            value = api.permit_worker(args.cycle_id, args.worker_id)
        elif args.command == "record-result":
            api.record_result(load(args.input.read_text()))
        print(json.dumps(value, ensure_ascii=False))
        return 1 if errors else 0
    except IO_ERRORS as exc:
        print(f"daily_ci: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
