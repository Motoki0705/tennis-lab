"""Local schedule, capacity, bootstrap and runtime supervision commands."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from lib.checkout_lock import (
    checkout_barrier,
    checkout_lock,
    registry_checkout_lock,
    require_checkout_job,
    source_checkout_barrier,
)
from parent.github import GitHub
from parent.policy import nonempty
from parent.state import state_directory
from worker_logs.sources import read_original

from .agent import RUNNER
from .diagnostics import README, DispatcherError, report_failure
from .runtime import gate, load
from .schedule import Schedule, local_time
from .service import recover_judgement, run_due, stop_judgement
from .state import DispatchStore
from .supervisor import Supervisor


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--state-dir", type=Path)
    result.add_argument("--parent-state-dir", type=Path)
    sub = result.add_subparsers(dest="command", required=True)
    init = sub.add_parser(
        "init", help="create disabled schedule; never overwrite existing state"
    )
    init.add_argument("--repo", required=True)
    init.add_argument("--cwd", type=Path, default=Path.cwd())
    init.add_argument("--agent-exec", type=Path, default=RUNNER)
    init.add_argument("--max-parents", type=int, default=1)
    init.add_argument("--max-children", type=int, default=2)
    init.add_argument("--judgement-timeout", type=float, default=1800)
    init.add_argument("--now")
    status = sub.add_parser("status")
    status.add_argument("--now")
    for name in ("enable", "disable", "shutdown", "recover-judgement"):
        command = sub.add_parser(name)
        if name == "recover-judgement":
            command.add_argument("--evidence-file", type=Path, required=True)
    config = sub.add_parser(
        "configure", help="limits for new parents; active child limits remain pinned"
    )
    config.add_argument("--max-parents", type=int, required=True)
    config.add_argument("--max-children", type=int, required=True)
    run = sub.add_parser(
        "run", help="one due slot, or one explicitly requested judgment"
    )
    run.add_argument("--explicit", action="store_true")
    run.add_argument("--now")
    supervise = sub.add_parser("supervise")
    supervise.add_argument("--once", action="store_true")
    accept = sub.add_parser(
        "accept",
        help="起動された親が予約を受理する",
        description="bootstrapの両stateを共通引数に指定して予約を受理する。続きは .agents/skills/parent-session/references/dispatch.md",
    )
    accept.add_argument(
        "--input",
        type=Path,
        required=True,
        help="起動指示末尾のbootstrap_inputが指すJSONファイル",
    )
    stop = sub.add_parser("stop")
    stop.add_argument("--parent", type=int, required=True)
    resume = sub.add_parser("resume")
    resume.add_argument("--parent", type=int, required=True)
    resume.add_argument("--reason-file", type=Path, required=True)
    child = sub.add_parser(
        "start-child",
        help="保存済み委任の起動を予約する",
        description="delegate済みの委任IDを使う復旧入口。通常の自律親は parent start-worker を使う。",
    )
    child.add_argument("--parent", type=int, required=True)
    child.add_argument("--child", type=int, required=True)
    child.add_argument("--delegation-id", required=True)
    child.add_argument("--worker", help="primary or the named helper")
    child.add_argument("--prompt-file", type=Path, required=True)
    from lib.agent_settings import implementation_models

    try:
        models = implementation_models()
    except (OSError, ValueError):
        result.error("Cannot read implementation model settings")
    child.add_argument(
        "--model",
        help=f"initial model ({', '.join(models)}); resume inherits the saved model",
    )
    return result


def limits(parents: int, children: int) -> None:
    if parents < 1 or children < 1:
        raise DispatcherError(
            "invalid_limits",
            "Execution limits must be positive integers.",
            "Set positive values for `--max-parents` and `--max-children` and retry.",
        )


def execute(args: argparse.Namespace) -> dict[str, Any]:
    parent_directory = (args.parent_state_dir or state_directory()).absolute()
    schedule = Schedule(args.state_dir or parent_directory.parent / "dispatcher")
    store = DispatchStore(parent_directory, schedule)
    if args.command == "init":
        limits(args.max_parents, args.max_children)
        if not math.isfinite(args.judgement_timeout) or args.judgement_timeout <= 0:
            raise DispatcherError(
                "invalid_timeout",
                "Judgment timeout must be finite and positive.",
                "Set a positive number of seconds for `--judgement-timeout` and retry.",
            )
        with store.locked():
            store.save()
            schedule.initialize(
                {
                    "repo": nonempty(args.repo, "repository").casefold(),
                    "cwd": str(args.cwd.resolve()),
                    "runner": str(args.agent_exec.resolve()),
                    "max_parents": args.max_parents,
                    "max_children": args.max_children,
                    "judgement_timeout": args.judgement_timeout,
                },
                local_time(args.now),
            )
        return {"result": "initialized", "enabled": False}
    with store.locked():
        state = schedule.read_optional()
        if state is None:
            raise DispatcherError(
                "schedule_missing",
                "dispatcher/schedule.json is missing.",
                f"Verify the state directory. Restore a verified schedule backup for an existing installation or follow initial setup in {README}.",
                state_path="dispatcher/schedule.json",
            )
    supervisor = Supervisor(
        GitHub(state["config"]["repo"], store=store), store, schedule
    )
    if args.command == "status":
        return supervisor.status(local_time(args.now))
    if args.command in {"enable", "disable", "configure"}:
        with store.locked():
            state = schedule.read()
            if args.command == "configure":
                limits(args.max_parents, args.max_children)
                state["config"].update(
                    max_parents=args.max_parents, max_children=args.max_children
                )
            else:
                state["enabled"] = args.command == "enable"
            schedule.write(state)
        return {"result": args.command}
    if args.command == "run":
        return run_due(
            supervisor.gh, store, schedule, local_time(args.now), explicit=args.explicit
        )
    if args.command == "accept":
        return supervisor.accept(load(args.input), dict(os.environ))
    if args.command == "resume":
        supervisor.resume(args.parent, args.reason_file.read_text(encoding="utf-8"))
    elif args.command == "start-child":
        job = supervisor.start_child(
            args.parent,
            args.child,
            args.delegation_id,
            read_original(args.prompt_file),
            dict(os.environ),
            args.model,
            worker=args.worker,
        )
        return {"result": "reserved", "job_id": job}
    elif args.command == "recover-judgement":
        recover_judgement(
            store, schedule, args.evidence_file.read_text(encoding="utf-8")
        )
    elif args.command == "stop":
        supervisor.reconcile(stop=True, parent=args.parent)
    elif args.command == "shutdown":
        stop_judgement(store, schedule)
        supervisor.reconcile(stop=True)
    elif args.command == "supervise":
        stopping = False
        pausing = False

        def stop(signum: int, frame: object) -> None:
            nonlocal stopping
            stopping = True

        def pause(signum: int, frame: object) -> None:
            # Maintenance holds ops-code exclusively and has verified no jobs.
            nonlocal stopping, pausing
            pausing = stopping = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGHUP, pause)
        barrier = source_checkout_barrier()
        job_barrier = checkout_barrier(Path(state["config"]["cwd"]))
        with gate(schedule.directory / "supervisor.lock") as fd:
            if fd is None:
                return {"result": "busy"}
            while not stopping:
                with checkout_lock(
                    Path(state["config"]["cwd"]), blocking=False
                ) as job_fd:
                    if job_fd is not None:
                        with registry_checkout_lock(blocking=False) as code_fd:
                            if (
                                code_fd is not None
                                and not barrier.exists()
                                and not job_barrier.exists()
                            ):
                                supervisor.reconcile()
                                supervisor.assign()
                                supervisor.launch_reserved()
                if args.once:
                    break
                time.sleep(1)
            if stopping and not pausing:
                stop_judgement(store, schedule)
                supervisor.reconcile(stop=True)
    return {"result": args.command}


def main(argv: list[str] | None = None, *, _supervisor_launcher: bool = False) -> int:
    args = parser().parse_args(argv)
    if args.command == "run":
        require_checkout_job()
    if args.command == "supervise" and not _supervisor_launcher:
        print(
            "dispatcher supervise: use checkout_job.py --repo <main> --supervisor",
            file=sys.stderr,
        )
        return 75
    os.umask(0o077)
    try:
        result = execute(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2 if result.get("result") in {"failed", "recovery_required"} else 0
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(json.dumps(report_failure(exc)))
        return 2
