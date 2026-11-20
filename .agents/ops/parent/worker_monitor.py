"""Detached controller for manual worker CI and queue follow-ups."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from dispatcher.runtime import process_identity
from lib.ops_review.progression import CI_TIMEOUT, RateWait
from lib.ops_review.progression_backend import LiveBackend
from lib.ops_review.review_locks import pr_lease

from . import worker_followup, worker_slots
from .github import GitHub
from .policy import digest
from .snapshot import git
from .state import Store
from .worker_wait_sources import ci_result, queue_result

POLL_SECONDS = 10
FORWARDED_ENV = (
    "PATH",
    "CODEX_HOME",
    "GH_CONFIG_DIR",
    "XDG_CONFIG_HOME",
    "XDG_STATE_HOME",
)


def start(
    store: Store, repo: str, parent: int, child: int, worker: str, chain: str
) -> None:
    ops = Path(__file__).resolve().parents[1]
    # A transient service is forked by the user manager. setsid alone would
    # still leave a descendant of an agent runner subject to subreaper cleanup.
    options = [f"--setenv={name}" for name in FORWARDED_ENV if name in os.environ]
    queue = os.environ.get("TRAINING_QUEUE_DIR")
    options.append(
        f"--setenv=TRAINING_QUEUE_DIR={Path(queue).resolve() if queue else ''}"
    )
    result = subprocess.run(
        [
            "systemd-run",
            "--user",
            "--quiet",
            "--collect",
            f"--unit=tennis-lab-worker-wait-{chain}",
            "--service-type=notify",
            "--property=TimeoutStartSec=30s",
            "--property=UMask=0077",
            f"--working-directory={Path.cwd()}",
            f"--setenv=PYTHONPATH={ops}{os.pathsep}{ops.parents[1]}",
            *options,
            "--",
            sys.executable,
            "-m",
            "parent.worker_monitor",
            str(store.directory),
            repo,
            str(parent),
            str(child),
            worker,
            chain,
        ],
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            "could not start the user service for worker follow-up: "
            + result.stderr.strip()[:600]
        )


def _notify_ready() -> None:
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.sendto(
                b"READY=1", "\0" + address[1:] if address[0] == "@" else address
            )


@dataclass(frozen=True)
class Monitor:
    store: Store
    repo: str
    parent: int
    child: int
    worker: str
    chain: str

    @property
    def key(self) -> str:
        return f"{self.repo.casefold()}#{self.parent}"

    def bound(
        self, parents: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        row = parents.get(self.key)
        if not row or row.get("dispatch"):
            return None
        task = worker_slots.record(row, self.child, self.worker)
        record = worker_followup.current(task)
        if (
            record is None
            or record["id"] != self.chain
            or not worker_followup.same_owner(row, record)
            or record["state"] not in worker_followup.ACTIVE
        ):
            return None
        return task, record

    def claim(self) -> bool:
        with self.store.operation(self.key), self.store.locked():
            target = self.bound(self.store.parents)
            if target is None:
                return False
            _, record = target
            if record["state"] != "waiting" or record["monitor_started"]:
                return False
            record.update(monitor_started=True, process=process_identity(os.getpid()))
            self.store.save()
            return True

    def pending(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        with self.store.consistent_snapshot():
            target = self.bound(self.store.parents)
            if target is None:
                return None
            task, record = target
            if record["state"] != "waiting":
                return None
            if task.get("stopped") or self.store.parents[self.key]["tasks"][
                str(self.child)
            ].get("stopped"):
                raise ValueError("worker follow-up was stopped")
            return target

    def observe(
        self, task: dict[str, Any], record: dict[str, Any]
    ) -> dict[str, Any] | None:
        request = record["wait"]
        observe: Callable[[], dict[str, Any] | None]
        lease: AbstractContextManager[object]
        deadline: float | None
        if request["kind"] == "ci":
            backend = LiveBackend(
                {"request": {"repo": self.repo, "parent_issue": self.parent}},
                self.store,
            )
            lease = pr_lease(self.store.directory, self.repo, request["pr"])
            observe = partial(ci_result, backend, request, task["branch"])
            deadline = time.monotonic() + CI_TIMEOUT
        else:
            configured = os.environ.get("TRAINING_QUEUE_DIR")
            root = (
                Path(configured)
                if configured
                else Path(
                    git(
                        "rev-parse",
                        "--path-format=absolute",
                        "--git-common-dir",
                        cwd=Path(task["worktree"]),
                    )
                    .decode()
                    .strip()
                ).parent
                / ".training_queue"
            )
            lease = nullcontext()

            def observe_queue() -> dict[str, Any] | None:
                results = [queue_result(root, job) for job in request["job_ids"]]
                if any(result is None for result in results):
                    return None
                return {"kind": "queue", "jobs": results}

            observe = observe_queue
            deadline = None

        next_check = time.monotonic()
        with lease:
            while self.pending() is not None:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    raise ValueError("CI follow-up timed out on the reserved HEAD")
                if now >= next_check:
                    try:
                        result = observe()
                    except RateWait as exc:
                        next_check = now + max(
                            POLL_SECONDS, exc.resume_at - time.time()
                        )
                    else:
                        if result is not None:
                            return result
                        next_check = now + POLL_SECONDS
                time.sleep(min(POLL_SECONDS, max(0.0, next_check - time.monotonic())))
        return None

    def accept(
        self, observed: dict[str, Any], source: dict[str, Any]
    ) -> dict[str, Any] | None:
        with self.store.operation(self.key), self.store.locked():
            target = self.bound(self.store.parents)
            if target is None:
                return None
            task, record = target
            if record != source or record["state"] != "waiting":
                raise ValueError("worker wait changed while observing the result")
            request = record["wait"]
            record["observation"] = observed
            if task.get("stopped") or self.store.parents[self.key]["tasks"][
                str(self.child)
            ].get("stopped"):
                record.update(state="needs_parent", reason="worker_stopped")
            elif (
                request["kind"] == "ci"
                and observed["status"] == "success"
                and not request["resume_on_success"]
            ):
                record["state"] = "done"
            else:
                event = digest(
                    ["ci", request["pr"], request["head"]]
                    if request["kind"] == "ci"
                    else ["queue", sorted(request["job_ids"])]
                )
                if event in record["resumptions"]:
                    record.update(state="needs_parent", reason="event_already_consumed")
                elif len(record["resumptions"]) >= worker_followup.RESUME_LIMIT:
                    record.update(state="needs_parent", reason="automatic_resume_limit")
                else:
                    record["resumptions"].append(event)
                    record["state"] = "resuming"
            self.store.save()
            return record if record["state"] == "resuming" else None

    def fail(self, reason: str, *, unclaimed_only: bool = False) -> bool:
        with self.store.operation(self.key), self.store.locked():
            target = self.bound(self.store.parents)
            if target is None:
                return False
            _, record = target
            if unclaimed_only and record["monitor_started"]:
                return False
            record.update(state="error", reason=reason)
            self.store.save()
            return True

    def run(self) -> None:
        from .manual_workers import run_worker

        while target := self.pending():
            result = self.observe(*target)
            if result is None:
                return
            record = self.accept(result, target[1])
            if record is None:
                return
            path = Path(record["run_dir"]) / "followup.md"
            with path.open("x", encoding="utf-8") as output:
                output.write(resume_prompt(record["wait"], result))
            status, response = run_worker(
                GitHub(self.repo, store=self.store),
                self.store,
                self.parent,
                record["owner"],
                child=self.child,
                prompt=path,
                resume=True,
                restart_evidence=None,
                planning=record["phase"] == "planning",
                worker=self.worker,
                automatic=self.chain,
            )
            if status:
                self.fail(str(response.get("launch_state", "worker_runtime_failed")))
                return


def resume_prompt(wait: dict[str, Any], result: dict[str, Any]) -> str:
    if wait["kind"] == "queue":
        action = (
            "全jobが終端した。成果物を検証し、指定された記録を更新する。"
            "失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。"
        )
    elif result["status"] == "failure":
        action = (
            "対象HEADのPython CIが失敗した。失敗job・step・ログ要点を参照し、"
            "同じbranchとPRで修正する。修正をpushしたら新HEADを宣言する。"
        )
    else:
        action = (
            "対象HEADのPython CIが成功した。宣言した後続作業を行う。\n"
            + wait["next_step"]
        )
    return (
        "## parent CLI が確認した待機結果\n\n"
        + action
        + "\n親の判断が必要なら、理由と必要な判断を返す。\n\n"
        + json.dumps(result, ensure_ascii=False, indent=2)
        + "\n"
    )


def main() -> int:
    from lib.checkout_lock import guard_interactive_cli

    guard_interactive_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_dir", type=Path)
    parser.add_argument("repo")
    parser.add_argument("parent", type=int)
    parser.add_argument("child", type=int)
    parser.add_argument("worker")
    parser.add_argument("chain")
    args = parser.parse_args()
    monitor = Monitor(
        Store(args.state_dir),
        args.repo,
        args.parent,
        args.child,
        args.worker,
        args.chain,
    )
    if not monitor.claim():
        return 0
    try:
        _notify_ready()
        monitor.run()
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        monitor.fail(str(exc))
        print(f"worker follow-up: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
