"""Public runner operations and a lifetime lock inherited by its launcher."""

from __future__ import annotations

import fcntl
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from parent.policy import object_value

from .records import read_json

OPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS / "lib"))
from opsproc import process_identity  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = object_value(read_json(path.read_text(encoding="utf-8")))
    return value


def alive(identity: dict[str, Any] | None) -> bool:
    return identity is not None and process_identity(identity["pid"]) == identity


@contextmanager
def gate(path: Path) -> Iterator[int | None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
        else:
            yield stream.fileno()


class Runtime:
    def __init__(self, runner: Path) -> None:
        self.runner = runner.resolve()
        self.stopping = False
        self.launcher: subprocess.Popen[bytes] | None = None

    def request_stop(self) -> None:
        self.stopping = True
        if self.launcher is not None:
            self.launcher.send_signal(signal.SIGTERM)

    def stop_requested(self) -> bool:
        return self.stopping

    def inspect(self, directory: Path, *, stop: bool = False) -> dict[str, Any]:
        proc = subprocess.run(
            [
                str(self.runner),
                "stop" if stop else "status",
                "--run-dir",
                str(directory),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode:
            raise ValueError(
                "runner status/stop failed; inspect private runtime records"
            )
        summary = object_value(read_json(proc.stdout))
        record = load(directory / "result.json")
        if (
            summary.get("schema_version") != 1
            or record.get("schema_version") != 1
            or not summary.get("run_id")
            or summary["run_id"] != record.get("run_id")
            or any(
                type(summary.get(k)) is not bool
                for k in ("runner_alive", "agent_alive", "agent_group_alive")
            )
        ):
            raise ValueError("invalid runner status evidence")
        return {"summary": summary, "record": record}

    def ended(self, evidence: dict[str, Any]) -> bool:
        summary, record = evidence["summary"], evidence["record"]
        return bool(
            record.get("result") in {"completed", "failed"}
            and record.get("cleanup_complete") is True
            and not any(
                summary[k] for k in ("runner_alive", "agent_alive", "agent_group_alive")
            )
        )

    def spawn(
        self,
        store_dir: Path,
        key: str,
        job_id: str,
        directory: Path,
        *,
        schedule_dir: Path,
    ) -> None:
        # The worker claims under Store.locked(). Two launchers may reach Popen
        # after a crash; only one can transition a reservation to starting.
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (directory / "worker.log").open("ab") as log:
            subprocess.Popen(
                [
                    sys.executable,
                    str(OPS / "dispatcher/worker.py"),
                    "--store",
                    str(store_dir),
                    "--state-dir",
                    str(schedule_dir),
                    "--key",
                    key,
                    "--job",
                    job_id,
                ],
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

    def execute(self, job: dict[str, Any], gate_fd: int) -> int:
        if self.stop_requested():
            return 143
        directory = Path(job["directory"])
        cmd = [str(self.runner)]
        if job["resume_from"] is not None:
            cmd += ["resume", "--resume-from", job["resume_from"]]
        cmd += [
            "--role",
            job["role"],
            "--mode",
            "write",
            "--cwd",
            job["cwd"],
            "--task-id",
            job["task_id"],
            "--prompt-file",
            str(directory / "prompt.md"),
            "--run-dir",
            str(directory / "runtime"),
        ]
        if job.get("branch"):
            cmd += ["--branch", job["branch"]]
        if job.get("model"):
            cmd += ["--model", job["model"]]
        with (
            (directory / "runner.stdout").open("wb") as out,
            (directory / "runner.stderr").open("wb") as err,
            subprocess.Popen(cmd, stdout=out, stderr=err, pass_fds=(gate_fd,)) as proc,
        ):
            self.launcher = proc
            # A TERM received during Popen must also stop the newly created
            # launcher, before result.json necessarily exists.
            if self.stop_requested():
                self.request_stop()
            return proc.wait()
