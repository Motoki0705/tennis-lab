"""A single read-only Codex judgement turn, with the shared runner's evidence checks."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib.agent_events import runtime_errors_failed
from parent.policy import object_value

from .records import read_json
from .runtime import Runtime, load

RUNNER = Path(__file__).resolve().parents[1] / "lib/agent_exec.sh"
OUTPUT_SCHEMA = Path(__file__).with_name("judgement.schema.json").resolve()


@dataclass
class StopRequest:
    requested: bool = False
    started: bool = False

    def check(self) -> None:
        if self.requested:
            raise InterruptedError("scheduled judgment was explicitly stopped")


def _execute(
    command: list[str], runtime_dir: Path, stop: StopRequest | None
) -> subprocess.CompletedProcess[str]:
    if stop is not None:
        stop.check()
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        if stop is not None:
            stop.started = True
        sent_stop = False
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.1 if stop else None)
                return subprocess.CompletedProcess(
                    command, process.returncode, stdout, stderr
                )
            except subprocess.TimeoutExpired:
                if stop is None or not stop.requested or sent_stop:
                    continue
                path = runtime_dir / "result.json"
                # The runner installs its stop handler before spawning the CLI.
                # A request during startup must wait for that ownership handoff.
                if path.exists():
                    record = load(path)
                    if (
                        record.get("agent_spawned") is True
                        and record.get("result") == "running"
                    ):
                        Runtime(Path(command[0])).inspect(runtime_dir, stop=True)
                        sent_stop = True


def judge(
    root: Path,
    run_dir: Path,
    timeout: float,
    runner: Path = RUNNER,
    *,
    stop: StopRequest | None = None,
) -> Any:
    if stop is not None:
        stop.check()
    input_dir = (run_dir / "agent-input").resolve()
    prompt = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8")
    prompt += f"\n\n固定入力の索引: {input_dir / 'index.json'}\n"
    prompt_path = run_dir / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    runtime_dir = run_dir / "agent-runtime"
    schema_sha256 = hashlib.sha256(OUTPUT_SCHEMA.read_bytes()).hexdigest()
    result = _execute(
        [
            str(runner),
            "--agent",
            "codex",
            "--profile",
            "issue_judgement",
            "--mode",
            "read-only",
            "--cwd",
            str(root),
            "--read-dir",
            str(input_dir),
            "--json-schema",
            str(OUTPUT_SCHEMA),
            "--task-id",
            "issue-start-judgement",
            "--prompt-file",
            str(prompt_path),
            "--run-dir",
            str(runtime_dir),
            "--timeout",
            str(timeout),
        ],
        runtime_dir,
        stop,
    )
    (run_dir / "runner.stdout").write_text(result.stdout, encoding="utf-8")
    (run_dir / "runner.stderr").write_text(result.stderr, encoding="utf-8")
    if stop is not None:
        stop.check()
    if result.returncode:
        diagnostic = result.stderr.strip()[:2048]
        failure_path = runtime_dir / "result.json"
        if failure_path.is_file():
            failure = object_value(read_json(failure_path.read_text(encoding="utf-8")))
            diagnostic = f"turn_end_reason={failure.get('turn_end_reason')}; " + str(
                failure.get("error") or diagnostic
            )
        raise ValueError(
            f"judgement runner failed with exit {result.returncode}; {diagnostic}; "
            "inspect private runtime logs"
        )
    summary = object_value(read_json(result.stdout))
    evidence = object_value(
        read_json((runtime_dir / "result.json").read_text(encoding="utf-8"))
    )
    if (
        any(
            r.get("schema_version") != 1
            or r.get("result") != "completed"
            or r.get("turn_end_reason") != "completed"
            or r.get("exit_code") != 0
            or r.get("task_success") is not None
            for r in (summary, evidence)
        )
        or summary.get("run_id") != evidence.get("run_id")
        or not evidence.get("run_id")
    ):
        raise ValueError(
            "judgement runner did not provide matching completed turn evidence"
        )
    contract = object_value(evidence.get("contract"))
    settings = object_value(contract.get("settings"))
    selected = object_value(settings.get("selected"))
    runtime = object_value(evidence.get("runtime"))
    effective = object_value(runtime.get("effective"))
    model = object_value(effective.get("model"))
    if (
        contract.get("mode") != "read-only"
        or contract.get("cwd") != str(root)
        or contract.get("read_dirs") != [str(input_dir)]
        or contract.get("json_schema_sha256") != schema_sha256
        or contract.get("task_id") != "issue-start-judgement"
        or settings.get("role") != "issue_judgement"
        or settings.get("agent") != "codex"
        or settings.get("profile") != "issue_judgement"
        or settings.get("overrides") != {}
        or selected != settings.get("defaults")
        or model.get("status") != "observed"
        or model.get("value") != selected["model"]
        or runtime_errors_failed(evidence)
        or evidence.get("cleanup_complete") is not True
    ):
        raise ValueError("judgement runtime contract/effective model/cleanup mismatch")
    return read_json((runtime_dir / "final.md").read_text(encoding="utf-8"))
