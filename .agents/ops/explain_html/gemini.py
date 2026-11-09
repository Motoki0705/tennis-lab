"""Tool-free, pinned agy turns with private evidence and caller cancellation.

This is deliberately separate from the issue-supplement read-only tool policy.
No credentials or user settings are copied or changed. Session strict mode and
an all-tools denial hook are checked again against the resulting stream.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Any
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from agent_events import parse_json  # noqa: E402
from agent_output import OutputSchema  # noqa: E402
from agent_settings import DEFAULT_CONFIG, read_pins  # noqa: E402
from agy_capture import capture_stream  # noqa: E402
from agy_headless import DEFAULT_AGY, MODEL_LABELS  # noqa: E402
from agy_permissions import readonly_environment  # noqa: E402
from opsproc import CommandError, ProcessCancelled, run_process_group  # noqa: E402

from .paths import SafeDirectory  # noqa: E402
from .schema import ContractError  # noqa: E402

MODEL = "gemini-3.8-flash-high"
EFFORT = "high"


@dataclass(frozen=True)
class TurnEvidence:
    selected_model: str
    selected_effort: str
    observed_model: str
    observed_effort: str
    observed_version: str
    elapsed_seconds: float
    usage: dict[str, int] | None


@dataclass(frozen=True)
class TurnResult:
    output: Any
    evidence: TurnEvidence


def private_write(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def check_cancel(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise ContractError("cancelled")


class Gemini:
    def __init__(
        self,
        private_root: Path,
        *,
        agy: Path = DEFAULT_AGY,
        model: str = MODEL,
        effort: str = EFFORT,
    ):
        if model != MODEL or effort != EFFORT:
            raise ContractError("model_mismatch")
        self.agy = agy.absolute()
        self.private_root = private_root.absolute()
        with SafeDirectory(self.private_root, create=True) as directory:
            if stat.S_IMODE(os.fstat(directory.fd).st_mode) & 0o077:
                raise ContractError("unsafe_path")
        self.version = read_pins(DEFAULT_CONFIG)["AGY_VERSION"]

    def turn(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        timeout: float,
        cancel: Event | None = None,
    ) -> TurnResult:
        if not prompt.strip() or not math.isfinite(timeout) or timeout <= 0:
            raise ContractError("invalid_request")
        check_cancel(cancel)
        started = time.monotonic()
        run = self.private_root / uuid4().hex
        run.mkdir(mode=0o700)
        private_write(run / "prompt.md", prompt)
        private_write(run / "schema.json", json.dumps(schema, ensure_ascii=False))
        output_schema = OutputSchema(run / "schema.json", agent="agy")
        settings = {
            "toolPermission": "strict",
            "agentMode": "plan",
            "allowNonWorkspaceAccess": False,
            "enableTerminalSandbox": False,
            "permissions": {
                "allow": [],
                "deny": [
                    "command(*)",
                    "write_file(*)",
                    "read_url(*)",
                    "execute_url(*)",
                    "mcp(*)",
                ],
            },
        }
        private_write(run / "settings.json", json.dumps(settings))
        private_write(run / "hooks.jsonl", "")
        private_write(
            run / "selected.json",
            json.dumps({"model": MODEL, "effort": EFFORT, "version": self.version}),
        )

        def remaining() -> float:
            check_cancel(cancel)
            seconds = timeout - (time.monotonic() - started)
            if seconds <= 0:
                raise ContractError("timeout")
            return seconds

        try:
            with tempfile.TemporaryDirectory(prefix="explain-html-work-") as temporary:
                work = Path(temporary)
                hook_command = shlex.join(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(Path(__file__).with_name("no_tools.py")),
                        str(run / "hooks.jsonl"),
                    ]
                )
                (work / ".agents").mkdir(mode=0o700)
                hook_path = work / ".agents/hooks.json"
                private_write(
                    hook_path,
                    json.dumps(
                        {
                            "explain-html-no-tools": {
                                "PreToolUse": [
                                    {
                                        "matcher": ".*",
                                        "hooks": [
                                            {"type": "command", "command": hook_command}
                                        ],
                                    }
                                ]
                            }
                        }
                    ),
                )
                immutable = {
                    path: hashlib.sha256(path.read_bytes()).digest()
                    for path in (hook_path, run / "settings.json", run / "schema.json")
                }
                env = readonly_environment(os.environ, cwd=work)
                # Prevent environment-level redirection to unqualified profiles.
                for key in (
                    "ANTIGRAVITY_GEMINI_DIR",
                    "ANTIGRAVITY_APP_DATA_DIR",
                    "ANTIGRAVITY_CONFIG_REL_DIR",
                ):
                    env.pop(key, None)
                fixed = [
                    "--model",
                    MODEL,
                    "--effort",
                    EFFORT,
                    "--mode",
                    "plan",
                    "--settings",
                    str(run / "settings.json"),
                ]
                # Reserve CLI-created logs with private modes before launch too.
                private_write(run / "agy.log", "")
                private_write(run / "effort.log", "")

                def invoke(args: list[str], name: str, stdin: Any = None) -> str:
                    path = run / name
                    private_write(path, "")
                    with path.open("w", encoding="utf-8") as stream:
                        run_process_group(
                            [str(self.agy), *args],
                            cwd=work,
                            timeout=remaining(),
                            env=env,
                            stdout_file=stream,
                            stdin_file=stdin,
                            cancel=cancel,
                        )
                    return path.read_text(encoding="utf-8")

                def version(name: str) -> None:
                    observed = re.findall(
                        r"\b\d+\.\d+\.\d+(?:[-+][\w.-]+)?\b",
                        invoke(["--version"], name),
                    )
                    if observed != [self.version]:
                        raise ContractError("model_mismatch")

                invocation_failed = False
                try:
                    version("version-before.txt")
                    effort_raw = invoke(
                        [
                            "-p",
                            "/effort",
                            *fixed,
                            "--output-format",
                            "json",
                            "--log-file",
                            str(run / "effort.log"),
                        ],
                        "effort.json",
                    )
                    effort = parse_json(effort_raw)
                    data = effort.get("command", {}).get("data", {})
                    if (
                        effort.get("status") != "SUCCESS"
                        or "finished_reason" in effort
                        or effort.get("command", {}).get("name") != "effort"
                        or effort.get("num_turns") != 0
                        or data.get("adjustable") is not True
                        or data.get("current") != EFFORT
                        or not isinstance(data.get("available"), list)
                        or EFFORT not in data.get("available", [])
                    ):
                        raise ContractError("model_mismatch")
                    private_write(
                        run / "input.jsonl",
                        json.dumps(
                            {"event": "user", "message": {"content": prompt}},
                            ensure_ascii=False,
                        )
                        + "\n",
                    )
                    with (run / "input.jsonl").open(encoding="utf-8") as stdin:
                        raw = invoke(
                            [
                                "--input-format",
                                "stream-json",
                                *fixed,
                                "--disable-slash-commands",
                                "--output-format",
                                "stream-json",
                                "--json-schema",
                                str(run / "schema.json"),
                                "--log-file",
                                str(run / "agy.log"),
                            ],
                            "stream.jsonl",
                            stdin,
                        )
                    version("version-after.txt")
                except BaseException:
                    invocation_failed = True
                    raise
                finally:
                    integrity: dict[str, bool] = {}
                    for path, expected in immutable.items():
                        try:
                            integrity[path.name] = (
                                not path.is_symlink()
                                and path.is_file()
                                and stat.S_IMODE(path.stat().st_mode) == 0o600
                                and hashlib.sha256(path.read_bytes()).digest()
                                == expected
                            )
                        except OSError:
                            integrity[path.name] = False
                    integrity["no_tool_attempts"] = not (
                        run / "hooks.jsonl"
                    ).read_text()
                    private_write(run / "integrity.json", json.dumps(integrity))
                    # Preserve cancellation/timeout classification after group
                    # cleanup; retain denial/integrity evidence on those failures.
                    if not invocation_failed and not all(integrity.values()):
                        raise ContractError("invalid_model_output")

                log = (run / "agy.log").read_text(encoding="utf-8")
                labels = re.findall(
                    r'Propagating selected model override to backend: label="([^"\r\n]+)"',
                    log,
                )
                if not labels or set(labels) != {MODEL_LABELS[MODEL]}:
                    raise ContractError("model_mismatch")
                capture = capture_stream(raw, log)
                events = [parse_json(line) for line in raw.splitlines() if line.strip()]
                if (
                    sum(event.get("event") == "init" for event in events) != 1
                    or capture["init"] is None
                ):
                    raise ContractError("invalid_model_output")
                if capture["init"].get("model") != MODEL:
                    raise ContractError("model_mismatch")
                if (
                    capture["init"].get("permission_mode") != "strict"
                    or capture["tool_counts"]
                    or capture["subagent_steps"]
                ):
                    raise ContractError("invalid_model_output")
                payload = capture["result"]
                if (
                    not isinstance(payload, dict)
                    or payload.get("status") != "SUCCESS"
                    or "finished_reason" in payload
                    or payload.get("denied_actions")
                ):
                    raise ContractError("invalid_model_output")
                output = payload.get("structured_output")
                output_schema.validate(output)
                usage = payload.get("usage")
                if usage is not None and (
                    not isinstance(usage, dict)
                    or any(
                        type(value) is not int or value < 0 for value in usage.values()
                    )
                ):
                    raise ContractError("invalid_model_output")
                check_cancel(cancel)
                evidence = TurnEvidence(
                    MODEL,
                    EFFORT,
                    labels[-1],
                    data["current"],
                    self.version,
                    time.monotonic() - started,
                    usage,
                )
                private_write(run / "evidence.json", json.dumps(asdict(evidence)))
                return TurnResult(output, evidence)
        except ProcessCancelled as exc:
            raise ContractError("cancelled") from exc
        except CommandError as exc:
            private_write(run / "error.txt", str(exc))
            code = (
                "timeout"
                if isinstance(exc.__cause__, subprocess.TimeoutExpired)
                else "generation_failed"
            )
            raise ContractError(code) from exc
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            private_write(run / "error.txt", str(exc))
            raise ContractError("invalid_model_output") from exc
        finally:
            for path in run.iterdir():
                if path.is_file() and not path.is_symlink():
                    path.chmod(0o600)
