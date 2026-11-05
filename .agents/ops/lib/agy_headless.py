"""One read-only agy turn, with pinned version/model and validated JSON output."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_events import parse_json
from agent_output import OutputSchema
from agent_settings import read_pins
from agy_capture import capture_stream, public_command_summary, restore_hook_commands
from agy_hook import install_command_hook
from agy_permissions import command_permissions, readonly_environment
from opsproc import CommandError, run_process_group

DEFAULT_CONFIG = Path(__file__).with_name("agents.conf")
DEFAULT_AGY = Path("/var/lib/tennis-lab-actions/home/.local/bin/agy")
DEFAULT_TIMEOUT = 600
INLINE_PROMPT_BYTES = 96 * 1024
MODEL_LABELS = {"gemini-3.8-flash-high": "Gemini 3.8 Flash (High)"}
_READONLY_MODE = {
    "toolPermission": "request-review",
    "agentMode": "plan",
    "allowNonWorkspaceAccess": False,
    "enableTerminalSandbox": False,
}


@dataclass(frozen=True)
class AgyConfig:
    version: str
    url: str
    sha256: str
    model: str


@dataclass(frozen=True)
class AgyResult:
    structured_output: Any
    usage: dict[str, int] | None
    denied_actions: list[dict[str, str]]
    observed_model: str
    observed_version: str
    duration_seconds: float | None
    elapsed_seconds: float
    command_summary: dict[str, Any] = field(default_factory=dict)


class AgyError(RuntimeError):
    """A turn failed; raw diagnostics stay local and must not be published blindly."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        denied_actions: list[dict[str, str]] | None = None,
        command_summary: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.denied_actions = [] if denied_actions is None else denied_actions
        self.command_summary = {} if command_summary is None else command_summary


def load_config(config_file: Path = DEFAULT_CONFIG) -> AgyConfig:
    """Use only literal agents.conf pins, never same-named environment variables."""
    try:
        pins = read_pins(config_file)
        config = AgyConfig(
            version=pins["AGY_VERSION"],
            url=pins["AGY_URL"],
            sha256=pins["AGY_SHA256"],
            model=pins["ISSUE_REWRITE_AGY_MODEL"],
        )
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", config.version):
            raise ValueError("invalid AGY_VERSION")
        if not re.fullmatch(r"[0-9a-f]{64}", config.sha256):
            raise ValueError("invalid AGY_SHA256")
        if not config.url.startswith(("https://", "file://")) or any(
            char.isspace() for char in config.url
        ):
            raise ValueError(
                "AGY_URL requires an HTTPS URL (file:// for local fixtures)"
            )
        if config.model not in MODEL_LABELS:
            raise ValueError("ISSUE_REWRITE_AGY_MODEL needs a known log label")
        return config
    except (OSError, UnicodeError, KeyError, ValueError) as exc:
        raise AgyError("invalid_input", f"invalid agy config: {exc}") from exc


def readonly_settings() -> dict[str, Any]:
    """Deny shell syntax, then allow only complete enumerated command forms."""
    commands = command_permissions()
    return {
        **_READONLY_MODE,
        "permissions": {
            "allow": commands["allow"],
            "deny": [
                "write_file(*)",
                "read_url(*)",
                "execute_url(*)",
                "unsandboxed(*)",
                "mcp(*)",
                *commands["deny"],
            ],
        },
    }


def _check_user_settings(cwd: Path) -> None:
    """Reject widening persistent settings before even running agy --version."""
    try:
        home = os.environ.get("HOME", "")
        if not home or not Path(home).is_absolute():
            raise ValueError("HOME must be an absolute directory")
        path = Path(home) / ".gemini/antigravity-cli/settings.json"
        try:
            path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("user settings must be a regular JSON file")
        settings = parse_json(path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("user settings must be a JSON object")
        known = {
            *_READONLY_MODE,
            "permissions",
            "$schema",
            "model",
            "trustedWorkspaces",
        }
        if settings.keys() - known:
            raise ValueError("user settings contain unsupported top-level keys")
        if settings.get("toolPermission", "request-review") not in (
            "request-review",
            "strict",
        ):
            raise ValueError("user toolPermission must be request-review or strict")
        if settings.get("agentMode", "") not in ("", "default", "plan"):
            raise ValueError("remove user agentMode or set it to plan")
        for key in ("allowNonWorkspaceAccess", "enableTerminalSandbox"):
            if key in settings and settings[key] is not _READONLY_MODE[key]:
                raise ValueError("user " + key + " differs from the fixed setting")
        if "$schema" in settings and not isinstance(settings["$schema"], str):
            raise ValueError("user $schema must be a string")
        if "model" in settings and not isinstance(settings["model"], str):
            raise ValueError("user model must be a string")
        trusted = settings.get("trustedWorkspaces", [])
        if not isinstance(trusted, list) or any(
            not isinstance(workspace, str) for workspace in trusted
        ):
            raise ValueError("user trustedWorkspaces must be a list of strings")
        if trusted:
            # Check directory boundaries, including both lexical and symlink aliases.
            work_paths = (Path(os.path.abspath(cwd)), cwd.resolve())
            for workspace in trusted:
                root = cwd / Path(os.path.expanduser(workspace))
                roots = (Path(os.path.abspath(root)), root.resolve())
                if any(
                    work.is_relative_to(parent)
                    for work in work_paths
                    for parent in roots
                ):
                    raise ValueError(
                        "cwd is in user trustedWorkspaces; use an untrusted work directory"
                    )
        permissions = settings.get("permissions", {})
        if not isinstance(permissions, dict) or permissions.keys() - {
            "allow",
            "ask",
            "deny",
        }:
            raise ValueError("user permissions contain unsupported keys or types")
        if permissions.get("allow", []) != []:
            raise ValueError("user permissions.allow must be an empty list")
        for key in ("ask", "deny"):
            rules = permissions.get(key, [])
            if not isinstance(rules, list) or any(
                not isinstance(rule, str) or not rule.strip() for rule in rules
            ):
                raise ValueError(
                    "user permissions restrictions must be lists of nonempty strings"
                )
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        raise AgyError(
            "unsafe_user_settings", "user settings rejected: " + str(exc)
        ) from exc


def _observed_version(agy: Path, cwd: Path, timeout: float, env: dict[str, str]) -> str:
    output: str = run_process_group(
        [str(agy), "--version"], cwd=cwd, timeout=timeout, env=env
    )
    versions: list[str] = re.findall(
        r"\b[0-9]+\.[0-9]+\.[0-9]+(?:[-+][\w.-]+)?\b", output
    )
    if len(versions) != 1:
        raise AgyError("version_mismatch", "agy --version did not report one version")
    return versions[0]


def _observed_model(log_file: Path, model: str, denied: list[dict[str, str]]) -> str:
    labels: list[str] = re.findall(
        r'Propagating selected model override to backend: label="([^"\r\n]+)"',
        log_file.read_text(encoding="utf-8"),
    )
    if not labels or set(labels) != {MODEL_LABELS[model]}:
        raise AgyError(
            "model_mismatch",
            "agy log does not confirm the pinned model",
            denied_actions=denied,
        )
    return labels[-1]


def _capture_evidence(log: Path, stream: Path, hooks: Path) -> dict[str, Any]:
    decoding_errors = []

    def read(path: Path) -> str:
        raw = path.read_bytes()
        try:
            return raw.decode("utf-8")
        except UnicodeError:
            decoding_errors.append("invalid UTF-8 in private evidence")
            return raw.decode("utf-8", errors="replace")

    capture = capture_stream(
        read(stream),
        read(log),
        strict=False,
    )
    decisions = []
    for line in read(hooks).splitlines():
        try:
            entry = parse_json(line)
            if (
                not isinstance(entry, dict)
                or entry.get("decision") not in ("allow", "deny")
                or not isinstance(entry.get("reason"), str)
                or "step" not in entry
                or "command" not in entry
                or type(entry["step"]) not in (int, type(None))
                or not (entry["command"] is None or isinstance(entry["command"], str))
            ):
                raise ValueError("invalid command hook record")
            decisions.append(entry)
        except ValueError:
            # A killed hook can leave a partial line. Retain earlier decisions
            # and the original timeout/CLI classification; success rejects it.
            capture["errors"].append("invalid command hook record")
    capture["errors"].extend(decoding_errors)
    capture["hook_decisions"] = decisions
    try:
        restore_hook_commands(capture, decisions, require_hook=True)
    except ValueError as exc:
        capture["hook_verification_errors"] = [str(exc)]
    return capture


def _captured_denials(capture: dict[str, Any]) -> list[dict[str, str]]:
    result = capture.get("result")
    actions = result.get("denied_actions", []) if isinstance(result, dict) else []
    denied = (
        [
            dict(action)
            for action in actions
            if isinstance(action, dict)
            and all(
                isinstance(k, str) and isinstance(v, str) for k, v in action.items()
            )
        ]
        if isinstance(actions, list)
        else []
    )
    for entry in capture.get("hook_decisions", []):
        if entry["decision"] == "deny":
            reading = entry.get("tool") == "read_file"
            action = {
                "action": "read_file" if reading else "command",
                "display_name": "ReadFile" if reading else "RunCommand",
                "reason": entry["reason"],
            }
            if isinstance(entry.get("command"), str):
                action["command"] = entry["command"]
            if reading and isinstance(entry.get("path"), str):
                action["path"] = entry["path"]
            if action not in denied:
                denied.append(action)
    return denied


def _session_settings_integrity(path: Path, expected_digest: str) -> dict[str, Any]:
    observed = None
    mode = None
    try:
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISREG(metadata.st_mode):
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        pass  # Missing/unreadable evidence is a mismatch, never an implicit pass.
    return {
        "expected_sha256": expected_digest,
        "observed_sha256": observed,
        "mode": mode,
        "unchanged": observed == expected_digest and mode == 0o600,
    }


def run_headless(
    *,
    agy: Path,
    cwd: Path,
    prompt_file: Path,
    json_schema: Path,
    log_file: Path,
    timeout: float = DEFAULT_TIMEOUT,
    model: str | None = None,
    config_file: Path = DEFAULT_CONFIG,
) -> AgyResult:
    """Run one turn in a caller-owned disposable workspace, retaining private evidence."""
    config = load_config(config_file)
    selected_model = config.model if model is None else model
    if selected_model != config.model:
        raise AgyError("invalid_input", "model differs from ISSUE_REWRITE_AGY_MODEL")
    if not math.isfinite(timeout) or timeout <= 0:
        raise AgyError("invalid_input", "timeout must be finite and positive")
    _check_user_settings(cwd)
    with ExitStack() as files:
        try:
            agy, cwd = agy.resolve(strict=True), cwd.resolve(strict=True)
            if not cwd.is_dir():
                raise ValueError("cwd must be a directory")
            if Path(__file__).resolve().is_relative_to(cwd):
                raise ValueError(
                    "cwd must be a disposable workspace outside the controller"
                )
            prompt = prompt_file.read_text(encoding="utf-8")
            if not prompt.strip():
                raise ValueError("prompt must not be empty")
            schema = OutputSchema(json_schema, agent="agy")
            env = readonly_environment(os.environ, cwd=cwd)
            log_file = log_file.absolute()
            if log_file.resolve().is_relative_to(cwd):
                raise ValueError("private evidence must be outside the workspace")
            log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            paths = [
                log_file,
                *(
                    log_file.with_name(log_file.name + suffix)
                    for suffix in (
                        ".stream.jsonl",
                        ".hooks.jsonl",
                        ".commands.json",
                        ".settings.json",
                    )
                ),
            ]
            handles = []
            for path in paths:
                handle = files.enter_context(path.open("x", encoding="utf-8"))
                path.chmod(0o600)
                handles.append(handle)
            stream_file, hooks_file = paths[1:3]
            settings_file = paths[4]
            settings_raw = json.dumps(readonly_settings())
            handles[4].write(settings_raw)
            handles[4].flush()
            settings_digest = hashlib.sha256(settings_raw.encode("utf-8")).hexdigest()
            agents_existed = (cwd / ".agents").exists()
            owned_hook = install_command_hook(cwd, hooks_file)
        except (OSError, ValueError, UnicodeError) as exc:
            raise AgyError("invalid_input", f"invalid agy input: {exc}") from exc

        started = time.monotonic()
        deadline = started + timeout

        def remaining() -> float:
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise AgyError("timeout", "agy turn exceeded timeout")
            return seconds

        def check_version() -> str:
            _check_user_settings(cwd)
            observed = _observed_version(agy, cwd, remaining(), env)
            if observed != config.version:
                raise AgyError(
                    "version_mismatch", "agy version differs from AGY_VERSION"
                )
            return observed

        denied: list[dict[str, str]] = []
        command_summary: dict[str, Any] = {}
        try:
            try:
                check_version()
                with tempfile.TemporaryDirectory(prefix="agy-schema-") as directory:
                    pinned_schema = Path(directory) / "schema.json"
                    pinned_schema.write_bytes(schema.raw)
                    stdin = None
                    prompt_args = ["-p", prompt]
                    if len(prompt.encode("utf-8")) > INLINE_PROMPT_BYTES:
                        prompt_args = ["--input-format", "stream-json"]
                        input_file = Path(directory) / "prompt.jsonl"
                        stdin = files.enter_context(
                            input_file.open("x+", encoding="utf-8")
                        )
                        input_file.chmod(0o600)
                        stdin.write(
                            json.dumps(
                                {"event": "user", "message": {"content": prompt}},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        stdin.flush()
                        stdin.seek(0)
                    _check_user_settings(cwd)
                    run_process_group(
                        [
                            str(agy),
                            *prompt_args,
                            "--model",
                            selected_model,
                            "--mode",
                            "plan",
                            "--settings",
                            str(settings_file),
                            "--output-format",
                            "stream-json",
                            "--json-schema",
                            str(pinned_schema),
                            "--log-file",
                            str(log_file),
                        ],
                        cwd=cwd,
                        timeout=remaining(),
                        env=env,
                        stdout_file=handles[1],
                        stdin_file=stdin,
                    )
                observed_version = check_version()
            finally:
                # stdout goes straight to disk, including partial output before a
                # nonzero exit or process-group timeout. Never recover from prose.
                integrity = _session_settings_integrity(settings_file, settings_digest)
                try:
                    capture = _capture_evidence(log_file, stream_file, hooks_file)
                    capture["session_settings_integrity"] = integrity
                    denied = _captured_denials(capture)
                    command_summary = public_command_summary(capture)
                    json.dump(capture, handles[3], indent=2)
                    handles[3].flush()
                finally:
                    changed = False
                    for path, expected in owned_hook.items():
                        try:
                            changed |= (
                                path.is_symlink()
                                or path.read_bytes() != expected
                                or stat.S_IMODE(path.stat().st_mode) != 0o600
                            )
                        finally:
                            path.unlink(missing_ok=True)
                    if not agents_existed:
                        try:
                            (cwd / ".agents").rmdir()
                        except OSError as exc:
                            if exc.errno != errno.ENOTEMPTY:
                                raise
                    if changed:
                        raise AgyError(
                            "invalid_output", "workspace command hook was modified"
                        )
                if not integrity["unchanged"]:
                    raise AgyError(
                        "invalid_output", "session settings file was modified"
                    )
                if capture.get("hook_verification_errors"):
                    raise AgyError(
                        "command_hook_mismatch",
                        "tool stream and hook decisions do not reconcile",
                    )

            try:
                if capture["errors"]:
                    raise ValueError("invalid agy stream")
                payload = capture["result"]
                if not isinstance(payload, dict):
                    raise ValueError("agy output must contain one result object")
                actions = payload.get("denied_actions", [])
                if not isinstance(actions, list) or any(
                    not isinstance(action, dict)
                    or not all(
                        isinstance(k, str) and isinstance(v, str)
                        for k, v in action.items()
                    )
                    for action in actions
                ):
                    raise ValueError("invalid denied_actions")
                if payload.get("status") != "SUCCESS":
                    raise ValueError("agy status is not SUCCESS")
                if "finished_reason" in payload:
                    raise ValueError(
                        "agy turn ended early (finished_reason is present)"
                    )
                if (
                    "structured_output" not in payload
                    or payload["structured_output"] is None
                ):
                    raise ValueError("agy structured_output is missing")
                schema.validate(payload["structured_output"])
                usage = payload.get("usage")
                if usage is not None and (
                    not isinstance(usage, dict)
                    or any(
                        type(value) is not int or value < 0 for value in usage.values()
                    )
                ):
                    raise ValueError("invalid usage")
                duration = payload.get("duration_seconds")
                if duration is not None and (
                    type(duration) not in (int, float)
                    or not math.isfinite(duration)
                    or duration < 0
                ):
                    raise ValueError("invalid duration_seconds")
            except (ValueError, UnicodeError) as exc:
                raise AgyError("invalid_output", str(exc)) from exc
            observed_model = _observed_model(log_file, selected_model, denied)
            return AgyResult(
                structured_output=payload["structured_output"],
                usage=usage,
                denied_actions=denied,
                observed_model=observed_model,
                observed_version=observed_version,
                duration_seconds=duration,
                elapsed_seconds=time.monotonic() - started,
                command_summary=command_summary,
            )
        except AgyError as exc:
            exc.denied_actions = denied
            exc.command_summary = command_summary
            raise
        except CommandError as exc:
            reason = (
                "timeout"
                if isinstance(exc.__cause__, subprocess.TimeoutExpired)
                else "cli_error"
            )
            raise AgyError(
                reason, str(exc), denied_actions=denied, command_summary=command_summary
            ) from exc
        except (OSError, UnicodeError, ValueError) as exc:
            raise AgyError(
                "invalid_output",
                f"agy evidence could not be read: {exc}",
                denied_actions=denied,
                command_summary=command_summary,
            ) from exc
