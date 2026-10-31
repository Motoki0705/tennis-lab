"""One read-only agy turn, with pinned version/model and validated JSON output."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_events import parse_json
from agent_output import OutputSchema
from agent_settings import read_pins
from agy_permissions import command_permissions
from opsproc import CommandError, run_process_group

DEFAULT_CONFIG = Path(__file__).with_name("agents.conf")
DEFAULT_AGY = Path("/var/lib/tennis-lab-actions/home/.local/bin/agy")
MODEL_LABELS = {"gemini-3.8-flash-high": "Gemini 3.8 Flash (High)"}


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


class AgyError(RuntimeError):
    """A turn failed; raw diagnostics stay local and must not be published blindly."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        denied_actions: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.denied_actions = [] if denied_actions is None else denied_actions


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
        }
    }


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


def run_headless(
    *,
    agy: Path,
    cwd: Path,
    prompt_file: Path,
    json_schema: Path,
    log_file: Path,
    timeout: float,
    model: str | None = None,
    config_file: Path = DEFAULT_CONFIG,
) -> AgyResult:
    """Run one new conversation. No retries, resume, prose fallback or model fallback."""
    config = load_config(config_file)
    selected_model = config.model if model is None else model
    if selected_model != config.model:
        raise AgyError("invalid_input", "model differs from ISSUE_REWRITE_AGY_MODEL")
    if not math.isfinite(timeout) or timeout <= 0:
        raise AgyError("invalid_input", "timeout must be finite and positive")
    try:
        agy = agy.resolve(strict=True)
        cwd = cwd.resolve(strict=True)
        if not cwd.is_dir():
            raise ValueError("cwd must be a directory")
        prompt = prompt_file.read_text(encoding="utf-8")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        schema = OutputSchema(json_schema, agent="agy")
        log_file = log_file.absolute()
        log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Never accept evidence appended by a previous turn or through a symlink.
        with log_file.open("x", encoding="utf-8"):
            log_file.chmod(0o600)
    except (OSError, ValueError, UnicodeError) as exc:
        raise AgyError("invalid_input", f"invalid agy input: {exc}") from exc

    env = {
        **os.environ,
        "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GH_PAGER": "cat",
        "RIPGREP_CONFIG_PATH": "/dev/null",
    }
    started = time.monotonic()
    deadline = started + timeout

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise AgyError("timeout", "agy turn exceeded timeout")
        return seconds

    def check_version() -> str:
        observed = _observed_version(agy, cwd, remaining(), env)
        if observed != config.version:
            raise AgyError("version_mismatch", "agy version differs from AGY_VERSION")
        return observed

    denied: list[dict[str, str]] = []
    try:
        check_version()
        # Pin the exact schema bytes validated above, even if the input is edited.
        with tempfile.TemporaryDirectory(prefix="agy-schema-") as directory:
            pinned_schema = Path(directory) / "schema.json"
            pinned_schema.write_bytes(schema.raw)
            output = run_process_group(
                [
                    str(agy),
                    "-p",
                    prompt,
                    "--model",
                    selected_model,
                    "--mode",
                    "plan",
                    "--settings",
                    json.dumps(readonly_settings()),
                    "--output-format",
                    "json",
                    "--json-schema",
                    str(pinned_schema),
                    "--log-file",
                    str(log_file),
                ],
                cwd=cwd,
                timeout=remaining(),
                env=env,
            )
        observed_version = check_version()
        try:
            payload = parse_json(output)
            if not isinstance(payload, dict):
                raise ValueError("agy output must be a JSON object")
            actions = payload.get("denied_actions", [])
            if not isinstance(actions, list) or any(
                not isinstance(action, dict)
                or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in action.items()
                )
                for action in actions
            ):
                raise ValueError("invalid denied_actions")
            denied = actions
            if payload.get("status") != "SUCCESS":
                raise ValueError("agy status is not SUCCESS")
            if "finished_reason" in payload:
                raise ValueError("agy turn ended early (finished_reason is present)")
            if (
                "structured_output" not in payload
                or payload["structured_output"] is None
            ):
                raise ValueError("agy structured_output is missing")
            schema.validate(payload["structured_output"])
            usage = payload.get("usage")
            if usage is not None and (
                not isinstance(usage, dict)
                or any(type(value) is not int or value < 0 for value in usage.values())
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
            raise AgyError("invalid_output", str(exc), denied_actions=denied) from exc
        observed_model = _observed_model(log_file, selected_model, denied)
        return AgyResult(
            structured_output=payload["structured_output"],
            usage=usage,
            denied_actions=denied,
            observed_model=observed_model,
            observed_version=observed_version,
            duration_seconds=duration,
            elapsed_seconds=time.monotonic() - started,
        )
    except CommandError as exc:
        reason = (
            "timeout"
            if isinstance(exc.__cause__, subprocess.TimeoutExpired)
            else "cli_error"
        )
        raise AgyError(reason, str(exc), denied_actions=denied) from exc
    except (OSError, UnicodeError) as exc:
        raise AgyError(
            "invalid_output", f"agy evidence could not be read: {exc}"
        ) from exc
