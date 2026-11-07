"""Resolve the headless role/job pins without loading the caller's environment."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).with_name("agents.conf")


def _parse_pins(raw: bytes) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in raw.decode().splitlines():
        fields = shlex.split(line, comments=True)
        if not fields:
            continue
        if len(fields) != 1 or not re.fullmatch(r"[A-Z][A-Z0-9_]*=.+", fields[0]):
            raise ValueError("agents.conf must contain literal NAME=value assignments")
        key, value = fields[0].split("=", 1)
        if key in pins or not value.strip():
            raise ValueError("agents.conf contains duplicate/empty pins")
        pins[key] = value
    return pins


def read_pins(config: Path) -> dict[str, str]:
    """Read literal pins for all runtimes without evaluating shell or environment."""
    return _parse_pins(config.read_bytes())


def _implementation_models(pins: dict[str, str]) -> tuple[str, str]:
    models = []
    for key in ("CODEX_MODEL", "CODEX_ALTERNATE_MODEL"):
        value = pins.get(key)
        if not value:
            raise ValueError(f"agents.conf must set {key}")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*", value):
            raise ValueError(f"invalid model pin: {key}")
        models.append(value)
    return models[0], models[1]


def implementation_models(config: Path | None = None) -> tuple[str, str]:
    """The implementation CLI choices, from the runner's literal config."""
    return _implementation_models(read_pins(config or DEFAULT_CONFIG))


def implementation_model(
    model: str | None, *, previous: str | None = None, config: Path | None = None
) -> str:
    """Validate a selection and preserve the original model on continuation."""
    if previous is not None and model is not None and model != previous:
        raise ValueError("continuation cannot change the worker model")
    settings = resolve(
        config or DEFAULT_CONFIG,
        "codex",
        None,
        previous if previous is not None else model,
        None,
        "write",
    )
    return str(settings["selected"]["model"])


def resolve(
    config: Path,
    agent: str,
    profile: str | None,
    model: str | None,
    effort: str | None,
    mode: str,
) -> dict[str, Any]:
    raw = config.read_bytes()
    pins = _parse_pins(raw)
    if profile is not None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", profile):
            raise ValueError("invalid --profile")
        if model is not None or effort is not None:
            raise ValueError("--profile cannot be combined with --model/--effort")
    prefix = (profile.upper() + "_" if profile else "") + agent.upper() + "_"

    def pin(name: str) -> str:
        key = prefix + name
        if not pins.get(key):
            raise ValueError(f"agents.conf must set {key}")
        return pins[key]

    defaults = {"model": pin("MODEL"), "effort": pin("EFFORT")}
    if agent == "codex":
        defaults["service_tier"] = pin("SERVICE_TIER")
    selected = {**defaults}
    overrides = {
        k: v for k, v in {"model": model, "effort": effort}.items() if v is not None
    }
    selected.update(overrides)
    if any(
        not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*", value)
        for value in (defaults["model"], selected["model"])
    ):
        raise ValueError("invalid model pin/override")
    levels = {"low", "medium", "high", "xhigh", "max"}
    if defaults["effort"] not in levels or selected["effort"] not in levels:
        raise ValueError("invalid effort pin/override")
    if agent == "codex":
        if selected["service_tier"] not in {"default", "fast", "flex", "priority"}:
            raise ValueError("invalid service tier")
        if profile is None:
            if selected["model"] not in _implementation_models(pins):
                raise ValueError(
                    "implementation model must be the default or configured alternate"
                )
            if defaults["effort"] != "max" or selected["effort"] != "max":
                raise ValueError("implementation effort must be max")
    threads = None
    if profile == "ops_report":
        if agent != "codex" or mode != "read-only":
            raise ValueError("ops_report requires codex in read-only mode")
        value = pins.get("OPS_REPORT_MAX_THREADS", "")
        if not value.isdecimal() or int(value) < 2:
            raise ValueError("agents.conf must set OPS_REPORT_MAX_THREADS >= 2")
        threads = int(value)
    return {
        "source": str(config.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "role": profile or ("parent" if agent == "claude" else "implementation"),
        "agent": agent,
        "profile": profile,
        "defaults": defaults,
        "overrides": overrides,
        "selected": selected,
        "selection_sources": {
            k: "task" if k in overrides else prefix + k.upper() for k in selected
        },
        "max_threads": threads,
    }


def execution_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Compare runtime inputs, retaining config provenance in the original record.

    A configuration path, comment or unrelated profile does not change the
    selected agent/model/effort/permissions. Unknown fields remain significant.
    """
    provenance = {"source", "sha256", "defaults", "overrides", "selection_sources"}
    return {
        **contract,
        "settings": {
            k: v for k, v in contract["settings"].items() if k not in provenance
        },
    }


def command(
    settings: dict[str, Any],
    mode: str,
    cwd: Path,
    out: Path,
    session: str | None,
    read_dirs: list[str],
    json_schema: Path | None = None,
    mcp_config: Path | None = None,
) -> list[str]:
    selected = settings["selected"]
    if settings["agent"] == "claude":
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            selected["model"],
            "--effort",
            selected["effort"],
            "--strict-mcp-config",
        ]
        if mode == "read-only":
            cmd += ["--tools", "Read", "Grep", "Glob", "--permission-mode", "dontAsk"]
        else:
            cmd += ["--permission-mode", "bypassPermissions"]
        for directory in read_dirs:
            cmd += ["--add-dir", directory]
        if json_schema is not None:
            cmd += [
                "--json-schema",
                json.dumps(
                    json.loads(json_schema.read_text(encoding="utf-8")),
                    ensure_ascii=False,
                ),
            ]
        if session:
            cmd += ["--resume", session]
        return cmd
    # WSL drvfs CODEX_HOME cannot host the daemon's Unix socket. Keep the
    # supported CLI feature override on both initial and resumed invocations.
    cmd = ["codex", "--disable", "daemon_auto_start", "exec", "--cd", str(cwd)]
    if mode == "read-only":
        cmd += ["--sandbox", "read-only"]
    if session:
        cmd += ["resume"]
    cmd += [
        "--json",
        "--output-last-message",
        str(out),
        "--model",
        selected["model"],
        "-c",
        "model_reasoning_effort=" + json.dumps(selected["effort"]),
        "-c",
        "service_tier=" + json.dumps(selected["service_tier"]),
    ]
    if mode == "write":
        cmd += ["--dangerously-bypass-approvals-and-sandbox"]
    if json_schema is not None:
        cmd += ["--output-schema", str(json_schema)]
    if mcp_config is not None:
        if settings["profile"] not in {"review", "ops_report"} or mode != "read-only":
            raise ValueError("MCP connection requires a read-only review/audit profile")
        servers = json.loads(mcp_config.read_text(encoding="utf-8"))
        if not isinstance(servers, dict) or not servers:
            raise ValueError("MCP config must name at least one host-provided server")
        for name, server in servers.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not isinstance(
                server, dict
            ):
                raise ValueError("invalid MCP server name/configuration")
            if set(server) - {
                "command",
                "args",
                "env",
                "tool_timeout_sec",
                "startup_timeout_sec",
            }:
                raise ValueError("unsupported MCP server setting")
            for key, value in server.items():
                if key == "env":
                    if not isinstance(value, dict) or any(
                        not re.fullmatch(r"[A-Z_][A-Z0-9_]*", k)
                        or not isinstance(v, str)
                        for k, v in value.items()
                    ):
                        raise ValueError("invalid MCP environment")
                    for env_key, env_value in value.items():
                        cmd += [
                            "-c",
                            f"mcp_servers.{name}.env.{env_key}="
                            + json.dumps(env_value),
                        ]
                else:
                    cmd += ["-c", f"mcp_servers.{name}.{key}=" + json.dumps(value)]
    if settings["profile"] == "review":
        cmd += ["--disable", "multi_agent"]
    if settings["profile"] == "ops_report":
        cmd += [
            "--enable",
            "multi_agent",
            "-c",
            f"agents.max_threads={settings['max_threads']}",
            "-c",
            "agents.max_depth=1",
        ]
    if session:
        cmd += [session]
    return [*cmd, "-"]
