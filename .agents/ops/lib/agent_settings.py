"""Resolve the headless role/job pins without loading the caller's environment."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any


def resolve(
    config: Path,
    agent: str,
    profile: str | None,
    model: str | None,
    effort: str | None,
    mode: str,
) -> dict[str, Any]:
    raw = config.read_bytes()
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
            alternate = pins.get("CODEX_ALTERNATE_MODEL")
            if not alternate:
                raise ValueError("agents.conf must set CODEX_ALTERNATE_MODEL")
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*", alternate):
                raise ValueError("invalid alternate model pin")
            if selected["model"] not in {defaults["model"], alternate}:
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


def command(
    settings: dict[str, Any], mode: str, cwd: Path, out: Path, session: str | None
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
