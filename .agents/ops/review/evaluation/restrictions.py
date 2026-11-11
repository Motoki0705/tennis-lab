"""Evaluation-only Codex launcher. The shared review runner stays unchanged."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from worker_logs.journal import atomic

DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "plugins",
    "remote_plugin",
    "hooks",
    "memories",
    "workspace_dependencies",
    "goals",
    "tool_suggest",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "browser_annotation_api",
    "computer_use",
    "in_app_browser",
    "in_app_local_automation",
    "in_app_chat",
    "image_generation",
    "view_image",
    "multi_agent",
    "multi_agent_v2",
    "skill_search",
    "skill_mcp_dependency_install",
    "code_mode",
    "code_mode_host",
)


def inspect_cli(binary: Path) -> tuple[list[str], str]:
    version = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if not version:
        raise ValueError("Codex CLI version is unavailable")
    features = subprocess.run(
        [str(binary), "features", "list"],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    available = {
        line.split()[0] for line in features.stdout.splitlines() if line.split()
    }
    if set(DISABLED_FEATURES) - available:
        raise ValueError("Codex lacks required evaluation tool restrictions")
    servers: Any = json.loads(
        subprocess.run(
            [str(binary), "mcp", "list", "--json"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        ).stdout
    )
    if not isinstance(servers, list) or any(
        not isinstance(server, dict) or not isinstance(server.get("name"), str)
        for server in servers
    ):
        raise ValueError("cannot enumerate inherited MCP servers")
    return sorted({server["name"] for server in servers}), version


def restricted_arguments(
    arguments: list[str], inherited_servers: list[str]
) -> list[str]:
    if "exec" not in arguments or any(
        arg in {"resume", "--search", "--dangerously-bypass-approvals-and-sandbox"}
        for arg in arguments
    ):
        raise ValueError("only a fresh, read-only comparison is supported")
    if (
        arguments.count("--sandbox") != 1
        or arguments[-1] == "--sandbox"
        or arguments[arguments.index("--sandbox") + 1] != "read-only"
    ):
        raise ValueError("evaluation requires the common runner's read-only sandbox")
    config_keys = {
        "model_reasoning_effort",
        "service_tier",
        "mcp_servers.review_evaluation.command",
        "mcp_servers.review_evaluation.args",
        "mcp_servers.review_evaluation.env.PYTHONPATH",
        "mcp_servers.review_evaluation.tool_timeout_sec",
        "mcp_servers.review_evaluation.startup_timeout_sec",
    }
    value_flags = {
        "--cd",
        "--sandbox",
        "--output-last-message",
        "--model",
        "--output-schema",
        "--disable",
        "-c",
        "--config",
    }
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"exec", "--json", "-"}:
            index += 1
            continue
        if argument not in value_flags or index + 1 >= len(arguments):
            raise ValueError("unregistered evaluation CLI argument")
        value = arguments[index + 1]
        if argument == "--model" and value not in {"gpt-6-astra", "gpt-6.1-sol"}:
            raise ValueError("unregistered evaluation model")
        if (
            argument in {"-c", "--config"}
            and value.partition("=")[0] not in config_keys
        ):
            raise ValueError("unregistered evaluation CLI configuration override")
        if argument == "--disable" and value not in {
            *DISABLED_FEATURES,
            "daemon_auto_start",
        }:
            raise ValueError("unregistered evaluation feature override")
        index += 2
    flags = [
        "-c",
        'web_search="disabled"',
        "-c",
        'developer_instructions=""',
        "-c",
        "project_doc_max_bytes=0",
    ]
    for feature in DISABLED_FEATURES:
        flags += ["--disable", feature]
    for name in inherited_servers:
        if name == "review_evaluation":
            raise ValueError("inherited MCP name conflicts with the evaluation server")
        flags += ["-c", f"mcp_servers.{json.dumps(name)}.enabled=false"]
    return [*flags, *arguments]


def main() -> int:
    try:
        root = Path(os.environ["REVIEW_EVALUATION_LAUNCH"])
        config = json.loads((root / "restricted-launch.json").read_text())
        command = restricted_arguments(sys.argv[1:], config["inherited_servers"])
        atomic(
            root / "restrictions.json",
            {
                "schema_version": 1,
                "disabled_features": list(DISABLED_FEATURES),
                "web_search": "disabled",
                "disabled_mcp_servers": config["inherited_servers"],
                "allowed_mcp_servers": ["review_evaluation"],
                "fresh_session": True,
            },
        )
        os.execv(config["codex_binary"], [config["codex_binary"], *command])
    except (ValueError, OSError, KeyError, IndexError) as exc:
        print(f"review evaluation launcher: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
