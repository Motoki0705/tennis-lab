"""Resolve the headless role/job pins without loading the caller's environment."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).with_name("agents.conf")
MODEL_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*")
CONTINUATION_MODELS_PIN = "CODEX_CONTINUATION_MODELS"
SELECTABLE_MODELS_PIN = "CODEX_SELECTABLE_MODELS"


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


def implementation_models(config: Path | None = None) -> tuple[str, ...]:
    """Current model choices for a new implementation worker, default first."""
    pins = read_pins(config or DEFAULT_CONFIG)
    model = pins.get("CODEX_MODEL")
    if not model or not MODEL_NAME.fullmatch(model):
        raise ValueError("agents.conf must set a valid CODEX_MODEL")
    selectable = _model_list(pins, SELECTABLE_MODELS_PIN)
    if model in selectable:
        raise ValueError(f"{SELECTABLE_MODELS_PIN} must not repeat CODEX_MODEL")
    return (model, *sorted(selectable))


def check_saved_implementation_model(model: str) -> None:
    """A retired model's terminal evidence can still be reconciled and handed off."""
    if model not in {"gpt-6-astra", "gpt-6.2-astra"}:
        implementation_model(None, previous=model)


def _model_list(pins: dict[str, str], key: str) -> set[str]:
    raw = pins.get(key)
    if raw is None:
        return set()
    models = raw.split(",")
    if len(models) != len(set(models)) or any(
        not MODEL_NAME.fullmatch(model) for model in models
    ):
        raise ValueError(f"invalid model list: {key}")
    return set(models)


def implementation_model(
    model: str | None, *, previous: str | None = None, config: Path | None = None
) -> str:
    """Validate a selection and preserve the original model on continuation."""
    settings = resolve(
        config or DEFAULT_CONFIG,
        "codex",
        None,
        model,
        None,
        "write",
        saved_model=previous,
    )
    return str(settings["selected"]["model"])


def resolve(
    config: Path,
    agent: str,
    profile: str | None,
    model: str | None,
    effort: str | None,
    mode: str,
    *,
    review_selection: dict[str, Any] | None = None,
    repair: dict[str, Any] | None = None,
    saved_model: str | None = None,
    saved_review: dict[str, str] | None = None,
) -> dict[str, Any]:
    if repair is not None:
        from lib.repair_turn import validate_request

        validate_request(repair)
        if (agent, profile, mode) != ("codex", None, "write"):
            raise ValueError("repair requires write-mode implementation Codex")
        if effort not in (None, "high"):
            raise ValueError("repair effort must be high")
        effort = "high"
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
    if saved_model is not None:
        if (agent, profile) != ("codex", None):
            raise ValueError("saved worker model requires implementation Codex")
        if model is not None and model != saved_model:
            raise ValueError("continuation cannot change the worker model")
        selected["model"] = saved_model
    continuation_pin = None
    if review_selection is not None and (agent, profile, mode) != (
        "codex",
        "review",
        "read-only",
    ):
        raise ValueError("review selection requires read-only Codex review profile")
    if agent == "codex" and profile == "review":
        from lib.review_settings import CONTINUATION_PAIRS_PIN, runner_selection

        selected, continuing = runner_selection(pins, review_selection, saved_review)
        if continuing:
            continuation_pin = CONTINUATION_PAIRS_PIN
    if any(
        not isinstance(value, str) or not MODEL_NAME.fullmatch(value)
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
            continuations = _model_list(pins, CONTINUATION_MODELS_PIN)
            selectable = _model_list(pins, SELECTABLE_MODELS_PIN)
            if selectable & continuations:
                raise ValueError(
                    f"{SELECTABLE_MODELS_PIN} and {CONTINUATION_MODELS_PIN} overlap"
                )
            if selected["model"] != pins["CODEX_MODEL"] and (
                selected["model"] not in selectable
            ):
                if selected["model"] not in continuations:
                    raise ValueError(
                        "implementation model must be the configured Sol model"
                    )
                if saved_model is None:
                    raise ValueError("implementation model is continuation-only")
                continuation_pin = CONTINUATION_MODELS_PIN
            if defaults["effort"] != "max" or selected["effort"] != (
                "high" if repair is not None else "max"
            ):
                raise ValueError("implementation effort must be max")
    threads = None
    if (
        profile in {"ops_report", "ops_scout", "issue_judgement", "memory_autodream"}
        and mode != "read-only"
    ):
        raise ValueError(f"{profile} requires read-only mode")
    if profile == "ops_report":
        value = pins.get("OPS_REPORT_MAX_THREADS", "")
        if not value.isdecimal() or int(value) < 2:
            raise ValueError("agents.conf must set OPS_REPORT_MAX_THREADS >= 2")
        threads = int(value)
    settings = {
        "source": str(config.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "role": profile or ("parent" if agent == "claude" else "implementation"),
        "agent": agent,
        "profile": profile,
        "defaults": defaults,
        "overrides": overrides,
        "selected": selected,
        "selection_sources": {
            k: "review_selection"
            if review_selection is not None
            else "task"
            if k in overrides
            else prefix + k.upper()
            for k in selected
        },
        "max_threads": threads,
    }
    if agent == "codex" and profile == "review":
        settings["web_search"] = "disabled"
    if continuation_pin is not None:
        settings["model_continuation"] = continuation_pin
    return settings


def execution_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Compare runtime inputs, retaining config provenance in the original record.

    A configuration path, comment or unrelated profile does not change the
    selected agent/model/effort/permissions. Unknown fields remain significant.
    """
    provenance = {
        "source",
        "sha256",
        "defaults",
        "overrides",
        "selection_sources",
        "model_continuation",
    }
    return {
        **contract,
        "settings": {
            k: v for k, v in contract["settings"].items() if k not in provenance
        },
    }


def session_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Compare a lineage, allowing only an explicitly identified repair's effort."""
    value = execution_contract(contract)
    if "repair" in value:
        from lib.repair_turn import validate_contract

        validate_contract(contract)
        value.pop("repair")
        value["settings"] = {
            **value["settings"],
            "selected": {**value["settings"]["selected"], "effort": "max"},
        }
    return value


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
        if mcp_config is not None:
            servers = json.loads(mcp_config.read_text(encoding="utf-8"))
            if (
                settings["profile"] != "ops_report"
                or not isinstance(servers, dict)
                or set(servers) != {"ops_report"}
                or not isinstance(servers["ops_report"], dict)
                or set(servers["ops_report"]) - {"command", "args", "env"}
            ):
                raise ValueError(
                    "Claude audit requires the host-provided ops_report MCP server"
                )
            cmd += [
                "--mcp-config",
                json.dumps({"mcpServers": servers}),
                "--allowedTools",
                "mcp__ops_report__*",
            ]
        if session:
            cmd += ["--resume", session]
        return cmd
    # WSL drvfs CODEX_HOME cannot host the daemon's Unix socket. Keep the
    # supported CLI feature override on both initial and resumed invocations.
    memory_autodream = settings["profile"] == "memory_autodream"
    cmd = ["codex", "--disable", "daemon_auto_start", "exec", "--cd", str(cwd)]
    if mode == "read-only" and not memory_autodream:
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
        if settings["profile"] not in {"review", "ops_scout"} or mode != "read-only":
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
        cmd += ["-c", "web_search=" + json.dumps(settings["web_search"])]
    if memory_autodream:
        if session or read_dirs:
            raise ValueError(
                "memory_autodream requires a fresh turn with only the pinned checkout"
            )
        executable = shutil.which("codex")
        if executable is None:
            raise ValueError("memory_autodream requires the Codex CLI")
        # The sandbox re-execs the CLI. Grant the exact executable, not its host home.
        cmd[0] = str(Path(executable).resolve(strict=True))
        roots = ", ".join(
            f'{json.dumps(root, ensure_ascii=False)}="read"'
            for root in dict.fromkeys((":minimal", str(cwd), cmd[0]))
        )
        cmd += [
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "-c",
            'default_permissions="memory_autodream"',
            "-c",
            f"permissions={{memory_autodream={{filesystem={{{roots}}},network={{enabled=false}}}}}}",
            "-c",
            'approval_policy="never"',
            "-c",
            'web_search="disabled"',
            "-c",
            "project_doc_max_bytes=0",
            "-c",
            'shell_environment_policy.inherit="none"',
            "-c",
            'shell_environment_policy.set.PATH="/usr/bin:/bin"',
            "--disable",
            "apps",
            "--disable",
            "plugins",
            "--disable",
            "hooks",
            "--disable",
            "memories",
        ]
    if settings["profile"] in {
        "review",
        "ops_scout",
        "issue_judgement",
        "memory_autodream",
    }:
        cmd += ["--disable", "multi_agent"]
    if session:
        cmd += [session]
    return [*cmd, "-"]
