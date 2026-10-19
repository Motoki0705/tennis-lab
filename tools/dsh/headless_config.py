"""Validate the pinned headless configuration without credentials or API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

DIRECTORY = Path(__file__).resolve().parent
RUNTIME = DIRECTORY / "runtime.mjs"
PATCH = DIRECTORY / "headless.patch.yml"
MODES = ("read-only", "workspace-write")
MODEL = {
    "provider": "deepseek-official",
    "model": "deepseek-flash",
    "reasoningEffort": "high",
}
ADAPTER = {
    "baseURL": "https://api.deepseek.com/anthropic",
    "apiKeyEnv": "DEEPSEEK_API_KEY",
    "thinking": "enabled",
    "reasoningEffort": "high",
}
SHARED_PATCH: list[dict[str, Any]] = [
    {"id": "agent-default-model", "config": MODEL},
    {"id": "llm-deepseek", "config": ADAPTER},
    {"id": "session-log-deepseek", "config": {"enabled": False}},
    {"id": "session-telemetry-otel", "disabled": True},
]
PLUGIN_NAMES = {
    "agent-default-model": "dsh-agent-default-model",
    "llm-deepseek": "dsh-llm-deepseek-api-key",
    "session-log-deepseek": "dsh-session-log-deepseek",
    "session-telemetry-otel": "dsh-session-telemetry-otel",
    "sandbox-policy": "dsh-sandbox-policy",
    "approval": "dsh-user-approval",
    "permission": "dsh-permission-presets",
}


class ConfigError(ValueError):
    """A safe diagnostic; arbitrary dsh output must never enter public output."""


class DumpLoader(yaml.SafeLoader):
    """Read the dump's !!js values as inert data, never as Python/JS code."""


def _js_expression(loader: DumpLoader, node: yaml.ScalarNode) -> dict[str, str]:
    return {"__unevaluated_js__": loader.construct_scalar(node)}


DumpLoader.add_constructor("tag:yaml.org,2002:js", _js_expression)


def permission_mode(value: str | None = None) -> str:
    """Reject an unsafe inherited mode even when an explicit safe mode is given."""
    ambient = os.environ.get("DSH_PERMISSION_MODE")
    for candidate in (ambient, value):
        if candidate is not None and candidate not in MODES:
            raise ConfigError(
                "DSH_PERMISSION_MODE must be read-only or workspace-write; danger-full-access is forbidden."
            )
    return value if value is not None else ambient or "read-only"


def permission_patch(mode: str) -> list[dict[str, Any]]:
    return [
        {"id": "sandbox-policy", "config": {"mode": mode}},
        {"id": "approval", "config": {"policy": "ask"}},
        {
            "id": "permission",
            "config": {
                "defaultPreset": mode,
                "presets": {mode: {"sandbox": mode, "approval": "ask"}},
            },
        },
    ]


def _same(left: Any, right: Any) -> bool:
    # JSON also distinguishes false from 0 and true from 1 (Python == does not).
    try:
        return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
    except (TypeError, ValueError):
        return False


def _load_patch(path: Path, expected: list[dict[str, Any]]) -> str:
    try:
        content = path.read_bytes()
        parsed = yaml.safe_load(content)
        if not _same(parsed, expected):
            raise ConfigError(
                "Headless patch does not match the pinned model/safety policy."
            )
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ConfigError(
            "Invalid headless patch; restore the reviewed literal policy files."
        ) from error
    return hashlib.sha256(content).hexdigest()


def headless_arguments(mode: str, patch: Path = PATCH) -> list[str]:
    """The final CLI overlays; later overlays would invalidate this contract."""
    mode = permission_mode(mode)
    _load_patch(patch, SHARED_PATCH)
    mode_path = DIRECTORY / f"{mode}.patch.yml"
    _load_patch(mode_path, permission_patch(mode))
    return [
        "--profile",
        "headless",
        "--patch",
        str(patch.resolve()),
        "--patch",
        str(mode_path),
    ]


def headless_environment(
    mode: str, home: Path, source: dict[str, str]
) -> dict[str, str]:
    """For callers that explicitly own credentials and a dedicated DSH_HOME."""
    mode = permission_mode(mode)
    if "DSH_PERMISSION_MODE" in source and source["DSH_PERMISSION_MODE"] not in MODES:
        raise ConfigError(
            "DSH_PERMISSION_MODE must be read-only or workspace-write; danger-full-access is forbidden."
        )
    return {
        **source,
        "DSH_HOME": str(home.resolve()),
        "DSH_TELEMETRY_DISABLED": "1",
        "DSH_PERMISSION_MODE": mode,
    }


def inspect_dump(text: str, mode: str) -> None:
    """Check composed rows, including plugin identity and literal safety values."""
    try:
        rows = yaml.load(text, Loader=DumpLoader)
    except yaml.YAMLError as error:
        raise ConfigError("Invalid YAML in dsh configuration dump.") from error
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ConfigError("Expected an entry list in dsh configuration dump.")
    for expected in SHARED_PATCH + permission_patch(mode):
        identifier = expected["id"]
        name = "@deepseek-ai/" + PLUGIN_NAMES[identifier]
        matches = [
            row
            for row in rows
            if row.get("id") == identifier or row.get("name") == name
        ]
        if len(matches) != 1:
            raise ConfigError(f"Missing or duplicate configuration row: {identifier}.")
        row = matches[0]
        if row.get("id") != identifier or row.get("name") != name or row.get("group"):
            raise ConfigError(f"Unexpected plugin identity: {identifier}.")
        if "disabled" in expected:
            valid = row.get("disabled") is True
        else:
            valid = row.get("disabled", False) is False and _same(
                row.get("config"), expected["config"]
            )
        if not valid:
            raise ConfigError(f"Effective headless policy mismatch: {identifier}.")


def check_config(mode: str, patch: Path = PATCH) -> dict[str, Any]:
    """Run only --dump-config through runtime.mjs in empty home/cwd directories."""
    mode = permission_mode(mode)
    arguments = headless_arguments(mode, patch)
    node = shutil.which("node")
    if node is None:
        raise ConfigError(
            "Node.js is missing; follow tools/dsh/README.md to install the pinned runtime."
        )
    with tempfile.TemporaryDirectory(prefix="dsh-headless-check-") as temporary:
        root = Path(temporary)
        home, cwd = root / "home", root / "cwd"
        home.mkdir(mode=0o700)
        cwd.mkdir(mode=0o700)
        # An allowlist excludes all inherited keys, provider overrides, NODE_OPTIONS,
        # and existing DSH_HOME/.env. The dump composes without booting plugins.
        environment = headless_environment(
            mode, home, {"PATH": os.defpath, "HOME": str(home), "TMPDIR": temporary}
        )
        try:
            result = subprocess.run(
                [node, str(RUNTIME), "exec", "--", *arguments, "--dump-config"],
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ConfigError(
                "dsh configuration dump could not finish; check the local runtime installation."
            ) from error
        if result.returncode != 0 or result.stderr.strip():
            # Missing targets can be warnings with exit 0; those also fail closed.
            raise ConfigError(
                f"dsh configuration dump failed or warned (exit {result.returncode}); check runtime/patch files."
            )
        inspect_dump(result.stdout, mode)
    return {
        **MODEL,
        "session_log_upload": False,
        "telemetry_disabled": True,
        "permission_mode": mode,
        "approval_policy": "ask",
        "headless_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "permission_patch_sha256": hashlib.sha256(
            (DIRECTORY / f"{mode}.patch.yml").read_bytes()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--patch", type=Path, default=PATCH)
    arguments = parser.parse_args()
    try:
        report = check_config(permission_mode(arguments.mode), arguments.patch)
    except ConfigError as error:
        print(f"dsh configuration: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
