"""Exercise config inspection and one smoke with a local, API-free dsh fixture."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tools/dsh"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

FAKE_CLI = """
import { appendFileSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = dirname(dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const behavior = JSON.parse(readFileSync(join(root, 'behavior.json'), 'utf8'));
appendFileSync(join(root, 'calls.ndjson'), JSON.stringify({
  args, cwd: process.cwd(), home: process.env.DSH_HOME,
  cwd_files: readdirSync(process.cwd()), home_files: readdirSync(process.env.DSH_HOME),
  key_present: Object.hasOwn(process.env, 'DEEPSEEK_API_KEY'),
  another_key_present: Object.hasOwn(process.env, 'ANOTHER_API_KEY'),
  telemetry: process.env.DSH_TELEMETRY_DISABLED, mode: process.env.DSH_PERMISSION_MODE,
  node_options_present: Object.hasOwn(process.env, 'NODE_OPTIONS'),
}) + '\\n');
if (args[0] === '-V') { console.log('0.2.0-rc.2'); process.exit(0); }
if (args.includes('--dump-config')) {
  const modePatch = args[args.lastIndexOf('--patch') + 1];
  process.stdout.write(readFileSync(join(root, basename(modePatch, '.patch.yml') + '.dump.yml')));
  if (behavior.dump_warning) console.error('fixture: unknown patch target');
  process.exit(behavior.dump_exit || 0);
}
if (!args.includes('--json') || args.at(-1) !== '-') process.exit(99);
const stdin = readFileSync(0, 'utf8');
writeFileSync(join(root, 'stdin.txt'), stdin);
if (behavior.modify_worktree) writeFileSync('unexpected-change.txt', 'mutation');
process.stdout.write(readFileSync(join(root, 'events.ndjson')));
console.error('private diagnostic /internal/fixture/path fixture-secret-value');
process.exit(behavior.exit_code || 0);
"""


def _dump(mode: str) -> list[dict[str, Any]]:
    """Independent fixture for the pinned CLI's composed entry-list contract."""
    rows: list[dict[str, Any]] = [
        {
            "id": "agent-default-model",
            "name": "@deepseek-ai/dsh-agent-default-model",
            "config": {
                "provider": "deepseek-official",
                "model": "deepseek-flash",
                "reasoningEffort": "high",
            },
        },
        {
            "id": "llm-deepseek",
            "name": "@deepseek-ai/dsh-llm-deepseek-api-key",
            "config": {
                "baseURL": "https://api.deepseek.com/anthropic",
                "apiKeyEnv": "DEEPSEEK_API_KEY",
                "thinking": "enabled",
                "reasoningEffort": "high",
            },
        },
        {
            "id": "session-log-deepseek",
            "name": "@deepseek-ai/dsh-session-log-deepseek",
            "config": {"enabled": False},
        },
        {
            "id": "session-telemetry-otel",
            "name": "@deepseek-ai/dsh-session-telemetry-otel",
            "disabled": True,
        },
        {
            "id": "sandbox-policy",
            "name": "@deepseek-ai/dsh-sandbox-policy",
            "config": {"mode": mode},
        },
        {
            "id": "approval",
            "name": "@deepseek-ai/dsh-user-approval",
            "config": {"policy": "ask"},
        },
        {
            "id": "permission",
            "name": "@deepseek-ai/dsh-permission-presets",
            "config": {
                "defaultPreset": mode,
                "presets": {mode: {"sandbox": mode, "approval": "ask"}},
            },
        },
    ]
    return rows


def _events() -> list[dict[str, Any]]:
    return [
        {
            "type": "session",
            "sessionId": "session-fixture",
            "cwd": "/internal/fixture/path",
        },
        {"type": "status", "phase": "turn_start", "turn": 1},
        {"type": "status", "phase": "step_start", "turn": 1, "step": 1},
        {
            "type": "tool_call",
            "callId": "call-1",
            "tool": "read",
            "input": {"file_path": "README.md", "limit": 20},
        },
        {
            "type": "tool_result",
            "callId": "call-1",
            "status": "completed",
            "result": "fixture README",
        },
        {
            "type": "status",
            "phase": "step_end",
            "turn": 1,
            "step": 1,
            "usage": {
                "inputTokens": 100,
                "cacheReadTokens": 40,
                "cacheWriteTokens": 0,
                "outputTokens": 10,
                "totalTokens": 150,
            },
        },
        {
            "type": "status",
            "phase": "turn_end",
            "turn": 1,
            "reason": {"kind": "completed"},
        },
        {
            "type": "final",
            "text": "DSH_READ_ONLY_OK\nA fixture summary /internal/fixture/path fixture-secret-value",
        },
    ]


@pytest.fixture
def owner(tmp_path: Path) -> Path:
    directory = tmp_path / "runtime owner" / "tools/dsh"
    directory.mkdir(parents=True)
    for name in (
        "runtime.mjs",
        "package.json",
        "headless.patch.yml",
        "read-only.patch.yml",
        "workspace-write.patch.yml",
        "headless_config.py",
        "smoke.py",
    ):
        shutil.copy2(SOURCE / name, directory / name)
    package = directory / "node_modules/@deepseek-ai/dsh"
    (package / "lib").mkdir(parents=True)
    (package / "lib/bin.js").write_text(FAKE_CLI)
    (package / "package.json").write_text(
        json.dumps(
            {
                "name": "@deepseek-ai/dsh",
                "version": "0.2.0-rc.2",
                "type": "module",
                "bin": {"dsh": "lib/bin.js"},
            }
        )
    )
    for mode in ("read-only", "workspace-write"):
        (package / f"{mode}.dump.yml").write_text(yaml.safe_dump(_dump(mode)))
    (package / "behavior.json").write_text("{}")
    _set_events(directory, _events())
    return directory


def _package(owner: Path) -> Path:
    return owner / "node_modules/@deepseek-ai/dsh"


def _set_events(owner: Path, events: list[dict[str, Any]]) -> None:
    (_package(owner) / "events.ndjson").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, str]:
    assert NODE is not None
    home = tmp_path / "caller home"
    home.mkdir()
    (home / ".credentials.yaml").write_text(
        "version: 1\nrefs:\n  DEEPSEEK_API_KEY: fixture-secret-value\n"
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return {
        "PATH": str(Path(NODE).parent) + os.pathsep + os.defpath,
        "HOME": str(home),
        "DSH_HOME": str(home),
        "TMPDIR": str(scratch),
    }


def _run(
    owner: Path,
    environment: dict[str, str],
    entry: str,
    *args: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = (
        [sys.executable, "-m", "tools.dsh.smoke"]
        if entry == "smoke.py"
        else [sys.executable, str(owner / entry)]
    )
    directory = owner.parents[1] if entry == "smoke.py" else cwd or owner
    return subprocess.run(
        [*command, *args],
        env=environment,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _calls(owner: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (_package(owner) / "calls.ndjson").read_text().splitlines()
    ]


@pytest.mark.parametrize("mode", ["read-only", "workspace-write"])
def test_check_dump_isolated_from_keys_cwd_home_and_node_options(
    owner: Path,
    environment: dict[str, str],
    tmp_path: Path,
    mode: str,
) -> None:
    caller = tmp_path / "caller cwd"
    caller.mkdir()
    (caller / ".env").write_text("DEEPSEEK_API_KEY=fixture-secret-value\n")
    environment.update(
        DEEPSEEK_API_KEY="fixture-secret-value",
        ANOTHER_API_KEY="another-fixture-key",
        NODE_OPTIONS="--require=/nonexistent/unsafe.js",
        DSH_TELEMETRY_DISABLED="0",
    )

    result = _run(owner, environment, "headless_config.py", "--mode", mode, cwd=caller)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["model"] == "deepseek-flash"
    assert report["reasoningEffort"] == "high"
    assert report["permission_mode"] == mode
    assert report["session_log_upload"] is False
    assert report["telemetry_disabled"] is True
    assert "fixture-secret-value" not in result.stdout + result.stderr
    calls = _calls(owner)
    assert len(calls) == 2
    for call in calls:
        assert call["key_present"] is False
        assert call["another_key_present"] is False
        assert call["node_options_present"] is False
        assert call["telemetry"] == "1"
        assert call["mode"] == mode
        assert call["cwd_files"] == []
        assert call["home_files"] == []
        assert not Path(call["home"]).exists()
        assert not Path(call["cwd"]).exists()
    assert calls[-1]["args"][-1] == "--dump-config"


@pytest.mark.parametrize(
    "replacement",
    [
        "not: a-list",
        "[unfinished",
        "- id: agent-default-model\n  config: 2026-10-19\n",
        "model: deepseek-v4-pro",
        "reasoningEffort: max",
        "enabled: true",
        "enabled: 0",
        "- id: extra-plugin\n  config: !!js process.env.DEEPSEEK_API_KEY\n",
    ],
)
def test_bad_patch_stops_before_dsh(
    owner: Path, environment: dict[str, str], replacement: str
) -> None:
    patch = owner / "invalid.patch.yml"
    original = (owner / "headless.patch.yml").read_text()
    if replacement.startswith("model:"):
        content = original.replace("model: deepseek-flash", replacement)
    elif replacement.startswith("reasoningEffort:"):
        content = original.replace("reasoningEffort: high", replacement)
    elif replacement.startswith("enabled:"):
        content = original.replace("enabled: false", replacement)
    else:
        content = replacement
    patch.write_text(content)

    result = _run(owner, environment, "headless_config.py", "--patch", str(patch))

    assert result.returncode == 1
    assert "Invalid headless patch" in result.stderr
    assert not (_package(owner) / "calls.ndjson").exists()


@pytest.mark.parametrize(
    ("row", "field", "value"),
    [
        (
            0,
            "config",
            {"provider": "other", "model": "deepseek-flash", "reasoningEffort": "high"},
        ),
        (
            0,
            "config",
            {
                "provider": "deepseek-official",
                "model": "deepseek-v4-pro",
                "reasoningEffort": "high",
            },
        ),
        (
            0,
            "config",
            {
                "provider": "deepseek-official",
                "model": "deepseek-flash",
                "reasoningEffort": "max",
            },
        ),
        (0, "disabled", True),
        (1, "config", {"thinking": "disabled"}),
        (2, "config", {"enabled": True}),
        (2, "config", {"enabled": 0}),
        (3, "disabled", False),
        (4, "config", {"mode": "danger-full-access"}),
        (5, "config", {"policy": "never"}),
        (6, "name", "@untrusted/permission"),
    ],
)
def test_effective_dump_must_match_independently_of_valid_patch(
    owner: Path,
    environment: dict[str, str],
    row: int,
    field: str,
    value: Any,
) -> None:
    rows = _dump("read-only")
    rows[row][field] = value
    (_package(owner) / "read-only.dump.yml").write_text(yaml.safe_dump(rows))

    result = _run(owner, environment, "headless_config.py")

    assert result.returncode == 1
    assert "headless" in result.stderr or "plugin identity" in result.stderr
    assert len(_calls(owner)) == 2


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "invalid-yaml", "warning", "nonzero"]
)
def test_dump_errors_and_skipped_targets_fail_closed(
    owner: Path, environment: dict[str, str], failure: str
) -> None:
    rows = _dump("read-only")
    if failure == "missing":
        rows.pop(2)
    if failure == "duplicate":
        rows.append(rows[0])
    (_package(owner) / "read-only.dump.yml").write_text(
        "[unfinished" if failure == "invalid-yaml" else yaml.safe_dump(rows)
    )
    (_package(owner) / "behavior.json").write_text(
        json.dumps(
            {
                "dump_warning": failure == "warning",
                "dump_exit": 1 if failure == "nonzero" else 0,
            }
        )
    )

    result = _run(owner, environment, "headless_config.py")

    assert result.returncode == 1
    assert result.stdout == ""


def test_dump_expressions_are_inert(
    owner: Path, environment: dict[str, str], tmp_path: Path
) -> None:
    marker = tmp_path / "must-not-be-written"
    dump = _package(owner) / "read-only.dump.yml"
    with dump.open("a") as stream:
        stream.write(
            f"- id: unused\n  config: !!js process.getBuiltinModule('fs').writeFileSync('{marker}', 'bad')\n"
        )
    result = _run(owner, environment, "headless_config.py")
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("mode", ["danger-full-access", "invalid", ""])
def test_unsafe_inherited_permission_refused(
    owner: Path, environment: dict[str, str], mode: str
) -> None:
    environment["DSH_PERMISSION_MODE"] = mode
    result = _run(owner, environment, "headless_config.py", "--mode", "read-only")
    assert result.returncode == 1
    assert "danger-full-access is forbidden" in result.stderr
    assert not (_package(owner) / "calls.ndjson").exists()


@pytest.fixture
def worktree(tmp_path: Path, environment: dict[str, str]) -> Path:
    main = tmp_path / "repo"
    main.mkdir()
    subprocess.run(["git", "init", "-q", str(main)], env=environment, check=True)
    (main / "README.md").write_text("# Smoke fixture\nA small README.\n")
    (main / ".gitignore").write_text("/.dsh-runs/\n")
    subprocess.run(["git", "-C", str(main), "add", "."], env=environment, check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(main),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        env=environment,
        check=True,
    )
    linked = main / ".claude/worktrees/smoke"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-qb", "smoke", str(linked)],
        env=environment,
        check=True,
    )
    return linked


def _smoke(
    owner: Path, environment: dict[str, str], worktree: Path, records: Path
) -> subprocess.CompletedProcess[str]:
    return _run(
        owner,
        {**environment, "DEEPSEEK_API_KEY": "fixture-secret-value"},
        "smoke.py",
        "--worktree",
        str(worktree),
        "--records-dir",
        str(records),
    )


def test_missing_key_does_not_discover_stored_key_or_invoke_dsh(
    owner: Path, environment: dict[str, str], tmp_path: Path
) -> None:
    result = _run(owner, environment, "smoke.py", "--worktree", str(tmp_path))
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["error_code"] == "MISSING_CREDENTIAL"
    assert "DEEPSEEK_API_KEY" in report["guidance"]
    assert "DSH_HOME" in report["guidance"]
    assert "repo/task/log" in report["guidance"]
    assert "fixture-secret-value" not in result.stdout + result.stderr
    assert not (_package(owner) / "calls.ndjson").exists()


def test_smoke_defaults_to_ignored_records_in_target_worktree(
    owner: Path,
    environment: dict[str, str],
    worktree: Path,
) -> None:
    result = _run(
        owner,
        {**environment, "DEEPSEEK_API_KEY": "fixture-secret-value"},
        "smoke.py",
        "--worktree",
        str(worktree),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    records = worktree / ".dsh-runs" / report["run_id"]
    assert (records / "events.ndjson").exists()
    assert report["worktree_unchanged"] is True
    assert (records / "git-after.status").read_bytes() == b""


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        ("dirty", "DIRTY_WORKTREE"),
        ("unignored", "UNSAFE_RECORDS"),
        ("danger", "CONFIG_ERROR"),
        ("main", "INVALID_WORKTREE"),
    ],
)
def test_smoke_preflight_guards_do_not_launch_a_task(
    owner: Path,
    environment: dict[str, str],
    worktree: Path,
    tmp_path: Path,
    failure: str,
    code: str,
) -> None:
    records = tmp_path / "records"
    if failure == "dirty":
        (worktree / "README.md").write_text("dirty")
    elif failure == "unignored":
        records = worktree / "public-records"
    elif failure == "danger":
        environment["DSH_PERMISSION_MODE"] = "danger-full-access"
    elif failure == "main":
        worktree = worktree.parents[2]
    result = _smoke(owner, environment, worktree, records)
    assert result.returncode == 1
    assert json.loads(result.stdout)["error_code"] == code
    assert not records.exists()
    assert not (_package(owner) / "stdin.txt").exists()


def test_multiple_step_usage_preserves_missing_bucket_and_sums_disjoint_counts(
    owner: Path,
    environment: dict[str, str],
    worktree: Path,
    tmp_path: Path,
) -> None:
    events = _events()
    events[6:6] = [
        {"type": "status", "phase": "step_start", "turn": 1, "step": 2},
        {
            "type": "status",
            "phase": "step_end",
            "turn": 1,
            "step": 2,
            "usage": {"inputTokens": 200, "outputTokens": 20, "cacheWriteTokens": 0},
        },
    ]
    _set_events(owner, events)
    records = tmp_path / "records"
    result = _smoke(owner, environment, worktree, records)
    assert result.returncode == 0, result.stdout + result.stderr
    usage = json.loads((records / "usage.json").read_text())
    assert usage["totals"]["inputTokens"] == 300
    assert usage["totals"]["outputTokens"] == 30
    assert usage["totals"]["cacheReadTokens"] is None
    assert usage["totals"]["totalTokens"] is None
    assert usage["observed_token_estimate_range_usd"] is None


def test_smoke_records_conditions_usage_and_private_output(
    owner: Path,
    environment: dict[str, str],
    worktree: Path,
    tmp_path: Path,
) -> None:
    records = tmp_path / "private records"
    result = _smoke(owner, environment, worktree, records)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert report["worktree_unchanged"] is True
    assert report["turn_end_reason"] == "completed"
    assert report["exit_code"] == 0
    for private_text in (
        "fixture-secret-value",
        "/internal/fixture/path",
        str(worktree),
        str(records),
    ):
        assert private_text not in result.stdout + result.stderr
        assert private_text not in (records / "summary.json").read_text()
    metadata = json.loads((records / "metadata.json").read_text())
    assert metadata["session_id"] == "session-fixture"
    assert metadata["cwd"] == str(worktree)
    assert Path(metadata["dsh_home"]).parent == records
    assert (
        (records / "git-before.status").read_bytes()
        == (records / "git-after.status").read_bytes()
        == b""
    )
    assert (records / "final.md").read_text().startswith("DSH_READ_ONLY_OK\n")
    assert "private diagnostic" in (records / "stderr.log").read_text()
    usage = json.loads((records / "usage.json").read_text())
    assert usage["totals"]["inputTokens"] == 100  # excludes cacheReadTokens
    assert usage["totals"]["cacheReadTokens"] == 40
    assert usage["totals"]["reasoningTokens"] is None
    assert usage["observed_token_estimate_range_usd"]["off_peak"] == pytest.approx(
        (100 * 0.15 + 40 * 0.003 + 10 * 0.6) / 1e6
    )
    assert usage["observed_token_estimate_range_usd"]["peak"] == pytest.approx(
        2 * usage["observed_token_estimate_range_usd"]["off_peak"]
    )
    assert usage["billing_total_usd"] is None
    assert usage["complete"] is False
    assert usage["pricing"]["checked_date"] == "2026-10-19"
    assert (
        usage["pricing"]["source"]
        == "https://api-docs.deepseek.com/quick_start/pricing/"
    )
    assert records.stat().st_mode & 0o777 == 0o700
    assert all(
        p.stat().st_mode & 0o777 == 0o600 for p in records.iterdir() if p.is_file()
    )
    task_call = _calls(owner)[-1]
    assert task_call["cwd"] == str(worktree)
    assert task_call["mode"] == "read-only"
    assert task_call["telemetry"] == "1"
    assert task_call["home_files"] == []
    assert task_call["key_present"] is True
    assert task_call["args"][-2:] == ["--json", "-"]
    assert "Read README.md" in (_package(owner) / "stdin.txt").read_text()


@pytest.mark.parametrize(
    "missing", ["cacheReadTokens", "cacheWriteTokens", "usage", "cache-write-priced"]
)
def test_missing_usage_never_becomes_zero_or_a_complete_estimate(
    owner: Path,
    environment: dict[str, str],
    worktree: Path,
    tmp_path: Path,
    missing: str,
) -> None:
    events = _events()
    step = events[5]
    if missing == "usage":
        step.pop("usage")
    elif missing == "cache-write-priced":
        step["usage"]["cacheWriteTokens"] = 5
    else:
        step["usage"].pop(missing)
    _set_events(owner, events)
    records = tmp_path / "records"
    result = _smoke(owner, environment, worktree, records)
    assert result.returncode == 0, result.stdout
    usage = json.loads((records / "usage.json").read_text())
    assert usage["observed_token_estimate_range_usd"] is None
    assert usage["complete"] is False
    if missing in {"cacheReadTokens", "cacheWriteTokens"}:
        assert usage["totals"][missing] is None
        assert missing in usage["missing_billable_buckets"]
    if missing == "usage":
        assert all(value is None for value in usage["totals"].values())


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("auth", "AUTH"),
        ("missing-key-event", "MISSING_CREDENTIAL"),
        ("permission", "PERMISSION_DENIED"),
        ("bad-json", "INVALID_NDJSON"),
        ("missing-final", "INCOMPLETE_RUN"),
        ("empty-final", "UNEXPECTED_FINAL"),
        ("nonzero", "DSH_FAILED"),
        ("other-tool", "UNEXPECTED_TOOL"),
        ("mutation", "WORKTREE_CHANGED"),
    ],
)
def test_api_tool_stream_and_worktree_failures_cannot_pass(
    owner: Path,
    environment: dict[str, str],
    worktree: Path,
    tmp_path: Path,
    failure: str,
    expected_code: str,
) -> None:
    events = _events()
    behavior: dict[str, Any] = {}
    if failure in {"auth", "missing-key-event"}:
        events[6]["reason"] = {
            "kind": "error",
            "error": {
                "code": "AUTH" if failure == "auth" else "MISSING_CREDENTIAL",
                "message": "private failure fixture-secret-value /internal/fixture/path",
            },
        }
        events[-1]["text"] = ""
        behavior["exit_code"] = 1
    elif failure == "permission":
        events[4].update(
            status="error",
            result="FS_SANDBOX_DENIED [sandbox: file access denied under read-only mode]",
        )
    elif failure == "missing-final":
        events.pop()
    elif failure == "empty-final":
        events[-1]["text"] = ""
    elif failure == "nonzero":
        behavior["exit_code"] = 7
    elif failure == "other-tool":
        events[3]["tool"] = "bash"
    elif failure == "mutation":
        behavior["modify_worktree"] = True
    _set_events(owner, events)
    if failure == "bad-json":
        (_package(owner) / "events.ndjson").write_text('{"type": "session"')
    (_package(owner) / "behavior.json").write_text(json.dumps(behavior))
    records = tmp_path / "records"
    result = _smoke(owner, environment, worktree, records)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["passed"] is False
    assert report["error_code"] == expected_code
    assert "fixture-secret-value" not in result.stdout + result.stderr
    assert "/internal/fixture/path" not in result.stdout + result.stderr
    assert (records / "metadata.json").exists()
    assert (records / "usage.json").exists()
    if failure == "permission":
        assert "Do not enable danger-full-access" in report["guidance"]
        assert report["exit_code"] == 0
        assert report["turn_end_reason"] == "completed"


@pytest.mark.parametrize("mode", ["read-only", "workspace-write"])
def test_installed_pinned_config_without_api(
    owner: Path, environment: dict[str, str], mode: str
) -> None:
    if not (SOURCE / "node_modules/@deepseek-ai/dsh/lib/bin.js").is_file():
        pytest.skip(
            "Local pinned dsh is not installed; offline fixture tests still run"
        )
    result = _run(SOURCE, environment, "headless_config.py", "--mode", mode)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["permission_mode"] == mode


def test_installed_read_only_filesystem_denies_write_without_api(
    environment: dict[str, str], tmp_path: Path
) -> None:
    library = SOURCE / "node_modules/@deepseek-ai/dsh-fs-sandbox/lib/index.js"
    if not library.is_file():
        pytest.skip("Local pinned dsh is not installed")
    assert NODE is not None
    target = tmp_path / "must-not-exist"
    script = """
import { pathToFileURL } from 'node:url';
const [library, target] = process.argv.slice(1);
const { SandboxedFileSystem } = await import(pathToFileURL(library));
try {
  await SandboxedFileSystem.prototype.checkedTarget.call({}, { displayPath: target },
    { mode: 'read-only', workspaceRoot: process.cwd() });
  process.exit(99);
} catch (error) {
  if (error.code !== 'FS_SANDBOX_DENIED') throw error;
  console.log(error.code);
}
"""
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script, str(library), str(target)],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "FS_SANDBOX_DENIED"
    assert not target.exists()
