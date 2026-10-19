"""Offline runner contracts using a fake pinned CLI and disposable worktrees."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tools/dsh"
NODE = shutil.which("node")
FILE_KEY = "fixture-file-secret-value-only"
AMBIENT_KEY = "fixture-ambient-value-must-not-be-used"
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

FAKE_CLI = r"""
import { appendFileSync, readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
const root = dirname(dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const behavior = JSON.parse(readFileSync(join(root, 'behavior.json'), 'utf8'));
const call = {
  args, cwd: process.cwd(), home: process.env.DSH_HOME,
  key_present: Object.hasOwn(process.env, 'DEEPSEEK_API_KEY'),
  file_key: process.env.DEEPSEEK_API_KEY === 'fixture-file-secret-value-only',
  ambient_key: process.env.DEEPSEEK_API_KEY === 'fixture-ambient-value-must-not-be-used',
  telemetry: process.env.DSH_TELEMETRY_DISABLED, mode: process.env.DSH_PERMISSION_MODE,
  recursive_marker: process.env.TENNIS_LAB_DSH_ACTIVE,
  unsafe_env: ['NODE_OPTIONS', 'NODE_PATH', 'DSH_TOOLS_MODE', 'DEEPSEEK_BASE_URL', 'ANOTHER_API_KEY', 'GIT_DIR'].filter(name => Object.hasOwn(process.env, name)),
};
appendFileSync(join(root, 'calls.ndjson'), JSON.stringify(call) + '\n');
if (args[0] === '-V') { console.log('0.2.0-rc.2'); process.exit(0); }
if (args.includes('--dump-config')) {
  if (behavior.dump_hang) {
    const descendant=spawn(process.execPath,['-e','process.on("SIGTERM",()=>{});setInterval(()=>{},1000)'],{stdio:'inherit'});
    process.on('SIGTERM',()=>{});
    writeFileSync(join(root,'descendants.json'),JSON.stringify([process.pid,descendant.pid]));
    writeFileSync(join(root,'ready'),String(process.pid));
    await new Promise(()=>setInterval(()=>{},1000));
  }
  const mode = basename(args[args.lastIndexOf('--patch') + 1], '.patch.yml');
  process.stdout.write(readFileSync(join(root, mode + '.dump.yml')));
  if (behavior.dump_warning) console.error('fixture config warning');
  process.exit(behavior.dump_exit || 0);
}
if (!args.includes('--json') || args.at(-1) !== '-') process.exit(99);
writeFileSync(join(root, 'stdin.bin'), readFileSync(0));
const statePath = join(process.env.DSH_HOME, '.fake-state.json');
const state = existsSync(statePath) ? JSON.parse(readFileSync(statePath)) : {};
const sid = args.includes('--session-id') ? args[args.indexOf('--session-id') + 1] : `session-fixture-${Object.keys(state).length + 1}`;
const turn = (state[sid]?.turn || 0) + 1;
state[sid] = { turn, cwd: process.cwd(), mode: process.env.DSH_PERMISSION_MODE };
writeFileSync(statePath, JSON.stringify(state), { mode: 0o600 });
if (behavior.modify) {
  writeFileSync('README.md', '# Changed\n');
  writeFileSync('new file "quote".txt', 'created\n');
  writeFileSync('existing untracked.txt', 'changed again\n');
}
let events = behavior.events || [
  {type:'session',sessionId:sid,cwd:process.cwd()},
  {type:'status',phase:'turn_start',turn},
  {type:'status',phase:'step_start',turn,step:1},
  {type:'status',phase:'step_end',turn,step:1,usage:{inputTokens:100,cacheReadTokens:40,cacheWriteTokens:0,outputTokens:10,totalTokens:150}},
  {type:'status',phase:'turn_end',turn,reason:{kind:'completed'}},
  {type:'final',text:behavior.final ?? 'A completed fixture turn; parent must verify the task.'},
];
events = JSON.parse(JSON.stringify(events).replaceAll('__cwd__', process.cwd()).replaceAll('__sid__', sid).replaceAll('"__turn__"', String(turn)));
if (behavior.api_error) {
  events.at(-2).reason = {kind:'error',error:{code:'AUTH',message:'private provider message'}};
  events.at(-1).text = '';
}
if (behavior.aborted) events.at(-2).reason = {kind:'aborted'};
if (behavior.stop_reason) events.at(-2).reason = {kind:behavior.stop_reason};
if (behavior.missing_usage) delete events[3].usage[behavior.missing_usage];
if (behavior.no_usage) delete events[3].usage;
if (behavior.cache_write) events[3].usage.cacheWriteTokens = 5;
if (behavior.truncated_text) events.splice(3, 0, {type:'text',text:'shortened',truncated:true});
if (behavior.hang) {
  process.stdout.write((behavior.exit_while_child ? events : events.slice(0,4)).map(e => JSON.stringify(e) + '\n').join(''));
  const tool = spawn(process.execPath, ['-e', `process.on('SIGTERM', () => {}); const {spawn}=require('child_process'); const fs=require('fs'); const grand=spawn(process.execPath,['-e','process.on("SIGTERM",()=>{});setInterval(()=>{},1000)'],{stdio:'inherit',detached:${Boolean(behavior.detached_hang)}}); fs.writeFileSync(${JSON.stringify(join(root, 'descendants.json'))},JSON.stringify([process.pid,grand.pid])); setInterval(()=>{},1000);`], {stdio:'inherit',detached:Boolean(behavior.detached_hang)});
  process.on('SIGTERM', () => {});
  writeFileSync(join(root, 'ready'), String(process.pid));
  if (behavior.exit_while_child) {
    while (!existsSync(join(root,'descendants.json'))) await new Promise(resolve=>setTimeout(resolve,10));
    process.exit(0);
  }
  setInterval(() => {}, 1000);
} else {
  if (behavior.raw !== undefined) process.stdout.write(behavior.raw);
  else process.stdout.write(events.map(e => JSON.stringify(e) + '\n').join(''));
  if (behavior.echo_key) {
    const key = process.env.DEEPSEEK_API_KEY;
    process.stderr.write(key.slice(0, 9));
    await new Promise(resolve => setTimeout(resolve, 30));
    process.stderr.write(key.slice(9));
  } else console.error('private fixture diagnostic');
  process.exit(behavior.api_error ? 1 : behavior.exit_code || 0);
}
"""


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def _dump(mode: str) -> list[dict[str, Any]]:
    rows = yaml.safe_load((SOURCE / "headless.patch.yml").read_text()) + yaml.safe_load(
        (SOURCE / f"{mode}.patch.yml").read_text()
    )
    names = {
        "agent-default-model": "dsh-agent-default-model",
        "llm-deepseek": "dsh-llm-deepseek-api-key",
        "session-log-deepseek": "dsh-session-log-deepseek",
        "session-telemetry-otel": "dsh-session-telemetry-otel",
        "sandbox-policy": "dsh-sandbox-policy",
        "approval": "dsh-user-approval",
        "permission": "dsh-permission-presets",
    }
    for row in rows:
        row["name"] = "@deepseek-ai/" + names[row["id"]]
    return list(rows)


@pytest.fixture
def fixture(tmp_path: Path) -> dict[str, Any]:
    owner = tmp_path / "runtime owner"
    directory = owner / "tools/dsh"
    directory.mkdir(parents=True)
    for path in SOURCE.iterdir():
        if path.suffix in {".py", ".yml", ".sh", ".mjs"} or path.name == "package.json":
            shutil.copy2(path, directory / path.name)
    (owner / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
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
    (package / "behavior.json").write_text("{}")
    # The runner's helper uses the public pinned read-only backend API. These
    # modules model that API without requiring real dsh/npm dependencies.
    modules = {
        "cordis": "export class Context {}",
        "dsh-session-persistence-jsonl": "export default class Persistence { constructor(context, config) { this.root = config.root; } }",
        "dsh-session": "export const SessionId = value => value; export class Session { static create() {} }",
        "dsh-session-query": """
import {readFileSync} from 'node:fs'; import {join,dirname} from 'node:path';
export async function readColdSessionLog(persistence, id) {
 const state=JSON.parse(readFileSync(join(dirname(persistence.root),'.fake-state.json')))[id];
 if (!state) throw new Error('missing fixture session');
 return {header:{version:4,id,cwd:state.cwd,origin:state.origin,parentSession:state.parentSession,agentPreset:state.agentPreset},inheritedEventCount:0,events:[
   {seq:0,type:'permission/preset',data:{preset:state.preset ?? state.mode}},
   {seq:1,type:'sandbox/mode',data:{mode:state.mode}},
   {seq:2,type:'approval/policy',data:{policy:state.approval ?? 'ask'}},
   {seq:3,type:'turn/start',data:{turn:state.turn}},
   {seq:4,type:'turn/end',data:{turn:state.turn,reason:{kind:'completed'}}},
 ]};
}
""",
        "dsh-session-format-catalog": "export const currentSessionMessageProjections = {};",
    }
    for name, source in modules.items():
        module = directory / "node_modules/@deepseek-ai" / name
        module.mkdir()
        (module / "index.js").write_text(source)
        (module / "package.json").write_text(
            json.dumps(
                {
                    "type": "module",
                    "exports": {
                        ".": "./index.js",
                        "./message-projections": "./index.js",
                    },
                }
            )
        )
    for mode in ("read-only", "workspace-write"):
        (package / f"{mode}.dump.yml").write_text(yaml.safe_dump(_dump(mode)))
    main = tmp_path / "repo"
    main.mkdir()
    _git(main, "init", "-q")
    (main / "README.md").write_text("# Fixture\n")
    (main / ".gitignore").write_text("/.dsh-runs/\n")
    _git(main, "add", ".")
    _git(
        main,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    worktree = main / ".claude/worktrees/runner"
    _git(main, "worktree", "add", "-qb", "runner", str(worktree))
    caller = tmp_path / "caller home"
    caller.mkdir(mode=0o700)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    key = tmp_path / "fake-key"
    key.write_text(FILE_KEY + "\n")
    key.chmod(0o600)
    task = tmp_path / "task.md"
    task.write_text(
        "# Task\nA \"quote\", 'apostrophe', $HOME, `uname`, $(id)\n日本語\n"
        + "long text\n" * 20000
    )
    assert NODE is not None
    environment = {
        "PATH": str(Path(NODE).parent) + os.pathsep + os.defpath,
        "HOME": str(caller),
        "TMPDIR": str(scratch),
        "DEEPSEEK_API_KEY": AMBIENT_KEY,
        "DSH_HOME": str(caller),
        "DSH_TELEMETRY_DISABLED": "0",
        "DSH_TOOLS_MODE": "unsafe",
        "NODE_OPTIONS": "--require=/must-not-be-loaded.js",
        "NODE_PATH": "/must-not-be-used",
        "DEEPSEEK_BASE_URL": "http://must-not-be-used.invalid",
        "ANOTHER_API_KEY": "another-fake-key",
        "GIT_DIR": "/must-not-influence-worktree-checks",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return {
        "owner": owner,
        "package": package,
        "main": main,
        "worktree": worktree,
        "environment": environment,
        "key": key,
        "task": task,
        "home": tmp_path / "persistent-home",
    }


def _command(fixture: dict[str, Any], **changes: Any) -> list[str]:
    options = {
        "task": fixture["task"],
        "worktree": fixture["worktree"],
        "mode": "workspace-write",
        "timeout": "10",
        "key-file": fixture["key"],
        "dsh-home": fixture["home"],
        **changes,
    }
    return [
        str(fixture["owner"] / "tools/dsh/run.sh"),
        *[
            piece
            for key, value in options.items()
            if value is not None
            for piece in ("--" + key, str(value))
        ],
    ]


def _run(
    fixture: dict[str, Any], **changes: Any
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    result = subprocess.run(
        _command(fixture, **changes),
        env=fixture["environment"],
        cwd=fixture["task"].parent,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert result.stderr == "", result.stderr
    assert len(result.stdout.splitlines()) == 1
    assert FILE_KEY not in result.stdout and AMBIENT_KEY not in result.stdout
    return result, json.loads(result.stdout)


def _behavior(fixture: dict[str, Any], **settings: Any) -> None:
    (fixture["package"] / "behavior.json").write_text(json.dumps(settings))


def _calls(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    path = fixture["package"] / "calls.ndjson"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def _records(report: dict[str, Any]) -> Path:
    return Path(report["records_dir"])


def test_stdin_cwd_fixed_arguments_environment_and_private_records(
    fixture: dict[str, Any],
) -> None:
    # The environment key and .env/home credentials all deliberately disagree.
    (fixture["worktree"] / ".env").write_text(
        "DEEPSEEK_API_KEY=fixture-dot-env-value\n"
    )
    (Path(fixture["environment"]["HOME"]) / ".credentials.yaml").write_text(
        "DEEPSEEK_API_KEY: fixture-credentials-value\n"
    )
    result, report = _run(fixture, task="task.md")
    assert result.returncode == 0, report
    assert report["result"] == "completed"
    assert report["exit_code"] == 0 and report["task_success"] is None
    assert "parent must verify" in report["meaning"]
    assert (fixture["package"] / "stdin.bin").read_bytes() == fixture[
        "task"
    ].read_bytes()
    call = _calls(fixture)[-1]
    assert call["args"][-2:] == ["--json", "-"]
    assert call["args"][:2] == ["--profile", "headless"]
    assert len([arg for arg in call["args"] if arg == "--patch"]) == 2
    assert call["cwd"] == str(fixture["worktree"])
    assert call["home"] == str(fixture["home"])
    assert call["file_key"] and not call["ambient_key"]
    assert call["mode"] == "workspace-write" and call["telemetry"] == "1"
    assert call["recursive_marker"] == "1"
    assert call["unsafe_env"] == []
    assert all(not call["key_present"] for call in _calls(fixture)[:-1])
    records = _records(report)
    assert (records / "final.md").read_text().startswith("A completed fixture turn")
    assert (records / "events.ndjson").is_file()
    assert records.stat().st_mode & 0o777 == 0o700
    assert records.parent.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in records.iterdir())
    assert fixture["home"].stat().st_mode & 0o777 == 0o700
    assert report["diff_summary"]["changed"] is False


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "directory",
        "symlink",
        "fifo",
        "open-mode",
        "execute-bit",
        "empty",
        "whitespace",
        "multiline",
        "non-utf8",
        "owner",
    ],
)
def test_bad_key_files_fail_before_cli_and_never_use_ambient_key(
    fixture: dict[str, Any], failure: str
) -> None:
    key = fixture["key"]
    if failure in {"missing", "directory", "symlink", "fifo"}:
        key.unlink()
    if failure == "directory":
        key.mkdir(mode=0o700)
    if failure == "symlink":
        key.symlink_to(fixture["task"])
    if failure == "fifo":
        os.mkfifo(key, 0o600)
    if failure == "open-mode":
        key.chmod(0o640)
    if failure == "execute-bit":
        key.chmod(0o700)
    if failure == "empty":
        key.write_bytes(b"")
    if failure == "whitespace":
        key.write_bytes(b" \n ")
    if failure == "multiline":
        key.write_text("fixture\nsecond-key")
    if failure == "non-utf8":
        key.write_bytes(b"\xff")
    if failure == "owner":
        # Ownership cannot be changed portably by an unprivileged fixture.
        script = "from pathlib import Path; from unittest.mock import patch; from tools.dsh.run_state import read_key, RunError; import os;\nwith patch('tools.dsh.run_state.os.getuid', return_value=os.getuid()+1):\n try: read_key(Path(__import__('sys').argv[1]))\n except RunError as error: print(error.code)\n"
        result = subprocess.run(
            [sys.executable, "-c", script, str(key)],
            cwd=fixture["owner"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "INVALID_KEY_FILE"
        return
    result, report = _run(fixture)
    assert result.returncode != 0
    assert report["error_code"] == "INVALID_KEY_FILE"
    assert "install -m 600 /dev/null" in report["guidance"]
    assert _calls(fixture) == []
    assert (_records(report) / "metadata.json").is_file()


@pytest.mark.parametrize(
    "failure",
    [
        "main",
        "subdirectory",
        "missing",
        "outside-layout",
        "danger",
        "ambient-danger",
        "recursive",
        "native-recursive",
        "unignored",
        "public-records",
        "key-in-repo",
        "home-in-repo",
    ],
)
def test_rejected_invocations_do_not_launch_task(
    fixture: dict[str, Any], failure: str
) -> None:
    options: dict[str, Any] = {}
    if failure == "main":
        options["worktree"] = fixture["main"]
    elif failure == "subdirectory":
        subdirectory = fixture["worktree"] / "child"
        subdirectory.mkdir()
        options["worktree"] = subdirectory
    elif failure == "missing":
        options["worktree"] = fixture["main"] / "nonexistent"
    elif failure == "outside-layout":
        outside = fixture["main"].parent / "outside"
        _git(fixture["main"], "worktree", "add", "-qb", "outside", str(outside))
        options["worktree"] = outside
    elif failure == "danger":
        options["mode"] = "danger-full-access"
    elif failure == "ambient-danger":
        fixture["environment"]["DSH_PERMISSION_MODE"] = "danger-full-access"
    elif failure == "recursive":
        fixture["environment"]["TENNIS_LAB_DSH_ACTIVE"] = "1"
    elif failure == "native-recursive":
        fixture["environment"]["DSH_SHELL"] = "1"
    elif failure == "unignored":
        (fixture["worktree"] / ".gitignore").write_text("")
    elif failure == "public-records":
        (fixture["worktree"] / ".dsh-runs").mkdir(mode=0o755)
    elif failure == "key-in-repo":
        options["key-file"] = fixture["worktree"] / "key"
    elif failure == "home-in-repo":
        options["dsh-home"] = fixture["worktree"] / "home"
    result, report = _run(fixture, **options)
    assert result.returncode != 0
    assert report["error_code"]
    assert not (fixture["package"] / "stdin.bin").exists()


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "-inf", "1e1000", "text"])
def test_timeout_must_be_positive_and_finite(
    fixture: dict[str, Any], timeout: str
) -> None:
    result, report = _run(fixture, timeout=timeout)
    assert result.returncode != 0
    assert report["error_code"] == "INVALID_ARGUMENTS"
    assert _calls(fixture) == []


@pytest.mark.parametrize(
    ("settings", "expected", "code"),
    [
        ({}, "completed", 0),
        ({"api_error": True}, "api_error", 10),
        ({"aborted": True}, "aborted", 13),
        ({"stop_reason": "max-tokens"}, "turn_stopped", 13),
        ({"stop_reason": "blocked"}, "turn_stopped", 13),
        ({"exit_code": 9}, "dsh_failed", 12),
        ({"raw": '{"type":"session"'}, "invalid_ndjson", 11),
        ({"raw": "not json\n"}, "invalid_ndjson", 11),
        ({"raw": ""}, "invalid_ndjson", 11),
    ],
)
def test_result_states(
    fixture: dict[str, Any], settings: dict[str, Any], expected: str, code: int
) -> None:
    _behavior(fixture, **settings)
    result, report = _run(fixture)
    assert result.returncode == code, report
    assert report["result"] == expected
    assert report["exit_code"] == code
    assert report["task_success"] is None
    metadata = json.loads((_records(report) / "metadata.json").read_text())
    assert metadata["result"] == expected
    assert (_records(report) / "usage.json").is_file()


def test_continuation_unknown_id_cwd_and_per_run_usage(fixture: dict[str, Any]) -> None:
    first_result, first = _run(fixture)
    assert first_result.returncode == 0
    second_result, second = _run(fixture, **{"session-id": first["session_id"]})
    assert second_result.returncode == 0, second
    assert first["run_id"] != second["run_id"]
    assert first["session_id"] == second["session_id"]
    assert second["turn"] == 2
    assert first["usage_totals"] == second["usage_totals"]
    usage = json.loads((_records(second) / "usage.json").read_text())
    assert usage["totals"]["inputTokens"] == 100
    assert usage["turns"][0]["turn"] == 2
    assert [run["turn"] for run in usage["session_runs"]] == [1, 2]
    assert usage["session"]["totals"]["inputTokens"] == 200
    calls = len(_calls(fixture))
    result, rejected = _run(fixture, **{"session-id": "session-unknown"})
    assert result.returncode != 0 and rejected["error_code"] == "UNKNOWN_SESSION"
    assert (_records(rejected) / "metadata.json").exists()
    other = fixture["main"] / ".claude/worktrees/other"
    _git(fixture["main"], "worktree", "add", "-qb", "other", str(other))
    result, rejected = _run(
        fixture, worktree=other, **{"session-id": first["session_id"]}
    )
    assert result.returncode != 0 and rejected["error_code"] == "SESSION_CWD_MISMATCH"
    assert len(_calls(fixture)) == calls


def test_git_evidence_captures_existing_dirty_and_untracked_changes(
    fixture: dict[str, Any],
) -> None:
    (fixture["worktree"] / "README.md").write_text("already dirty\n")
    (fixture["worktree"] / "existing untracked.txt").write_text("untracked before\n")
    _behavior(fixture, modify=True)
    result, report = _run(fixture)
    assert result.returncode == 0
    changes = report["diff_summary"]
    assert changes["changed"] is True
    assert changes["tracked_before"] == changes["tracked_after"] == ["README.md"]
    assert changes["untracked_before"] == ["existing untracked.txt"]
    assert changes["untracked_after"] == [
        "existing untracked.txt",
        'new file "quote".txt',
    ]
    assert set(changes["changed_paths"]) == {
        "README.md",
        "existing untracked.txt",
        'new file "quote".txt',
    }
    records = _records(report)
    assert b"already dirty" in (records / "git-before.diff").read_bytes()
    assert b"Changed" in (records / "git-after.diff").read_bytes()
    before = json.loads((records / "git-before.json").read_text())
    after = json.loads((records / "git-after.json").read_text())
    assert before["untracked"][0]["sha256"] != after["untracked"][0]["sha256"]


@pytest.mark.parametrize(
    "missing", [None, "cacheReadTokens", "cacheWriteTokens", "all", "cache-write"]
)
def test_usage_estimate_and_missing_buckets(
    fixture: dict[str, Any], missing: str | None
) -> None:
    _behavior(
        fixture,
        missing_usage=missing
        if missing in {"cacheReadTokens", "cacheWriteTokens"}
        else None,
        no_usage=missing == "all",
        cache_write=missing == "cache-write",
    )
    result, report = _run(fixture)
    assert result.returncode == 0
    usage = json.loads((_records(report) / "usage.json").read_text())
    assert usage["billing_total_usd"] is None and usage["complete"] is False
    assert usage["pricing"]["currency"] == "USD"
    assert usage["pricing"]["checked_date"] == "2026-10-19"
    if missing:
        assert report["observed_token_estimate_range_usd"] is None
        if missing != "cache-write":
            assert report["usage_missing"] is True
    else:
        assert usage["totals"]["inputTokens"] == 100
        assert usage["totals"]["cacheReadTokens"] == 40
        assert usage["totals"]["reasoningTokens"] is None
        assert report["observed_token_estimate_range_usd"]["off_peak"] == pytest.approx(
            (100 * 0.15 + 40 * 0.003 + 10 * 0.6) / 1e6
        )
        assert report["usage_missing"] is False


def _wait_file(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while not path.exists():
        if process.poll() is not None:
            pytest.fail(f"Wrapper exited early: {process.communicate()}")
        if time.monotonic() >= deadline:
            process.kill()
            pytest.fail("Fake process did not become ready")
        time.sleep(0.03)


def _live(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("termination", ["timeout", "SIGINT", "SIGTERM"])
@pytest.mark.parametrize("detached", [False, True])
def test_timeout_and_interrupt_kill_descendants_and_preserve_prefix_usage(
    fixture: dict[str, Any],
    termination: str,
    detached: bool,
) -> None:
    _behavior(fixture, hang=True, detached_hang=detached)
    process = subprocess.Popen(
        _command(fixture, timeout="2" if termination == "timeout" else "20"),
        env=fixture["environment"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_file(fixture["package"] / "ready", process)
    _wait_file(fixture["package"] / "descendants.json", process)
    descendants = json.loads((fixture["package"] / "descendants.json").read_text())
    descendants.append(int((fixture["package"] / "ready").read_text()))
    if termination != "timeout":
        process.send_signal(getattr(signal, termination))
    stdout, stderr = process.communicate(timeout=15)
    assert stderr == ""
    report = json.loads(stdout)
    assert report["result"] == (
        "timeout" if termination == "timeout" else "interrupted"
    )
    assert process.returncode == (
        124 if termination == "timeout" else 128 + getattr(signal, termination)
    )
    assert all(not _live(pid) for pid in descendants)
    assert report["usage_totals"]["inputTokens"] == 100
    assert report["usage_missing"] is True
    assert (_records(report) / "metadata.json").is_file()
    # The lock is released even after interruption; a fresh turn can run.
    _behavior(fixture)
    assert _run(fixture)[0].returncode == 0


def test_normal_exit_cleans_up_detached_orphaned_tools(fixture: dict[str, Any]) -> None:
    _behavior(fixture, hang=True, detached_hang=True, exit_while_child=True)
    result, report = _run(fixture)
    assert result.returncode == 0 and report["result"] == "completed"
    descendants = json.loads((fixture["package"] / "descendants.json").read_text())
    assert all(not _live(pid) for pid in descendants)
    assert all(not Path(f"/proc/{pid}").exists() for pid in descendants)


def test_cli_echo_of_key_is_redacted_before_recording(fixture: dict[str, Any]) -> None:
    _behavior(fixture, echo_key=True, final="fixture-file-secret-value-only")
    result, report = _run(fixture)
    assert result.returncode == 0
    for path in _records(report).iterdir():
        assert FILE_KEY.encode() not in path.read_bytes()
    assert (_records(report) / "final.md").read_text() == "<redacted-api-key>"
    assert (_records(report) / "stderr.log").read_text() == "<redacted-api-key>"


def test_invalid_final_unicode_is_a_recorded_failure(fixture: dict[str, Any]) -> None:
    _behavior(fixture, final="\ud800")
    result, report = _run(fixture)
    assert result.returncode == 11
    assert report["error_code"] == "INVALID_FINAL_ENCODING"
    assert (_records(report) / "metadata.json").is_file()


def test_legitimate_text_truncation_is_not_stream_corruption(
    fixture: dict[str, Any],
) -> None:
    _behavior(fixture, truncated_text=True)
    result, report = _run(fixture)
    assert result.returncode == 0
    assert (
        json.loads((_records(report) / "metadata.json").read_text())["truncated_events"]
        == 1
    )


def _turn_events(body: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"type": "session", "sessionId": "__sid__", "cwd": "__cwd__"},
        {"type": "status", "phase": "turn_start", "turn": 1},
        {"type": "status", "phase": "step_start", "turn": 1, "step": 1},
        *body,
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
            },
        },
        {
            "type": "status",
            "phase": "turn_end",
            "turn": 1,
            "reason": {"kind": "completed"},
        },
        {"type": "final", "text": "Lossless final answer.\n" * 2000},
    ]


@pytest.mark.parametrize(
    ("body", "types", "missing_fields"),
    [
        (
            [
                {"type": "tool_call", "callId": "c", "tool": "read", "truncated": True},
                {
                    "type": "tool_result",
                    "callId": "c",
                    "status": "completed",
                    "result": "done",
                },
            ],
            {"tool_call": 1},
            {"tool_call.input": 1},
        ),
        (
            [
                {"type": "text", "text": "short", "truncated": True},
                {"type": "text", "text": "short again", "truncated": True},
                {"type": "thinking", "text": "short", "truncated": True},
                {
                    "type": "tool_result",
                    "callId": "c",
                    "status": "completed",
                    "result": "short",
                    "truncated": True,
                },
            ],
            {"text": 2, "thinking": 1, "tool_result": 1},
            {},
        ),
        (
            [
                {"type": kind, "truncated": True}
                for kind in ("tool_call", "tool_result", "text", "thinking")
            ],
            {"tool_call": 1, "tool_result": 1, "text": 1, "thinking": 1},
            {
                "tool_call.callId": 1,
                "tool_call.tool": 1,
                "tool_call.input": 1,
                "tool_result.callId": 1,
                "tool_result.status": 1,
                "tool_result.result": 1,
                "text.text": 1,
                "thinking.text": 1,
            },
        ),
    ],
    ids=["tool-input-omitted", "strings-shortened", "body-type-only"],
)
def test_shortened_body_preserves_completion_final_usage_and_loss_record(
    fixture: dict[str, Any],
    body: list[dict[str, Any]],
    types: dict[str, int],
    missing_fields: dict[str, int],
) -> None:
    events = _turn_events(body)
    _behavior(fixture, events=events)
    result, report = _run(fixture)
    assert result.returncode == 0 and report["result"] == "completed"
    assert report["turn_end_reason"] == "completed"
    assert (_records(report) / "final.md").read_text() == events[-1]["text"]
    usage = json.loads((_records(report) / "usage.json").read_text())
    assert usage["stream_complete"] is True and usage["usage_complete"] is True
    assert usage["totals"]["inputTokens"] == 100
    assert usage["totals"]["cacheReadTokens"] == 40
    assert usage["totals"]["cacheWriteTokens"] == 0
    assert usage["totals"]["outputTokens"] == 10
    assert usage["turns"][0]["usage"]["totals"] == usage["totals"]
    assert usage["session"]["totals"] == usage["totals"]
    assert report["usage_missing"] is False
    assert report["observed_token_estimate_range_usd"] is not None
    for record in (
        report,
        json.loads((_records(report) / "summary.json").read_text()),
        json.loads((_records(report) / "metadata.json").read_text()),
    ):
        assert record["truncated_events"] == sum(types.values())
        assert record["truncated_event_types"] == types
        assert record["truncated_missing_fields"] == missing_fields


@pytest.mark.parametrize("loss", ["usage", "reason", "status", "session"])
def test_shortened_status_or_identity_does_not_prevent_reading_later_events(
    fixture: dict[str, Any], loss: str
) -> None:
    events = _turn_events([])
    if loss == "usage":
        events[3].pop("usage")
        events[3]["truncated"] = True
    elif loss == "reason":
        events[4].pop("reason")
        events[4]["truncated"] = True
    elif loss == "status":
        events[2] = {"type": "status", "truncated": True}
    else:
        events[0] = {"type": "session", "truncated": True}
    _behavior(fixture, events=events)
    result, report = _run(fixture)
    assert result.returncode == (0 if loss == "usage" else 11)
    assert report["result"] == ("completed" if loss == "usage" else "invalid_ndjson")
    assert (_records(report) / "final.md").read_text() == events[-1]["text"]
    metadata = json.loads((_records(report) / "metadata.json").read_text())
    assert metadata["ndjson_complete"] is True
    assert report["truncated_events"] == 1
    assert report["usage_missing"] is True
    usage = json.loads((_records(report) / "usage.json").read_text())
    if loss == "usage":
        assert usage["stream_complete"] is True
        assert all(value is None for value in usage["totals"].values())
        # Optional usage could already have been absent before serialization.
        assert report["truncated_missing_fields"] == {}
    else:
        assert usage["totals"]["inputTokens"] == 100
        assert usage["totals"]["outputTokens"] == 10
        assert usage["stream_complete"] is False
        assert report["truncated_missing_fields"]


@pytest.mark.parametrize(
    "failure", ["unmarked-input-loss", "wrong-call-id", "marked-final"]
)
def test_truncation_support_does_not_hide_corrupt_streams(
    fixture: dict[str, Any], failure: str
) -> None:
    body: list[dict[str, Any]] = [{"type": "tool_call", "callId": "c", "tool": "read"}]
    if failure == "wrong-call-id":
        body[0].update(callId=[], input={}, truncated=True)
    events = _turn_events(body if failure != "marked-final" else [])
    if failure == "marked-final":
        events[-1]["truncated"] = True
    _behavior(fixture, events=events)
    result, report = _run(fixture)
    assert result.returncode == 11 and report["result"] == "invalid_ndjson"
    assert (_records(report) / "final.md").read_text() == ""
    assert report["usage_missing"] is True
    assert report["truncated_events"] == 0
    metadata = json.loads((_records(report) / "metadata.json").read_text())
    assert metadata["ndjson_complete"] is False


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("mode", "danger-full-access", "SESSION_MODE_MISMATCH"),
        ("mode", "read-only", "SESSION_MODE_MISMATCH"),
        ("approval", "never", "SESSION_MODE_MISMATCH"),
        ("preset", "other", "SESSION_MODE_MISMATCH"),
        ("turn", 8, "SESSION_TURN_MISMATCH"),
        ("cwd", "/wrong/worktree", "SESSION_CWD_MISMATCH"),
        ("origin", "subagent", "INVALID_SESSION_STATE"),
        ("agentPreset", "unreviewed-agent", "INVALID_SESSION_STATE"),
    ],
)
def test_persisted_session_state_must_match_wrapper_registry(
    fixture: dict[str, Any],
    field: str,
    value: Any,
    expected: str,
) -> None:
    assert _run(fixture)[0].returncode == 0
    state_path = fixture["home"] / ".fake-state.json"
    saved = json.loads(state_path.read_text())
    identity = next(iter(saved))
    saved[identity][field] = value
    state_path.write_text(json.dumps(saved))
    (fixture["package"] / "stdin.bin").unlink()
    result, report = _run(fixture, **{"session-id": identity})
    assert result.returncode != 0
    assert report["error_code"] == expected
    assert not (fixture["package"] / "stdin.bin").exists()


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".credentials.yaml",
        "settings.yaml",
        "cordis.patch.yml",
        "profiles/headless/cordis.patch.yml",
    ],
)
def test_unmanaged_home_configuration_is_rejected_without_cli(
    fixture: dict[str, Any],
    name: str,
) -> None:
    fixture["home"].mkdir(mode=0o700)
    path = fixture["home"] / name
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text("unreviewed customization")
    result, report = _run(fixture)
    assert result.returncode != 0
    assert report["error_code"] == "UNSAFE_DSH_HOME"
    assert _calls(fixture) == []


def test_changed_managed_home_profile_is_rejected(fixture: dict[str, Any]) -> None:
    assert _run(fixture)[0].returncode == 0
    profile = fixture["home"] / "profiles/headless/cordis.yml"
    profile.parent.mkdir(mode=0o700, parents=True)
    profile.write_text("- id: arbitrary-plugin\n")
    calls = len(_calls(fixture))
    result, report = _run(fixture)
    assert result.returncode != 0 and report["error_code"] == "UNSAFE_DSH_HOME"
    assert len(_calls(fixture)) == calls


def test_default_external_key_and_home_paths_ignore_ambient_settings(
    fixture: dict[str, Any],
) -> None:
    caller = Path(fixture["environment"]["HOME"])
    key = caller / ".config/tennis-lab/deepseek-api-key"
    key.parent.mkdir(parents=True)
    key.write_text(FILE_KEY)
    key.chmod(0o600)
    result, report = _run(fixture, **{"key-file": None, "dsh-home": None})
    assert result.returncode == 0, report
    assert _calls(fixture)[-1]["home"] == str(
        caller / ".local/state/tennis-lab-agents/dsh-home"
    )
    assert _calls(fixture)[-1]["file_key"] is True


def test_home_lock_prevents_simultaneous_runs(fixture: dict[str, Any]) -> None:
    _behavior(fixture, hang=True)
    first = subprocess.Popen(
        _command(fixture, timeout="20"),
        env=fixture["environment"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_file(fixture["package"] / "ready", first)
        _wait_file(fixture["package"] / "descendants.json", first)
        result, report = _run(fixture)
        assert result.returncode != 0 and report["error_code"] == "DSH_HOME_BUSY"
        assert report["records_dir"] is not None
    finally:
        first.send_signal(signal.SIGTERM)
        first.communicate(timeout=15)


def test_interrupt_during_fresh_config_check_stops_its_process_group(
    fixture: dict[str, Any],
) -> None:
    _behavior(fixture, dump_hang=True)
    process = subprocess.Popen(
        _command(fixture),
        env=fixture["environment"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_file(fixture["package"] / "ready", process)
    descendants = json.loads((fixture["package"] / "descendants.json").read_text())
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=15)
    assert stderr == "" and process.returncode == 143
    report = json.loads(stdout)
    assert report["result"] == "interrupted"
    assert report["usage_missing"] is True
    assert all(not _live(pid) for pid in descendants)
    assert not (fixture["package"] / "stdin.bin").exists()
    assert (_records(report) / "metadata.json").is_file()


@pytest.mark.parametrize(
    "failure",
    [
        "missing-final",
        "missing-newline",
        "missing-step-end",
        "step-order",
        "other-turn",
        "session-cwd",
        "duplicate-session",
        "reason-shape",
        "no-turn-end",
    ],
)
def test_malformed_or_incomplete_intervals_cannot_pass(
    fixture: dict[str, Any], failure: str
) -> None:
    events: list[dict[str, Any]] = [
        {"type": "session", "sessionId": "__sid__", "cwd": "__cwd__"},
        {"type": "status", "phase": "turn_start", "turn": 1},
        {"type": "status", "phase": "step_start", "turn": 1, "step": 1},
        {
            "type": "status",
            "phase": "step_end",
            "turn": 1,
            "step": 1,
            "usage": {
                "inputTokens": 5,
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "outputTokens": 1,
            },
        },
        {
            "type": "status",
            "phase": "turn_end",
            "turn": 1,
            "reason": {"kind": "completed"},
        },
        {"type": "final", "text": "answer"},
    ]
    if failure == "missing-final":
        events.pop()
    elif failure == "missing-step-end":
        events.pop(3)
    elif failure == "step-order":
        events[2], events[3] = events[3], events[2]
    elif failure == "other-turn":
        events[3]["turn"] = 7
    elif failure == "session-cwd":
        events[0]["cwd"] = "/wrong"
    elif failure == "duplicate-session":
        events.insert(1, events[0])
    elif failure == "reason-shape":
        events[4]["reason"] = {"kind": ["completed"]}
    elif failure == "no-turn-end":
        events.pop(4)
    if failure == "missing-newline":
        _behavior(
            fixture,
            raw="\n".join(
                json.dumps(event)
                .replace("__sid__", "session-fixture-1")
                .replace("__cwd__", str(fixture["worktree"]))
                for event in events
            ),
        )
    else:
        _behavior(fixture, events=events)
    result, report = _run(fixture)
    assert result.returncode == 11, report
    assert report["result"] == "invalid_ndjson"
    assert report["usage_missing"] is True
