"""Exercise the pinned runtime with a local CLI fixture, without npm/API access."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "tools/dsh"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

# Exercise the production entrypoint at supported/unsupported Node boundaries
# without requiring a separate Node installation for each case. The fake dsh
# needs only basic Node APIs and never imports the real dsh or accesses a network.
LAUNCH = """
import { pathToFileURL } from 'node:url';
const [entry, version, nodeExecutable, ...args] = process.argv.slice(1);
Object.defineProperty(process.versions, 'node', { value: version });
if (nodeExecutable) Object.defineProperty(process, 'execPath', { value: nodeExecutable });
process.argv = [process.execPath, entry, ...args];
await import(pathToFileURL(entry).href);
"""

FAKE_CLI = """
import { appendFileSync, readdirSync, readFileSync } from 'node:fs';
const args = process.argv.slice(2);
const call = {
  args, cwd: process.cwd(), executable: process.argv[1], node: process.execPath,
  home: process.env.DSH_HOME, telemetry: process.env.DSH_TELEMETRY_DISABLED,
  api_key_present: Object.hasOwn(process.env, 'DEEPSEEK_API_KEY'),
};
appendFileSync(process.env.FAKE_CALLS, JSON.stringify(call) + '\\n');
if (args.length === 1 && args[0] === '-V') {
  if (call.api_key_present || call.telemetry !== '1' || call.cwd !== call.home ||
      readdirSync(call.home).length !== 0) process.exit(91);
  console.log(process.env.FAKE_DSH_VERSION || '0.2.0-rc.2');
  process.exit(Number(process.env.FAKE_VERSION_EXIT || 0));
}
if (args[0] === 'wait-for-signal') {
  setInterval(() => {}, 1000);
} else {
  console.error('fixture diagnostic');
  console.log(JSON.stringify({ ...call, stdin: readFileSync(0, 'utf8') }));
  process.exit(Number(process.env.FAKE_DSH_EXIT || 0));
}
"""


@pytest.fixture
def runtime(tmp_path: Path) -> Path:
    """A copied public entrypoint with no dependency on the real installation."""
    directory = tmp_path / "owner worktree" / "tools" / "dsh"
    directory.mkdir(parents=True)
    for name in ("runtime.mjs", "package.json"):
        shutil.copy2(RUNTIME / name, directory / name)
    package = directory / "node_modules/@deepseek-ai/dsh"
    (package / "lib").mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps(
            {
                "name": "@deepseek-ai/dsh",
                "version": "0.2.0-rc.2",
                "type": "module",
                "bin": {"dsh": "lib/bin.js"},
            }
        ),
        encoding="utf-8",
    )
    (package / "lib/bin.js").write_text(FAKE_CLI, encoding="utf-8")
    return directory / "runtime.mjs"


@pytest.fixture
def cli_env(tmp_path: Path) -> dict[str, str]:
    """Only dummy credentials enter a subprocess; no host secrets are copied."""
    home = tmp_path / "caller home"
    home.mkdir()
    return {
        "PATH": os.defpath,
        "HOME": str(home),
        "DSH_HOME": str(home),
        "DSH_TELEMETRY_DISABLED": "0",
        "DEEPSEEK_API_KEY": "fixture-only-sentinel",
        "FAKE_CALLS": str(tmp_path / "calls.ndjson"),
    }


def _command(
    runtime: Path,
    *args: str,
    version: str = "24.0.0",
    node_executable: str = "",
) -> list[str]:
    assert NODE is not None
    return [
        NODE, "--input-type=module", "--eval", LAUNCH,
        str(runtime), version, node_executable, *args,
    ]  # fmt: skip


def _run(
    runtime: Path,
    env: dict[str, str],
    *args: str,
    version: str = "24.0.0",
    cwd: Path | None = None,
    stdin: str = "",
    node_executable: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(runtime, *args, version=version, node_executable=node_executable),
        env=env,
        cwd=cwd or runtime.parent,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def _calls(env: dict[str, str]) -> list[dict[str, object]]:
    path = Path(env["FAKE_CALLS"])
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("18.20.0", False),
        ("20.19.0", False),
        ("22.18.99", False),
        ("22.19.0", True),
        ("22.19.1", True),
        ("22.20.0", True),
        ("22.99.99", True),
        ("23.0.0", False),
        ("23.99.0", False),
        ("24.0.0", True),
        ("24.14.1", True),
        ("25.0.0", True),
        ("100.0.0", True),
        ("24.0.0-rc.1", False),
        ("v24.0.0", False),
    ],
)
def test_supported_node_range_stops_before_launch(
    runtime: Path, cli_env: dict[str, str], version: str, supported: bool
) -> None:
    result = _run(runtime, cli_env, "check", version=version)

    if supported:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["node_version"] == version
    else:
        assert result.returncode == 1
        assert f"Unsupported Node.js {version}" in result.stderr
        assert "^22.19.0 || >=24.0.0" in result.stderr
        assert not Path(cli_env["FAKE_CALLS"]).exists()


def test_check_reports_local_real_paths_and_isolates_credentials(
    runtime: Path, cli_env: dict[str, str], tmp_path: Path
) -> None:
    other_worktree = tmp_path / "other worktree"
    other_worktree.mkdir()
    result = _run(runtime, cli_env, "check", cwd=other_worktree)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    # PATH may expose a version-manager wrapper; compare the binary that the
    # selected CLI actually used rather than the wrapper's filesystem path.
    assert report["node_executable"] == str(Path(str(_calls(cli_env)[0]["node"])).resolve())
    expected = runtime.parent / "node_modules/@deepseek-ai/dsh/lib/bin.js"
    assert report["dsh_executable"] == str(expected.resolve())
    assert report["dsh_version"] == "0.2.0-rc.2"
    assert len(_calls(cli_env)) == 1
    call = _calls(cli_env)[0]
    assert call["api_key_present"] is False
    assert call["telemetry"] == "1"
    assert call["cwd"] == call["home"]
    assert call["home"] != cli_env["DSH_HOME"]
    assert not Path(str(call["home"])).exists()


def test_missing_local_install_never_uses_global_or_parent_dependency(
    runtime: Path, cli_env: dict[str, str], tmp_path: Path
) -> None:
    package = runtime.parent / "node_modules/@deepseek-ai/dsh"
    parent_package = runtime.parent.parent / "node_modules/@deepseek-ai/dsh"
    parent_package.parent.mkdir(parents=True)
    shutil.move(str(package), str(parent_package))
    fake_bin = tmp_path / "global bin"
    fake_bin.mkdir()
    marker = tmp_path / "global-was-run"
    global_cli = fake_bin / "dsh"
    global_cli.write_text('#!/bin/sh\ntouch "$GLOBAL_MARKER"\n', encoding="utf-8")
    global_cli.chmod(0o755)
    cli_env.update(PATH=f"{fake_bin}:{os.defpath}", GLOBAL_MARKER=str(marker))

    result = _run(runtime, cli_env, "exec", "--", "-V")

    assert result.returncode == 1
    assert "Cannot read the local dsh installation" in result.stderr
    assert "install" in result.stderr
    assert not marker.exists()
    assert not Path(cli_env["FAKE_CALLS"]).exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"version": "0.2.1-rc.2"}, "dsh version mismatch"),
        ({"name": "another-package"}, "dsh version mismatch"),
        ({"bin": {"dsh": "../another.js"}}, "Unexpected dsh bin entry"),
    ],
)
def test_bad_local_metadata_fails_before_cli(
    runtime: Path, cli_env: dict[str, str], change: dict[str, object], message: str
) -> None:
    metadata = runtime.parent / "node_modules/@deepseek-ai/dsh/package.json"
    installed = json.loads(metadata.read_text())
    metadata.write_text(json.dumps(installed | change), encoding="utf-8")

    result = _run(runtime, cli_env, "check")

    assert result.returncode == 1
    assert message in result.stderr
    assert not Path(cli_env["FAKE_CALLS"]).exists()


def test_missing_cli_fails_loudly(runtime: Path, cli_env: dict[str, str]) -> None:
    (runtime.parent / "node_modules/@deepseek-ai/dsh/lib/bin.js").unlink()

    result = _run(runtime, cli_env, "check")

    assert result.returncode == 1
    assert "Missing local dsh executable" in result.stderr


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"FAKE_DSH_VERSION": "0.2.1-rc.2"}, "dsh CLI version mismatch"),
        ({"FAKE_VERSION_EXIT": "9"}, "Local dsh -V failed (exit 9)"),
    ],
)
def test_real_cli_version_is_checked_and_failure_does_not_execute_task(
    runtime: Path, cli_env: dict[str, str], changes: dict[str, str], message: str
) -> None:
    cli_env.update(changes)

    result = _run(runtime, cli_env, "exec", "--", "task")

    assert result.returncode == 1
    assert message in result.stderr
    assert [call["args"] for call in _calls(cli_env)] == [["-V"]]
    assert not Path(str(_calls(cli_env)[0]["home"])).exists()


def test_exec_keeps_cwd_arguments_streams_and_environment(
    runtime: Path, cli_env: dict[str, str], tmp_path: Path
) -> None:
    worktree = tmp_path / "target worktree"
    worktree.mkdir()
    args = ["--profile", "headless", "--json", "-", "quotes ' \" and $(literal)"]
    prompt = "task with\nmultiple lines and literal $HOME\n"

    result = _run(runtime, cli_env, "exec", "--", *args, cwd=worktree, stdin=prompt)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)  # no check JSON is mixed into dsh stdout
    assert report["args"] == args
    assert report["cwd"] == str(worktree)
    assert report["stdin"] == prompt
    assert report["home"] == cli_env["DSH_HOME"]
    assert report["api_key_present"] is True
    assert report["node"] == _calls(cli_env)[0]["node"]
    assert result.stderr == "fixture diagnostic\n"


def test_exec_preserves_nonzero_exit_code(runtime: Path, cli_env: dict[str, str]) -> None:
    cli_env["FAKE_DSH_EXIT"] = "7"

    result = _run(runtime, cli_env, "exec", "--", "task")

    assert result.returncode == 7
    assert json.loads(result.stdout)["args"] == ["task"]


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_exec_forwards_signals_and_does_not_leave_cli_running(
    runtime: Path, cli_env: dict[str, str], stop_signal: signal.Signals
) -> None:
    process = subprocess.Popen(
        _command(runtime, "exec", "--", "wait-for-signal"),
        env=cli_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            calls_path = Path(cli_env["FAKE_CALLS"])
            if calls_path.exists() and len(_calls(cli_env)) == 2:
                break
            assert process.poll() is None
            time.sleep(0.02)
        else:
            pytest.fail("fake dsh did not start")
        process.send_signal(stop_signal)
        process.communicate(timeout=5)
        # communicate also waits for the CLI to close the inherited pipes.
        assert process.returncode == -stop_signal
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


@pytest.mark.parametrize("npm_exit", [0, 12])
def test_install_uses_ci_in_runtime_directory_and_checks_only_after_success(
    runtime: Path, cli_env: dict[str, str], tmp_path: Path, npm_exit: int
) -> None:
    fake_bin = tmp_path / "fake node bin"
    fake_bin.mkdir()
    assert NODE is not None
    fake_node = fake_bin / "node"
    fake_node.symlink_to(Path(NODE).resolve())
    fake_npm = fake_bin / "npm"
    fake_npm.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$PWD" "$@" > "$NPM_CALLS"\n'
        'exit "$NPM_EXIT"\n',
        encoding="utf-8",
    )
    fake_npm.chmod(0o755)
    npm_calls = tmp_path / "npm-calls.txt"
    cli_env.update(NPM_CALLS=str(npm_calls), NPM_EXIT=str(npm_exit))

    result = _run(runtime, cli_env, "install", node_executable=str(fake_node))

    assert npm_calls.read_text().splitlines() == [
        str(runtime.parent), "ci", "--no-audit", "--no-fund",
    ]  # fmt: skip
    if npm_exit:
        assert result.returncode == 1
        assert "npm ci failed (exit 12)" in result.stderr
        assert not Path(cli_env["FAKE_CALLS"]).exists()
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["dsh_version"] == "0.2.0-rc.2"


@pytest.mark.parametrize(
    "args", [[], ["exec"], ["exec", "-V"], ["check", "extra"], ["latest"]]
)
def test_bad_arguments_do_not_launch_cli(
    runtime: Path, cli_env: dict[str, str], args: list[str]
) -> None:
    result = _run(runtime, cli_env, *args)

    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert not Path(cli_env["FAKE_CALLS"]).exists()
