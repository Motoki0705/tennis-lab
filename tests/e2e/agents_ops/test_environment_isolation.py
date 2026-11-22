"""Replay real ops CLI tests below a fake interactive Claude/Herdr shell."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.support.agents_ops.hook_process import hook_command

ROOT = Path(__file__).resolve().parents[3]


def test_ops_cli_tests_ignore_host_herdr_and_agent_ancestors(tmp_path: Path) -> None:
    calls = tmp_path / "host-herdr-calls"
    bin_dir = tmp_path / "host-bin"
    bin_dir.mkdir()
    herdr = bin_dir / "herdr"
    herdr.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        f"with Path({str(calls)!r}).open('a') as stream: stream.write('called\\n')\n"
        "raise SystemExit(98)\n"
    )
    herdr.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "HERDR_ENV": "1",
        "HERDR_PANE_ID": "host-pane",
        "HERDR_TAB_ID": "host-tab",
        "HERDR_WORKSPACE_ID": "host-workspace",
        "HERDR_SOCKET_PATH": str(tmp_path / "host.sock"),
        "HERDR_BIN_PATH": str(herdr),
        "HERDR_FUTURE_VARIABLE": "must-also-be-removed",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    }
    # Keep this actual named process above nested pytest, outside its boundary.
    # exec would reset the process name and lose the regression's extra ancestor.
    script = """
import ctypes, subprocess, sys
from pathlib import Path
assert ctypes.CDLL(None).prctl(15, b'claude', 0, 0, 0) == 0
assert Path('/proc/self/comm').read_text().strip() == 'claude'
raise SystemExit(subprocess.run(sys.argv[1:], check=False).returncode)
"""
    nodes = [
        "test_ops_report_triage.py::test_adoption_receipt_and_agreement_are_three_distinct_states",
        "test_board_doorbell.py::test_headless_root_hook_uses_native_ancestor_and_rejects_unowned_process",
        "test_herdr_labels.py::test_lifecycle_hook_cannot_claim_an_unrelated_agent_process",
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-q",
            "--basetemp",
            str(tmp_path / "nested-pytest"),
            *["tests/e2e/agents_ops/" + node for node in nodes],
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not calls.exists(), calls.read_text()


def test_hook_launcher_rejects_unreachable_boundary(tmp_path: Path) -> None:
    marker = tmp_path / "hook-called"
    hook = tmp_path / "hook.py"
    hook.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    command = hook_command(hook)
    command[2] = "0"  # No real ancestor can have PID 0.
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=5
    )
    assert result.returncode != 0
    assert "hook isolation boundary must be an ancestor" in result.stderr
    assert not marker.exists()
