"""Inspect real pinned session files locally, without booting dsh or an API."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "tools/dsh"
NODE = shutil.which("node")

CREATE_SESSION = """
import {join} from 'node:path';
import {Context} from '@deepseek-ai/cordis';
import JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl';
import {SessionId} from '@deepseek-ai/dsh-session';
const [home, cwd, mode] = process.argv.slice(1);
const context = new Context();
const persistence = new JsonlSessionPersistence(context, {root:join(home,'sessions'),compression:'zstd'});
const id = SessionId('session-native-fixture');
const handle = await persistence.create({version:4,id,createdAt:Date.now(),cwd,isSeeded:false,delegationDepth:0});
const events = [
  {type:'permission/preset',data:{preset:mode}},
  {type:'sandbox/mode',data:{mode}},
  {type:'approval/policy',data:{policy:'ask'}},
  {type:'turn/start',data:{turn:1}},
  {type:'turn/end',data:{turn:1,reason:{kind:'completed'}}},
].map((event,seq)=>({...event,seq,time:Date.now()}));
await handle.append(events);
await handle.flush();
await handle.close();
"""


@pytest.mark.parametrize("mode", ["read-only", "workspace-write"])
def test_pinned_backend_inspection_replays_real_zstd_log_without_writing(
    tmp_path: Path, mode: str
) -> None:
    if NODE is None or not (SOURCE / "node_modules/@deepseek-ai/dsh-session").is_dir():
        pytest.skip("Pinned runtime is not installed; fake runner tests still run")
    assert NODE is not None
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    environment = {
        "PATH": str(Path(NODE).parent) + os.pathsep + os.defpath,
        "HOME": str(home),
        "TMPDIR": str(tmp_path),
    }
    created = subprocess.run(
        [
            NODE,
            "--input-type=module",
            "--eval",
            CREATE_SESSION,
            str(home),
            str(tmp_path),
            mode,
        ],
        cwd=SOURCE,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert created.returncode == 0, created.stderr
    before = {
        str(path.relative_to(home)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in home.rglob("*")
        if path.is_file()
    }
    result = subprocess.run(
        [NODE, str(SOURCE / "session_state.mjs"), str(home), "session-native-fixture"],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "session_id": "session-native-fixture",
        "cwd": str(tmp_path),
        "mode": mode,
        "approval": "ask",
        "preset": mode,
        "last_turn": 1,
    }
    after = {
        str(path.relative_to(home)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in home.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_persistent_pinned_profile_is_inspected_on_each_run_without_api(
    tmp_path: Path,
) -> None:
    if (
        NODE is None
        or not (SOURCE / "node_modules/@deepseek-ai/dsh/lib/bin.js").is_file()
    ):
        pytest.skip("Pinned runtime is not installed")
    assert NODE is not None
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    environment = {
        "PATH": str(Path(NODE).parent) + os.pathsep + os.defpath,
        "HOME": str(home),
        "TMPDIR": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    script = """
import os, shutil, sys
from pathlib import Path
from tools.dsh import headless_config as config
from tools.dsh.runner import inspect_home_config
from tools.dsh.run_process import Interrupt
from tools.dsh.home_policy import remember_home, verify_home
os.umask(0o077)
home = Path(sys.argv[1])
environment = config.headless_environment('workspace-write', home, dict(os.environ))
command = [shutil.which('node'), str(config.RUNTIME), 'exec', '--', *config.headless_arguments('workspace-write')]
for _ in range(2):
 verify_home(home)
 inspect_home_config(command,worktree=home,environment=environment,mode='workspace-write',interrupt=Interrupt())
 remember_home(home)
print('profile_checked_twice')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(home)],
        cwd=SOURCE.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "profile_checked_twice"
