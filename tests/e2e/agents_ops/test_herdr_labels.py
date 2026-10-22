"""Exercise visible labels through fake Herdr/GitHub processes, never the host UI."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"

FAKE_HERDR = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
path = Path(os.environ['LABEL_FAKE_STATE'])
state = json.loads(path.read_text())
state['calls'].append(args)
if args[:1] == ['--session']: args = args[2:]
pane_id = os.environ.get('HERDR_PANE_ID', 'w1:p1')
if args[:2] == ['pane', 'get']: pane_id = args[2]
if args[:2] == ['pane', 'report-metadata']: pane_id = args[2]
if args[:2] == ['pane', 'process-info']: pane_id = args[args.index('--pane')+1]
pane = state['panes'][pane_id]
result = {'pane': pane}
if args[:2] == ['pane', 'report-metadata']:
    if state.get('reject'):
        path.write_text(json.dumps(state)); sys.exit(1)
    if not state.get('drop_write'):
        if '--display-agent' in args: pane['display_agent'] = args[args.index('--display-agent')+1]
        if '--clear-display-agent' in args: pane.pop('display_agent', None)
        tokens = pane.setdefault('tokens', {})
        for i,a in enumerate(args):
            if a == '--token':
                k,v = args[i+1].split('=',1); tokens[k] = v
            if a == '--clear-token': tokens.pop(args[i+1],None)
    result = None
elif args[:2] == ['pane', 'process-info']:
    result = {'process_info':{'foreground_processes':[{'pid':state['pid'], 'name':pane['agent']}]}}
elif args == ['agent', 'list']:
    result = {'agents':list(state['panes'].values())}
elif args[:2] not in (['pane','get'], ['pane','current']):
    raise SystemExit('unexpected Herdr command')
path.write_text(json.dumps(state))
if result is not None: print(json.dumps({'result':result}))
"""

FAKE_GH = r"""
import json,os,sys
from pathlib import Path
if os.environ.get('LABEL_REGISTRY_RACE'):
    p=Path(os.environ['XDG_STATE_HOME'])/'tennis-lab-agents/parent-sessions/registry.json'
    data=json.loads(p.read_text());data['parents']['o/r#42']['status']='ended';p.write_text(json.dumps(data))
if '--include' in sys.argv: print('HTTP/1.1 200 OK\n')
print(json.dumps({'number':42,'repository_url':'https://api.github.com/repos/o/r','title':'[表示設定] 親担当の表示','state':'open'}))
"""


@dataclass
class Labels:
    root: Path
    env: dict[str, str]
    state: Path
    registry: Path

    def read(self) -> dict[str, Any]:
        return json.loads(self.state.read_text())  # type: ignore[no-any-return]

    def write(self, state: dict[str, Any]) -> None:
        self.state.write_text(json.dumps(state))

    def set_registry(self, parents: dict[str, Any]) -> None:
        self.registry.parent.mkdir(parents=True, exist_ok=True)
        self.registry.write_text(json.dumps({"version": 1, "parents": parents}))

    def owner(
        self, status: str = "active", identity: str = "claude:test-owner"
    ) -> dict[str, Any]:
        return {
            "session": identity,
            "status": status,
            "generation": 1,
            "tasks": {},
            "reviews": [],
        }

    def run(
        self, *args: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "herdr_labels", *args],
            env=env or self.env,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def mutations(self) -> list[list[str]]:
        return [x for x in self.read()["calls"] if "report-metadata" in x]


@pytest.fixture
def labels(tmp_path: Path) -> Labels:
    tmp_path = tmp_path / "labels"
    tmp_path.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, script in [("herdr", FAKE_HERDR), ("gh", FAKE_GH)]:
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n{script}")
        path.chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HERDR_", "CODEX_", "CLAUDE_"))
    }
    state = tmp_path / "herdr.json"
    env.update(
        PATH=str(bin_dir) + os.pathsep + env["PATH"],
        PYTHONPATH=str(OPS),
        XDG_STATE_HOME=str(tmp_path / "state"),
        CLAUDE_CODE_SESSION_ID="test-owner",
        HERDR_ENV="1",
        HERDR_PANE_ID="w1:p1",
        LABEL_FAKE_STATE=str(state),
    )
    value = Labels(
        tmp_path,
        env,
        state,
        tmp_path / "state/tennis-lab-agents/parent-sessions/registry.json",
    )
    value.write(
        {
            "calls": [],
            "pid": os.getpid(),
            "panes": {
                "w1:p1": {
                    "pane_id": "w1:p1",
                    "terminal_id": "term-a",
                    "agent": "claude",
                    "tokens": {"other": "keep"},
                },
                "w1:p2": {
                    "pane_id": "w1:p2",
                    "terminal_id": "term-b",
                    "agent": "codex",
                    "focused": True,
                },
            },
        }
    )
    value.set_registry({"o/r#42": value.owner()})
    return value


def test_sync_uses_caller_not_focus_and_is_idempotent(labels: Labels) -> None:
    result = labels.run("sync")
    assert result.returncode == 0, result.stderr
    state = labels.read()
    pane = state["panes"]["w1:p1"]
    assert pane["display_agent"] == "#42 · 親"
    assert pane["tokens"]["tl_topic"] == "表示設定"
    assert pane["tokens"]["tl_engine"] == "claude"
    assert pane["tokens"]["other"] == "keep"
    assert "display_agent" not in state["panes"]["w1:p2"]
    assert "test-owner" not in result.stdout
    assert labels.run("sync").returncode == 0
    assert len(labels.mutations()) == 1


def test_custom_topic_persists_and_cjk_is_shortened(labels: Labels) -> None:
    result = labels.run(
        "sync", "--parent", "42", "--topic", "日本語表示幅確認用の非常に長い題名"
    )
    assert result.returncode == 0, result.stderr
    topic = json.loads(result.stdout)["topic"]
    assert topic.endswith("…") and len(topic) <= 11
    assert labels.run("sync").returncode == 0
    assert labels.read()["panes"]["w1:p1"]["tokens"]["tl_topic"] == topic


@pytest.mark.parametrize("status", ["ended", "released", "handoff_pending"])
def test_ended_or_handoff_clears_only_owned_metadata(
    labels: Labels, status: str
) -> None:
    assert labels.run("sync").returncode == 0
    labels.set_registry({"o/r#42": labels.owner(status)})
    result = labels.run("sync")
    assert result.returncode == 0, result.stderr
    pane = labels.read()["panes"]["w1:p1"]
    assert "display_agent" not in pane
    assert pane["tokens"] == {"other": "keep"}
    assert labels.run("clear").returncode == 0
    assert len(labels.mutations()) == 2


def test_wrong_parent_or_ambiguous_owner_never_updates(labels: Labels) -> None:
    assert labels.run("sync", "--parent", "99").returncode == 2
    labels.set_registry({"o/r#42": labels.owner(), "o/r#43": labels.owner()})
    assert labels.run("sync").returncode == 2
    assert not labels.mutations()


@pytest.mark.parametrize("event", ["sync", "start"])
def test_duplicate_claude_session_cannot_turn_helper_into_parent(
    labels: Labels, event: str
) -> None:
    state = labels.read()
    state["panes"]["w1:p1"].update(display_agent="#42 · 関連")
    state["panes"]["w1:p2"].update(
        agent="claude",
        display_agent="#42 · 親",
        tokens={"tl_owner": hashlib.sha256(b"claude:test-owner").hexdigest()},
    )
    labels.write(state)
    if event == "sync":
        result = labels.run("sync")
    else:
        result = subprocess.run(
            [sys.executable, str(OPS / "herdr_labels/claude_hook.py")],
            input=json.dumps(
                {"hook_event_name": "SessionStart", "session_id": "test-owner"}
            ),
            text=True,
            capture_output=True,
            env=labels.env,
            timeout=20,
            check=False,
        )
    assert result.returncode != 0
    assert "another live pane" in result.stderr
    assert not labels.mutations()
    assert labels.read()["panes"]["w1:p1"]["display_agent"] == "#42 · 関連"


@pytest.mark.parametrize(
    "updates",
    [
        {"CLAUDE_CODE_SESSION_ID": ""},
        {"CODEX_SESSION_ID": "another-root"},
        {"HERDR_PANE_ID": ""},
        {"HERDR_PANE_ID": "w1:p2"},
    ],
)
def test_invalid_identity_and_context_never_updates(
    labels: Labels, updates: dict[str, str]
) -> None:
    result = labels.run("sync", env={**labels.env, **updates})
    assert result.returncode == 2
    assert not labels.mutations()


def test_no_herdr_context_is_explicit_and_codex_child_is_rejected(
    labels: Labels,
) -> None:
    env = {k: v for k, v in labels.env.items() if not k.startswith("HERDR_")}
    result = labels.run("sync", env=env)
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "not_configured"
    env.pop("CLAUDE_CODE_SESSION_ID")
    env.update(CODEX_SESSION_ID="root", CODEX_THREAD_ID="child")
    assert labels.run("sync", env=env).returncode == 2
    assert not labels.mutations()


def test_bind_is_explicit_and_rejects_a_replaced_terminal(labels: Labels) -> None:
    env = {k: v for k, v in labels.env.items() if not k.startswith("HERDR_")}
    result = labels.run(
        "bind",
        "--herdr-session",
        "default",
        "--pane",
        "w1:p1",
        "--terminal",
        "term-a",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert labels.run("sync", env=env).returncode == 0
    state = labels.read()
    state["panes"]["w1:p1"]["terminal_id"] = "replacement"
    labels.write(state)
    assert labels.run("sync", env=env).returncode == 2
    assert len(labels.mutations()) == 1


def test_bind_rejects_pid_reuse_or_agent_replacement(labels: Labels) -> None:
    env = {k: v for k, v in labels.env.items() if not k.startswith("HERDR_")}
    result = labels.run(
        "bind",
        "--herdr-session",
        "default",
        "--pane",
        "w1:p1",
        "--terminal",
        "term-a",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    state = labels.read()
    state["pid"] = os.getppid()
    labels.write(state)
    assert labels.run("sync", env=env).returncode == 2
    assert not labels.mutations()


@pytest.mark.parametrize("failure", ["drop_write", "reject"])
def test_failure_or_readback_mismatch_is_not_success(
    labels: Labels, failure: str
) -> None:
    state = labels.read()
    state[failure] = True
    labels.write(state)
    assert labels.run("sync").returncode == 2


def test_registry_change_during_title_lookup_is_rejected(labels: Labels) -> None:
    result = labels.run("sync", env={**labels.env, "LABEL_REGISTRY_RACE": "1"})
    assert result.returncode == 2
    assert not labels.mutations()


@pytest.mark.parametrize("command", ["sync", "clear"])
def test_ordinary_commands_refuse_another_sessions_label(
    labels: Labels, command: str
) -> None:
    state = labels.read()
    state["panes"]["w1:p1"]["tokens"]["tl_owner"] = "someone-else"
    labels.write(state)
    assert labels.run(command).returncode == 2
    assert not labels.mutations()


@pytest.mark.parametrize("assigned", [False, True])
def test_claude_start_replaces_stale_predecessor_in_current_pane(
    labels: Labels, assigned: bool
) -> None:
    state = labels.read()
    pane = state["panes"]["w1:p1"]
    pane["display_agent"] = "#99 · 親"
    pane["tokens"].update(tl_owner="old-session", tl_parent="o/r#99", tl_topic="old")
    labels.write(state)
    if not assigned:
        labels.set_registry({})
    before = labels.registry.read_bytes()
    result = subprocess.run(
        [sys.executable, str(OPS / "herdr_labels/claude_hook.py")],
        input=json.dumps(
            {"hook_event_name": "SessionStart", "session_id": "test-owner"}
        ),
        text=True,
        capture_output=True,
        env=labels.env,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    after = labels.read()["panes"]["w1:p1"]
    assert after.get("display_agent") == ("#42 · 親" if assigned else None)
    assert after["tokens"]["other"] == "keep"
    assert labels.registry.read_bytes() == before
    assert "display_agent" not in labels.read()["panes"]["w1:p2"]


def test_claude_hooks_use_real_payload_and_never_release_registry(
    labels: Labels,
) -> None:
    export = labels.root / "env.sh"
    original = labels.registry.read_bytes()
    env = {k: v for k, v in labels.env.items() if k != "CLAUDE_CODE_SESSION_ID"}
    env["CLAUDE_ENV_FILE"] = str(export)
    for event in ("SessionStart", "SessionEnd"):
        result = subprocess.run(
            [sys.executable, str(OPS / "herdr_labels/claude_hook.py")],
            input=json.dumps({"hook_event_name": event, "session_id": "test-owner"}),
            text=True,
            capture_output=True,
            env=env,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    assert "export CLAUDE_CODE_SESSION_ID=test-owner" in export.read_text()
    assert labels.registry.read_bytes() == original
    assert "display_agent" not in labels.read()["panes"]["w1:p1"]


def test_parent_success_and_display_failure_are_distinguished(
    parent_env: ParentEnv,
    labels: Labels,
) -> None:
    # Reuse the real parent CLI's fake GitHub, with a separate fake Herdr executable.
    target = Path(parent_env.env["PATH"].split(os.pathsep)[0]) / "herdr"
    target.write_text(f"#!{sys.executable}\n{FAKE_HERDR}")
    target.chmod(0o755)
    extra = {
        "HERDR_ENV": "1",
        "HERDR_PANE_ID": "w1:p1",
        "LABEL_FAKE_STATE": str(labels.state),
    }
    state = labels.read()
    state["reject"] = True
    labels.write(state)
    result = parent_env.cli("register", ok=None, extra=extra)
    assert result.returncode == 1, result.stderr
    response = json.loads(result.stdout)
    assert response["assignment"] == "active"
    assert response["parent_action_applied"] is True
    assert response["display_sync"]["status"] == "error"
    state = labels.read()
    state["reject"] = False
    labels.write(state)
    env = {**parent_env.env, **extra}
    result = labels.run("sync", env=env)
    assert result.returncode == 0, result.stderr
    expected_owner = hashlib.sha256(b"claude:private-session-a-unique").hexdigest()
    assert labels.read()["panes"]["w1:p1"]["tokens"]["tl_owner"] == expected_owner
