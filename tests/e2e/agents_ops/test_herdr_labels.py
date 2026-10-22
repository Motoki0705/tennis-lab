"""Exercise visible labels through fake Herdr/GitHub processes, never the host UI."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
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
elif args == ['session','list','--json']:
    result = {'sessions':[{'name':'default','default':True,'running':True,'socket_path':'/fake/herdr.sock'}]}
elif args[:2] not in (['pane','get'], ['pane','current']):
    raise SystemExit('unexpected Herdr command')
path.write_text(json.dumps(state))
if os.environ.get('LABEL_WAITING_MARKER') and args[:2] == ['pane','process-info']:
    Path(os.environ['LABEL_WAITING_MARKER']).touch()
if result is not None: print(json.dumps({'result':result}))
"""

FAKE_GH = r"""
import json,os,sys,time
from pathlib import Path
if os.environ.get('LABEL_GH_ENTERED'):
    Path(os.environ['LABEL_GH_ENTERED']).touch()
    deadline=time.monotonic()+10
    while not Path(os.environ['LABEL_GH_RELEASE']).exists():
        if time.monotonic()>deadline: raise SystemExit('test release timed out')
        time.sleep(.01)
if os.environ.get('LABEL_PREPARE_RACE'):
    p=Path(os.environ['LABEL_FAKE_STATE']);data=json.loads(p.read_text())
    if os.environ['LABEL_PREPARE_RACE']=='process': data['pid']=os.getppid()
    else: data['panes']['w1:p1'].update(display_agent='#99 · 親',tokens={'tl_owner':'new-owner'})
    p.write_text(json.dumps(data))
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

    def hook(
        self, event: str, identity: str = "test-owner", **fields: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(OPS / "herdr_labels/claude_hook.py")],
            input=json.dumps(
                {"hook_event_name": event, "session_id": identity, **fields}
            ),
            text=True,
            capture_output=True,
            env=self.env,
            timeout=20,
            check=False,
        )


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
        HERDR_SOCKET_PATH="/fake/herdr.sock",
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
    # Model a parent whose pane was confirmed during register/initial migration.
    owners = tmp_path / "state/tennis-lab-agents/herdr-labels/owners"
    owners.mkdir(parents=True)
    (owners / (hashlib.sha256(b"claude:test-owner").hexdigest() + ".json")).write_text(
        json.dumps(
            {
                "version": 1,
                "scope": hashlib.sha256(b"/fake/herdr.sock").hexdigest(),
                "terminal_id": "term-a",
            }
        )
    )
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


@pytest.mark.parametrize("identity", ["test-owner", "new-session"])
@pytest.mark.parametrize("end_first", [False, True])
def test_clear_start_never_restores_registered_parent(
    labels: Labels, identity: str, end_first: bool
) -> None:
    assert labels.run("sync").returncode == 0
    original = labels.registry.read_bytes()
    if end_first:
        assert labels.hook("SessionEnd", reason="clear").returncode == 0
    result = labels.hook("SessionStart", identity, source="clear")
    assert result.returncode == 0, result.stderr
    pane = labels.read()["panes"]["w1:p1"]
    assert "display_agent" not in pane
    assert pane["tokens"] == {"other": "keep"}
    assert labels.registry.read_bytes() == original


@pytest.mark.parametrize("display", ["#42 · 関連", "親未登録"])
@pytest.mark.parametrize("event", ["clear", "SessionEnd", "SessionStart"])
def test_clear_removes_initial_migration_label(
    labels: Labels, display: str, event: str
) -> None:
    state = labels.read()
    pane = state["panes"]["w1:p1"]
    pane["display_agent"] = display
    pane["tokens"].update(tl_topic="初回設定", tl_engine="claude")
    if display == "#42 · 関連":
        pane["tokens"]["tl_parent"] = "o/r#42"
    # The actual parent may share this Claude ID in another pane.
    state["panes"]["w1:p2"].update(
        agent="claude",
        display_agent="#42 · 親",
        tokens={"tl_owner": hashlib.sha256(b"claude:test-owner").hexdigest()},
    )
    labels.write(state)
    original_registry = labels.registry.read_bytes()
    original_parent = state["panes"]["w1:p2"]
    result = (
        labels.run("clear")
        if event == "clear"
        else labels.hook(
            event, **{"source" if event == "SessionStart" else "reason": "clear"}
        )
    )
    assert result.returncode == 0, result.stderr
    after = labels.read()["panes"]
    assert "display_agent" not in after["w1:p1"]
    assert after["w1:p1"]["tokens"] == {"other": "keep"}
    assert after["w1:p1"]["agent"] == "claude"
    assert after["w1:p2"] == original_parent
    assert labels.registry.read_bytes() == original_registry


def test_clear_keeps_display_without_managed_metadata(labels: Labels) -> None:
    state = labels.read()
    state["panes"]["w1:p1"]["display_agent"] = "別の用途の表示"
    labels.write(state)
    result = labels.hook("SessionStart", "new-session", source="clear")
    assert result.returncode == 0, result.stderr
    assert labels.read()["panes"]["w1:p1"] == state["panes"]["w1:p1"]
    assert not labels.mutations()


@pytest.mark.parametrize("display", ["#42 · 親", "#42 · 関連", "親未登録"])
def test_compaction_preserves_display_and_runtime(labels: Labels, display: str) -> None:
    assert labels.run("sync", "--topic", "手動で選んだ題名").returncode == 0
    state = labels.read()
    state["panes"]["w1:p1"]["display_agent"] = display
    if display != "#42 · 親":
        state["panes"]["w1:p1"]["tokens"].pop("tl_owner")
        labels.set_registry({})
    labels.write(state)
    private = Path(labels.env["XDG_STATE_HOME"])
    records = {p: p.read_bytes() for p in private.rglob("*.json")}
    result = labels.hook("SessionStart", source="compact")
    assert result.returncode == 0, result.stderr
    assert labels.read() == state  # No Herdr calls, even read-only lookups.
    assert {p: p.read_bytes() for p in private.rglob("*.json")} == records
    assert json.loads(result.stdout)["herdr_label"]["status"] == "preserved"


@pytest.mark.parametrize("ending", ["clear", "SessionEnd"])
def test_clearing_parent_display_does_not_authorize_helper(
    labels: Labels, ending: str
) -> None:
    assert labels.run("sync").returncode == 0
    result = labels.run("clear") if ending == "clear" else labels.hook(ending)
    assert result.returncode == 0, result.stderr
    state = labels.read()
    state["panes"]["w1:p2"].update(agent="claude", display_agent="#42 · 関連")
    labels.write(state)
    helper_env = {**labels.env, "HERDR_PANE_ID": "w1:p2"}
    result = labels.run("sync", env=helper_env)
    assert result.returncode == 2
    assert "confirmed parent display" in result.stderr
    assert labels.read()["panes"]["w1:p2"]["display_agent"] == "#42 · 関連"


def test_unconfirmed_destination_requires_explicit_binding(labels: Labels) -> None:
    owners = labels.registry.parents[1] / "herdr-labels/owners"
    next(owners.glob("*.json")).unlink()
    assert labels.run("sync").returncode == 2
    assert labels.hook("SessionStart").returncode == 1
    assert not labels.mutations()
    bound = labels.run(
        "bind", "--herdr-session", "default", "--pane", "w1:p1", "--terminal", "term-a"
    )
    assert bound.returncode == 0, bound.stderr
    assert labels.run("sync").returncode == 0


def test_end_fences_old_commands_and_start_restores_same_parent(labels: Labels) -> None:
    assert labels.run("sync").returncode == 0
    assert labels.hook("SessionEnd").returncode == 0
    count = len(labels.mutations())
    assert labels.run("sync").returncode == 2
    assert len(labels.mutations()) == count
    assert labels.hook("SessionStart", source="resume").returncode == 0
    assert labels.read()["panes"]["w1:p1"]["display_agent"] == "#42 · 親"


def test_new_session_fences_late_commands_from_predecessor(labels: Labels) -> None:
    assert labels.run("sync").returncode == 0
    assert labels.hook("SessionStart", "new-session").returncode == 0
    count = len(labels.mutations())
    assert labels.run("sync").returncode == 2
    assert labels.hook("SessionEnd").returncode == 1
    assert len(labels.mutations()) == count
    assert "display_agent" not in labels.read()["panes"]["w1:p1"]


def test_lifecycle_hook_cannot_claim_an_unrelated_agent_process(labels: Labels) -> None:
    with subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(20)"]
    ) as sibling:
        try:
            state = labels.read()
            state["pid"] = sibling.pid
            labels.write(state)
            result = labels.hook("SessionStart")
            assert result.returncode == 1
            assert "not a descendant" in result.stderr
            assert not labels.mutations()
        finally:
            sibling.terminate()
            sibling.wait(timeout=5)


def test_headless_claude_child_cannot_clear_its_parent_pane(labels: Labels) -> None:
    assert labels.run("sync").returncode == 0
    count = len(labels.mutations())
    # A second Claude below the interactive agent owns this hook, despite
    # inheriting the parent's Herdr environment and being its descendant.
    script = """
import ctypes,json,subprocess,sys
ctypes.CDLL(None).prctl(15,b'claude',0,0,0)
r=subprocess.run([sys.executable,sys.argv[1]],input=json.dumps({'hook_event_name':'SessionStart','session_id':'headless-child'}),text=True)
raise SystemExit(r.returncode)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(OPS / "herdr_labels/claude_hook.py")],
        env=labels.env,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 1
    assert "nested agent" in result.stderr
    assert len(labels.mutations()) == count
    assert labels.read()["panes"]["w1:p1"]["display_agent"] == "#42 · 親"


@pytest.mark.parametrize("race", ["process", "metadata"])
def test_replacement_during_title_lookup_rejects_old_write(
    labels: Labels, race: str
) -> None:
    result = labels.run("sync", env={**labels.env, "LABEL_PREPARE_RACE": race})
    assert result.returncode == 2, result.stdout
    assert not labels.mutations()


def wait_file(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"test barrier timed out: {path.name}")
        time.sleep(0.01)


def test_session_start_is_serialized_after_inflight_old_sync(labels: Labels) -> None:
    entered, release, waiting = [
        labels.root / n for n in ("entered", "release", "waiting")
    ]
    old_env = {
        **labels.env,
        "LABEL_GH_ENTERED": str(entered),
        "LABEL_GH_RELEASE": str(release),
    }
    old = subprocess.Popen(
        [sys.executable, "-m", "herdr_labels", "sync"],
        env=old_env,
        cwd=labels.root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    new = None
    try:
        wait_file(entered)
        new = subprocess.Popen(
            [sys.executable, str(OPS / "herdr_labels/claude_hook.py")],
            env={**labels.env, "LABEL_WAITING_MARKER": str(waiting)},
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert new.stdin is not None
        new.stdin.write(
            json.dumps({"hook_event_name": "SessionStart", "session_id": "new-session"})
        )
        new.stdin.close()
        new.stdin = None
        wait_file(waiting)
        assert new.poll() is None
        release.touch()
        old_out = old.communicate(timeout=10)
        new_out = new.communicate(timeout=10)
        assert old.returncode == 0, old_out
        assert new.returncode == 0, new_out
        assert "display_agent" not in labels.read()["panes"]["w1:p1"]
        assert labels.run("sync").returncode == 2
    finally:
        release.touch()
        for proc in (old, new):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


def test_parent_fixture_blocks_inherited_host_herdr(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    marker = tmp_path / "host-herdr-called"
    host_bin = tmp_path / "host-bin"
    host_bin.mkdir()
    trap = host_bin / "herdr"
    trap.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\nraise SystemExit(98)\n"
    )
    trap.chmod(0o755)
    monkeypatch.setenv("PATH", str(host_bin) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("HERDR_PANE_ID", "host-pane")
    monkeypatch.setenv("HERDR_SOCKET_PATH", "/host/socket")
    parent = request.getfixturevalue("parent_env")
    assert not any(k.startswith("HERDR_") for k in parent.env)
    result = parent.cli("register")
    assert json.loads(result.stdout)["display_sync"]["status"] == "not_configured"
    failed = parent.cli(
        "register", ok=None, extra={"HERDR_ENV": "1", "HERDR_PANE_ID": "host-pane"}
    )
    assert failed.returncode == 1
    assert "exit 99" in failed.stdout
    assert not marker.exists()


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
