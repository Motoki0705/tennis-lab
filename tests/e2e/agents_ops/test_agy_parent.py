"""Manual agy ownership, display lifecycle and deferred board delivery."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_board import run_board
from tests.e2e.agents_ops.test_herdr_labels import Labels
from tests.e2e.agents_ops.test_herdr_labels import labels as labels
from tests.e2e.agents_ops.test_parent_session import OPS, ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.agy_tui import (
    CHILD,
    OTHER,
    ROOT,
    TRAJECTORY,
    Tui,
    foreground,
)
from tests.support.agents_ops.hook_process import hook_command

HOOK = OPS / "herdr_labels/agy_hook.py"
LAUNCH_ENV = "TENNIS_LAB_AGY_PARENT"

FAKE_AGY_LAUNCH = r"""
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

assert ctypes.CDLL(None).prctl(15, b"fake-agy-driver", 0, 0, 0) == 0
args = sys.argv[1:]
roots = [Path.cwd()]
if "--add-dir" in args:
    roots.append(Path(args[args.index("--add-dir") + 1]))
configs = [root / ".agents/hooks.json" for root in roots]
discovered = [path for path in configs if path.exists()]
snapshot = {"cwd": str(Path.cwd()), "roots": [str(root) for root in roots]}
if not discovered:
    print(json.dumps({**snapshot, "hooks": 0}))
    raise SystemExit(0)
assert len(discovered) == 1
path = discovered[0]
config = json.loads(path.read_text())
command = config["tennis-lab-parent-board"]["PreInvocation"][0]["command"]
if os.environ.get("TEST_REMOVE_AGY_VENV") == "1":
    venv = Path.cwd() / ".venv"
    assert venv.is_symlink()
    venv.unlink()
payload = {"hookEventName": "PreInvocation", "invocationNum": 0,
           "conversationId": os.environ["ANTIGRAVITY_CONVERSATION_ID"]}
result = subprocess.run(
    ["/bin/sh", "-c", command], cwd=path.parent, input=json.dumps(payload),
    capture_output=True, text=True, check=False,
)
print(json.dumps({**snapshot, "hooks": 1, "status": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr}))
raise SystemExit(result.returncode)
"""


def _agy_env(env: dict[str, str]) -> dict[str, str]:
    return {
        **{key: value for key, value in env.items() if not key.startswith("CLAUDE_")},
        LAUNCH_ENV: "1",
    }


def _value(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr + result.stdout
    value: dict[str, Any] = json.loads(result.stdout)
    return value


def _parent(
    parent: ParentEnv, tui: Tui, *args: str, **caller: Any
) -> subprocess.CompletedProcess[str]:
    env = caller.pop("env", _agy_env(parent.env))
    return tui.run(
        "-m:parent",
        "--repo",
        "o/r",
        "--parent",
        "1",
        *args,
        env=env,
        cwd=parent.repo,
        **caller,
    )


def test_agy_launcher_scopes_native_hook_and_reports_missing_venv(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "manual parent checkout"
    other = tmp_path / "other checkout"
    other.mkdir()
    shutil.copytree(
        OPS, checkout / ".agents/ops", ignore=shutil.ignore_patterns("__pycache__")
    )
    # Model native discovery of the shared project file if it is ever reintroduced.
    if (shared := OPS.parent / "hooks.json").is_file():
        shutil.copyfile(shared, checkout / ".agents/hooks.json")
    launcher = checkout / ".agents/ops/herdr_labels/agy-parent.sh"
    native = tmp_path / "agy"
    native.write_text(f"#!{sys.executable}\n{FAKE_AGY_LAUNCH}")
    native.chmod(0o755)
    env = {
        **{
            key: value
            for key, value in os.environ.items()
            if key not in {LAUNCH_ENV, "HERDR_ENV", "TENNIS_LAB_AGY_HOOK_ENTRY"}
        },
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "ANTIGRAVITY_CONVERSATION_ID": ROOT,
    }

    def launch(
        command: list[str], cwd: Path = checkout, **extra: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=cwd,
            env={**env, **extra},
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    for args, cwd in ((["-p", "test"], checkout), (["acp"], checkout), ([], other)):
        assert _value(launch([str(native), *args], cwd))["hooks"] == 0
    missing = launch([str(launcher)])
    assert missing.returncode == 1 and ".venv/bin/python is required" in missing.stderr
    assert missing.stdout == ""

    (checkout / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    for args in (["-p", "test"], ["acp"]):
        excluded = launch([str(launcher), *args])
        assert excluded.returncode == 1 and "interactive TUI" in excluded.stderr
        assert excluded.stdout == ""
    unverified = launch([str(launcher)])
    event = json.loads(unverified.stdout)
    assert unverified.returncode == event["status"] == 1 and event["hooks"] == 1
    assert "host process ancestry is unavailable" in event["stderr"]
    missing_during_hook = launch([str(launcher)], TEST_REMOVE_AGY_VENV="1")
    assert missing_during_hook.returncode == 1
    event = json.loads(missing_during_hook.stdout)
    assert event["hooks"] == 1 and event["status"] == 1
    assert ".venv/bin/python is required" in event["stderr"] and event["stdout"] == ""
    assert event["cwd"] == str(checkout)
    working, hook_workspace = (Path(path) for path in event["roots"])
    assert working == checkout and not hook_workspace.is_relative_to(checkout)
    assert not hook_workspace.exists()
    assert not (checkout / ".agents/hooks.json").exists()


@pytest.mark.parametrize(
    ("args", "tty", "conversation", "trajectory", "nested", "error"),
    [
        (("-p", "prompt"), True, ROOT, TRAJECTORY, False, "interactive TUI"),
        (("acp",), True, ROOT, TRAJECTORY, False, "interactive TUI"),
        ((), False, ROOT, TRAJECTORY, False, "foreground host TUI"),
        ((), True, ROOT, TRAJECTORY, True, "nested agy parent operations"),
        ((), True, CHILD, TRAJECTORY, False, "foreground TUI lock"),
        ((), True, ROOT, OTHER, False, "trajectory differs"),
    ],
    ids=[
        "print",
        "acp",
        "no-tty",
        "nested",
        "same-pid-child",
        "trajectory",
    ],
)
def test_agy_identity_rejects_unverified_callers_before_github(
    parent_env: ParentEnv,
    args: tuple[str, ...],
    tty: bool,
    conversation: str,
    trajectory: str,
    nested: bool,
    error: str,
) -> None:
    with foreground(parent_env.root / "agy-home", *args, tty=tty) as tui:
        assert tui.lock(ROOT)
        tui.metadata(ROOT)
        refused = _parent(
            parent_env,
            tui,
            "register",
            "--summary",
            "親の表示",
            conversation=conversation,
            trajectory=trajectory,
            nested=nested,
        )
        assert refused.returncode != 0 and error in refused.stderr
        if not args and not nested:
            status = _parent(
                parent_env,
                tui,
                "status",
                conversation=conversation,
                trajectory=trajectory,
            )
            assert (
                status.returncode == 2
                and status.stdout == ""
                and error in status.stderr
            )
        else:
            hook = tui.run(
                HOOK,
                "pre-invocation",
                env=_agy_env(parent_env.env),
                cwd=parent_env.repo,
                nested=nested,
                stdin=_hook_payload(ROOT),
            )
            assert _value(hook) == {} and hook.stderr == ""
    assert parent_env.calls() == []


@pytest.mark.parametrize("parent", [ROOT, None], ids=["subagent", "workflow"])
def test_agy_reopened_child_history_is_rejected_even_with_its_own_lock(
    parent_env: ParentEnv, parent: str | None
) -> None:
    with foreground(parent_env.root / "agy-home", "--conversation", CHILD) as tui:
        assert tui.lock(CHILD)
        tui.metadata(CHILD, root=ROOT, parent=parent)
        refused = _parent(
            parent_env, tui, "register", "--summary", "親の表示", conversation=CHILD
        )
        assert (
            refused.returncode != 0 and "require a root conversation" in refused.stderr
        )
    assert parent_env.calls() == []


def test_agy_metadata_requires_a_listener_owned_by_the_tui(
    parent_env: ParentEnv,
) -> None:
    with (
        foreground(parent_env.root / "agy-home") as tui,
        foreground(parent_env.root / "other-home") as unrelated,
    ):
        assert tui.lock(ROOT)
        unrelated.metadata(ROOT)
        refused = _parent(
            parent_env,
            tui,
            "register",
            "--summary",
            "親の表示",
            metadata_port=unrelated.port,
        )
        assert refused.returncode != 0
        assert "metadata endpoint is not owned by the foreground TUI" in refused.stderr
    assert parent_env.calls() == []


def test_agy_sandbox_cannot_use_identity_environment_without_host_ancestry(
    parent_env: ParentEnv,
) -> None:
    env = {**_agy_env(parent_env.env), "ANTIGRAVITY_CONVERSATION_ID": ROOT}
    for command, expected, stdin in (
        (
            [
                *hook_command("-m:parent"),
                "--repo",
                "o/r",
                "--parent",
                "1",
                "register",
                "--summary",
                "親の表示",
            ],
            2,
            None,
        ),
        (
            [*hook_command("-m:parent"), "--repo", "o/r", "--parent", "1", "status"],
            2,
            None,
        ),
        ([*hook_command(HOOK), "pre-invocation"], 1, json.dumps(_hook_payload(ROOT))),
    ):
        result = subprocess.run(
            command,
            env=env,
            cwd=parent_env.repo,
            capture_output=True,
            text=True,
            input=stdin,
            timeout=10,
            check=False,
        )
        assert result.returncode == expected and result.stdout == ""
        assert "host process ancestry is unavailable" in result.stderr
    assert parent_env.calls() == []


@pytest.mark.parametrize("obstruct", ["proc", "lock"])
def test_agy_hook_reports_verification_errors_without_an_empty_reply(
    parent_env: ParentEnv, obstruct: str
) -> None:
    with foreground(parent_env.root / "agy-home") as tui:
        env = _agy_env(parent_env.env)
        if obstruct == "proc":
            assert tui.lock(ROOT)
            env["TEST_AGY_UNREADABLE_STAT"] = str(tui.pid)
        hook = tui.run(
            HOOK,
            "pre-invocation",
            env=env,
            cwd=parent_env.repo,
            stdin=_hook_payload(ROOT),
        )
        status = _parent(parent_env, tui, "status", env=env)
        assert hook.returncode == 1 and hook.stdout == ""
        assert status.returncode == 2 and status.stdout == ""
        expected = (
            "host process ancestry is unavailable" if obstruct == "proc" else "TUI"
        )
        assert expected in hook.stderr and expected in status.stderr
    assert parent_env.calls() == []


def test_unmanaged_agy_cannot_register_an_unavailable_board_hook(
    parent_env: ParentEnv,
) -> None:
    with foreground(parent_env.root / "agy-home") as tui:
        assert tui.lock(ROOT)
        tui.metadata(ROOT)
        env = _agy_env(parent_env.env)
        del env[LAUNCH_ENV]
        result = _parent(parent_env, tui, "register", "--summary", "親の表示", env=env)
        assert result.returncode == 2 and "agy-parent.sh" in result.stderr
    assert parent_env.calls() == []


def test_agy_parent_register_handoff_and_release_use_the_current_root(
    parent_env: ParentEnv,
) -> None:
    with foreground(parent_env.root / "agy-home", "--continue") as tui:
        assert tui.lock(ROOT)
        tui.metadata(ROOT)
        tui.metadata(OTHER, root=OTHER)
        args = ("--note-file", parent_env.note())
        assert (
            _value(_parent(parent_env, tui, "register", "--summary", "試験の担当"))[
                "current_session_is_owner"
            ]
            is True
        )
        assert _value(_parent(parent_env, tui, "status"))["assignment"] == "active"
        _value(_parent(parent_env, tui, "handoff", *args))
        assert (
            _value(_parent(parent_env, tui, "status"))["assignment"]
            == "handoff_pending"
        )
        assert tui.lock(OTHER)
        _value(
            _parent(parent_env, tui, "handoff", "--accept", *args, conversation=OTHER)
        )
        record = (
            parent_env.root
            / "private-state/tennis-lab-agents/parent-sessions/registry.json"
        )
        owner = json.loads(record.read_text())["parents"]["o/r#1"]
        assert owner["session"] == f"agy:{OTHER}" and owner["generation"] == 2
        assert (
            _value(_parent(parent_env, tui, "status", conversation=OTHER))[
                "current_session_is_owner"
            ]
            is True
        )
        assert (
            _parent(parent_env, tui, "release", *args, conversation=ROOT).returncode
            != 0
        )
        _value(_parent(parent_env, tui, "release", *args, conversation=OTHER))
        assert (
            _value(_parent(parent_env, tui, "status", conversation=OTHER))["assignment"]
            == "released"
        )


def _place(
    parent: ParentEnv,
    labels: Labels,
    tui: Tui,
    pane: str = "w1:p1",
) -> dict[str, str]:
    fake_herdr = Path(parent.env["PATH"].split(":")[0]) / "herdr"
    fake_herdr.write_bytes((labels.root / "bin/herdr").read_bytes())
    state = labels.read()
    state.setdefault("pane_pids", {})[pane] = tui.pid
    state["panes"][pane].update(
        agent="agy",
        agent_status="idle",
        agent_session={"kind": "id", "agent": "agy", "value": CHILD},
    )
    labels.write(state)
    return {
        **_agy_env(parent.env),
        **{
            key: labels.env[key]
            for key in ("HERDR_ENV", "HERDR_SOCKET_PATH", "LABEL_FAKE_STATE")
        },
        "HERDR_PANE_ID": pane,
    }


def _status(
    tui: Tui, env: dict[str, str], cwd: Path, conversation: str = ROOT
) -> subprocess.CompletedProcess[str]:
    return tui.run(
        HOOK,
        "status-line",
        env=env,
        cwd=cwd,
        conversation=conversation,
        stdin={"conversation_id": conversation, "session_id": conversation},
    )


def _hook_payload(conversation: str, invocation: int = 0) -> dict[str, Any]:
    return {
        "hookEventName": "PreInvocation",
        "conversationId": conversation,
        "invocationNum": invocation,
    }


def _pre_invocation(
    tui: Tui,
    env: dict[str, str],
    cwd: Path,
    conversation: str = ROOT,
    invocation: int = 0,
) -> dict[str, Any]:
    return _value(
        tui.run(
            HOOK,
            "pre-invocation",
            env=env,
            cwd=cwd,
            conversation=conversation,
            stdin={**_hook_payload(conversation, invocation), "cwd": str(cwd)},
        )
    )


def test_agy_herdr_switch_and_exit_clear_only_the_original_pane(
    parent_env: ParentEnv,
    labels: Labels,
) -> None:
    p = parent_env
    opsproc = importlib.import_module("lib.opsproc")
    launcher = importlib.import_module("herdr_labels.agy_tui")
    home = p.root / "agy-home"
    with foreground(home) as tui:
        assert tui.lock(ROOT)
        tui.metadata(ROOT)
        env = _place(p, labels, tui)
        state = labels.read()
        state["fail_call"] = "pane get"
        labels.write(state)
        partial = _parent(p, tui, "register", "--summary", "試験の担当", env=env)
        assert partial.returncode == 1
        failure = json.loads(partial.stdout)
        assert failure["parent_action_applied"] is True
        assert "専用ランチャー" in failure["display_sync"]["repair"]
        assert "parent register" in failure["display_sync"]["repair"]
        state = labels.read()
        del state["fail_call"]
        labels.write(state)
        result = _value(_parent(p, tui, "register", "--summary", "試験の担当", env=env))
        assert result["display_sync"]["status"] == "updated"
        pane = labels.read()["panes"]["w1:p1"]
        display = pane["display_agent"]
        assert pane["tokens"]["tl_issue"] == "#1"
        assert pane["tokens"]["tl_engine"] == "agy"
        assert pane["tokens"]["tl_name"]
        assert tui.lock(OTHER)
        assert _status(tui, env, p.repo, OTHER).returncode == 0
        assert labels.read()["panes"]["w1:p1"]["tokens"] == {"other": "keep"}
        assert "display_agent" not in labels.read()["panes"]["w1:p1"]
        before = labels.mutations()
        assert (
            _status(tui, env, p.repo).returncode == 0
        )  # Delayed callback for the old UUID.
        assert labels.mutations() == before
        assert tui.lock(ROOT)
        assert _status(tui, env, p.repo).returncode == 0
        assert labels.read()["panes"]["w1:p1"]["display_agent"] == display
        previous = opsproc.process_identity(tui.pid)

    with foreground(home, "--conversation", ROOT) as replacement:
        assert replacement.lock(ROOT)
        env = _place(p, labels, replacement)
        assert _status(replacement, env, p.repo).returncode == 0
        before = labels.mutations()
        launcher.clear_after_exit(env, "term-a", previous)
        assert labels.mutations() == before
        assert labels.read()["panes"]["w1:p1"]["display_agent"] == display
        generation = opsproc.process_identity(replacement.pid)

    state = labels.read()
    state["panes"]["w1:p1"]["agent"] = None
    labels.write(state)
    launcher.clear_after_exit(env, "term-a", generation)
    pane = labels.read()["panes"]["w1:p1"]
    assert pane["tokens"] == {"other": "keep"} and "display_agent" not in pane


def test_agy_register_and_sync_refuse_a_different_pane_with_the_same_conversation(
    parent_env: ParentEnv, labels: Labels
) -> None:
    p = parent_env
    with (
        foreground(p.root / "caller-home") as caller,
        foreground(p.root / "other-home") as target,
    ):
        assert caller.lock(ROOT) and target.lock(ROOT)
        caller.metadata(ROOT)
        env = _place(p, labels, target)
        before = labels.read()["panes"]
        register = _parent(
            p,
            caller,
            "register",
            "--summary",
            "試験の担当",
            env=env,
            other_tui=target,
        )
        assert register.returncode == 1
        result = json.loads(register.stdout)
        assert result["parent_action_applied"] is True
        assert (
            "does not match the inherited Herdr pane" in result["display_sync"]["error"]
        )
        sync = caller.run(
            "-m:herdr_labels",
            "sync",
            "--parent",
            "1",
            env=env,
            cwd=p.repo,
            other_tui=target,
        )
        assert (
            sync.returncode == 2
            and "does not match the inherited Herdr pane" in sync.stderr
        )
        assert labels.read()["panes"] == before and not labels.mutations()
        display_state = Path(env["XDG_STATE_HOME"]) / "tennis-lab-agents/herdr-labels"
        assert not list(display_state.glob("owners/*.json"))
        assert not list(display_state.glob("panes/*.json"))


def test_agy_display_move_needs_the_lock_and_a_confirmed_vacancy(
    parent_env: ParentEnv,
    labels: Labels,
) -> None:
    p = parent_env
    home = p.root / "agy-home"
    with foreground(home) as old, foreground(home, "--conversation", ROOT) as new:
        assert old.lock(ROOT)
        old.metadata(ROOT)
        old_env = _place(p, labels, old)
        assert (
            _value(_parent(p, old, "register", "--summary", "試験の担当", env=old_env))[
                "display_sync"
            ]["status"]
            == "updated"
        )
        new_env = _place(p, labels, new, "w1:p2")
        assert not new.lock(ROOT)
        before = labels.mutations()
        duplicate = _status(new, new_env, p.repo)
        assert (
            duplicate.returncode == 1
            and "missing its conversation lock" in duplicate.stderr
        )
        assert labels.mutations() == before
        assert old.lock(OTHER) and new.lock(ROOT)
        assert (
            _status(new, new_env, p.repo).returncode != 0
        )  # Old pane still advertises it.
        assert labels.mutations() == before
        assert _status(old, old_env, p.repo, OTHER).returncode == 0
        assert _status(new, new_env, p.repo).returncode == 0
        panes = labels.read()["panes"]
        assert "display_agent" not in panes["w1:p1"]
        assert panes["w1:p2"]["tokens"]["tl_issue"] == "#1"


def test_agy_board_routes_unread_without_sending_or_reading_it(
    parent_env: ParentEnv,
    labels: Labels,
) -> None:
    p = parent_env
    with foreground(p.root / "agy-home") as tui:
        assert tui.lock(ROOT)
        tui.metadata(ROOT)
        env = _place(p, labels, tui)
        _value(_parent(p, tui, "register", "--summary", "試験の担当", env=env))
        _value(p.cli("register", parent="2"))

        def board(*args: str) -> dict[str, Any]:
            return _value(tui.run("-m:board", *args, env=env, cwd=p.repo))

        def peer(*args: str) -> dict[str, Any]:
            return _value(run_board(p.env, *args))

        name = board("whoami")["name"]
        roster = board("agents")["agents"]
        assert [(row["name"], row["engine"], row["state"]) for row in roster] == [
            (name, "agy", "idle")
        ]
        outgoing = board(
            "post",
            "--to",
            "human",
            "--kind",
            "info",
            "--subject",
            "担当からの発言",
            "--body-file",
            p.note("担当からの本文。"),
        )["post"]
        assert outgoing["sender"]["name"] == name
        incoming = peer(
            "post",
            "--to",
            name,
            "--kind",
            "question",
            "--subject",
            "確認の依頼",
            "--body-file",
            p.note("通知に載せない本文。"),
        )["post"]
        post_id = incoming["id"]
        recipient = incoming["recipients"][0]
        assert (
            recipient["delivery_state"],
            recipient["delivery_reason"],
            recipient["attempts"],
        ) == (
            "pending",
            "awaiting_hook",
            0,
        )
        retry = peer("dispatch", "--post", post_id)
        assert retry["status"] == "awaiting_hook"
        assert retry["recipients"][0]["attempted"] is False
        assert board("inbox", "--unread")["posts"][0]["id"] == post_id
        assert _pre_invocation(tui, env, p.repo, CHILD) == {}
        assert _pre_invocation(tui, env, p.repo, invocation=1) == {}
        notice = _pre_invocation(tui, env, p.repo)["injectSteps"][0]["systemMessage"][
            "systemMessage"
        ]
        assert "未読:1" in notice and "board inbox --unread" in notice
        assert "通知に載せない本文" not in notice
        assert tui.lock(OTHER)
        assert (
            _pre_invocation(tui, env, p.repo) == {}
        )  # Old conversation's delayed event.
        assert _pre_invocation(tui, env, p.repo, OTHER) == {}
        assert board("agents")["agents"] == []  # Herdr's cached UUID is insufficient.
        assert tui.lock(ROOT)
        assert (
            board("inbox", "--unread")["posts"][0]["recipients"][0]["read_at"] is None
        )
        read = board("read", post_id)["post"]
        assert read["body"] == "通知に載せない本文。"
        assert read["recipients"][0]["read_at"] is not None
        assert _pre_invocation(tui, env, p.repo) == {}
        assert board("inbox", "--unread")["posts"] == []
        with sqlite3.connect(
            p.root / "private-state/tennis-lab-agents/board/board.sqlite3"
        ) as connection:
            assert connection.execute(
                "SELECT engine, state, reason, attempts FROM deliveries WHERE post_id=?",
                (post_id,),
            ).fetchone() == ("agy", "pending", "awaiting_hook", 0)
        reply = board("reply", "--post", post_id, "--body-file", p.note("返信本文。"))[
            "post"
        ]
        assert reply["sender"]["name"] == name and reply["in_reply_to"] == post_id
        delivery = reply["recipients"][0]
        assert (
            delivery["delivery_state"],
            delivery["delivery_reason"],
            delivery["attempts"],
        ) == (
            "delivery_failed",
            "route_unavailable",
            1,
        )
