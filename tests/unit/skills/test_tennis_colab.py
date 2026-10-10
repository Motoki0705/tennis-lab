"""Colab ownership and worker persistence invariants, with real CPU subprocesses."""

from __future__ import annotations

import argparse
import importlib
import json
import shlex
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / ".agents/skills/tennis-colab/scripts"))
    return importlib.import_module("colab_core"), importlib.import_module(
        "colab_remote"
    )


def test_missing_registry_cannot_implicitly_allocate(
    modules: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    core, _ = modules
    session = core.Session("owned", state_root=tmp_path)
    session.save(phase="ready")
    with pytest.raises(core.ColabError, match="refusing an implicit allocation"):
        session.ssh_argv(["true"])
    core.atomic_json(session.registry, {"different": {"token": "PRIVATE"}})
    with pytest.raises(core.ColabError, match="refusing an implicit allocation"):
        session.ssh_argv(["true"])


def test_ssh_preserves_argv_and_uses_owned_registry_and_single_transport(
    modules: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    core, _ = modules
    binary = tmp_path / "colab"
    binary.touch()
    session = core.Session("owned", state_root=tmp_path, colab_bin=binary)
    session.save(phase="ready")
    core.atomic_json(session.registry, {"owned": {"token": "PRIVATE"}})
    command = ["python3", "-c", "print('quoted $HOME `literal`')"]
    argv = session.ssh_argv(command)
    assert shlex.split(argv[-1]) == command
    proxy = next(item for item in argv if item.startswith("ProxyCommand=")).split(
        "=", 1
    )[1]
    parsed = shlex.split(proxy)
    assert parsed[parsed.index("--config") + 1] == str(session.registry)
    assert parsed[parsed.index("--auth") + 1] == "oauth2"
    assert "--proxy-mode" in parsed and "--rm" not in parsed
    assert "ControlMaster=auto" in argv
    assert "PRIVATE" not in " ".join(argv)
    assert argv == session.ssh_argv(command)


@pytest.mark.parametrize(
    "url",
    [
        "https://token@github.com/x/y",
        "git@github.com:x/y",
        "https://github.com/x/y?token=x",
        "https://github.com/../x",
    ],
)
def test_git_credentials_and_ambiguous_urls_rejected(
    modules: tuple[ModuleType, ModuleType], url: str
) -> None:
    core, _ = modules
    with pytest.raises(core.ColabError):
        core.github_url(url)


@pytest.fixture
def worker(
    modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[tuple[ModuleType, Path, list[dict[str, Any]]]]:
    core, remote = modules
    root = tmp_path / "session"
    root.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    core.atomic_json(
        root / "session.json",
        {
            "session": "test",
            "repo_root": str(repo),
            "drive_remote": "test",
            "drive_root": "project",
            "sync_interval_seconds": 60,
        },
    )
    monkeypatch.setattr(remote, "ROOT", root)
    published: list[dict[str, Any]] = []

    class Store:
        def sync_tree(self) -> None:
            pass

        def publish_file(self, path: Path) -> None:
            state = json.loads(path.read_text())
            if state["status"] in {"completed", "failed", "cancelled"}:
                local = json.loads((path.parent.parent / "status.json").read_text())
                assert local["status"] == "finalizing", (
                    "Do not expose completion before durable publication"
                )
            published.append(state)

    monkeypatch.setattr(remote, "_store", lambda *a: Store())
    # Tests call the worker in-process; restore the test runner's signal handlers.
    handlers = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        yield remote, repo, published
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


@pytest.mark.parametrize("exit_code", [0, 17])
def test_remote_exit_code_and_completion_follow_final_persistence(
    worker: tuple[ModuleType, Path, list[dict[str, Any]]],
    exit_code: int,
) -> None:
    remote, repo, published = worker
    remote.prepare(
        {
            "job_id": "cpu-command",
            "argv": [
                sys.executable,
                "-c",
                f"print('result'); raise SystemExit({exit_code})",
            ],
            "persist": [],
        }
    )
    cfg = json.loads((remote.ROOT / "session.json").read_text())
    cfg["rclone_config"] = str(repo / "unused-test-config")
    (remote.ROOT / "session.json").write_text(json.dumps(cfg))
    assert remote.run_job("cpu-command") == exit_code
    state = json.loads((remote.ROOT / "jobs/cpu-command/status.json").read_text())
    assert state["returncode"] == exit_code
    assert state["status"] == ("completed" if exit_code == 0 else "failed")
    assert state["drive_saved_at"]
    assert published[0]["status"] == "starting"
    assert published[-1]["status"] == state["status"]
    assert remote.process_token(state["pid"]) is None
    assert "result" in remote.rpc({"action": "logs", "job_id": "cpu-command"})["text"]


def test_storage_failure_stops_running_process_and_never_reports_completion(
    worker: tuple[ModuleType, Path, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, repo, _ = worker
    cfg_path = remote.ROOT / "session.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.update(
        sync_interval_seconds=0.01, rclone_config=str(repo / "unused-test-config")
    )
    cfg_path.write_text(json.dumps(cfg))
    remote.prepare(
        {
            "job_id": "running",
            "argv": [sys.executable, "-c", "import time; time.sleep(120)"],
            "persist": [],
        }
    )
    calls = 0

    def fail_after_initial_save(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("storage unavailable")

    monkeypatch.setattr(remote, "sync_job", fail_after_initial_save)
    assert remote.run_job("running") == 1
    state = json.loads((remote.ROOT / "jobs/running/status.json").read_text())
    assert state["status"] == "save_failed"
    assert not state.get("drive_saved_at")
    assert remote.process_token(state["pid"]) is None


def test_pending_cancel_is_saved_and_prevents_execution(
    worker: tuple[ModuleType, Path, list[dict[str, Any]]],
) -> None:
    remote, _, published = worker
    remote.prepare(
        {
            "job_id": "pending",
            "argv": [sys.executable, "-c", "raise RuntimeError('must not run')"],
            "persist": [],
        }
    )
    assert remote.rpc({"action": "cancel", "job_id": "pending"})["requested"]
    assert published[-1]["status"] == "cancelled"
    with pytest.raises(RuntimeError, match="pending job"):
        remote.run_job("pending")


def test_sensitive_untracked_files_are_excluded_and_artifact_escape_rejected(
    worker: tuple[ModuleType, Path, list[dict[str, Any]]],
) -> None:
    remote, repo, _ = worker
    (repo / ".env").write_text("PRIVATE")
    (repo / "client_secret.json").write_text("PRIVATE")
    (repo / "tokenizer.py").write_text("# model code")
    (repo / "credentials.py").write_text("# credentials handling code")
    (repo / "fix.py").write_text("print('fix')")
    recovered = remote.diff({"include_untracked": True})
    assert ".env" in recovered["excluded"] and ".env" not in recovered["untracked"]
    assert "fix.py" in recovered["untracked"]
    assert "client_secret.json" in recovered["excluded"]
    assert "tokenizer.py" in recovered["untracked"]
    assert "credentials.py" in recovered["untracked"]
    with pytest.raises(RuntimeError, match="relative path"):
        remote._persist_path(remote.config(), "../session/secrets")


def test_input_assets_do_not_block_source_snapshot_or_next_command(
    worker: tuple[ModuleType, Path, list[dict[str, Any]]],
) -> None:
    remote, repo, _ = worker
    asset = repo / "data/example/shards/images.bin"
    asset.parent.mkdir(parents=True)
    with asset.open("wb") as stream:
        stream.truncate(17 * 1024 * 1024)
    (repo / "fix.py").write_text("print('fix')")
    state = remote.prepare({"job_id": "after-inputs", "argv": [sys.executable, "-c", "pass"], "persist": []})
    assert state["status"] == "pending"
    changes = json.loads((remote.ROOT / "jobs/after-inputs/source_changes.json").read_text())
    assert "fix.py" in changes["untracked"]
    assert "data/example/shards/images.bin" in changes["excluded"]
    assert "Drive" in changes["exclusion_reasons"]["data/example/shards/images.bin"]


def test_failed_source_snapshot_does_not_reserve_a_job_id(
    worker: tuple[ModuleType, Path, list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, _, _ = worker

    def fail_snapshot(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("snapshot failure")

    monkeypatch.setattr(remote, "diff", fail_snapshot)
    with pytest.raises(RuntimeError, match="snapshot failure"):
        remote.prepare({"job_id": "not-created", "argv": [sys.executable], "persist": []})
    assert not (remote.ROOT / "jobs/not-created").exists()


@pytest.mark.parametrize("unrecovered_code", [False, True])
def test_stop_allows_storage_exclusions_but_keeps_source_recovery_guard(
    modules: tuple[ModuleType, ModuleType], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, unrecovered_code: bool,
) -> None:
    frontend = importlib.import_module("colab")
    excluded = ["data/images.bin"] + (["private.key"] if unrecovered_code else [])
    monkeypatch.setattr(frontend, "recover_diff", lambda *a: {
        "excluded": excluded, "storage_excluded": ["data/images.bin"],
    })
    commands = []
    session = SimpleNamespace(
        name="owned", directory=tmp_path,
        state=lambda: {"phase": "ready"},
        rpc=lambda action: {"jobs": [{"job_id": "train", "status": "completed", "drive_saved_at": "saved"}]},
        close_transport=lambda: None,
        cli=lambda *args: commands.append(args),
        save=lambda **kwargs: None,
    )
    if unrecovered_code:
        with pytest.raises(RuntimeError, match="source files"):
            frontend.stop(argparse.Namespace(), session)
        assert not commands
    else:
        assert frontend.stop(argparse.Namespace(), session)["status"] == "stopped"
        assert commands == [("stop", "-s", "owned")]


@pytest.mark.parametrize("assignment_expired", [False, True])
def test_reconnect_rechecks_refreshed_ownership_and_never_allocates(
    modules: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch,
    assignment_expired: bool,
) -> None:
    frontend = importlib.import_module("colab")
    calls = []

    def require_registered() -> None:
        calls.append("guard")
        if assignment_expired and calls.count("guard") == 2:
            raise RuntimeError("assignment expired")

    session = SimpleNamespace(
        require_registered=require_registered,
        cli=lambda command, **kwargs: calls.append(command),
        rpc=lambda action, **kwargs: {"action": action, "jobs": []},
    )
    if assignment_expired:
        with pytest.raises(RuntimeError, match="assignment expired"):
            frontend.reconnect(session)
    else:
        assert frontend.reconnect(session)["connection"] == "reconnected"
    assert calls == ["guard", "sessions", "guard"]


def test_runner_output_has_one_live_writer_and_is_salvaged_after_exit(
    worker: tuple[ModuleType, Path, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, repo, _ = worker
    output = repo / "outputs/run"
    output.mkdir(parents=True)
    (output / "last.ckpt").write_bytes(b"finished checkpoint")
    copied: list[Path] = []

    class Store:
        def __init__(self, path: Path) -> None:
            self.path = path

        def sync_tree(self) -> None:
            copied.append(self.path)

        def publish_file(self, path: Path) -> None:
            pass

    monkeypatch.setattr(remote, "_store", lambda cfg, local, target: Store(local))
    remote.prepare(
        {
            "job_id": "runner",
            "argv": ["true"],
            "persist": [],
            "runner_outputs": ["outputs/run"],
        }
    )
    directory = remote.job_dir("runner")
    remote.sync_job(remote.config(), directory)
    assert output not in copied
    remote.sync_job(remote.config(), directory, final=True, completion="completed")
    assert output in copied
    with pytest.raises(RuntimeError, match="existing command"):
        remote.prepare({"job_id": "another", "argv": ["true"], "persist": []})
