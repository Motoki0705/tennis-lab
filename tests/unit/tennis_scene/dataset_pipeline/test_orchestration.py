"""Queue orchestration never launches a GPU workload outside the worker path."""

import shlex
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.tennis_scene.dataset_pipeline import orchestration as module


def test_all_sequence_and_source_overrides(tmp_path: Path) -> None:
    commands = module.build_commands("all", [], project=tmp_path, shared=tmp_path)
    assert [command[2].rsplit(".", 1)[-1] for command in commands] == [
        "import_broadcast_ball",
        "build_slcs_dataset",
        "build_slcs_dataset",
        "report_slcs_dataset_quality",
        "assemble_slcs_dataset",
    ]
    override = "dataset_output_directory=custom/clip"
    commands = module.build_commands(
        "broadcast", [override], project=tmp_path, shared=tmp_path
    )
    assert override not in commands[0]
    assert commands[1][-1] == override
    assert (
        module.build_commands("meiji", [override], project=tmp_path, shared=tmp_path)[
            0
        ][-1]
        == override
    )
    with pytest.raises(ValueError, match="Source-specific"):
        module.build_commands("all", [override], project=tmp_path, shared=tmp_path)


def test_enqueue_quotes_and_shared_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, shared = tmp_path / "checkout with spaces", tmp_path / "main"
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(module, "shared_repository_root", lambda _: shared)
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "worker_alive", lambda _: True)
    override = "output_dir=run/a b;$(touch /bad)"
    module.run_build(
        "meiji", [override], execute=False, project=project, environment={}
    )
    assert run.call_count == 1
    args, kwargs = run.call_args
    assert args[0][-2:] == ["--resource", "all"]
    command = shlex.split(args[0][3])
    assert command[-1] == override
    assert command[2] == str(project / ".venv/bin/python")
    assert kwargs["env"]["TRAINING_QUEUE_DIR"] == str(shared / ".training_queue")
    assert kwargs["cwd"] == project


def test_internal_guard_and_failure_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "shared_repository_root", lambda _: tmp_path)
    run = Mock(side_effect=subprocess.CalledProcessError(7, "failed"))
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(ValueError, match="reservation"):
        module.run_build("all", [], execute=True, project=tmp_path, environment={})
    run.assert_not_called()
    with pytest.raises(subprocess.CalledProcessError):
        module.run_build(
            "all",
            [],
            execute=True,
            project=tmp_path,
            environment={"TENNIS_RUN_ID": "job", "TENNIS_REPRO_DIR": "/repro"},
        )
    assert run.call_count == 1


def test_starts_missing_worker_and_handles_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "shared_repository_root", lambda _: tmp_path)
    monkeypatch.setattr(module, "worker_alive", Mock(side_effect=[False, True]))
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 1),
        ]
    )
    monkeypatch.setattr(module.subprocess, "run", run)
    module.run_build("all", [], execute=False, project=tmp_path, environment={})
    assert run.call_count == 2
    assert run.call_args.args[0][-1] == "start"


def test_worker_status_and_git_common_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not module.worker_alive(tmp_path)
    (tmp_path / "worker.pid").write_text("0")
    assert not module.worker_alive(tmp_path)
    (tmp_path / "worker.pid").write_text("42")
    kill = Mock(side_effect=ProcessLookupError)
    monkeypatch.setattr(module.os, "kill", kill)
    assert not module.worker_alive(tmp_path)
    kill.side_effect = None
    assert module.worker_alive(tmp_path)
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [], 0, stdout=str(tmp_path / ".git") + "\n"
        )
    )
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.shared_repository_root(tmp_path / "linked") == tmp_path
    assert run.call_args.args[0][-1] == "--git-common-dir"
