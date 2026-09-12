"""Exercise direct Drive outputs and monitoring with real CPU subprocesses."""

from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from scripts.colab.workflow import remote_runner as remote
from scripts.colab.workflow.common import WorkflowError
from scripts.colab.workflow.jobs import load_job


def request_for(script: Path) -> dict[str, Any]:
    return {
        "run_id": "live-test-0001",
        "request_digest": "a" * 64,
        "drive": {"mode": "mount", "root": "tennis_lab"},
        "job": {
            "output_storage": "drive",
            "timeout_seconds": 20,
            "argv": [sys.executable, str(script), "paths.output_root=outputs/colab"],
            "outputs": ["outputs/colab/test"],
        },
    }


@pytest.mark.parametrize("exit_code", [0, 7])
def test_job_writes_to_drive_and_retains_logs_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = repo / "train.py"
    script.write_text(
        """import json, os, pathlib, sys
root=pathlib.Path(sys.argv[1].split('=',1)[1])/'test'
root.mkdir(parents=True)
(root/'last.ckpt').write_bytes(b'checkpoint')
pathlib.Path(os.environ['TENNIS_LAB_COLAB_PROGRESS_PATH']).write_text(json.dumps({'global_step': 12}))
print('step 12 loss 0.25', flush=True)
"""
        + f"sys.exit({exit_code})\n"
    )
    request = request_for(script)
    monkeypatch.setattr(
        remote, "_mount_drive_root", lambda _request: tmp_path / "drive"
    )
    live = remote._prepare_live_output(request)
    assert live is not None
    status = {"attempt": 1}
    if exit_code:
        with pytest.raises(remote.RemoteWorkflowError, match="exit code 7"):
            remote._run_monitored_job(
                request, repo, workspace, status, dict(os.environ), live
            )
    else:
        remote._run_monitored_job(
            request, repo, workspace, status, dict(os.environ), live
        )
    assert (live / "outputs/colab/test/last.ckpt").read_bytes() == b"checkpoint"
    assert not (repo / "outputs").exists()
    assert "loss 0.25" in (live / "attempt-1.log").read_text()
    progress = json.loads((workspace / "progress.json").read_text())
    assert progress["training"]["global_step"] == 12
    assert progress["returncode"] == exit_code
    assert progress["state"] == ("failed" if exit_code else "job_succeeded")
    manifest, archive = remote._collect_outputs(request, repo, workspace / "bundle")
    assert manifest[0]["entries"][0]["path"] == "outputs/colab/test/last.ckpt"
    with tarfile.open(archive) as bundle:
        member = bundle.extractfile("outputs/colab/test/last.ckpt")
        assert member is not None
        assert member.read() == b"checkpoint"


def test_progress_reader_preserves_last_observation_during_replace_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "training-progress.json"
    path.write_text('{"global_step": 13}', encoding="utf-8")
    previous = {"global_step": 12}

    def disappear(_path: Path, _label: str) -> dict[str, Any]:
        path.unlink()
        raise remote.RemoteWorkflowError("cannot read training progress")

    monkeypatch.setattr(remote, "_read_json_object", disappear)

    assert remote._read_training_progress_observation(path, previous) == previous


def test_progress_reader_rejects_persistently_malformed_file(tmp_path: Path) -> None:
    path = tmp_path / "training-progress.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(remote.RemoteWorkflowError, match="cannot read training progress"):
        remote._read_training_progress_observation(path, {"global_step": 12})


def test_drive_live_directory_rejects_other_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = request_for(tmp_path / "train.py")
    monkeypatch.setattr(remote, "_mount_drive_root", lambda _request: tmp_path)
    remote._prepare_live_output(request)
    request["request_digest"] = "b" * 64
    with pytest.raises(remote.RemoteWorkflowError, match="another request"):
        remote._prepare_live_output(request)


def test_rclone_training_stays_local_and_receives_managed_artifact_store(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = repo / "train.py"
    script.write_text(
        """import json, os, pathlib, sys
root=pathlib.Path(sys.argv[1].split('=',1)[1])/'test'
root.mkdir(parents=True)
(root/'invocation.json').write_text(json.dumps({'argv': sys.argv[2:], 'rclone_config': os.environ.get('RCLONE_CONFIG')}))
""",
        encoding="utf-8",
    )
    request = {
        "run_id": "rclone-test-0001",
        "request_digest": "a" * 64,
        "drive": {"mode": "rclone", "root": "team/training", "remote": "drive"},
        "job": {
            "output_storage": "drive",
            "timeout_seconds": 20,
            "argv": [sys.executable, str(script), "paths.output_root=outputs/colab"],
            "outputs": ["outputs/colab/test"],
        },
    }
    environment = {**os.environ, "RCLONE_CONFIG": "/run/secrets/rclone.conf"}

    remote._run_monitored_job(
        request,
        repo,
        workspace,
        {"attempt": 1},
        environment,
        live_root=None,
    )

    invocation = json.loads(
        (repo / "outputs/colab/test/invocation.json").read_text(encoding="utf-8")
    )
    assert invocation["rclone_config"] == "/run/secrets/rclone.conf"
    assert invocation["argv"] == [
        "run.artifact_store.mode=rclone",
        "run.artifact_store.remote=drive",
        "run.artifact_store.remote_root=team/training/colab-live/rclone-test-0001/training",
        "run.artifact_store.sync_interval_seconds=60",
    ]
    assert remote._prepare_live_output(request) is None
    manifest, _ = remote._collect_outputs(request, repo, workspace / "bundle")
    assert manifest[0]["declared_path"] == "outputs/colab/test"


def test_drive_output_rejects_an_unmounted_local_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = request_for(tmp_path / "train.py")
    mountpoint = tmp_path / "drive"
    (mountpoint / "MyDrive").mkdir(parents=True)
    monkeypatch.setattr(remote, "DRIVE_MOUNTPOINT", mountpoint)

    with pytest.raises(remote.RemoteWorkflowError, match="not mounted"):
        remote._prepare_live_output(request)

    assert not (mountpoint / "MyDrive" / "tennis_lab").exists()


def test_training_catalog_requires_explicit_drive_output() -> None:
    root = Path(__file__).parents[3]
    for path in (root / "scripts/colab/workflows/jobs").glob("*.toml"):
        job = load_job(path)
        if job.module and ".scripts.train" in job.module:
            assert job.output_storage == "drive", job.name


def test_drive_output_contract_rejects_unmapped_path(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "scripts/colab/workflows/jobs/blcs.toml"
    path = tmp_path / source.name
    path.write_text(
        source.read_text().replace(
            'outputs = ["outputs/colab/blcs"]', 'outputs = ["elsewhere/blcs"]'
        )
    )
    with pytest.raises(WorkflowError, match="Drive outputs require"):
        load_job(path)


def test_progress_command_reads_contents_and_reports_staleness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse
    import subprocess

    from scripts.colab.workflow import cli

    request = request_for(tmp_path / "train.py")
    metadata = {
        "colab_config": str(tmp_path / "config.json"),
        "session": "test-session",
    }
    monkeypatch.setattr(cli, "_load_run", lambda *_args: (tmp_path, request, metadata))
    calls: list[list[str]] = []

    def invoke(
        config: Path, args: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        Path(args[-1]).write_text(
            json.dumps(
                {
                    "run_id": request["run_id"],
                    "request_digest": request["request_digest"],
                    "updated_at": "2020-01-01T00:00:00+00:00",
                    "state": "completed",
                    "training": {"global_step": 23},
                    "log_tail": ["first", "second"],
                }
            )
        )
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(cli, "_invoke", invoke)
    args = argparse.Namespace(
        state_dir=str(tmp_path),
        run_id=request["run_id"],
        interval=2,
        command="progress",
        watch=True,
        tail=1,
    )
    assert cli.command_progress(args) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["stale"] is True
    assert observed["training"]["global_step"] == 23
    assert calls[0][0] == "download"
    args.command = "logs"
    assert cli.command_progress(args) == 0
    assert capsys.readouterr().out.strip() == "second"
