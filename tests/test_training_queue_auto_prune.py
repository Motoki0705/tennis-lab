"""Opt-in training-queue checkpoint pruning tests (issue #545)."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from src.tasks.base.training.runner import BaseTrainingRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_SCRIPT = REPO_ROOT / ".agents/skills/training-queue/scripts/training_queue.sh"
PRUNE_SCRIPT = REPO_ROOT / ".agents/skills/training-queue/scripts/prune_ckpts.py"


def _run(
    cwd: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(QUEUE_SCRIPT), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def _wait_for_worker(
    cwd: Path,
    env: dict[str, str],
    *,
    done: int,
    failed: int,
) -> str:
    status = ""
    for _ in range(60):
        status = _run(cwd, env, "status").stdout
        if (
            "worker: stopped" in status
            and f"done={done}" in status
            and f"failed={failed}" in status
        ):
            return status
        time.sleep(0.5)
    raise AssertionError(f"worker did not finish: {status}")


def _write_valid_npz(repro_dir: Path) -> Path:
    npz = repro_dir / "predictions" / "pred_test.npz"
    npz.parent.mkdir(parents=True)
    np.savez(npz, scene_ids=np.asarray(["scene_001"]))
    return npz


def test_runner_records_absolute_checkpoint_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repro_dir = tmp_path / "repro"
    checkpoint_dir = tmp_path / "outputs" / "version_0" / "checkpoints"
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))

    BaseTrainingRunner()._record_ckpt_dir_pointer(checkpoint_dir)

    pointer = repro_dir / "output_dir.txt"
    assert pointer.read_text(encoding="utf-8") == f"{checkpoint_dir.resolve()}\n"


def test_runner_pointer_is_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(not_a_directory))

    BaseTrainingRunner()._record_ckpt_dir_pointer(tmp_path / "checkpoints")


def test_repro_pruner_dry_run_delete_and_keep(tmp_path: Path) -> None:
    repro_dir = tmp_path / "repro"
    checkpoint_dir = tmp_path / "outputs" / "version_0" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoints = [checkpoint_dir / "best.ckpt", checkpoint_dir / "last.ckpt"]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"checkpoint")
    npz = _write_valid_npz(repro_dir)
    (repro_dir / "output_dir.txt").write_text(f"{checkpoint_dir}\n", encoding="utf-8")

    dry_run = subprocess.run(
        [sys.executable, str(PRUNE_SCRIPT), "--repro-dir", str(repro_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "would delete 2 ckpt(s)" in dry_run.stdout
    assert all(checkpoint.exists() for checkpoint in checkpoints)

    delete = subprocess.run(
        [
            sys.executable,
            str(PRUNE_SCRIPT),
            "--repro-dir",
            str(repro_dir),
            "--delete",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert delete.returncode == 0, delete.stderr
    assert "pruned 2/2 ckpt(s)" in delete.stdout
    assert not any(checkpoint.exists() for checkpoint in checkpoints)
    assert npz.exists()

    keep_dir = tmp_path / "keep"
    keep_checkpoint_dir = tmp_path / "outputs" / "version_1" / "checkpoints"
    keep_checkpoint_dir.mkdir(parents=True)
    keep_checkpoint = keep_checkpoint_dir / "best.ckpt"
    keep_checkpoint.write_bytes(b"checkpoint")
    keep_dir.mkdir()
    (keep_dir / "output_dir.txt").write_text(
        f"{keep_checkpoint_dir}\n", encoding="utf-8"
    )
    keep = subprocess.run(
        [
            sys.executable,
            str(PRUNE_SCRIPT),
            "--repro-dir",
            str(keep_dir),
            "--delete",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert keep.returncode == 0, keep.stderr
    assert "no verified pred_test.npz" in keep.stdout
    assert keep_checkpoint.exists()


def test_queue_prunes_only_opted_in_successful_jobs(tmp_path: Path) -> None:
    repo = tmp_path / "worktree"
    repo.mkdir()
    helper = repo / "make_artifacts.py"
    helper.write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                "import numpy as np",
                "repro = Path(sys.argv[1])",
                "name = sys.argv[2]",
                "ckpt_dir = Path(sys.argv[3]) / name / 'checkpoints'",
                "ckpt_dir.mkdir(parents=True)",
                "(ckpt_dir / 'best.ckpt').write_bytes(b'checkpoint')",
                "(repro / 'predictions').mkdir(parents=True)",
                "np.savez(repro / 'predictions' / 'pred_test.npz', "
                "scene_ids=np.asarray(['scene_001']))",
                "(repro / 'output_dir.txt').write_text("
                "str(ckpt_dir.resolve()) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "make_artifacts.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    queue_dir = tmp_path / "shared-queue"
    output_dir = tmp_path / "outputs"
    env = {
        **os.environ,
        "TRAINING_QUEUE_DIR": str(queue_dir),
        "TRAINING_QUEUE_PYTHON": sys.executable,
    }

    def add(name: str, *, prune: bool, fail: bool = False) -> None:
        command = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(helper))} "
            f'"$TENNIS_REPRO_DIR" {shlex.quote(name)} '
            f"{shlex.quote(str(output_dir))}"
        )
        if fail:
            command += "; exit 7"
        args = [
            "add",
            command,
            "--name",
            name,
            "--provider",
            "claude",
            "--session",
            "session-545",
            "--issue",
            "545",
        ]
        if prune:
            args.append("--prune-ckpt")
        result = _run(repo, env, *args)
        assert result.returncode == 0, result.stderr

    add("success_prune", prune=True)
    add("success_default", prune=False)
    add("failed_prune", prune=True, fail=True)

    start = _run(repo, env, "start", "--idle-timeout", "2")
    assert start.returncode == 0, start.stderr
    _wait_for_worker(repo, env, done=2, failed=1)

    assert not (output_dir / "success_prune" / "checkpoints" / "best.ckpt").exists()
    assert (output_dir / "success_default" / "checkpoints" / "best.ckpt").exists()
    assert (output_dir / "failed_prune" / "checkpoints" / "best.ckpt").exists()

    done_jobs = sorted((queue_dir / "done").glob("*.job"))
    failed_jobs = sorted((queue_dir / "failed").glob("*.job"))
    assert len(done_jobs) == 2
    assert len(failed_jobs) == 1
    headers = {job.name: job.read_text(encoding="utf-8") for job in done_jobs}
    assert any(
        "# prune_ckpt: 1" in content and "success_prune" in name
        for name, content in headers.items()
    )
    assert any(
        "# prune_ckpt: 0" in content and "success_default" in name
        for name, content in headers.items()
    )
    assert "# prune_ckpt: 1" in failed_jobs[0].read_text(encoding="utf-8")

    repro_dirs = sorted((queue_dir / "repro").glob("*"))
    assert len(repro_dirs) == 3
    assert all((repro / "run.json").exists() for repro in repro_dirs)
    assert all(
        (repro / "predictions" / "pred_test.npz").exists() for repro in repro_dirs
    )
