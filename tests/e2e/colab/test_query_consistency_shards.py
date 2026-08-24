"""Command-level tests for the distributed Issue #790 encoder shards."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir

from scripts.colab.setup.enqueue_query_consistency_shard import (
    RuntimeRelocation,
    build_shard_plan,
)
from src.tasks.court_detection.experiments.query_consistency import (
    QueryConsistencyAblationConfig,
    build_query_consistency_manifest,
)

ROOT = Path(__file__).parents[3]
CONFIG_DIR = ROOT / "src/tasks/court_detection/configs"
ENQUEUE_MODULE = "scripts.colab.setup.enqueue_query_consistency_shard"
TRAIN_DIR = ROOT / "scripts/colab/train/2026-08-24"
COLAB_SCRIPTS = (
    TRAIN_DIR / "train_court_query_consistency_encoder_shard.sh",
    TRAIN_DIR / "train_court_query_consistency_encoder_colab0.sh",
    TRAIN_DIR / "train_court_query_consistency_encoder_colab1.sh",
)


def _manifest() -> dict[str, object]:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="run_query_consistency_ablation")
    return cast(
        dict[str, object],
        build_query_consistency_manifest(
            QueryConsistencyAblationConfig.from_config(config)
        ),
    )


def _runtime(tmp_path: Path) -> RuntimeRelocation:
    return RuntimeRelocation(
        python_executable=Path(sys.executable).resolve(),
        data_root=(tmp_path / "data").resolve(),
        external_asset_root=(tmp_path / "third_party").resolve(),
        output_root=(tmp_path / "outputs").resolve(),
        checkpoint_root=(tmp_path / "checkpoints").resolve(),
    )


def _write_fake_queue(tmp_path: Path) -> tuple[Path, Path]:
    capture = tmp_path / "queue-calls.txt"
    queue_script = tmp_path / "training_queue.sh"
    queue_script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
    status)
        printf '%s\n' 'worker: stopped' 'queued=0 running=0 done=3 failed=1'
        ;;
    add)
        {
            printf '%s\n' '__CALL__'
            printf '%s\n' "$@"
        } >> "${QUEUE_CAPTURE:?}"
        ;;
    *)
        exit 80
        ;;
esac
""",
        encoding="utf-8",
    )
    queue_script.chmod(0o755)
    return queue_script, capture


@pytest.mark.parametrize("script", COLAB_SCRIPTS)
def test_issue790_colab_scripts_have_valid_bash_syntax(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_wrappers_freeze_the_requested_colab_seed_mapping() -> None:
    colab0 = COLAB_SCRIPTS[1].read_text(encoding="utf-8")
    colab1 = COLAB_SCRIPTS[2].read_text(encoding="utf-8")
    shard = COLAB_SCRIPTS[0].read_text(encoding="utf-8")

    assert "colab-0 43" in colab0
    assert "colab-1 44" in colab1
    assert "colab-0 44" not in colab0
    assert "colab-1 43" not in colab1
    assert "queued=0 running=0 done=0 failed=0" in shard
    assert "--job-kind both" in shard
    assert "queued=0 running=0 done=8 failed=0" in shard


def test_training_shard_relocates_only_runtime_roots_and_keeps_depth_order(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    runtime = _runtime(tmp_path)
    plan = build_shard_plan(manifest, seeds=(43,), job_kind="train", runtime=runtime)
    jobs = cast(list[dict[str, object]], plan["jobs"])

    assert plan["source_manifest_sha256"] == manifest["manifest_sha256"]
    assert [job["run_id"] for job in jobs] == [
        "encoder-depth-01-seed-43",
        "encoder-depth-02-seed-43",
        "encoder-depth-04-seed-43",
        "encoder-depth-08-seed-43",
    ]
    for job in jobs:
        argv = cast(list[str], job["argv"])
        assert argv[0] == str(runtime.python_executable)
        assert f"paths.data_root={runtime.data_root}" in argv
        assert f"paths.external_asset_root={runtime.external_asset_root}" in argv
        assert f"paths.output_root={runtime.output_root}" in argv
        assert f"paths.checkpoint_root={runtime.checkpoint_root}" in argv
        assert "training.trainer.max_epochs=15" in argv
        assert "data/augmentation=pose_safe" in argv
        assert "loss=query_joint_both" in argv
        assert f"run.seed={job['seed']}" in argv


def test_local_profile_plan_covers_all_seeds_on_one_runtime(tmp_path: Path) -> None:
    plan = build_shard_plan(
        _manifest(),
        seeds=(42, 43, 44),
        job_kind="profile",
        runtime=_runtime(tmp_path),
    )
    jobs = cast(list[dict[str, object]], plan["jobs"])

    assert len(jobs) == 12
    assert {cast(int, job["seed"]) for job in jobs} == {42, 43, 44}
    for job in jobs:
        argv = cast(list[str], job["argv"])
        assert not any(token.startswith("paths.data_root=") for token in argv)
        assert any(token.startswith("paths.output_root=") for token in argv)
        assert "profile.device=cuda" in argv


def test_enqueue_cli_writes_bound_plan_and_four_attributed_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    queue_script, capture = _write_fake_queue(tmp_path)
    queue_dir = (tmp_path / "queue").resolve()
    runtime = _runtime(tmp_path)
    monkeypatch.setenv("QUEUE_CAPTURE", str(capture))

    subprocess.run(
        [
            sys.executable,
            "-m",
            ENQUEUE_MODULE,
            "--manifest",
            str(manifest_path),
            "--seed",
            "44",
            "--job-kind",
            "train",
            "--python-executable",
            str(runtime.python_executable),
            "--data-root",
            str(runtime.data_root),
            "--external-asset-root",
            str(runtime.external_asset_root),
            "--output-root",
            str(runtime.output_root),
            "--checkpoint-root",
            str(runtime.checkpoint_root),
            "--queue-script",
            str(queue_script),
            "--queue-dir",
            str(queue_dir),
            "--repository-root",
            str(ROOT),
            "--provider",
            "codex",
            "--session",
            "issue790-test",
            "--issue",
            "790",
        ],
        cwd=ROOT,
        check=True,
        env=dict(os.environ),
    )

    captured = capture.read_text(encoding="utf-8")
    assert captured.count("__CALL__") == 4
    assert captured.count("--provider\ncodex\n") == 4
    assert captured.count("--session\nissue790-test\n") == 4
    assert captured.count("--issue\n790\n") == 4
    plan = json.loads((queue_dir / "shard-plan.json").read_text(encoding="utf-8"))
    assert plan["seeds"] == [44]
    assert len(plan["jobs"]) == 4
