"""Command-level tests for the distributed Issue #790 DPT shards."""

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
    enqueue_shard_plan,
)
from src.tasks.court_detection.experiments.query_consistency import (
    QueryConsistencyAblationConfig,
    build_query_consistency_manifest,
)

ROOT = Path(__file__).parents[3]
CONFIG_DIR = ROOT / "src/tasks/court_detection/configs"
ENQUEUE_MODULE = "scripts.colab.setup.enqueue_query_consistency_shard"
TRAIN_DIR = ROOT / "scripts/colab/train/2026-08-25"
COLAB_SCRIPTS = (
    TRAIN_DIR / "train_court_query_scaling_grid_shard.sh",
    TRAIN_DIR / "train_court_query_scaling_grid_colab1.sh",
    TRAIN_DIR / "train_court_query_scaling_grid_colab2.sh",
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


def _write_fake_queue(tmp_path: Path, *, status: str = "queued=0 running=0 done=0 failed=0") -> tuple[Path, Path]:
    capture = tmp_path / "queue-calls.txt"
    queue_script = tmp_path / "training_queue.sh"
    queue_script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
    status)
        printf '%s\\n' 'worker: stopped' "${QUEUE_STATUS}"
        ;;
    add)
        {
            printf '%s\\n' '__CALL__'
            printf '%s\\n' "$@"
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


def test_colab_wrappers_select_the_two_manifest_halves() -> None:
    colab1 = COLAB_SCRIPTS[1].read_text(encoding="utf-8")
    colab2 = COLAB_SCRIPTS[2].read_text(encoding="utf-8")
    shard = COLAB_SCRIPTS[0].read_text(encoding="utf-8")
    assert "exec bash" in colab1 and "colab1" in colab1
    assert "exec bash" in colab2 and "colab2" in colab2
    assert 'START_INDEX=0' in shard and 'END_INDEX=12' in shard
    assert 'START_INDEX=12' in shard and 'END_INDEX=24' in shard
    assert "run_query_consistency_ablation" in shard
    assert "enqueue_query_consistency_shard" in shard


def test_training_shard_relocates_only_runtime_roots_and_keeps_grid_order(
    tmp_path: Path,
) -> None:
    plan = build_shard_plan(
        _manifest(), seeds=(42,), job_kind="train", runtime=_runtime(tmp_path), start_index=0, end_index=12
    )
    jobs = cast(list[dict[str, object]], plan["jobs"])
    assert len(jobs) == 12
    assert [job["run_id"] for job in jobs[:4]] == [
        "scaling-input-256-depth-01-dpt-tiny-seed-42",
        "scaling-input-256-depth-01-dpt-small-seed-42",
        "scaling-input-256-depth-01-dpt-base-seed-42",
        "scaling-input-256-depth-01-dpt-large-seed-42",
    ]
    for job in jobs:
        argv = cast(list[str], job["argv"])
        assert argv[0] == str(_runtime(tmp_path).python_executable)
        assert f"paths.data_root={_runtime(tmp_path).data_root}" in argv
        assert f"paths.output_root={_runtime(tmp_path).output_root}" in argv
        assert f"paths.checkpoint_root={_runtime(tmp_path).checkpoint_root}" in argv
        assert "training.trainer.max_epochs=15" in argv
        assert "data/augmentation=pose_safe" in argv
        assert "model.task_encoder.depth=1" in argv or "model.task_encoder.depth=8" in argv


def test_enqueue_cli_writes_bound_plan_and_twelve_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    queue_script, capture = _write_fake_queue(tmp_path)
    queue_dir = (tmp_path / "queue").resolve()
    runtime = _runtime(tmp_path)
    monkeypatch.setenv("QUEUE_CAPTURE", str(capture))
    monkeypatch.setenv("QUEUE_STATUS", "queued=0 running=0 done=0 failed=0")
    subprocess.run(
        [
            sys.executable,
            "-m",
            ENQUEUE_MODULE,
            "--manifest",
            str(manifest_path),
            "--seed",
            "42",
            "--job-kind",
            "train",
            "--start-index",
            "12",
            "--end-index",
            "24",
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
            "colab",
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
    assert captured.count("__CALL__") == 12
    plan = json.loads((queue_dir / "shard-plan.json").read_text(encoding="utf-8"))
    assert plan["seeds"] == [42]
    assert plan["start_index"] == 12
    assert plan["end_index"] == 24
    assert len(plan["jobs"]) == 12


def test_enqueue_rejects_nonfresh_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    queue_script, _ = _write_fake_queue(
        tmp_path, status="queued=0 running=0 done=1 failed=0"
    )
    monkeypatch.setenv("QUEUE_STATUS", "queued=0 running=0 done=1 failed=0")
    with pytest.raises(RuntimeError, match="fresh and stopped"):
        enqueue_shard_plan(
            build_shard_plan(_manifest(), seeds=(42,), job_kind="train", runtime=runtime, end_index=1),
            queue_script=queue_script,
            queue_dir=(tmp_path / "queue").resolve(),
            repository_root=ROOT,
            provider="colab",
            session="test",
            issue=790,
        )
