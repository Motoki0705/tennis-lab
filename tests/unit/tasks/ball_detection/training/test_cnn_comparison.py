import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from src.tasks.ball_detection.scripts.advance_cnn_campaign import (
    advance,
    normalize_plan,
    queue_once,
)
from src.tasks.ball_detection.training.heatmap_pretraining.checkpoint import SCHEMA
from src.tasks.ball_detection.training.posttraining.comparison import (
    compare_pretraining,
)
from tests.unit.tasks.ball_detection.models.test_mdd_pretrain import small_config


def candidates() -> dict[str, dict[str, str]]:
    return {v: dict(run_id=f"{v}-s42-v1", job_name=f"i1050-{v}-s42-u60000", smoke_id=v, prefetch_mode="overlap")
            for v in ("convnext_v2", "fasternet")}


def test_queue_reuses_an_existing_job_without_resubmitting(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    (queue / "running").mkdir(parents=True)
    existing = queue / "running" / "123_456_i1050-case.job"
    existing.write_text("running")
    with patch("subprocess.run", side_effect=AssertionError("must not resubmit")):
        job = queue_once(dict(queue_directory=str(queue)), name="i1050-case", argv=[], issue=1050)
    assert job == existing.name


def test_posttraining_is_not_enqueued_before_all_pretraining_finishes(tmp_path: Path) -> None:
    plan = dict(code_root=str(tmp_path), baseline_run=str(tmp_path / "baseline"), manifest="manifest",
                training_root=str(tmp_path / "train"), smoke_root=str(tmp_path / "smoke"), posttraining_run=str(tmp_path / "post"),
                candidates=candidates(), posttraining_prefetch_mode="serial")
    with patch("src.tasks.ball_detection.scripts.advance_cnn_campaign.queue_once", return_value="queued.job") as enqueue:
        result = advance(plan, tmp_path)
    assert enqueue.call_count == 2
    assert set(result["pending"]) == {"residual", "convnext_v2", "fasternet"}
    assert "posttraining_job" not in result


def test_comparison_refuses_partial_or_undeclared_cnn_sets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="three"):
        compare_pretraining({"residual": tmp_path}, tmp_path / "manifest.json")
    runs = {name: tmp_path / name for name in ("residual", "convnext_v2", "fasternet")}
    with pytest.raises(FileNotFoundError):
        compare_pretraining(runs, tmp_path / "manifest.json")


def test_only_selected_cnn_is_used_for_posttraining(tmp_path: Path) -> None:
    runs = [tmp_path / "baseline", tmp_path / "train/convnext_v2-s42-v1", tmp_path / "train/fasternet-s42-v1"]
    for run in runs:
        run.mkdir(parents=True)
        (run / "COMPLETED.json").write_text(json.dumps({}))
    plan = dict(code_root=str(tmp_path), baseline_run=str(runs[0]), manifest="manifest",
                training_root=str(tmp_path / "train"), smoke_root=str(tmp_path / "smoke"), posttraining_run=str(tmp_path / "post"),
                candidates=candidates(), posttraining_prefetch_mode="serial")
    report = dict(winner="fasternet", selected_run=str(runs[2]))
    with patch("src.tasks.ball_detection.scripts.advance_cnn_campaign.compare_pretraining", return_value=report), \
         patch("src.tasks.ball_detection.scripts.advance_cnn_campaign.save_comparison"), \
         patch("src.tasks.ball_detection.scripts.advance_cnn_campaign.queue_once", return_value="queued.job") as enqueue:
        result = advance(plan, tmp_path)
    assert enqueue.call_count == 3
    assert result["winner"] == "fasternet"
    command = enqueue.call_args.kwargs["argv"]
    assert command[command.index("--pretraining-run") + 1] == str(runs[2])
    assert "--image-prefetch" not in command


def test_selection_uses_common_then_full_error_and_rejects_budget_or_seed_differences(tmp_path: Path) -> None:
    recipe: dict[str, Any] = dict(epochs=1, windows_per_epoch=6, batch_size=1, seed=42, learning_rate=.0002, warmup_updates=0,
        schedule="cosine", weight_decay=.01, gradient_clip=1., manifest_sha256="same-data", selection_scope="common",
        selection_metric="macro_mean_error_px", sigma_ratio=.012, focal_gamma=2., input_contract={}, image_decode={},
        supervision="observed", test_usage="none", runtime=dict(precision="bf16", compile_mode="default", image_prefetch=True))
    runs = {v: tmp_path / v for v in ("residual", "convnext_v2", "fasternet")}
    records: dict[str, dict[str, Any]] = {}
    for v, common, full in (("residual", 4., 3.), ("convnext_v2", 2., 3.), ("fasternet", 2., 1.)):
        runs[v].mkdir()
        (runs[v] / "COMPLETED.json").write_text(json.dumps(dict(global_step=6)))
        (runs[v] / "train.jsonl").write_text(json.dumps(dict(epoch=0, windows=6, train_seconds=2.)) + "\n")
        records[v] = dict(schema=SCHEMA, model_config=asdict(replace(small_config(), encoder_variant=v)),
            recipe=dict(recipe), epoch=0, state_dict={"weights": torch.zeros(4)},
            validation=dict(scopes=dict(common=dict(macro_mean_error_px=common), full=dict(macro_mean_error_px=full))))

    def completed(run: Path, manifest: Path) -> tuple[Path, dict]:
        return run / "epoch-000.pt", records[run.name]

    with patch("src.tasks.ball_detection.training.posttraining.comparison.completed_pretraining", side_effect=completed):
        report = compare_pretraining(runs, tmp_path / "manifest")
        assert report["winner"] == "fasternet"
        records["convnext_v2"]["recipe"]["runtime"] = dict(recipe["runtime"], image_prefetch=False)
        report = compare_pretraining(runs, tmp_path / "manifest")
        assert report["same_image_prefetch"] is False
        assert "not a CNN-only comparison" in report["limitation"]
        records["convnext_v2"]["recipe"]["seed"] = 43
        with pytest.raises(ValueError, match="differs"):
            compare_pretraining(runs, tmp_path / "manifest")
        records["convnext_v2"]["recipe"]["seed"] = 42
        (runs["fasternet"] / "COMPLETED.json").write_text(json.dumps(dict(global_step=5)))
        with pytest.raises(ValueError, match="budget"):
            compare_pretraining(runs, tmp_path / "manifest")


def test_recovery_uses_new_identity_and_reuses_the_other_running_job(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    (queue / "failed").mkdir(parents=True)
    (queue / "running").mkdir()
    (queue / "failed/1_i1050-convnext_v2-s42-u60000.job").write_text("failed")
    existing = queue / "running/2_i1050-fasternet-s42-u60000.job"
    existing.write_text("running")
    specs = candidates()
    specs["convnext_v2"] = dict(run_id="convnext_v2-s42-v2-serial", job_name="i1050-convnext_v2-s42-serial-v2",
                                smoke_id="convnext_v2-serial-v2", prefetch_mode="serial")
    plan = dict(code_root=str(tmp_path), baseline_run=str(tmp_path / "baseline"), manifest="manifest", thread_id="test",
                queue_directory=str(queue), training_root=str(tmp_path / "train"), smoke_root=str(tmp_path / "smoke"),
                posttraining_run=str(tmp_path / "post"), candidates=specs, posttraining_prefetch_mode="serial")
    with patch("subprocess.run") as execute:
        execute.return_value.stdout = "queued: 3_i1050-convnext_v2-s42-serial-v2.job"
        result = advance(plan, tmp_path)
    assert execute.call_count == 1
    assert result["candidate_jobs"]["fasternet"] == existing.name
    command = execute.call_args.args[0]
    assert "convnext_v2-s42-v2-serial" in command[3]
    assert "--prefetch-mode serial" in command[3]
    with pytest.raises(RuntimeError, match="inspection"):
        queue_once(plan, name="i1050-convnext_v2-s42-u60000", argv=[], issue=1050)


def test_plan_v1_has_explicit_migration_and_v2_rejects_unsafe_ids() -> None:
    plan = dict(schema="mdd_cnn_campaign.v1", code_root="code", code_commit="hash", queue_directory="queue",
                manifest="manifest", manifest_sha256="hash", baseline_run="baseline", training_root="train",
                smoke_root="smoke", posttraining_run="post", thread_id="thread", model_config_sha256={})
    normalized = normalize_plan(plan)
    assert normalized["candidates"] == candidates()
    assert normalized["posttraining_prefetch_mode"] == "overlap"
    normalized["candidates"]["convnext_v2"]["run_id"] = "../escape"
    with pytest.raises(ValueError, match="plain names"):
        normalize_plan(normalized)
