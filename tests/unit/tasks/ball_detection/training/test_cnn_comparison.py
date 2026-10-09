import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from src.tasks.ball_detection.scripts.advance_cnn_campaign import advance, queue_once
from src.tasks.ball_detection.training.heatmap_pretraining.checkpoint import SCHEMA
from src.tasks.ball_detection.training.posttraining.comparison import (
    compare_pretraining,
)
from tests.unit.tasks.ball_detection.models.test_mdd_pretrain import small_config


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
                training_root=str(tmp_path / "train"), smoke_root=str(tmp_path / "smoke"), posttraining_run=str(tmp_path / "post"))
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
                training_root=str(tmp_path / "train"), smoke_root=str(tmp_path / "smoke"), posttraining_run=str(tmp_path / "post"))
    report = dict(winner="fasternet", selected_run=str(runs[2]))
    with patch("src.tasks.ball_detection.scripts.advance_cnn_campaign.compare_pretraining", return_value=report), \
         patch("src.tasks.ball_detection.scripts.advance_cnn_campaign.save_comparison"), \
         patch("src.tasks.ball_detection.scripts.advance_cnn_campaign.queue_once", return_value="queued.job") as enqueue:
        result = advance(plan, tmp_path)
    assert enqueue.call_count == 3
    assert result["winner"] == "fasternet"
    command = enqueue.call_args.kwargs["argv"]
    assert command[command.index("--pretraining-run") + 1] == str(runs[2])


def test_selection_uses_common_then_full_error_and_rejects_budget_or_seed_differences(tmp_path: Path) -> None:
    recipe = dict(epochs=1, windows_per_epoch=6, batch_size=1, seed=42, learning_rate=.0002, warmup_updates=0,
        schedule="cosine", weight_decay=.01, gradient_clip=1., manifest_sha256="same-data", selection_scope="common",
        selection_metric="macro_mean_error_px", sigma_ratio=.012, focal_gamma=2., input_contract={}, image_decode={},
        supervision="observed", test_usage="none", runtime=dict(precision="bf16", compile_mode="default"))
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
        records["convnext_v2"]["recipe"]["seed"] = 43
        with pytest.raises(ValueError, match="differs"):
            compare_pretraining(runs, tmp_path / "manifest")
        records["convnext_v2"]["recipe"]["seed"] = 42
        (runs["fasternet"] / "COMPLETED.json").write_text(json.dumps(dict(global_step=5)))
        with pytest.raises(ValueError, match="budget"):
            compare_pretraining(runs, tmp_path / "manifest")
