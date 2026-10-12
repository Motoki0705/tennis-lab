import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from src.tasks.ball_detection.checkpoints.pretraining import SCHEMA
from src.tasks.ball_detection.evaluation.comparison import (
    compare_pretraining,
)
from tests.unit.tasks.ball_detection.models.mdd.test_cnn import small_config


def test_comparison_refuses_partial_or_undeclared_cnn_sets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="three"):
        compare_pretraining({"residual": tmp_path}, tmp_path / "manifest.json")
    runs = {name: tmp_path / name for name in ("residual", "convnext_v2", "fasternet")}
    with pytest.raises(FileNotFoundError):
        compare_pretraining(runs, tmp_path / "manifest.json")




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

    with patch("src.tasks.ball_detection.evaluation.comparison.completed_pretraining", side_effect=completed):
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
