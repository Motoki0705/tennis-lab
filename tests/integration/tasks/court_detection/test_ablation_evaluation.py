"""CPU integration of strict reconstruction, split metrics and paired figures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from src.tasks.court_detection.data.bundle_state import serialize_target_bundle
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.evaluation.ablation import evaluate_checkpoint
from src.tasks.court_detection.evaluation.comparison import (
    BACKBONES,
    compare_evaluations,
)
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from tests.integration.tasks.court_detection.test_composable_pipeline import (
    _compose_mixed,
    _configure_tiny_model,
    _prepare_court_roots,
)


def test_checkpoint_evaluation_separates_real_pose_and_enforces_matched_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare_court_roots(tmp_path / "data", calibrated_pose=True)
    config = _compose_mixed(root)
    config.loss.pose.enabled = True
    for kind in ("translation", "rotation", "focal"):
        config.loss.pose[f"{kind}_weight"] = 1.0
    _configure_tiny_model(config, monkeypatch)
    data = CourtDetectionDataModule(config)
    with pytest.raises(ValueError, match="has no 'test' split"):
        data.source_eval_dataloader("tennis_court_detector", "test")
    with pytest.raises(ValueError, match="train augmentation"):
        data.source_eval_dataloader("synthetic_court", "train")
    module = CourtDetectionLightningModule(config, target_bundle=data.target_bundle_spec)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({
        "hyper_parameters": {"config": config, "target_bundle_state": serialize_target_bundle(data.target_bundle_spec)},
        "state_dict": module.state_dict(), "epoch": 0, "global_step": 0,
    }, checkpoint)
    result_dir = tmp_path / "evaluation"
    result = evaluate_checkpoint(checkpoint, result_dir, device="cpu")
    synthetic = result["splits"]["synthetic_court-test"]
    real = result["splits"]["tennis_court_detector-val"]
    assert synthetic["pose_supervised"] and not real["pose_supervised"]
    assert synthetic["count"] == real["count"] == 1
    assert synthetic["sample_ids"][0].startswith("V3:")
    assert real["sample_ids"] == ["court_val"]
    assert "pose_translation_l2_m" in synthetic["metrics"]
    assert not any("pose" in key for key in real["metrics"])
    assert len(synthetic["qualitative"][0]["files"]) == 5
    assert len(real["qualitative"][0]["files"]) == 4
    assert all((result_dir / "synthetic_court-test" / name).is_file() for name in synthetic["qualitative"][0]["files"].values())

    inputs = []
    for backbone in BACKBONES:
        destination = tmp_path / backbone
        shutil.copytree(result_dir, destination)
        copied = json.loads((destination / "evaluation.json").read_text())
        copied["config"]["model"]["encoder"]["backbone_name"] = backbone
        (destination / "evaluation.json").write_text(json.dumps(copied))
        inputs.append(destination / "evaluation.json")
    report = compare_evaluations(inputs, tmp_path / "comparison")
    assert "tennis_court_detector-val" in report.read_text()
    assert len(list(report.parent.glob("*.png"))) == 9
    modified = json.loads(inputs[-1].read_text())
    modified["splits"]["synthetic_court-test"]["sample_ids"] = ["different-sample"]
    inputs[-1].write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="conditions differ"):
        compare_evaluations(inputs, tmp_path / "invalid")
