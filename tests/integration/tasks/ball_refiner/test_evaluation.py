"""Real cached pilot -> checksum-bound held-aside validation -> persisted HDR report."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_refiner.data.evidence_cache import generate_evidence_cache
from src.tasks.ball_refiner.evaluation.configuration import EvaluationConfig
from src.tasks.ball_refiner.evaluation.runner import run_evaluation
from src.tasks.ball_refiner.training.runner import run_training
from tests.integration.tasks.ball_refiner.test_training import config_for
from tests.integration.tasks.ball_refiner.test_training import (
    pilot_inputs as pilot_inputs,  # shared real-detector fixture
)
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip

CONFIG = Path(__file__).resolve().parents[4] / "src/tasks/ball_refiner/configs"


@pytest.fixture(scope="module")
def completed_pilot(pilot_inputs, tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluation-pilot")
    shutil.copytree(pilot_inputs / "store", root / "store")
    for index in (2, 3):
        write_store_clip(root / "store", f"meiji/val/clip_{index:03d}/cam0",
                         [frame(i, ball()) for i in range(13)], source="meiji", split="val")
    generate_evidence_cache(store_directory=root / "store", checkpoint=pilot_inputs / "detector.ckpt", output=root / "evidence",
                            splits=("train", "val"), sources=("tracknet", "meiji", "chat_annotation"),
                            device="cpu", subpixel_refine=True, stride=2, batch_size=2,
                            candidates=BallCandidateConfig(max_candidates=3, nms_kernel=3, patch_size=3))
    cfg = config_for(root, root / "training")
    cfg.training.gap_lengths = [1]
    return run_training(cfg)


def evaluation_config(training, output):
    with initialize_config_dir(config_dir=str(CONFIG), version_base=None):
        cfg = compose(config_name="evaluate_pilot")
    cfg.paths.artifact_root = str(training.parent)
    cfg.paths.output_root = str(output.parent)
    cfg.evaluate.training_run = training.name
    cfg.evaluate.samples, cfg.evaluate.bootstrap_repetitions = 64, 20
    cfg.evaluate.chunk_size = 3
    cfg.compile.enabled = False
    cfg.run.device, cfg.run.output_dir = "cpu", output.name
    return cfg


def test_completed_checkpoint_evaluates_only_calibration_and_persists_recomputable_predictions(completed_pilot, tmp_path):
    cfg = evaluation_config(completed_pilot, tmp_path / "evaluation")
    output = run_evaluation(cfg)
    data = json.loads((completed_pilot / "data_manifest.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["partition"] == "calibration"
    assert manifest["clip_ids"] == data["validation"]["calibration"]
    assert set(manifest["clip_ids"]).isdisjoint(data["validation"]["selection"])
    assert json.loads((output / "run_state.json").read_text())["status"] == "complete"
    report = json.loads((output / "metrics.json").read_text())
    assert len(report["observed"]["temporal_camera_groups"]) == 2
    assert "paired_mean_error_delta_px" in report["observed"]["metrics"]
    assert "paired_mean_error_delta_px" not in report["evidence_gap"]["metrics"]
    observed_sites = report["observed_at_gap_length_1"]["frames"]
    assert observed_sites == report["evidence_gap_at_gap_length_1"]["frames"]
    assert observed_sites < report["observed"]["frames"]
    saved_coverage = []
    for item in manifest["artifacts"]:
        with np.load(output / item["path"], allow_pickle=False) as arrays:
            assert len(arrays["means"]) == item["frames"]
            assert len(arrays["scored_frame_index"]) == item["scored_frames"]
            assert arrays["area_px2"].shape == (item["scored_frames"], 3)
            assert np.isfinite(arrays["hdr_log_threshold_uv"]).all()
            if item["condition"] == "observed":
                saved_coverage.extend(arrays["coverage"][:, 1])
    assert report["observed"]["metrics"]["coverage_mass_0.9"]["value"] == pytest.approx(np.mean(saved_coverage))
    with pytest.raises(FileExistsError, match="already contains"):
        run_evaluation(cfg)


@pytest.mark.parametrize("filename,field,value,match", [
    ("run_state.json", "status", "training", "completed training"),
    ("best.json", "checkpoint_sha256", "0" * 64, "checksum"),
])
def test_incomplete_or_corrupted_training_is_rejected_before_output(completed_pilot, tmp_path, filename, field, value, match):
    source = tmp_path / "copy"
    shutil.copytree(completed_pilot, source)
    changed = json.loads((source / filename).read_text())
    changed[field] = value
    (source / filename).write_text(json.dumps(changed))
    cfg = evaluation_config(source, tmp_path / "rejected")
    with pytest.raises(ValueError, match=match):
        run_evaluation(cfg)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("key,value", [("partition", "test"), ("samples", 2), ("levels", [.5, .5]),
                                       ("bootstrap_repetitions", 1), ("bootstrap_confidence", float("nan"))])
def test_evaluation_configuration_rejects_test_or_invalid_statistics(tmp_path, key, value):
    cfg = evaluation_config(tmp_path / "training", tmp_path / "unused")
    cfg.evaluate[key] = value
    with pytest.raises(ValueError):
        EvaluationConfig.from_config(cfg)
