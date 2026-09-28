"""Tiny real detector cache -> pilot -> checkpoint and unique-frame validation."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair
from src.tasks.ball_refiner.data.evidence_cache import generate_evidence_cache
from src.tasks.ball_refiner.refiner_2d import build_ball_refiner_2d
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.tasks.ball_refiner.training.evaluation import (
    metric_rows,
    predict_clip,
    summarize_rows,
)
from src.tasks.ball_refiner.training.runner import prepare_data, run_training
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip

CONFIG = Path(__file__).resolve().parents[4] / "src/tasks/ball_refiner/configs"


@pytest.fixture(scope="module")
def pilot_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("pilot")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    cfg = OmegaConf.create({
        "model": {"name": "conv_next_unet", "input_mode": "rgb", "in_channels": 3, "num_classes": 1,
                  "num_frames": 4, "input_layout": "bcthw", "dims": [4, 8, 16, 32], "depth": 1,
                  "drop_path_prob": 0.0, "mdd_a": 0.2, "mdd_b": 0.15},
        "data": {"image_size": [32, 32], "augmentation": {"normalize_imagenet": {
            "enabled": False, "mean": [0.1, 0.2, 0.3], "std": [0.5, 0.6, 0.7],
        }}},
    })
    pair = build_ball_detection_pair(cfg)
    checkpoint = root / "detector.ckpt"
    torch.save({"hyper_parameters": {"config": OmegaConf.to_container(cfg)},
                "state_dict": {f"model.{key}": value for key, value in pair.model.state_dict().items()}}, checkpoint)
    for source in ("tracknet", "meiji", "chat_annotation"):
        write_store_clip(root / "store", f"{source}/train/clip", [
            frame(i, ball("out_of_frame", None) if i == 0 else ball()) for i in range(15)
        ], source=source, split="train")
        for index in range(2):
            write_store_clip(root / "store", f"{source}/val/clip_{index:03d}/cam0", [
                frame(i, *([] if i == 0 else [ball()])) for i in range(13)
            ], source=source, split="val")
    write_store_clip(root / "store", "tracknet/short/clip", [frame(i, ball()) for i in range(6)])
    generate_evidence_cache(store_directory=root / "store", checkpoint=checkpoint, output=root / "evidence",
                            splits=("train", "val"), sources=("tracknet", "meiji", "chat_annotation"),
                            device="cpu", subpixel_refine=True, stride=2, batch_size=2,
                            candidates=BallCandidateConfig(max_candidates=3, nms_kernel=3, patch_size=3))
    yield root
    torch.set_num_threads(previous)


def config_for(root, output):
    with initialize_config_dir(config_dir=str(CONFIG), version_base=None):
        cfg = compose(config_name="train")
    cfg.paths.data_root = cfg.paths.cache_root = str(root)
    cfg.paths.output_root = str(output.parent)
    cfg.data.store, cfg.data.evidence = "store", "evidence"
    cfg.data.window_length, cfg.data.stride = 9, 4
    cfg.training.gap_lengths = [1, 4]
    cfg.training.batch_size, cfg.training.steps_per_epoch = 2, 2
    cfg.training.epochs, cfg.training.max_steps = 3, 2
    cfg.model.hidden_dim, cfg.model.attention_heads, cfg.model.patch_size = 16, 2, 3
    cfg.model.dropout = 0.0
    cfg.compile.enabled = False
    cfg.run.output_dir, cfg.run.device = output.name, "cpu"
    return cfg


def test_real_cache_training_selects_and_restores_checkpoint(pilot_inputs, tmp_path):
    cfg = config_for(pilot_inputs, tmp_path / "training")
    runtime = PilotConfig.from_config(cfg)
    output = run_training(cfg)
    state = json.loads((output / "run_state.json").read_text())
    assert state["status"] == "complete" and state["steps"] == 2
    assert state["best_epoch"] == 0  # step cap takes precedence over epochs
    manifest = json.loads((output / "data_manifest.json").read_text())
    assert manifest["excluded"] == [{"clip_id": "tracknet/short/clip", "reason": "short_clip", "frames": 6}]
    assert len(manifest["validation"]["calibration"]) == 1
    best = json.loads((output / "best.json").read_text())
    checkpoint = torch.load(output / best["checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["schema"] == "ball_refiner_2d_checkpoint.v1"
    pair = build_ball_refiner_2d(runtime.model)
    pair.model.load_state_dict(checkpoint["state_dict"], strict=True)
    torch.manual_seed(runtime.seed)
    initial = build_ball_refiner_2d(runtime.model)
    assert any(not torch.equal(value, initial.model.state_dict()[key]) for key, value in checkpoint["state_dict"].items())
    _, clips, _ = prepare_data(runtime)
    predicted = predict_clip(pair, clips[0], runtime, device=torch.device("cpu"), gap=np.zeros(13, dtype=np.bool_))
    file = output / best["predictions"] / f"clip-{clips[0].record.index:05d}-observed.npz"
    with np.load(file) as saved:
        np.testing.assert_allclose(predicted.means[0], saved["means"], atol=1e-6)
        assert saved["means"].shape[0] == 13
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["observed"]["position_frames"] == 12
    assert metrics["observed"]["negative_frames"] == 0
    assert metrics["evidence_gap"]["position_frames"] < 12
    assert np.isfinite(metrics["selection_nll_uv"])
    with pytest.raises(FileExistsError, match="already contains"):
        run_training(cfg)


def test_dry_run_does_not_build_model_and_only_loads_selection(pilot_inputs, tmp_path, monkeypatch):
    import src.tasks.ball_refiner.training.runner as runner
    cfg = config_for(pilot_inputs, tmp_path / "preflight")
    cfg.run.dry_run = True
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not create a model")
    monkeypatch.setattr(runner, "build_ball_refiner_2d", forbidden)
    output = run_training(cfg)
    assert json.loads((output / "run_state.json").read_text())["status"] == "dry_run_complete"
    dataset, selection, manifest = prepare_data(PilotConfig.from_config(cfg))
    assert all(clip.record.split == "train" for clip in dataset.clips)
    assert set(clip.record.clip_id for clip in selection) == set(manifest["validation"]["selection"])
    assert set(manifest["validation"]["calibration"]).isdisjoint(clip.record.clip_id for clip in selection)


def test_unique_frame_predictions_and_metric_aggregation_ignore_batch_partition(pilot_inputs, tmp_path):
    runtime = PilotConfig.from_config(config_for(pilot_inputs, tmp_path / "unused"))
    _, selection, _ = prepare_data(runtime)
    pair = build_ball_refiner_2d(runtime.model)
    clip = selection[0]
    gap = np.zeros(clip.record.frame_count, dtype=np.bool_)
    a = predict_clip(pair, clip, runtime, device=torch.device("cpu"), gap=gap)
    b = predict_clip(pair, clip, replace(runtime, training=replace(runtime.training, batch_size=1)), device=torch.device("cpu"), gap=gap)
    torch.testing.assert_close(a.means, b.means, atol=1e-6, rtol=1e-5)
    mask = np.ones_like(gap)
    first = mask.copy()
    first[5:] = False
    full = summarize_rows([metric_rows(a, clip, mask)])
    split = summarize_rows([metric_rows(a, clip, first), metric_rows(a, clip, mask & ~first)])
    assert full == split
    expected_jacobian = np.log((clip.record.source_width - 1) * (clip.record.source_height - 1))
    assert full["position_nll_px"] - full["position_nll_uv"] == pytest.approx(expected_jacobian, abs=1e-5)


@pytest.mark.parametrize("key,value", [
    ("model.use_pose", True), ("data.short_clip_policy", "pad"),
    ("training.max_steps", 0), ("training.gap_lengths", [9]),
    ("training.learning_rate", float("nan")), ("data.sources", ["meiji", "meiji"]),
    ("compile.mode", "reduce-overhead"), ("run.device", "auto"),
])
def test_invalid_pilot_configuration_fails_before_output(pilot_inputs, tmp_path, key, value):
    cfg = config_for(pilot_inputs, tmp_path / "rejected")
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        run_training(cfg)
    assert not (tmp_path / "rejected").exists()


def test_unknown_or_missing_training_key_is_rejected(pilot_inputs, tmp_path):
    cfg = config_for(pilot_inputs, tmp_path / "unused")
    del cfg.training.max_steps
    with pytest.raises(ValueError):
        PilotConfig.from_config(cfg)
    cfg = config_for(pilot_inputs, tmp_path / "unused")
    OmegaConf.set_struct(cfg, False)
    cfg.training.typo = 2
    with pytest.raises(ValueError):
        PilotConfig.from_config(cfg)
