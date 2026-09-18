"""End-to-end CPU smoke test from issue #634 data to a trainable loss."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from src.tasks.base.model_io import bind_model_io
from src.tasks.slcs.data.dataset import SLCSDataConfig, SLCSWindowDataset, collate_slcs
from src.tasks.slcs.data.quality import QualityConfig
from src.tasks.slcs.data.splits import generate_video_splits, save_split_file
from src.tasks.slcs.evaluation.evaluate import (
    evaluate_split,
    evaluation_context,
    save_evaluation,
)
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.tasks.slcs.model_io import SLCSModelIOAdapter, SLCSModelIOSpec
from src.tasks.slcs.models.slcs_model import SLCSFusionModel
from src.tasks.slcs.training.lightning_module import SLCSLightningModule
from src.tasks.slcs.training.losses import (
    SLCSLoss,
    SLCSLossConfig,
    build_slcs_loss_inputs,
)
from tests.support.tasks.slcs.dataset import (
    DEFAULT_FIXTURE_DINO_SPEC,
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)


@pytest.mark.parametrize("num_frames", [6, 18])
def test_dataset_model_loss_backward_smoke(tmp_path: Path, num_frames: int) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "dataset",
        SLCSFixtureDatasetConfig(videos=("video_000", "video_001"), num_frames=num_frames),
    )
    split_file = index.root / "splits.json"
    assignments = generate_video_splits(index, val_ratio=0.0, test_ratio=0.0, seed=0)
    save_split_file(split_file, assignments, seed=0, val_ratio=0.0, test_ratio=0.0)
    dataset = SLCSWindowDataset(
        dataset_root=index.root,
        split_file=split_file,
        split="train",
        stride=8,
        config=SLCSDataConfig(
            window_size=8,
            train_stride=8,
            eval_stride=8,
            num_players=2,
            num_court_kp=14,
            require_dino=True,
            cache_dino_tokens=True,
            on_incomplete="error",
            dino_spec=DEFAULT_FIXTURE_DINO_SPEC,
            quality=QualityConfig(
                min_player_confidence=0.3,
                min_ball_cameras=1,
                label_weight_power=1.0,
                min_window_label_ratio=0.1,
            ),
        ),
    )
    batch = collate_slcs([dataset[0]])
    model = SLCSFusionModel(
        hidden_dim=32,
        num_shared_layers=1,
        num_position_layers=0,
        num_rotation_layers=0,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        rope_dim=8,
        rope_theta_time=10000.0,
        rope_theta_entity=10000.0,
        attention_type="mha",
        ffn_type="swiglu",
        num_players=2,
        num_court_kp=14,
        max_seq_len=8,
        invisible_init_std=0.02,
        dino_embed_dim=8,
        dino_grid_h=3,
        dino_grid_w=4,
        dino_patch_downsample_factor=1,
        dino_cross_attn_every=1,
        log_b_min=-6.0,
        log_b_max=3.0,
    )
    adapter = SLCSModelIOAdapter(
        SLCSModelIOSpec(
            num_players=2,
            num_court_kp=14,
            max_seq_len=8,
            dino_num_tokens=12,
            dino_encoded_num_tokens=12,
            dino_embed_dim=8,
            log_b_min=-6.0,
            log_b_max=3.0,
        )
    )
    adapter.validate_model(model)
    model_io = bind_model_io(model, adapter)
    call = model_io.build_call(batch)
    targets = adapter.build_training_targets(batch)
    prediction = model_io.decode_output(model_io.execute_call(call))
    terms = SLCSLoss(
        SLCSLossConfig(
            player_position_weight=1.0,
            player_rotation_weight=1.0,
            player_angle_weight=0.5,
            ball_position_weight=1.0,
            player_position_nll_weight=0.5,
            player_rotation_nll_weight=0.25,
            ball_position_nll_weight=0.5,
            player_position_smoothness_weight=1.0,
            ball_position_smoothness_weight=1.0,
            ground_penetration_weight=1.0,
            smoothness_order=3,
        )
    )(build_slcs_loss_inputs(prediction, targets))
    loss = terms["total"]
    assert torch.isfinite(loss)
    assert terms
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    # Complete one tiny CPU epoch over all remaining windows.
    for sample_index in range(1, len(dataset)):
        optimizer.zero_grad()
        training_batch = collate_slcs([dataset[sample_index]])
        training_output = model_io.run(training_batch)
        training_targets = adapter.build_training_targets(training_batch)
        epoch_loss = (training_output.player_position - training_targets.target_player_position).square().mean()
        epoch_loss.backward()
        optimizer.step()
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))

    class InferenceModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.model_adapter = adapter
            self.model_io = model_io

    predictor = SLCSPredictor(cast(SLCSLightningModule, InferenceModule()), torch.device("cpu"))
    baseline: dict[str, np.ndarray] | None = None
    for mode in ("full", "no_rgb", "detector_gap", "rgb_only"):
        report, arrays = evaluate_split(
            predictor,
            dataset_root=index.root,
            split_file=split_file,
            split="train",
            data_config=dataset.config,
            batch_size=3,
            input_mode=mode,
        )
        context = evaluation_context(checkpoint, input_mode=mode)
        metrics_path, arrays_path = save_evaluation(tmp_path / mode, report, arrays, context=context)
        assert context["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert json.loads(metrics_path.read_text())["context"] == context
        with np.load(arrays_path, allow_pickle=False) as saved:
            assert len(set(saved["scene_ids"])) == len(dataset)
            for i, meta in enumerate(dataset.metas):
                sample = dataset[i]
                for target_key in ("target_player_position", "target_player_rotation", "target_ball_position"):
                    np.testing.assert_array_equal(saved[target_key][i], sample[target_key].numpy())
                assert saved["scene_ids"][i] == dataset.prediction_ids[i]
                assert saved["video_ids"][i] == meta.video_id
                assert saved["clip_ids"][i] == meta.clip_id
                assert saved["camera_ids"][i] == meta.camera_id
                assert saved["window_start"][i] == meta.window_start
                assert saved["window_length"][i] == meta.window_length
                np.testing.assert_array_equal(saved["frame_idx"][i, :meta.window_length],
                    np.arange(meta.window_start, meta.window_start + meta.window_length))
                assert (saved["frame_idx"][i, meta.window_length:] == -1).all()
            assert np.isfinite(saved["pred_player_position"]).all()
            assert np.isfinite(saved["pred_ball_position"]).all()
            assert saved["pred_player_rotation"].shape == saved["target_player_rotation"].shape
            for key in ("target_player_position", "target_player_rotation", "target_ball_position",
                        "player_mask", "ball_mask", "player_weight", "ball_weight", "padding_mask", "frame_idx"):
                if baseline is not None:
                    np.testing.assert_array_equal(saved[key], baseline[key])
        if baseline is None:
            baseline = arrays

    with pytest.raises(ValueError, match="RGB-only evaluation requires RGB tokens"):
        evaluate_split(
            predictor,
            dataset_root=index.root,
            split_file=split_file,
            split="train",
            data_config=replace(dataset.config, require_dino=False),
            batch_size=3,
            input_mode="rgb_only",
        )


def test_real_rgb_profile_normal_fit_saves_monitored_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise production logging/checkpoint callbacks, which fast_dev_run skips."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from src.tasks.slcs.training.runner import SLCSTrainingRunner

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    index = build_slcs_dataset_fixture(
        tmp_path / "data" / "fixture",
        SLCSFixtureDatasetConfig(videos=("video_000", "video_001"), num_frames=6),
    )
    assignments = {"video_000": "train", "video_001": "val"}
    save_split_file(index.root / "splits.json", assignments, seed=0, val_ratio=0.5, test_ratio=0.0)
    config_dir = Path(__file__).parents[4] / "src/tasks/slcs/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="train_real_rgb",
            overrides=[
                f"paths.data_root={tmp_path / 'data'}",
                f"paths.output_root={tmp_path / 'outputs'}",
                "run.output_dir=slcs/train/cpu_callback_smoke/s42-001",
                "run.gpus=0",
                "data.dataset_root=fixture",
                "data.split_file=fixture/splits.json",
                "data.batch_size=1",
                "data.num_workers=0",
                "data.pin_memory=false",
                "data.window_size=8",
                "data.train_stride=8",
                "data.eval_stride=8",
                "data.dino.image_height=48",
                "data.dino.image_width=64",
                "data.dino.embed_dim=8",
                "model.hidden_dim=32",
                "model.num_shared_layers=1",
                "model.ffn_dim=64",
                "model.rope_dim=8",
                "model.dino_patch_downsample_factor=1",
                "training.trainer.max_epochs=1",
                "training.warmup_steps=0",
                "training.trainer.precision=32-true",
                "training.trainer.log_every_n_steps=1",
                "training.trainer.enable_model_summary=false",
            ],
        )
    runner = SLCSTrainingRunner()
    runtime = runner.validate_runtime_config(config)
    datamodule = runner.build_datamodule(config)
    steps = runner.resolve_steps_per_epoch(config, datamodule, train_loader=None)
    assert steps == 1
    module = runner.build_lightning_module(config, datamodule, steps_per_epoch=steps)
    logger = runner.build_logger(config, tmp_path / "outputs" / str(config.run.output_dir))
    assert isinstance(logger, TensorBoardLogger)
    callbacks = runner.build_callbacks(config, datamodule, logger)
    monitored = [c for c in callbacks if isinstance(c, ModelCheckpoint) and c.monitor is not None]
    assert len(monitored) == 1
    checkpoint = monitored[0]
    # Do not override the monitor: this must test the real profile's configured key.
    assert checkpoint.monitor == runtime.training.checkpoint.monitor
    trainer = runner.build_trainer(config, callbacks, logger)
    assert not trainer.fast_dev_run
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        trainer.fit(module, datamodule=datamodule)
    finally:
        torch.set_num_threads(old_threads)
    assert trainer.global_step == 1
    assert checkpoint.monitor in trainer.callback_metrics
    assert checkpoint.best_model_score is not None
    assert torch.isfinite(checkpoint.best_model_score)
    assert checkpoint.best_model_path
    saved_path = Path(checkpoint.best_model_path)
    assert saved_path.is_file() and saved_path.stat().st_size > 0
    saved = torch.load(saved_path, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 0 and saved["global_step"] == 1
    assert saved["optimizer_states"] and saved["state_dict"]
    logger.experiment.flush()
    hparams = OmegaConf.load(Path(logger.log_dir) / "hparams.yaml")
    assert hparams.config.training.checkpoint.monitor == checkpoint.monitor
    assert hparams.config.model.hidden_dim == 32
    events = EventAccumulator(logger.log_dir).Reload()
    assert checkpoint.monitor in events.Tags()["scalars"]
    assert np.isfinite(events.Scalars(checkpoint.monitor)[-1].value)
