"""End-to-end CPU smoke test from issue #634 data to a trainable loss."""

from __future__ import annotations

from pathlib import Path

import torch

from src.tasks.base.model_io import bind_model_io
from src.tasks.slcs.data.dataset import SLCSDataConfig, SLCSWindowDataset, collate_slcs
from src.tasks.slcs.data.quality import QualityConfig
from src.tasks.slcs.data.splits import generate_recording_splits, save_split_file
from src.tasks.slcs.model_io import SLCSModelIOAdapter, SLCSModelIOSpec
from src.tasks.slcs.models.slcs_model import SLCSFusionModel
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


def test_dataset_model_loss_backward_smoke(tmp_path: Path) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "dataset",
        SLCSFixtureDatasetConfig(recordings=("recording",), num_frames=8),
    )
    split_file = index.root / "splits.json"
    assignments = generate_recording_splits(index, val_ratio=0.0, test_ratio=0.0, seed=0)
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
