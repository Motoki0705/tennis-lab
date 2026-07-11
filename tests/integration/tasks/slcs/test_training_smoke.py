"""End-to-end CPU smoke test from issue #634 data to a trainable loss."""

from __future__ import annotations

from pathlib import Path

import torch

from src.tasks.slcs.data.dataset import SLCSDataConfig, SLCSWindowDataset, collate_slcs
from src.tasks.slcs.data.splits import generate_recording_splits, save_split_file
from src.tasks.slcs.data.synthetic import (
    DEFAULT_TEST_DINO_SPEC,
    SyntheticDatasetConfig,
    build_synthetic_dataset,
)
from src.tasks.slcs.models.slcs_model import SLCSFusionModel
from src.tasks.slcs.training.losses import SLCSLoss


def test_dataset_model_loss_backward_smoke(tmp_path: Path) -> None:
    index = build_synthetic_dataset(
        tmp_path / "dataset",
        SyntheticDatasetConfig(recordings=("recording",), num_frames=8),
    )
    split_file = index.root / "splits.json"
    assignments = generate_recording_splits(index, val_ratio=0.0, test_ratio=0.0, seed=0)
    save_split_file(split_file, assignments, seed=0, val_ratio=0.0, test_ratio=0.0)
    dataset = SLCSWindowDataset(
        dataset_root=index.root,
        split_file=split_file,
        split="train",
        config=SLCSDataConfig(
            window_size=8,
            train_stride=8,
            eval_stride=8,
            dino_spec=DEFAULT_TEST_DINO_SPEC,
        ),
    )
    batch = collate_slcs([dataset[0]])
    model = SLCSFusionModel(
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_seq_len=8,
        dino_embed_dim=8,
        dino_grid_h=3,
        dino_grid_w=4,
        dino_cross_attn_every=1,
    )
    prediction = model(
        **{
            key: batch[key]
            for key in (
                "player_kp",
                "player_kp_vis",
                "player_valid",
                "ball_uv",
                "ball_vis",
                "court_kp",
                "court_vis",
                "frame_mask",
                "dino_tokens",
                "dino_frame_idx",
                "dino_valid",
            )
        }
    )
    terms = SLCSLoss()(prediction, batch)
    loss = terms["total"]
    assert torch.isfinite(loss)
    assert terms
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
