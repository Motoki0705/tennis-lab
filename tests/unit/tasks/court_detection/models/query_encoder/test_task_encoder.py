"""Tests for one-query task-encoder taps."""

from __future__ import annotations

import torch

from src.tasks.court_detection.configuration import CourtQueryTaskEncoderConfig
from src.tasks.court_detection.models.query_encoder.contracts import PatchTokenBatch
from src.tasks.court_detection.models.query_encoder.task_encoder import (
    CourtQueryTaskEncoder,
)


def test_task_encoder_emits_one_query_and_patch_only_unique_taps() -> None:
    config = CourtQueryTaskEncoderConfig(
        hidden_dim=32,
        depth=3,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        rope_dim=8,
        rope_theta=10000.0,
        tap_indices=(0, 2),
    )
    encoder = CourtQueryTaskEncoder(input_dim=12, config=config)
    patch_batch = PatchTokenBatch(
        tokens=torch.randn(2, 6, 12),
        original_hw=(8, 12),
        padded_hw=(8, 12),
        padding_hw=(0, 0),
        grid_hw=(2, 3),
        patch_size=4,
    )

    output = encoder(patch_batch)
    loss = output.pose_query.square().mean() + sum(
        tap.patch_tokens.square().mean() for tap in output.taps
    )
    loss.backward()

    assert encoder.pose_query.shape == (1, 1, 32)
    assert output.pose_query.shape == (2, 32)
    assert tuple(tap.layer_index for tap in output.taps) == (0, 2)
    assert [tap.patch_tokens.shape for tap in output.taps] == [
        (2, 6, 32),
        (2, 6, 32),
    ]
    assert not any("position" in name for name, _ in encoder.named_parameters())
    assert encoder.pose_query.grad is not None
