"""Contracts for the DINOv3 patch-token temporal ball detector."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from src.tasks.ball_detection.models import build_ball_detection_model
from src.tasks.ball_detection.models.dinov3_rope import (
    DINOv3RoPEBallDetector,
    build_spatiotemporal_positions,
)
from src.tasks.ball_detection.models.input_adapter import to_model_input
from src.utils.models.loading import (
    DINOv3BackboneAdapter,
    DINOv3TrainMode,
    configure_dinov3_trainability,
    load_dinov3_backbone,
)


class _FakeDINOv3(nn.Module):
    embed_dim = 24
    patch_size = 16

    def __init__(self, *, depth: int = 4) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward_features(self, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = tokens + block(tokens)
        return {"x_norm_patchtokens": self.norm(tokens)}


def _make_model(
    *,
    num_frames: int = 1,
    train_mode: DINOv3TrainMode = "frozen",
    last_n_blocks: int = 0,
    decoder_layers: int = 2,
    gradient_checkpointing: bool = False,
) -> DINOv3RoPEBallDetector:
    return DINOv3RoPEBallDetector(
        num_frames=num_frames,
        image_size=(32, 48),
        backbone=DINOv3BackboneAdapter(_FakeDINOv3()),
        backbone_train_mode=train_mode,
        backbone_last_n_blocks=last_n_blocks,
        decoder_dim=24,
        decoder_layers=decoder_layers,
        decoder_heads=4,
        decoder_ffn_dim=48,
        decoder_rope_dim=6,
        decoder_gradient_checkpointing=gradient_checkpointing,
        head_min_channels=8,
    )


def test_spatiotemporal_positions_match_token_flatten_order() -> None:
    positions = build_spatiotemporal_positions(
        num_frames=2,
        patch_height=2,
        patch_width=3,
    )
    assert positions.tolist() == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 2],
        [1, 0, 0],
        [1, 0, 1],
        [1, 0, 2],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 2],
    ]


def test_forward_shape_supports_single_and_multiple_frames() -> None:
    for num_frames in (1, 3):
        model = _make_model(num_frames=num_frames)
        logits = model(torch.randn(2, num_frames, 3, 32, 48))
        assert logits.shape == (2, 1, num_frames, 32, 48)


def test_default_resolution_head_outputs_288_by_512() -> None:
    model = DINOv3RoPEBallDetector(
        image_size=(288, 512),
        backbone=DINOv3BackboneAdapter(_FakeDINOv3()),
        decoder_dim=24,
        decoder_layers=1,
        decoder_heads=4,
        decoder_ffn_dim=48,
        decoder_rope_dim=6,
        head_min_channels=8,
    )
    logits = model(torch.randn(1, 1, 3, 288, 512))
    assert logits.shape == (1, 1, 1, 288, 512)


def test_every_decoder_layer_receives_three_axis_rope() -> None:
    model = _make_model(num_frames=2, decoder_layers=3)
    observed: list[torch.Tensor] = []
    handles = []

    def capture(
        _module: nn.Module,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        observed.append(kwargs["freqs_cis"])

    for block in model.decoder:
        handles.append(
            block.attn.register_forward_pre_hook(capture, with_kwargs=True)
        )
    try:
        model(torch.randn(1, 2, 3, 32, 48))
    finally:
        for handle in handles:
            handle.remove()

    assert len(observed) == 3
    assert all(freqs.shape == (12, 3) for freqs in observed)


def test_backward_updates_decoder_but_not_frozen_backbone() -> None:
    model = _make_model(num_frames=2, train_mode="frozen")
    model.train()
    model(torch.randn(1, 2, 3, 32, 48)).mean().backward()

    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert model.token_projection.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.heatmap_head.parameters())


def test_decoder_gradient_checkpointing_supports_backward() -> None:
    model = _make_model(
        num_frames=2,
        train_mode="full",
        gradient_checkpointing=True,
    )
    model.train()
    model(torch.randn(1, 2, 3, 32, 48)).mean().backward()
    assert model.token_projection.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.backbone.parameters())


def test_last_n_blocks_trainability_is_explicit() -> None:
    adapter = DINOv3BackboneAdapter(_FakeDINOv3(depth=4))
    configure_dinov3_trainability(
        adapter,
        train_mode="last_n_blocks",
        last_n_blocks=2,
    )
    blocks = adapter.transformer_blocks()
    assert not any(parameter.requires_grad for parameter in blocks[0].parameters())
    assert not any(parameter.requires_grad for parameter in blocks[1].parameters())
    assert all(parameter.requires_grad for parameter in blocks[2].parameters())
    assert all(parameter.requires_grad for parameter in blocks[3].parameters())
    final_norm = cast(_FakeDINOv3, adapter.module).norm
    assert all(parameter.requires_grad for parameter in final_norm.parameters())

    configure_dinov3_trainability(adapter, train_mode="full")
    assert all(parameter.requires_grad for parameter in adapter.parameters())


def test_t1_state_dict_loads_strictly_into_multiframe_model() -> None:
    phase1 = _make_model(num_frames=1)
    phase2 = _make_model(num_frames=4)
    load_result = phase2.load_state_dict(phase1.state_dict(), strict=True)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


def test_model_factory_and_btchw_input_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    config = OmegaConf.create(
        {
            "model": {
                "name": "dinov3_rope",
                "input_mode": "rgb",
                "input_layout": "btchw",
                "in_channels": 3,
                "num_classes": 1,
                "num_frames": 2,
                "image_size": [32, 48],
                "backbone": {
                    "train_mode": "frozen",
                    "last_n_blocks": 0,
                },
                "decoder": {
                    "dim": 24,
                    "num_layers": 1,
                    "num_heads": 4,
                    "ffn_dim": 48,
                    "rope_dim": 6,
                    "rope_base": [10000.0, 10000.0, 10000.0],
                },
                "heatmap_head": {"min_channels": 8},
            }
        }
    )
    monkeypatch.setattr(
        "src.tasks.ball_detection.models.dinov3_rope.load_dinov3_backbone",
        lambda **_kwargs: DINOv3BackboneAdapter(_FakeDINOv3()),
    )
    model = build_ball_detection_model(config)
    images = torch.randn(1, 2, 3, 32, 48)
    assert to_model_input(images, config.model) is images
    assert model(images).shape == (1, 1, 2, 32, 48)


def test_shared_loader_validates_checkpoint_strictly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "dinov3"
    repository.mkdir()
    checkpoint_path = tmp_path / "backbone.pth"
    expected = _FakeDINOv3()
    torch.save({"model": expected.state_dict()}, checkpoint_path)
    monkeypatch.setattr(
        torch.hub,
        "load",
        lambda *_args, **_kwargs: _FakeDINOv3(),
    )

    loaded = load_dinov3_backbone(
        repository_path=repository,
        checkpoint_path=checkpoint_path,
        strict=True,
    )
    assert loaded.embed_dim == 24
    assert loaded.patch_size == 16

    invalid_checkpoint = tmp_path / "invalid.pth"
    torch.save({"model": {}}, invalid_checkpoint)
    with pytest.raises(RuntimeError):
        load_dinov3_backbone(
            repository_path=repository,
            checkpoint_path=invalid_checkpoint,
            strict=True,
        )
