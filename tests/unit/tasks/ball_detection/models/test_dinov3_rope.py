from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.tasks.ball_detection.model_io.adapters import (
    BallModelIOAdapter,
    DINOv3BallExecutionBoundary,
)
from src.tasks.ball_detection.model_io.contracts import (
    BallModelInputSpec,
    BallModelIOError,
)
from src.tasks.ball_detection.models.dinov3_rope import DINOv3RoPEBallDetector
from src.tasks.base.model_io import bind_model_io
from src.utils.models.loading import DINOv3BackboneAdapter
from src.utils.models.lora import LoRAConfig


class _FakeDINOv3(nn.Module):
    embed_dim = 8
    patch_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity()])
        self.grad_enabled: bool | None = None
        self.invalid_response = False

    def forward_features(self, inputs: torch.Tensor) -> object:
        self.grad_enabled = torch.is_grad_enabled()
        if self.invalid_response:
            return {}
        token_count = (inputs.shape[-2] // self.patch_size) * (
            inputs.shape[-1] // self.patch_size
        )
        return {
            "x_norm_patchtokens": inputs.new_zeros(
                inputs.shape[0],
                token_count,
                self.embed_dim,
            )
        }


def _model(
    fake: _FakeDINOv3,
    *,
    gradient_checkpointing: bool = False,
) -> DINOv3RoPEBallDetector:
    return DINOv3RoPEBallDetector(
        backbone_repository_path=Path("/unused/dinov3"),
        backbone_checkpoint_path=Path("/unused/dinov3/checkpoint.pth"),
        in_channels=3,
        num_classes=1,
        num_frames=2,
        image_size=(8, 8),
        backbone_name="fake",
        backbone_strict=True,
        backbone_train_mode="frozen",
        backbone_last_n_blocks=0,
        backbone_lora=LoRAConfig(
            enabled=False,
            rank=2,
            alpha=2.0,
            dropout=0.0,
            target_modules=("qkv",),
        ),
        decoder_dim=8,
        decoder_layers=1,
        decoder_heads=2,
        decoder_head_dim=4,
        decoder_ffn_dim=16,
        decoder_rope_dim=4,
        decoder_rope_base=(10000.0, 10000.0, 10000.0),
        decoder_dropout=0.0,
        decoder_attention_type="mha",
        decoder_n_kv_heads=None,
        decoder_ffn_type="swiglu",
        decoder_gradient_checkpointing=gradient_checkpointing,
        head_min_channels=2,
        backbone=DINOv3BackboneAdapter(fake),
    )


def _pair(
    model: DINOv3RoPEBallDetector,
) -> tuple[DINOv3RoPEBallDetector, BallModelIOAdapter]:
    adapter = BallModelIOAdapter(
        BallModelInputSpec(
            model_name="dinov3_rope",
            input_mode="rgb",
            input_layout="btchw",
            in_channels=3,
            num_classes=1,
            configured_frames=2,
            image_size_hw=(8, 8),
            minimum_spatial_size=None,
            mdd_gain=1.0,
            mdd_offset=0.0,
        ),
        expected_model_type=DINOv3RoPEBallDetector,
        minimum_frames=1,
        execution_boundary=DINOv3BallExecutionBoundary(frozen_backbone=True),
    )
    adapter.validate_model_pair(model)
    return model, adapter


def test_dinov3_boundary_prepares_tokens_and_valid_shape_forward() -> None:
    fake = _FakeDINOv3()
    model, adapter = _pair(_model(fake))
    pair = bind_model_io(model, adapter)

    probability = pair.run(torch.zeros(1, 2, 3, 8, 8))

    assert fake.grad_enabled is False
    assert probability.shape == (1, 2, 8, 8)
    assert torch.all((probability >= 0.0) & (probability <= 1.0))


def test_invalid_dinov3_response_fails_before_detector_forward() -> None:
    fake = _FakeDINOv3()
    fake.invalid_response = True
    model, adapter = _pair(_model(fake))
    pair = bind_model_io(model, adapter)
    model_entries = 0

    def count_entry(
        module: nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        nonlocal model_entries
        _ = (module, args)
        model_entries += 1

    model.register_forward_pre_hook(count_entry)

    with pytest.raises(BallModelIOError, match="missing required x_norm_patchtokens"):
        pair.run(torch.zeros(1, 2, 3, 8, 8))

    assert model_entries == 0


def test_decoder_checkpoint_execution_is_selected_on_mode_change() -> None:
    model = _model(_FakeDINOv3(), gradient_checkpointing=True)

    assert model._decoder_block_executor.__name__ == "_checkpoint_decoder_block"
    model.eval()
    assert model._decoder_block_executor.__name__ == "_run_decoder_block"
    model.train()
    assert model._decoder_block_executor.__name__ == "_checkpoint_decoder_block"
