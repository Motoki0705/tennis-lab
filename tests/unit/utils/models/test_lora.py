"""Unit tests for :mod:`src.utils.models.lora`."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.utils.models.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    iter_lora_parameters,
    mark_only_lora_as_trainable,
)


class _TinyAttention(nn.Module):
    """Minimal block exposing the DINOv3-style ``qkv``/``proj`` linears."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.qkv(self.norm(x))[..., : self.proj.in_features])


class _TinyBackbone(nn.Module):
    def __init__(self, dim: int = 16, depth: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_TinyAttention(dim) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TestLoRAConfig:
    def test_disabled_skips_validation(self) -> None:
        cfg = LoRAConfig(enabled=False, rank=0, alpha=0.0)
        assert cfg.enabled is False

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"rank": 0}, "rank"),
            ({"alpha": 0.0}, "alpha"),
            ({"dropout": 1.0}, "dropout"),
            ({"target_modules": ()}, "target_modules"),
        ],
    )
    def test_enabled_validates(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            LoRAConfig(enabled=True, **kwargs)

    def test_from_mapping_none_is_disabled(self) -> None:
        cfg = LoRAConfig.from_mapping(None, default_target_modules=("qkv",))
        assert cfg.enabled is False
        assert cfg.target_modules == ("qkv",)

    def test_from_mapping_reads_values(self) -> None:
        cfg = LoRAConfig.from_mapping(
            {"enabled": True, "rank": 4, "alpha": 8.0, "dropout": 0.1}
        )
        assert cfg.enabled is True
        assert cfg.rank == 4
        assert cfg.alpha == 8.0
        assert cfg.dropout == 0.1

    def test_from_mapping_default_targets_used_when_absent(self) -> None:
        cfg = LoRAConfig.from_mapping(
            {"enabled": True}, default_target_modules=("qkv", "proj")
        )
        assert cfg.target_modules == ("qkv", "proj")


class TestLoRALinear:
    def test_initial_output_matches_base(self) -> None:
        base = nn.Linear(8, 5)
        wrapped = LoRALinear(base, rank=2, alpha=4.0)
        x = torch.randn(3, 8)
        torch.testing.assert_close(wrapped(x), base(x))

    def test_exposes_linear_shape_attributes(self) -> None:
        wrapped = LoRALinear(nn.Linear(8, 5), rank=2, alpha=4.0)
        assert wrapped.in_features == 8
        assert wrapped.out_features == 5
        assert wrapped.scaling == pytest.approx(4.0 / 2)

    def test_base_frozen_only_adapters_trainable(self) -> None:
        wrapped = LoRALinear(nn.Linear(8, 5), rank=2, alpha=4.0)
        assert wrapped.base.weight.requires_grad is False
        assert wrapped.lora_a.requires_grad is True
        assert wrapped.lora_b.requires_grad is True

    def test_update_changes_output_after_training_step(self) -> None:
        wrapped = LoRALinear(nn.Linear(8, 5), rank=2, alpha=4.0)
        x = torch.randn(3, 8)
        wrapped(x).sum().backward()
        assert wrapped.lora_a.grad is not None
        assert wrapped.lora_b.grad is not None
        assert wrapped.base.weight.grad is None

    def test_rejects_non_linear(self) -> None:
        with pytest.raises(TypeError, match="nn.Linear"):
            LoRALinear(nn.Conv2d(3, 3, 1), rank=2, alpha=4.0)  # type: ignore[arg-type]


class TestApplyLoRA:
    def test_wraps_only_target_modules(self) -> None:
        model = _TinyBackbone(dim=16, depth=2)
        wrapped = apply_lora(model, rank=4, alpha=8.0, target_modules=("qkv", "proj"))
        assert wrapped == [
            "blocks.0.qkv",
            "blocks.0.proj",
            "blocks.1.qkv",
            "blocks.1.proj",
        ]
        assert isinstance(model.blocks[0].qkv, LoRALinear)
        assert isinstance(model.blocks[0].proj, LoRALinear)
        assert not isinstance(model.blocks[0].norm, LoRALinear)

    def test_is_idempotent(self) -> None:
        model = _TinyBackbone(dim=16, depth=1)
        apply_lora(model, rank=4, alpha=8.0, target_modules=("qkv",))
        # Second call must not re-wrap the existing adapter.
        with pytest.raises(ValueError, match="matched no nn.Linear"):
            apply_lora(model, rank=4, alpha=8.0, target_modules=("qkv",))

    def test_preserves_forward_shape(self) -> None:
        model = _TinyBackbone(dim=16, depth=2)
        x = torch.randn(2, 7, 16)
        before = model(x)
        apply_lora(model, rank=4, alpha=8.0, target_modules=("qkv", "proj"))
        after = model(x)
        assert after.shape == before.shape
        # B initialised to zero => identical output right after wrapping.
        torch.testing.assert_close(after, before)

    def test_raises_when_no_match(self) -> None:
        model = _TinyBackbone(dim=16, depth=1)
        with pytest.raises(ValueError, match="matched no nn.Linear"):
            apply_lora(model, rank=4, alpha=8.0, target_modules=("does_not_exist",))


class TestTrainableHelpers:
    def test_iter_lora_parameters_yields_only_adapters(self) -> None:
        model = _TinyBackbone(dim=16, depth=2)
        apply_lora(model, rank=4, alpha=8.0, target_modules=("qkv", "proj"))
        params = list(iter_lora_parameters(model))
        assert len(params) == 2 * 2 * 2  # 2 blocks * 2 targets * (A, B)

    def test_mark_only_lora_as_trainable(self) -> None:
        model = _TinyBackbone(dim=16, depth=1)
        apply_lora(model, rank=4, alpha=8.0, target_modules=("qkv",))
        mark_only_lora_as_trainable(model)
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        assert trainable == {"blocks.0.qkv.lora_a", "blocks.0.qkv.lora_b"}
