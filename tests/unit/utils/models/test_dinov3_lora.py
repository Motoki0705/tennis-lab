"""Tests for LoRA wiring in :mod:`src.utils.models.loading.dinov3`.

These exercise the trainability/LoRA helpers against a tiny fake backbone so
they stay fast and do not require the vendored DINOv3 weights.
"""

from __future__ import annotations

import pytest
from torch import nn

from src.utils.models.loading.dinov3 import (
    DINOv3BackboneAdapter,
    apply_dinov3_lora,
    configure_dinov3_trainability,
)
from src.utils.models.lora import LoRAConfig, LoRALinear


class _FakeBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3)
        self.attn.proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * 2)
        self.mlp.fc2 = nn.Linear(dim * 2, dim)


class _FakeDINOv3(nn.Module):
    """Mimic the attributes the adapter and LoRA helpers rely on."""

    def __init__(self, dim: int = 16, depth: int = 2) -> None:
        super().__init__()
        self.embed_dim = dim
        self.patch_size = 16
        self.blocks = nn.ModuleList(_FakeBlock(dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)


def _make_adapter(dim: int = 16, depth: int = 2) -> DINOv3BackboneAdapter:
    return DINOv3BackboneAdapter(_FakeDINOv3(dim=dim, depth=depth))


class TestApplyDINOv3LoRA:
    def test_requires_enabled_config(self) -> None:
        with pytest.raises(ValueError, match="enabled"):
            apply_dinov3_lora(_make_adapter(), LoRAConfig(enabled=False))

    def test_freezes_base_and_wraps_targets(self) -> None:
        adapter = _make_adapter(depth=2)
        wrapped = apply_dinov3_lora(
            adapter,
            LoRAConfig(
                enabled=True,
                rank=4,
                alpha=8.0,
                target_modules=("qkv", "proj", "fc1", "fc2"),
            ),
        )
        assert len(wrapped) == 2 * 4  # 2 blocks * 4 target linears
        assert isinstance(adapter.module.blocks[0].attn.qkv, LoRALinear)
        assert isinstance(adapter.module.blocks[0].mlp.fc2, LoRALinear)

        trainable = {name for name, p in adapter.named_parameters() if p.requires_grad}
        assert trainable
        assert all(name.endswith(("lora_a", "lora_b")) for name in trainable)


class TestConfigureTrainabilityWithLoRA:
    def test_lora_overrides_train_mode(self) -> None:
        adapter = _make_adapter(depth=1)
        configure_dinov3_trainability(
            adapter,
            train_mode="frozen",
            lora=LoRAConfig(enabled=True, rank=2, alpha=4.0, target_modules=("qkv",)),
        )
        assert isinstance(adapter.module.blocks[0].attn.qkv, LoRALinear)
        trainable = {name for name, p in adapter.named_parameters() if p.requires_grad}
        assert trainable == {
            "module.blocks.0.attn.qkv.lora_a",
            "module.blocks.0.attn.qkv.lora_b",
        }

    def test_disabled_lora_falls_back_to_train_mode(self) -> None:
        adapter = _make_adapter(depth=1)
        configure_dinov3_trainability(
            adapter,
            train_mode="full",
            lora=LoRAConfig(enabled=False),
        )
        assert not isinstance(adapter.module.blocks[0].attn.qkv, LoRALinear)
        assert all(p.requires_grad for p in adapter.parameters())
