"""DINOv3 backbone owned by the additive Court query variant."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from torch import Tensor, nn

from src.tasks.court_detection.configuration import CourtQueryBackboneConfig
from src.utils.models.loading import (
    DINOv3BackboneAdapter,
    DINOv3TrainMode,
    configure_dinov3_trainability,
    load_dinov3_backbone,
)
from src.utils.models.lora import LoRAConfig


class CourtQueryDINOv3Backbone(nn.Module):
    """Load/train DINOv3 but expose only its normalized patch-token call."""

    def __init__(
        self,
        config: CourtQueryBackboneConfig,
        *,
        backbone: DINOv3BackboneAdapter,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.train_mode = config.train_mode
        self.lora_enabled = config.lora.enabled
        domain_lora = config.lora
        lora = LoRAConfig(
            enabled=domain_lora.enabled,
            rank=domain_lora.rank,
            alpha=domain_lora.alpha,
            dropout=domain_lora.dropout,
            target_modules=domain_lora.target_modules,
        )
        train_mode: DINOv3TrainMode = (
            "last_n_blocks" if config.train_mode == "last_n" else config.train_mode
        )
        configure_dinov3_trainability(
            self.backbone,
            train_mode=train_mode,
            last_n_blocks=config.last_n_blocks,
            lora=lora,
        )
        dynamic_module = cast(Any, self.backbone.module)
        forward_features = dynamic_module.forward_features
        if not callable(forward_features):
            raise TypeError("DINOv3 backbone must expose callable forward_features.")
        self._forward_features = cast(Callable[[Tensor], object], forward_features)

    @classmethod
    def from_config(
        cls,
        config: CourtQueryBackboneConfig,
    ) -> CourtQueryDINOv3Backbone:
        backbone = load_dinov3_backbone(
            repository_path=config.repository_path,
            checkpoint_path=config.checkpoint_path,
            backbone_name=config.backbone_name,
            strict=config.strict,
        )
        return cls(config, backbone=backbone)

    @property
    def embed_dim(self) -> int:
        return cast(int, self.backbone.embed_dim)

    @property
    def patch_size(self) -> int:
        return cast(int, self.backbone.patch_size)

    @property
    def frozen_execution(self) -> bool:
        return self.train_mode == "frozen" and not self.lora_enabled

    def execute_patch_features(self, images: Tensor) -> object:
        """Execute the dynamic API; the model-I/O boundary validates its result."""
        return cast(Any, self._forward_features)(images)

    def train(self, mode: bool = True) -> CourtQueryDINOv3Backbone:
        super().train(mode)
        if self.frozen_execution:
            self.backbone.eval()
        return self


__all__ = ["CourtQueryDINOv3Backbone"]
