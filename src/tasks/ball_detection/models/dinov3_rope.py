"""DINOv3 patch-token ball detector with three-axis rotary attention."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.utils.models.components import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
)
from src.utils.models.components.rope import RopeBaseLike
from src.utils.models.loading import (
    DEFAULT_DINOV3_CHECKPOINT,
    DEFAULT_DINOV3_LORA_TARGET_MODULES,
    DEFAULT_DINOV3_REPOSITORY,
    DINOv3BackboneAdapter,
    DINOv3TrainMode,
    configure_dinov3_trainability,
    load_dinov3_backbone,
)
from src.utils.models.lora import LoRAConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_spatiotemporal_positions(
    *,
    num_frames: int,
    patch_height: int,
    patch_width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build row-major integer ``(time, y, x)`` coordinates for flattened tokens."""
    if min(num_frames, patch_height, patch_width) <= 0:
        raise ValueError("num_frames, patch_height, and patch_width must be positive.")
    time = torch.arange(num_frames, device=device, dtype=torch.long)
    y_coord = torch.arange(patch_height, device=device, dtype=torch.long)
    x_coord = torch.arange(patch_width, device=device, dtype=torch.long)
    time_grid, y_grid, x_grid = torch.meshgrid(
        time,
        y_coord,
        x_coord,
        indexing="ij",
    )
    return torch.stack((time_grid, y_grid, x_grid), dim=-1).reshape(-1, 3)


class FrameSharedHeatmapHead(nn.Module):
    """Apply one spatial upsampling head independently to every time step."""

    def __init__(
        self,
        *,
        in_channels: int,
        patch_size: int,
        out_channels: int = 1,
        min_channels: int = 32,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or min_channels <= 0:
            raise ValueError("Heatmap head channel counts must be positive.")
        if patch_size <= 0 or patch_size & (patch_size - 1):
            raise ValueError("patch_size must be a positive power of two.")

        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(int(math.log2(patch_size))):
            next_channels = max(min_channels, current_channels // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        current_channels,
                        next_channels,
                        kernel_size=2,
                        stride=2,
                        bias=False,
                    ),
                    nn.GroupNorm(1, next_channels),
                    nn.GELU(),
                ]
            )
            current_channels = next_channels
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``(B*T, C, Hp, Wp)`` patch grids to dense frame logits."""
        if features.ndim != 4:
            raise ValueError(
                f"Heatmap head expects (B*T, C, Hp, Wp), got {tuple(features.shape)}."
            )
        return self.network(features)


class DINOv3RoPEBallDetector(nn.Module):
    """Bidirectional global patch-token decoder for temporal ball detection."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 1,
        num_frames: int = 1,
        image_size: tuple[int, int] = (288, 512),
        backbone_repository_path: str | Path = DEFAULT_DINOV3_REPOSITORY,
        backbone_checkpoint_path: str | Path = DEFAULT_DINOV3_CHECKPOINT,
        backbone_name: str = "dinov3_vitb16",
        backbone_strict: bool = True,
        backbone_train_mode: DINOv3TrainMode = "frozen",
        backbone_last_n_blocks: int = 0,
        backbone_lora: LoRAConfig | None = None,
        decoder_dim: int = 256,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_ffn_dim: int | None = None,
        decoder_rope_dim: int | None = None,
        decoder_rope_base: RopeBaseLike = (10_000.0, 10_000.0, 10_000.0),
        decoder_dropout: float = 0.0,
        decoder_gradient_checkpointing: bool = False,
        head_min_channels: int = 32,
        backbone: DINOv3BackboneAdapter | None = None,
    ) -> None:
        super().__init__()
        self._validate_init_args(
            in_channels=in_channels,
            num_classes=num_classes,
            num_frames=num_frames,
            image_size=image_size,
            decoder_dim=decoder_dim,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            decoder_rope_dim=decoder_rope_dim,
            decoder_dropout=decoder_dropout,
        )
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.num_frames = int(num_frames)
        self.image_size = tuple(int(value) for value in image_size)
        self.backbone_train_mode = backbone_train_mode
        self.backbone_lora = backbone_lora
        self.backbone_lora_enabled = bool(
            backbone_lora is not None and backbone_lora.enabled
        )
        self.decoder_gradient_checkpointing = bool(decoder_gradient_checkpointing)
        self.decoder_rope_base = decoder_rope_base

        self.backbone = backbone or load_dinov3_backbone(
            repository_path=backbone_repository_path,
            checkpoint_path=backbone_checkpoint_path,
            backbone_name=backbone_name,
            strict=backbone_strict,
        )
        configure_dinov3_trainability(
            self.backbone,
            train_mode=self.backbone_train_mode,
            last_n_blocks=backbone_last_n_blocks,
            lora=backbone_lora,
        )
        self.patch_size = self.backbone.patch_size
        if any(size % self.patch_size != 0 for size in self.image_size):
            raise ValueError(
                f"image_size={self.image_size} must be divisible by patch_size={self.patch_size}."
            )

        self.token_projection = nn.Linear(self.backbone.embed_dim, decoder_dim)
        self.token_dropout = nn.Dropout(decoder_dropout)
        block_config = TransformerBlockConfig(
            dim=decoder_dim,
            n_heads=decoder_heads,
            ffn_dim=decoder_ffn_dim,
            rope_dim=decoder_rope_dim,
            attn_dropout=decoder_dropout,
        )
        self.decoder = nn.ModuleList(
            TransformerBlock(block_config) for _ in range(decoder_layers)
        )
        self.decoder_norm = RMSNorm(decoder_dim)
        self.heatmap_head = FrameSharedHeatmapHead(
            in_channels=decoder_dim,
            patch_size=self.patch_size,
            out_channels=self.num_classes,
            min_channels=head_min_channels,
        )

    @staticmethod
    def _validate_init_args(
        *,
        in_channels: int,
        num_classes: int,
        num_frames: int,
        image_size: tuple[int, int],
        decoder_dim: int,
        decoder_layers: int,
        decoder_heads: int,
        decoder_rope_dim: int | None,
        decoder_dropout: float,
    ) -> None:
        if in_channels != 3:
            raise ValueError("DINOv3RoPEBallDetector requires 3-channel RGB input.")
        for name, value in (
            ("num_classes", num_classes),
            ("num_frames", num_frames),
            ("decoder_dim", decoder_dim),
            ("decoder_layers", decoder_layers),
            ("decoder_heads", decoder_heads),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("image_size must contain two positive integers.")
        if decoder_dim % decoder_heads != 0:
            raise ValueError("decoder_dim must be divisible by decoder_heads.")
        head_dim = decoder_dim // decoder_heads
        if decoder_rope_dim is not None:
            if decoder_rope_dim <= 0 or decoder_rope_dim % 2 != 0:
                raise ValueError("decoder_rope_dim must be a positive even integer.")
            if decoder_rope_dim > head_dim:
                raise ValueError(
                    "decoder_rope_dim cannot exceed the per-head dimension."
                )
        if not 0.0 <= decoder_dropout < 1.0:
            raise ValueError("decoder_dropout must be in [0, 1).")

    @classmethod
    def from_config(cls, config: DictConfig) -> DINOv3RoPEBallDetector:
        """Create the model from a composed Hydra config."""
        model_cfg = config.get("model", {}) or {}
        backbone_cfg = model_cfg.get("backbone", {}) or {}
        decoder_cfg = model_cfg.get("decoder", {}) or {}
        head_cfg = model_cfg.get("heatmap_head", {}) or {}
        image_size = _parse_pair(
            model_cfg.get("image_size", [288, 512]),
            name="model.image_size",
        )
        rope_base = _parse_rope_base(decoder_cfg.get("rope_base", 10_000.0))
        return cls(
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("num_classes", 1)),
            num_frames=int(model_cfg.get("num_frames", 1)),
            image_size=image_size,
            backbone_repository_path=backbone_cfg.get(
                "repository_path",
                DEFAULT_DINOV3_REPOSITORY,
            ),
            backbone_checkpoint_path=backbone_cfg.get(
                "checkpoint_path",
                DEFAULT_DINOV3_CHECKPOINT,
            ),
            backbone_name=str(backbone_cfg.get("name", "dinov3_vitb16")),
            backbone_strict=bool(backbone_cfg.get("strict", True)),
            backbone_train_mode=cast(
                DINOv3TrainMode,
                str(backbone_cfg.get("train_mode", "frozen")),
            ),
            backbone_last_n_blocks=int(backbone_cfg.get("last_n_blocks", 0)),
            backbone_lora=LoRAConfig.from_mapping(
                backbone_cfg.get("lora"),
                default_target_modules=DEFAULT_DINOV3_LORA_TARGET_MODULES,
            ),
            decoder_dim=int(decoder_cfg.get("dim", 256)),
            decoder_layers=int(decoder_cfg.get("num_layers", 4)),
            decoder_heads=int(decoder_cfg.get("num_heads", 8)),
            decoder_ffn_dim=_optional_int(decoder_cfg.get("ffn_dim")),
            decoder_rope_dim=_optional_int(decoder_cfg.get("rope_dim")),
            decoder_rope_base=rope_base,
            decoder_dropout=float(decoder_cfg.get("dropout", 0.0)),
            decoder_gradient_checkpointing=bool(
                decoder_cfg.get("gradient_checkpointing", False)
            ),
            head_min_channels=int(head_cfg.get("min_channels", 32)),
        )

    def train(self, mode: bool = True) -> DINOv3RoPEBallDetector:
        """Keep a frozen backbone deterministic while training the decoder."""
        super().train(mode)
        if self.backbone_train_mode == "frozen" and not self.backbone_lora_enabled:
            self.backbone.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Return logits ``(B, C, T, H, W)`` from ``(B, T, 3, H, W)`` frames."""
        self._validate_forward_input(frames)
        batch_size, num_frames, channels, height, width = frames.shape
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size

        flat_frames = frames.reshape(
            batch_size * num_frames,
            channels,
            height,
            width,
        )
        patch_tokens = self._extract_patch_tokens(flat_frames)
        expected_tokens = patch_height * patch_width
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "DINOv3 patch-token count does not match the input grid: "
                f"expected {expected_tokens}, got {patch_tokens.shape[1]}."
            )

        tokens = patch_tokens.reshape(
            batch_size,
            num_frames * expected_tokens,
            self.backbone.embed_dim,
        )
        tokens = self.token_dropout(self.token_projection(tokens))
        positions = build_spatiotemporal_positions(
            num_frames=num_frames,
            patch_height=patch_height,
            patch_width=patch_width,
            device=frames.device,
        )
        rope_dim = self.decoder[0].attn.rope_dim
        freqs_cis = precompute_freqs_cis_nd(
            dim=rope_dim,
            pos=positions,
            base=self.decoder_rope_base,
        )
        for block in self.decoder:
            if self.decoder_gradient_checkpointing and self.training:
                tokens = checkpoint(
                    block,
                    tokens,
                    freqs_cis=freqs_cis,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens, freqs_cis=freqs_cis)
        tokens = self.decoder_norm(tokens)

        patch_grids = (
            tokens.reshape(
                batch_size,
                num_frames,
                patch_height,
                patch_width,
                tokens.shape[-1],
            )
            .permute(0, 1, 4, 2, 3)
            .reshape(
                batch_size * num_frames,
                tokens.shape[-1],
                patch_height,
                patch_width,
            )
        )
        frame_logits = self.heatmap_head(patch_grids)
        if frame_logits.shape[-2:] != (height, width):
            raise RuntimeError(
                "Heatmap head output size does not match the input size: "
                f"{tuple(frame_logits.shape[-2:])} != {(height, width)}."
            )
        return (
            frame_logits.reshape(
                batch_size,
                num_frames,
                self.num_classes,
                height,
                width,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    def _extract_patch_tokens(self, flat_frames: torch.Tensor) -> torch.Tensor:
        if self.backbone_train_mode == "frozen" and not self.backbone_lora_enabled:
            with torch.no_grad():
                return self.backbone.forward_features(flat_frames)["x_norm_patchtokens"]
        return self.backbone.forward_features(flat_frames)["x_norm_patchtokens"]

    def _validate_forward_input(self, frames: torch.Tensor) -> None:
        if frames.ndim != 5:
            raise ValueError(
                "DINOv3RoPEBallDetector expects (B, T, C, H, W), "
                f"got {tuple(frames.shape)}."
            )
        if frames.shape[2] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {frames.shape[2]}."
            )
        if tuple(frames.shape[-2:]) != self.image_size:
            raise ValueError(
                f"Expected image_size={self.image_size}, got {tuple(frames.shape[-2:])}."
            )


def _parse_pair(value: Sequence[int], *, name: str) -> tuple[int, int]:
    parsed = tuple(int(item) for item in value)
    if len(parsed) != 2:
        raise ValueError(f"{name} must contain exactly two integers.")
    return parsed[0], parsed[1]


def _parse_rope_base(value: float | Sequence[float]) -> RopeBaseLike:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(float(item) for item in value)
    return float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "DINOv3RoPEBallDetector",
    "FrameSharedHeatmapHead",
    "build_spatiotemporal_positions",
]
