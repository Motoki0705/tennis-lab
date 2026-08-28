"""DINOv3 patch-token ball detector with three-axis rotary attention."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.tasks.ball_detection.configuration import BallRuntimePaths, validate_model
from src.utils.models.components import (
    RMSNorm,
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.models.components.ffn_layers import FFNType
from src.utils.models.loading import (
    DINOv3BackboneAdapter,
    DINOv3TrainMode,
    configure_dinov3_trainability,
    load_dinov3_backbone,
)
from src.utils.models.lora import LoRAConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig


DecoderBlockExecutor = Callable[
    [nn.Module, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def _run_decoder_block(
    block: nn.Module,
    tokens: torch.Tensor,
    freqs_cis: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        block(tokens, freqs_cis=freqs_cis, attn_mask=attn_mask),
    )


def _checkpoint_decoder_block(
    block: nn.Module,
    tokens: torch.Tensor,
    freqs_cis: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        checkpoint(
            block,
            tokens,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            use_reentrant=False,
        ),
    )


def build_spatiotemporal_positions(
    *,
    num_frames: int,
    patch_height: int,
    patch_width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build row-major integer ``(time, y, x)`` coordinates for flattened tokens."""
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
        out_channels: int,
        min_channels: int,
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
        return cast(torch.Tensor, self.network(features))


class DINOv3RoPEBallDetector(nn.Module):
    """Bidirectional global patch-token decoder for temporal ball detection."""

    def __init__(
        self,
        *,
        backbone_repository_path: str | Path,
        backbone_checkpoint_path: str | Path,
        in_channels: int,
        num_classes: int,
        num_frames: int,
        image_size: tuple[int, int],
        backbone_name: str,
        backbone_strict: bool,
        backbone_train_mode: DINOv3TrainMode,
        backbone_last_n_blocks: int,
        backbone_lora: LoRAConfig | None,
        decoder_dim: int,
        decoder_layers: int,
        decoder_heads: int,
        decoder_head_dim: int,
        decoder_ffn_dim: int,
        decoder_rope_dim: int,
        decoder_rope_base: float | tuple[float, ...],
        decoder_dropout: float,
        decoder_attention_type: Literal["mha", "gqa"],
        decoder_n_kv_heads: int | None,
        decoder_ffn_type: FFNType,
        decoder_gradient_checkpointing: bool,
        head_min_channels: int,
        backbone: DINOv3BackboneAdapter | None = None,
    ) -> None:
        super().__init__()
        repository_path = Path(backbone_repository_path)
        checkpoint_path = Path(backbone_checkpoint_path)
        if not repository_path.is_absolute() or not checkpoint_path.is_absolute():
            raise ValueError(
                "DINOv3 repository and checkpoint paths must be absolute; "
                f"got repository={repository_path}, checkpoint={checkpoint_path}."
            )
        self._validate_init_args(
            in_channels=in_channels,
            num_classes=num_classes,
            num_frames=num_frames,
            image_size=image_size,
            decoder_dim=decoder_dim,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            decoder_head_dim=decoder_head_dim,
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
        self._decoder_block_executor: DecoderBlockExecutor = _run_decoder_block

        self.backbone = backbone or load_dinov3_backbone(
            repository_path=repository_path,
            checkpoint_path=checkpoint_path,
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
            head_dim=decoder_head_dim,
            rope_dim=decoder_rope_dim,
            attn_dropout=decoder_dropout,
            attention_type=decoder_attention_type,
            n_kv_heads=decoder_n_kv_heads,
            rope_base=float(
                decoder_rope_base[0]
                if isinstance(decoder_rope_base, tuple)
                else decoder_rope_base
            ),
            ffn_type=decoder_ffn_type,
        )
        self.decoder = nn.ModuleList(
            TransformerBlock(block_config) for _ in range(decoder_layers)
        )
        self.decoder_rope_dim = decoder_rope_dim
        self.decoder_rope = RotaryFrequencyComputer(
            dim=decoder_rope_dim,
            base=decoder_rope_base,
            n_axes=3,
        )
        self.decoder_norm = RMSNorm(decoder_dim)
        self.heatmap_head = FrameSharedHeatmapHead(
            in_channels=decoder_dim,
            patch_size=self.patch_size,
            out_channels=self.num_classes,
            min_channels=head_min_channels,
        )
        self.train(self.training)

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
        decoder_head_dim: int,
        decoder_rope_dim: int,
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
        if decoder_head_dim != decoder_dim // decoder_heads:
            raise ValueError(
                "decoder_head_dim must equal decoder_dim // decoder_heads."
            )
        if decoder_rope_dim <= 0 or decoder_rope_dim % 2 != 0:
            raise ValueError("decoder_rope_dim must be a positive even integer.")
        if decoder_rope_dim > decoder_head_dim:
            raise ValueError("decoder_rope_dim cannot exceed the per-head dimension.")
        if not 0.0 <= decoder_dropout < 1.0:
            raise ValueError("decoder_dropout must be in [0, 1).")

    @classmethod
    def from_config(cls, config: DictConfig) -> DINOv3RoPEBallDetector:
        """Create the model from a composed Hydra config."""
        model_cfg = validate_model(config)
        backbone_cfg = cast(dict[str, object], model_cfg["backbone"])
        decoder_cfg = cast(dict[str, object], model_cfg["decoder"])
        head_cfg = cast(dict[str, object], model_cfg["heatmap_head"])
        paths = BallRuntimePaths.from_config(config)
        image_size = _parse_pair(
            cast(Sequence[int], model_cfg["image_size"]),
            name="model.image_size",
        )
        rope_base = _parse_rope_base(
            cast(float | Sequence[float], decoder_cfg["rope_base"])
        )
        return cls(
            in_channels=int(cast(int, model_cfg["in_channels"])),
            num_classes=int(cast(int, model_cfg["num_classes"])),
            num_frames=int(cast(int, model_cfg["num_frames"])),
            image_size=image_size,
            backbone_repository_path=paths.external_asset(
                str(backbone_cfg["repository_path"])
            ),
            backbone_checkpoint_path=paths.external_asset(
                str(backbone_cfg["checkpoint_path"])
            ),
            backbone_name=str(backbone_cfg["name"]),
            backbone_strict=bool(backbone_cfg["strict"]),
            backbone_train_mode=cast(
                DINOv3TrainMode,
                str(backbone_cfg["train_mode"]),
            ),
            backbone_last_n_blocks=int(cast(int, backbone_cfg["last_n_blocks"])),
            backbone_lora=LoRAConfig.from_mapping(
                cast(dict[str, Any], backbone_cfg["lora"]),
            ),
            decoder_dim=int(cast(int, decoder_cfg["dim"])),
            decoder_layers=int(cast(int, decoder_cfg["num_layers"])),
            decoder_heads=int(cast(int, decoder_cfg["num_heads"])),
            decoder_head_dim=int(cast(int, decoder_cfg["head_dim"])),
            decoder_ffn_dim=int(cast(int, decoder_cfg["ffn_dim"])),
            decoder_rope_dim=int(cast(int, decoder_cfg["rope_dim"])),
            decoder_rope_base=rope_base,
            decoder_dropout=float(cast(float, decoder_cfg["dropout"])),
            decoder_attention_type=cast(
                Literal["mha", "gqa"], decoder_cfg["attention_type"]
            ),
            decoder_n_kv_heads=cast(int | None, decoder_cfg["n_kv_heads"]),
            decoder_ffn_type=cast(FFNType, decoder_cfg["ffn_type"]),
            decoder_gradient_checkpointing=bool(decoder_cfg["gradient_checkpointing"]),
            head_min_channels=int(cast(int, head_cfg["min_channels"])),
        )

    def train(self, mode: bool = True) -> DINOv3RoPEBallDetector:
        """Resolve train/eval execution once when module mode changes."""
        super().train(mode)
        if self.backbone_train_mode == "frozen" and not self.backbone_lora_enabled:
            self.backbone.eval()
        self._decoder_block_executor = (
            _checkpoint_decoder_block
            if self.decoder_gradient_checkpointing and mode
            else _run_decoder_block
        )
        return self

    def forward(
        self,
        frames: torch.Tensor,
        patch_tokens: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode boundary-prepared DINO tokens into dense ball logits."""
        batch_size, num_frames, _channels, height, width = frames.shape
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        expected_tokens = patch_height * patch_width
        tokens = patch_tokens.reshape(
            batch_size,
            num_frames * expected_tokens,
            self.backbone.embed_dim,
        )
        tokens = self.token_dropout(self.token_projection(tokens))
        for block in self.decoder:
            tokens = self._decoder_block_executor(
                block,
                tokens,
                freqs_cis,
                attn_mask,
            )
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
        output = (
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
        return cast(torch.Tensor, output)


def _parse_pair(value: Sequence[int], *, name: str) -> tuple[int, int]:
    parsed = tuple(int(item) for item in value)
    if len(parsed) != 2:
        raise ValueError(f"{name} must contain exactly two integers.")
    return parsed[0], parsed[1]


def _parse_rope_base(
    value: float | Sequence[float],
) -> float | tuple[float, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(float(item) for item in value)
    return float(value)


__all__ = [
    "DINOv3RoPEBallDetector",
    "FrameSharedHeatmapHead",
    "build_spatiotemporal_positions",
]
