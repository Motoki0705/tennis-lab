"""DINOv3 + spatial Transformer + multiscale DPT, dense and camera-pose heads."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.tasks.court_detection.configuration import (
    DPT_CHANNELS_BY_SIZE,
    CourtDecoderConfig,
    CourtDenseHeadBranchConfig,
    CourtDenseHeadConfig,
    CourtEncoderConfig,
    CourtModelConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
)
from src.tasks.court_detection.geometry.pose import POSE10D_RAW_ORDER
from src.utils.models.blocks import Conv2dWiseWiseBlock
from src.utils.models.components import (
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.models.components.ffn_layers import (
    SUPPORTED_FFN_TYPES,
    FFNType,
    default_ffn_dim,
)
from src.utils.models.loading import (
    DINOv3BackboneAdapter,
    DINOv3TrainMode,
    configure_dinov3_trainability,
    load_dinov3_backbone,
)
from src.utils.models.lora import LoRAConfig

IntermediateLayerMode = Literal["uniform", "last"]


class CourtDINOv3Encoder(nn.Module):
    """DINOv3 ViT encoder that exposes DPT-style multi-layer feature maps."""

    requires_prepared_features = True

    def __init__(
        self,
        *,
        out_indices: Sequence[int],
        in_channels: int,
        repository_path: str | Path | None,
        checkpoint_path: str | Path | None,
        backbone_name: str | None,
        strict: bool | None,
        train_mode: DINOv3TrainMode,
        last_n_blocks: int,
        lora: LoRAConfig,
        layer_mode: IntermediateLayerMode,
        backbone: DINOv3BackboneAdapter | None,
    ) -> None:
        super().__init__()
        if backbone is None and None in (
            repository_path,
            checkpoint_path,
            backbone_name,
            strict,
        ):
            raise ValueError(
                "CourtDINOv3Encoder requires explicit DINOv3 asset settings."
            )
        self._validate_init_args(
            in_channels=in_channels,
            out_indices=out_indices,
            layer_mode=layer_mode,
        )
        self.in_channels = int(in_channels)
        self.train_mode = train_mode
        self.lora_enabled = lora.enabled
        self.layer_mode = layer_mode

        if backbone is None:
            assert repository_path is not None and checkpoint_path is not None
            assert backbone_name is not None and strict is not None
            backbone = load_dinov3_backbone(
                repository_path=Path(repository_path),
                checkpoint_path=Path(checkpoint_path),
                backbone_name=backbone_name,
                strict=strict,
            )
        self.backbone = backbone
        configure_dinov3_trainability(
            self.backbone,
            train_mode=train_mode,
            last_n_blocks=last_n_blocks,
            lora=lora,
        )
        self.patch_size = self.backbone.patch_size
        self.feature_channels = tuple([self.backbone.embed_dim] * 4)
        self.out_indices = tuple(out_indices)
        get_intermediate_layers = getattr(
            self.backbone.module,
            "get_intermediate_layers",
            None,
        )
        if not callable(get_intermediate_layers):
            raise TypeError(
                "DINOv3 backbone must expose get_intermediate_layers for DPT decoding."
            )
        self._get_intermediate_layers = cast(
            Callable[..., tuple[torch.Tensor, ...]],
            get_intermediate_layers,
        )

    @staticmethod
    def _validate_init_args(
        *,
        in_channels: int,
        out_indices: Sequence[int] | None,
        layer_mode: str,
    ) -> None:
        if in_channels != 3:
            raise ValueError("CourtDINOv3Encoder requires 3-channel RGB input.")
        if layer_mode not in {"uniform", "last"}:
            raise ValueError("layer_mode must be one of ['uniform', 'last'].")
        if out_indices is not None and len(tuple(out_indices)) != 4:
            raise ValueError("out_indices must contain exactly four layer indices.")

    def train(self, mode: bool = True) -> CourtDINOv3Encoder:
        """Keep a frozen backbone deterministic while training the decoder."""
        super().train(mode)
        if self.train_mode == "frozen" and not self.lora_enabled:
            self.backbone.eval()
        return self


def _build_dinov3_encoder(
    *, in_channels: int, config: CourtEncoderConfig
) -> CourtDINOv3Encoder:
    if None in (
        config.repository_path,
        config.checkpoint_path,
        config.backbone_name,
        config.strict,
        config.train_mode,
        config.last_n_blocks,
        config.out_indices,
        config.layer_mode,
        config.lora,
    ):
        raise AssertionError("Validated DINOv3 encoder configuration is incomplete.")
    domain_lora = config.lora
    assert domain_lora is not None
    lora = LoRAConfig(
        enabled=domain_lora.enabled,
        rank=domain_lora.rank,
        alpha=domain_lora.alpha,
        dropout=domain_lora.dropout,
        target_modules=domain_lora.target_modules,
    )
    return CourtDINOv3Encoder(
        in_channels=in_channels,
        repository_path=cast("Path", config.repository_path),
        checkpoint_path=cast("Path", config.checkpoint_path),
        backbone_name=cast("str", config.backbone_name),
        strict=cast("bool", config.strict),
        train_mode=cast("DINOv3TrainMode", config.train_mode),
        last_n_blocks=cast("int", config.last_n_blocks),
        lora=lora,
        out_indices=cast("tuple[int, ...]", config.out_indices),
        layer_mode=cast("IntermediateLayerMode", config.layer_mode),
        backbone=None,
    )


def build_court_encoder(
    *,
    config: CourtEncoderConfig,
    in_channels: int,
) -> CourtDINOv3Encoder:
    """Build the requested court encoder."""

    if config.name != "dinov3":
        raise ValueError("Court detection only supports the DINOv3 encoder.")
    return _build_dinov3_encoder(in_channels=in_channels, config=config)


class DPTFeatureFusionBlock(nn.Module):
    """RefineNet-style residual fusion block used by DPT decoders."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        self.skip_block = Conv2dWiseWiseBlock(channels, channels)
        self.output_block = Conv2dWiseWiseBlock(channels, channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = x + self.skip_block(skip)
        return cast("torch.Tensor", self.output_block(x))


class CourtDPTDecoder(nn.Module):
    """DPT decoder for ViT features reassembled at multiple image scales."""

    def __init__(
        self,
        *,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        reassemble_factors: Sequence[float],
    ) -> None:
        super().__init__()
        self.encoder_channels = tuple(int(channel) for channel in encoder_channels)
        self.decoder_channels = int(decoder_channels)
        self.reassemble_factors = tuple(float(factor) for factor in reassemble_factors)
        self._validate_init_args(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,
            reassemble_factors=self.reassemble_factors,
        )

        self.output_channels = self.decoder_channels
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, self.decoder_channels, kernel_size=1, bias=False
                ),
                nn.GroupNorm(1, self.decoder_channels),
                nn.GELU(),
            )
            for in_channels in self.encoder_channels
        )
        self.reassembly = nn.ModuleList(
            nn.Identity()
            if factor == 1.0
            else nn.Upsample(
                scale_factor=factor,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            for factor in self.reassemble_factors
        )
        self.fusion_blocks = nn.ModuleList(
            DPTFeatureFusionBlock(self.decoder_channels)
            for _ in range(len(self.encoder_channels))
        )

    @staticmethod
    def _validate_init_args(
        *,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        reassemble_factors: Sequence[float],
    ) -> None:
        if len(encoder_channels) != 4:
            raise ValueError(
                "CourtDPTDecoder expects four encoder feature levels, "
                f"got {len(encoder_channels)}."
            )
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive.")
        if len(reassemble_factors) != 4:
            raise ValueError("reassemble_factors must contain exactly four values.")
        if any(factor <= 0.0 for factor in reassemble_factors):
            raise ValueError("reassemble_factors must be positive.")

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        projected_feats = [
            _apply_tensor_module(
                reassemble,
                _apply_tensor_module(projection, feat),
            )
            for projection, reassemble, feat in zip(
                self.projections,
                self.reassembly,
                feats,
                strict=True,
            )
        ]

        deepest_fusion = cast("DPTFeatureFusionBlock", self.fusion_blocks[-1])
        x = deepest_fusion.output_block(projected_feats[-1])
        for block, skip in zip(
            reversed(self.fusion_blocks[:-1]),
            reversed(projected_feats[:-1]),
            strict=True,
        ):
            x = cast("DPTFeatureFusionBlock", block)(x, skip)
        return cast("torch.Tensor", x)


def _apply_tensor_module(module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Invoke a module whose configured contract is tensor-to-tensor."""
    return cast("torch.Tensor", module(tensor))


def build_court_decoder(
    *,
    config: CourtDecoderConfig,
    encoder_channels: Sequence[int],
) -> CourtDPTDecoder:
    """Build one decoder from the strict typed Court decoder contract."""

    if config.name == "dpt":
        if config.size is None:
            raise ValueError("DPT decoder requires an explicit size preset.")
        decoder_channels = _parse_decoder_channel_scalar(config.channels)
        expected_channels = DPT_CHANNELS_BY_SIZE[config.size]
        if decoder_channels != expected_channels:
            raise ValueError(
                "DPT decoder channels disagree with its size preset: "
                f"size={config.size!r} requires {expected_channels}, "
                f"got {decoder_channels}."
            )
        return CourtDPTDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            reassemble_factors=_require_reassemble_factors(config.reassemble_factors),
        )
    raise ValueError(f"Unsupported court decoder: {config.name}")


def _require_reassemble_factors(value: Sequence[float] | None) -> Sequence[float]:
    if value is None:
        raise ValueError("DPT decoder requires reassemble_factors.")
    return value


def _parse_decoder_channel_scalar(value: Sequence[int] | int) -> int:
    if isinstance(value, int):
        return value
    raise ValueError("DPT decoder expects one scalar channel count.")


class CourtDenseResidualBlock(nn.Module):
    """Spatial residual adapter with depthwise context and pointwise mixing."""

    def __init__(self, channels: int, *, normalization_groups: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("Dense residual block channels must be positive.")
        if normalization_groups <= 0 or channels % normalization_groups:
            raise ValueError(
                "Dense residual block channels must be divisible by "
                "normalization_groups."
            )
        self.network = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(normalization_groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(normalization_groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast("Tensor", self.activation(features + self.network(features)))


class CourtDenseResidualHead(nn.Module):
    """Adapt one shared DPT feature map to one dense target representation."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        config: CourtDenseHeadBranchConfig,
        normalization_groups: int,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("Dense head input/output channels must be positive.")
        if config.hidden_channels % normalization_groups:
            raise ValueError(
                "Dense head hidden_channels must be divisible by normalization_groups."
            )
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                config.hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(normalization_groups, config.hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(
                CourtDenseResidualBlock(
                    config.hidden_channels,
                    normalization_groups=normalization_groups,
                )
                for _ in range(config.depth)
            )
        )
        self.output = nn.Conv2d(
            config.hidden_channels,
            output_channels,
            kernel_size=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast("Tensor", self.output(self.blocks(self.stem(features))))


def build_court_dense_head(
    *,
    kind: CourtTargetKind,
    input_channels: int,
    output_channels: int,
    config: CourtDenseHeadConfig,
) -> nn.Module:
    """Build a task-specific residual adapter."""
    if config.name != "residual":
        raise ValueError("Court detection requires residual dense heads.")
    if config.normalization_groups is None or kind not in config.branches:
        raise ValueError("Residual dense head configuration is incomplete.")
    return CourtDenseResidualHead(
        input_channels=input_channels,
        output_channels=output_channels,
        config=config.branches[kind],
        normalization_groups=config.normalization_groups,
    )


@dataclass(frozen=True, slots=True)
class TransformerEncoderOutput:
    """Output contract for :class:`CourtTransformerEncoder`.

    ``spatial`` keeps the input grid; ``pose_query`` has shape ``[B, C]``.
    """

    spatial: Tensor
    pose_query: Tensor | None

    @property
    def spatial_feature_map(self) -> Tensor:
        """Descriptive alias used by the hierarchical model API."""

        return self.spatial

    def __iter__(self) -> Iterator[Tensor | None]:
        """Allow the convenient ``spatial, query = encoder(features)`` form."""

        yield self.spatial
        yield self.pose_query


ConfigLike: TypeAlias = object


def _config_value(config: ConfigLike, name: str, default: Any) -> Any:
    """Read a dataclass or mapping value without silently coercing its type."""

    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _require_exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}.")
    return value


def _require_positive_float(value: object, *, name: str) -> float:
    if type(value) not in (float, int) or not cast("float | int", value) > 0.0:
        raise ValueError(f"{name} must be a positive number, got {value!r}.")
    result = float(cast("float | int", value))
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _require_rope_base(value: object, *, name: str) -> float | tuple[float, ...]:
    """Validate a scalar or explicit two-axis RoPE base without coercion."""

    if type(value) in (float, int):
        return _require_positive_float(value, name=name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if len(values) not in (1, 2):
            raise ValueError(f"{name} must contain one or two values.")
        validated = tuple(
            _require_positive_float(item, name=f"{name}[{index}]")
            for index, item in enumerate(values)
        )
        return validated
    raise TypeError(f"{name} must be a positive scalar or sequence.")


def _resolve_config(
    config: ConfigLike | None,
    *,
    dim: int | None,
    depth: int | None,
    num_heads: int | None,
    heads: int | None,
    rope_dim: int | None,
    ffn_dim: int | None,
    rope_base: float | Sequence[float],
    dropout: float,
) -> tuple[
    int,
    int | None,
    int,
    int,
    int,
    int,
    float | tuple[float, ...],
    float,
]:
    """Validate the patch Transformer parameters at construction."""

    if config is not None:
        if dim is None:
            dim = _config_value(config, "dim", _config_value(config, "hidden_dim", dim))
        if depth is None:
            depth = _config_value(config, "depth", depth)
        if num_heads is None:
            num_heads = _config_value(
                config,
                "num_heads",
                _config_value(config, "heads", num_heads),
            )
        if rope_dim is None:
            rope_dim = _config_value(config, "rope_dim", rope_dim)
        if ffn_dim is None:
            ffn_dim = _config_value(config, "ffn_dim", ffn_dim)
        rope_base = _config_value(
            config, "rope_base", _config_value(config, "rope_theta", rope_base)
        )
        dropout = _config_value(
            config, "dropout", _config_value(config, "attn_dropout", dropout)
        )
        configured_head_dim = _config_value(config, "head_dim", None)
        if configured_head_dim is not None and (
            type(configured_head_dim) is not int or configured_head_dim <= 0
        ):
            raise ValueError("head_dim must be a positive int.")
        attention_type = _config_value(config, "attention_type", "mha")
        ffn_type = _config_value(config, "ffn_type", "swiglu")
        rope_type = _config_value(config, "rope_type", "2d")
        if attention_type != "mha":
            raise ValueError(
                "Court intermediate Transformer requires attention_type='mha'."
            )
        if type(ffn_type) is not str or ffn_type not in SUPPORTED_FFN_TYPES:
            raise ValueError(
                "Court intermediate Transformer ffn_type must be one of "
                f"{sorted(SUPPORTED_FFN_TYPES)!r}."
            )
        if rope_type != "2d":
            raise ValueError("Court intermediate Transformer requires rope_type='2d'.")

    if dim is None:
        raise ValueError("Transformer token dimension (dim) is required.")
    dim = _require_exact_int(dim, name="dim", minimum=1)
    # A depth is required; disabled/identity encoders are unsupported.
    if depth is not None:
        depth = _require_exact_int(depth, name="depth", minimum=0)
    if num_heads is None:
        num_heads = 8
    num_heads = _require_exact_int(num_heads, name="num_heads", minimum=1)
    if dim % num_heads:
        raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
    head_dim = dim // num_heads
    configured_head_dim = (
        _config_value(config, "head_dim", None) if config is not None else None
    )
    if configured_head_dim is not None and configured_head_dim != head_dim:
        raise ValueError(
            f"head_dim must equal dim / num_heads: {configured_head_dim} != {head_dim}."
        )
    if rope_dim is None:
        rope_dim = head_dim
    rope_dim = _require_exact_int(rope_dim, name="rope_dim", minimum=1)
    if rope_dim % 4 or rope_dim > head_dim:
        raise ValueError(
            "rope_dim must be positive, divisible by four for 2-D RoPE, "
            f"and <= head_dim={head_dim}; got {rope_dim}."
        )
    if ffn_dim is None:
        ffn_dim = default_ffn_dim(dim)
    ffn_dim = _require_exact_int(ffn_dim, name="ffn_dim", minimum=1)
    rope_base = _require_rope_base(rope_base, name="rope_base")
    if type(dropout) not in (float, int) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout!r}.")
    return dim, depth, num_heads, head_dim, rope_dim, ffn_dim, rope_base, float(dropout)


def build_patch_positions(grid_hw: tuple[int, int], *, device: torch.device) -> Tensor:
    """Return row-major ``(y, x)`` positions with a zero-position query first."""

    if (
        type(grid_hw) is not tuple
        or len(grid_hw) != 2
        or any(type(value) is not int or value <= 0 for value in grid_hw)
    ):
        raise ValueError("grid_hw must be a tuple of two positive integers.")
    height, width = grid_hw
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.long),
        torch.arange(width, device=device, dtype=torch.long),
        indexing="ij",
    )
    patch_positions = torch.stack((rows, columns), dim=-1).reshape(-1, 2)
    return torch.cat(
        (torch.zeros(1, 2, device=device, dtype=torch.long), patch_positions)
    )


class CourtTransformerEncoder(nn.Module):
    """MHA + patch-only 2-D RoPE + SwiGLU over one deepest feature map."""

    def __init__(
        self,
        dim: int | None = None,
        *,
        channels: int | None = None,
        token_dim: int | None = None,
        config: ConfigLike | None = None,
        depth: int | None = 8,
        num_heads: int | None = 8,
        heads: int | None = None,
        rope_dim: int | None = None,
        ffn_dim: int | None = None,
        rope_base: float | Sequence[float] = 10000.0,
        rope_theta: float | None = None,
        dropout: float = 0.0,
        attn_dropout: float | None = None,
        ffn_type: FFNType = "swiglu",
    ) -> None:
        super().__init__()
        aliases = tuple(value for value in (channels, token_dim) if value is not None)
        if aliases and dim is not None and any(value != dim for value in aliases):
            raise ValueError("dim, channels, and token_dim disagree.")
        if len(set(aliases)) > 1:
            raise ValueError("channels and token_dim disagree.")
        if dim is None and aliases:
            dim = aliases[0]
        if heads is not None:
            if num_heads != 8 and num_heads != heads:
                raise ValueError("num_heads and heads disagree.")
            num_heads = heads
        if rope_theta is not None:
            rope_base = rope_theta
        if attn_dropout is not None:
            dropout = attn_dropout
        # The public defaults are 8, while a supplied typed config owns these
        # values.  ``None`` is used only at this private boundary so an
        # explicit config depth of zero remains distinguishable from the
        # constructor's enabled default.
        resolved_depth = None if config is not None and depth == 8 else depth
        resolved_heads = None if config is not None and num_heads == 8 else num_heads
        (
            self.dim,
            self.depth,
            self.num_heads,
            self.head_dim,
            self.rope_dim,
            self.ffn_dim,
            self.rope_base,
            self.dropout,
        ) = _resolve_config(
            config,
            dim=dim,
            depth=resolved_depth,
            num_heads=resolved_heads,
            heads=heads,
            rope_dim=rope_dim,
            ffn_dim=ffn_dim,
            rope_base=rope_base,
            dropout=dropout,
        )
        raw_ffn_type = _config_value(config, "ffn_type", ffn_type)
        if type(raw_ffn_type) is not str or raw_ffn_type not in SUPPORTED_FFN_TYPES:
            raise ValueError(
                "Court intermediate Transformer ffn_type must be one of "
                f"{sorted(SUPPORTED_FFN_TYPES)!r}."
            )
        self.ffn_type = cast(FFNType, raw_ffn_type)

        if self.depth is None or self.depth <= 0:
            raise ValueError("Court Transformer depth must be positive.")

        assert self.depth is not None
        self.pose_query = nn.Parameter(torch.empty(1, 1, self.dim))
        nn.init.normal_(self.pose_query, std=0.02)
        self.frequency_computer = RotaryFrequencyComputer(
            dim=self.rope_dim,
            base=self.rope_base,
            n_axes=2,
        )
        block_config = TransformerBlockConfig(
            dim=self.dim,
            n_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            head_dim=self.head_dim,
            rope_dim=self.rope_dim,
            attn_dropout=self.dropout,
            attention_type="mha",
            n_kv_heads=None,
            rope_base=(
                self.rope_base[0]
                if isinstance(self.rope_base, tuple)
                else self.rope_base
            ),
            ffn_type=self.ffn_type,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(block_config) for _ in range(self.depth)
        )

    @property
    def enabled(self) -> bool:
        """Whether this instance has a learned query and Transformer blocks."""

        return self.depth not in (None, 0)

    def _validate_input(self, features: Tensor) -> tuple[int, int, int, int]:
        if features.ndim != 4:
            raise ValueError(
                "Transformer features must have shape [B,C,H,W], "
                f"got {tuple(features.shape)}."
            )
        batch, channels, height, width = (int(value) for value in features.shape)
        if min(batch, channels, height, width) <= 0:
            raise ValueError("Transformer feature dimensions must all be positive.")
        if channels != self.dim:
            raise ValueError(
                f"Transformer feature channels must equal dim={self.dim}, got {channels}."
            )
        if not features.is_floating_point():
            raise TypeError("Transformer features must use a floating-point dtype.")
        parameter = next(self.parameters(), None)
        if parameter is not None:
            if features.device != parameter.device:
                raise ValueError(
                    "Transformer features and parameters must share a device: "
                    f"{features.device} != {parameter.device}."
                )
            if features.dtype != parameter.dtype:
                raise TypeError(
                    "Transformer features and parameters must share a dtype: "
                    f"{features.dtype} != {parameter.dtype}."
                )
        return batch, channels, height, width

    @staticmethod
    def _validate_patch_valid_mask(
        patch_valid_mask: Tensor | None,
        *,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        if patch_valid_mask is None:
            return torch.ones(
                batch,
                height,
                width,
                dtype=torch.bool,
                device=device,
            )
        if patch_valid_mask.shape != (batch, height, width):
            raise ValueError(
                "patch_valid_mask must have shape [B,H,W] matching the feature "
                f"grid; expected {(batch, height, width)}, got "
                f"{tuple(patch_valid_mask.shape)}."
            )
        if patch_valid_mask.dtype is not torch.bool:
            raise TypeError("patch_valid_mask must use torch.bool dtype.")
        if patch_valid_mask.device != device:
            raise ValueError(
                "patch_valid_mask and Transformer features must share a device."
            )
        if bool(torch.any(~patch_valid_mask.flatten(1).any(dim=1))):
            raise ValueError(
                "patch_valid_mask must keep at least one patch per sample."
            )
        return patch_valid_mask

    def forward(
        self,
        features: Tensor,
        *,
        patch_valid_mask: Tensor | None = None,
    ) -> TransformerEncoderOutput:
        """Transform valid deepest-grid tokens and return map plus pose query."""

        batch, _, height, width = self._validate_input(features)
        valid_grid = self._validate_patch_valid_mask(
            patch_valid_mask,
            batch=batch,
            height=height,
            width=width,
            device=features.device,
        )
        patch_tokens = features.flatten(2).transpose(1, 2)
        query = self.pose_query.expand(batch, -1, -1)
        tokens = torch.cat((query, patch_tokens), dim=1)
        token_valid = torch.cat(
            (
                torch.ones(batch, 1, dtype=torch.bool, device=features.device),
                valid_grid.flatten(1),
            ),
            dim=1,
        )
        tokens = torch.where(token_valid.unsqueeze(-1), tokens, 0.0)
        positions = build_patch_positions((height, width), device=features.device)
        frequencies = self.frequency_computer(positions)
        attention_mask = token_valid.unsqueeze(1).expand(
            batch,
            tokens.shape[1],
            tokens.shape[1],
        )
        for block in self.blocks:
            tokens = block(
                tokens,
                freqs_cis=frequencies,
                attn_mask=attention_mask,
            )
            tokens = torch.where(token_valid.unsqueeze(-1), tokens, 0.0)
        spatial = tokens[:, 1:].transpose(1, 2).reshape(batch, self.dim, height, width)
        return TransformerEncoderOutput(spatial=spatial, pose_query=tokens[:, 0])


@dataclass(frozen=True, slots=True)
class CourtRawPoseOutput:
    """Un-decoded camera pose in the immutable ``pose10d`` scalar order."""

    values: Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(POSE10D_RAW_ORDER):
            raise ValueError("Raw Court pose must have exact shape (B,10).")
        if self.values.shape[0] <= 0:
            raise ValueError("Raw Court pose batch size must be positive.")
        if not self.values.is_floating_point():
            raise TypeError("Raw Court pose must be floating point.")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("Raw Court pose must contain only finite values.")


@dataclass(frozen=True, slots=True)
class CourtModelOutput:
    """Raw output of the hierarchical model.

    ``pose`` is optional so the dense-only target/loss path keeps the original
    mapping contract.  Pose-enabled models must return the typed raw output;
    model-I/O validates the target/loss combination before computing losses.
    """

    dense_logits: Mapping[CourtTargetKind, Tensor]
    pose: CourtRawPoseOutput | None = None

    def __post_init__(self) -> None:
        logits = dict(self.dense_logits)
        if not logits:
            raise ValueError("Court model output requires a non-empty dense mapping.")
        for kind, value in logits.items():
            if kind not in {"kp", "seg", "line", "semantic_line"}:
                raise ValueError(f"Unknown Court dense output kind: {kind!r}.")
            if not isinstance(value, Tensor) or value.ndim != 4:
                raise ValueError(f"Court {kind} logits must be rank-4 Tensor.")
            if not value.is_floating_point():
                raise TypeError(f"Court {kind} logits must be floating point.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Court {kind} logits must be finite.")
            if self.pose is not None and value.shape[0] != self.pose.values.shape[0]:
                raise ValueError(
                    f"Court {kind} logits batch must match the pose output batch."
                )
        object.__setattr__(self, "dense_logits", MappingProxyType(logits))

    @property
    def dense_outputs(self) -> Mapping[CourtTargetKind, Tensor]:
        """Explicit alias used by the model-facing hierarchy API."""
        return self.dense_logits

    @property
    def raw_pose(self) -> CourtRawPoseOutput | None:
        """Return the optional raw pose branch without decoding it."""
        return self.pose


class CourtPose10DHead(nn.Module):
    """Regress the raw ten-scalar camera pose from a global feature."""

    def __init__(self, *, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or depth <= 0:
            raise ValueError("Pose head dimensions and depth must be positive.")
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth - 1):
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.GELU()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, len(POSE10D_RAW_ORDER)))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, features: Tensor) -> CourtRawPoseOutput:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("Pose head input must have shape (B,input_dim).")
        return CourtRawPoseOutput(self.network(features))


CourtFeatures = tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]


@dataclass(frozen=True, slots=True)
class CourtHierarchicalOutput:
    """Auxiliary output exposed by ``forward_with_pose`` when enabled."""

    dense_outputs: Mapping[CourtTargetKind, Tensor]
    spatial_feature_map: Tensor
    pose_query: Tensor
    pose_raw: Tensor

    @property
    def dense_logits(self) -> Mapping[CourtTargetKind, Tensor]:
        """Alias matching the typed model-I/O raw-output vocabulary."""

        return self.dense_outputs


class CourtHierarchicalModel(nn.Module):
    """Run the DINOv3/Transformer/DPT trunk with dense heads and a pose query."""

    transformer_encoder: CourtTransformerEncoder
    pose_head: CourtPose10DHead

    def __init__(
        self,
        config: CourtModelConfig,
        target_bundle: CourtTargetBundleSpec,
    ) -> None:
        super().__init__()
        if not target_bundle.targets:
            raise ValueError("Court model requires a non-empty target bundle.")
        self.in_channels = config.in_channels
        self.target_bundle_spec = target_bundle

        self.encoder = build_court_encoder(
            config=config.encoder,
            in_channels=self.in_channels,
        )
        self.decoder = build_court_decoder(
            config=config.decoder,
            encoder_channels=self.encoder.feature_channels,
        )

        transformer_config = config.transformer_encoder
        if not transformer_config.enabled:
            raise ValueError("Court detection requires the spatial Transformer.")
        deepest_dim = int(self.encoder.feature_channels[-1])
        if transformer_config.dim != deepest_dim:
            raise ValueError(
                "Transformer dimension must match the deepest encoder feature: "
                f"{transformer_config.dim} != {deepest_dim}."
            )
        assert transformer_config.depth is not None
        assert transformer_config.num_heads is not None
        assert transformer_config.rope_dim is not None
        assert transformer_config.ffn_dim is not None
        assert transformer_config.rope_theta is not None
        assert transformer_config.dropout is not None
        assert transformer_config.ffn_type is not None
        self.transformer_encoder = CourtTransformerEncoder(
            dim=deepest_dim,
            depth=transformer_config.depth,
            num_heads=transformer_config.num_heads,
            rope_dim=transformer_config.rope_dim,
            ffn_dim=transformer_config.ffn_dim,
            ffn_type=transformer_config.ffn_type,
            rope_theta=transformer_config.rope_theta,
            dropout=transformer_config.dropout,
        )
        self.pose_head = CourtPose10DHead(
            input_dim=deepest_dim,
            hidden_dim=deepest_dim,
            depth=2,
        )
        self.heads = nn.ModuleDict(
            {
                kind: build_court_dense_head(
                    kind=kind,
                    input_channels=self.decoder.output_channels,
                    output_channels=spec.output_channels,
                    config=config.dense_head,
                )
                for kind, spec in target_bundle.targets.items()
            }
        )

    @property
    def output_channels(self) -> Mapping[CourtTargetKind, int]:
        output_channels: Mapping[CourtTargetKind, int] = (
            self.target_bundle_spec.head_channels
        )
        return output_channels

    @property
    def transformer_enabled(self) -> bool:
        """The supported model always includes the Transformer/query/pose branch."""

        return True

    @classmethod
    def from_config(
        cls,
        config: CourtModelConfig,
        target_bundle: CourtTargetBundleSpec,
    ) -> CourtHierarchicalModel:
        return cls(config, target_bundle)

    def forward(
        self,
        x: Tensor,
        feature_1: Tensor | None = None,
        feature_2: Tensor | None = None,
        feature_3: Tensor | None = None,
        feature_4: Tensor | None = None,
        patch_valid_mask: Tensor | None = None,
    ) -> CourtModelOutput:
        """Decode prepared DINOv3 features into dense logits and camera pose."""
        features: CourtFeatures = (feature_1, feature_2, feature_3, feature_4)
        resolved = self._feature_forward_values(x, features)
        output, transformed = self._decode_with_transformer(
            x, resolved, patch_valid_mask
        )
        assert transformed.pose_query is not None
        return CourtModelOutput(
            dense_logits=output, pose=self.pose_head(transformed.pose_query)
        )

    def forward_with_pose(
        self,
        x: Tensor,
        feature_1: Tensor | None = None,
        feature_2: Tensor | None = None,
        feature_3: Tensor | None = None,
        feature_4: Tensor | None = None,
        patch_valid_mask: Tensor | None = None,
    ) -> CourtHierarchicalOutput:
        """Return dense heads and the intermediate spatial/query/raw-pose tensors."""

        features: CourtFeatures = (feature_1, feature_2, feature_3, feature_4)
        resolved_features = self._feature_forward_values(x, features)
        output, transformed_features = self._decode_with_transformer(
            x,
            resolved_features,
            patch_valid_mask,
        )
        assert transformed_features.pose_query is not None
        assert hasattr(self, "pose_head")
        pose_raw = self.pose_head(transformed_features.pose_query)
        spatial = transformed_features.spatial
        return CourtHierarchicalOutput(
            dense_outputs=MappingProxyType(output),
            spatial_feature_map=spatial,
            pose_query=transformed_features.pose_query,
            pose_raw=pose_raw.values,
        )

    def _feature_forward_values(
        self,
        x: Tensor,
        features: CourtFeatures,
    ) -> CourtFeatures:
        if any(feature is None for feature in features):
            raise ValueError(
                "Prepared-feature DINOv3 route requires all four feature maps."
            )
        return features

    def _decode_with_transformer(
        self,
        x: Tensor,
        features: CourtFeatures,
        patch_valid_mask: Tensor | None,
    ) -> tuple[dict[CourtTargetKind, Tensor], TransformerEncoderOutput]:
        deepest = features[-1]
        if deepest is None:
            raise ValueError(
                "Enabled intermediate Transformer requires the deepest feature map."
            )
        transformed = self.transformer_encoder(
            deepest,
            patch_valid_mask=patch_valid_mask,
        )
        if transformed.pose_query is None:
            raise RuntimeError(
                "Enabled Transformer unexpectedly returned no pose query."
            )
        transformed_features: CourtFeatures = (
            features[0],
            features[1],
            features[2],
            transformed.spatial,
        )
        return self._decode(x, transformed_features), transformed

    def _decode(
        self,
        x: Tensor,
        features: CourtFeatures,
    ) -> dict[CourtTargetKind, Tensor]:
        decoded = self.decoder(features)
        return {
            kind: F.interpolate(
                self.heads[kind](decoded),
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for kind in self.target_bundle_spec.kinds
        }
