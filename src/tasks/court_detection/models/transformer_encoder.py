"""Patch-token Transformer used by the optional hierarchical court model trunk.

The module deliberately owns only the token mixer.  It consumes the deepest
feature map as ``[B, C, H, W]``, prepends one learned camera-pose query, and
returns the transformed spatial map together with that query.  The query is
positionless: its two-dimensional rotary coordinate is exactly ``(0, 0)``.

The optional configuration is kept task-local because the legacy
``court_hierarchical`` model has no Transformer configuration.  A caller that
does not request a depth (``None`` or ``0``) gets a true identity and no
learned query parameter.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import torch
from torch import Tensor, nn

from src.utils.models.components import (
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
    apply_rotary_emb,
)
from src.utils.models.components.ffn_layers import default_ffn_dim


@dataclass(frozen=True, slots=True)
class TransformerEncoderOutput:
    """Output contract for :class:`CourtTransformerEncoder`.

    ``spatial`` is always a feature map with the same shape as the input.  For
    an enabled encoder, ``pose_query`` has shape ``[B, C]``.  The identity
    variant has no query and reports ``None`` rather than manufacturing a
    parameter or a zero-valued prediction.
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


# This alias documents the exact type accepted by the flexible config helper
# below without coupling this module to the still-evolving Hydra schema.
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
    """Resolve supported config spellings at one strict construction boundary.

    ``intermediate_transformer`` config has intentionally been kept separate
    from the legacy model config.  Supporting both ``num_heads`` and ``heads``
    here lets the typed schema choose its preferred spelling while preserving
    one implementation contract.
    """

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
        rope_base = _config_value(config, "rope_base", _config_value(config, "rope_theta", rope_base))
        dropout = _config_value(config, "dropout", _config_value(config, "attn_dropout", dropout))
        configured_head_dim = _config_value(config, "head_dim", None)
        if configured_head_dim is not None and (
            type(configured_head_dim) is not int or configured_head_dim <= 0
        ):
            raise ValueError("head_dim must be a positive int.")
        attention_type = _config_value(config, "attention_type", "mha")
        ffn_type = _config_value(config, "ffn_type", "swiglu")
        rope_type = _config_value(config, "rope_type", "2d")
        if attention_type != "mha":
            raise ValueError("Court intermediate Transformer requires attention_type='mha'.")
        if ffn_type != "swiglu":
            raise ValueError("Court intermediate Transformer requires ffn_type='swiglu'.")
        if rope_type != "2d":
            raise ValueError("Court intermediate Transformer requires rope_type='2d'.")

    if dim is None:
        raise ValueError("Transformer token dimension (dim) is required.")
    dim = _require_exact_int(dim, name="dim", minimum=1)
    # None and zero both mean the explicit identity variant.
    if depth is not None:
        depth = _require_exact_int(depth, name="depth", minimum=0)
    if num_heads is None:
        num_heads = 8
    num_heads = _require_exact_int(num_heads, name="num_heads", minimum=1)
    if dim % num_heads:
        raise ValueError(
            f"dim={dim} must be divisible by num_heads={num_heads}."
        )
    head_dim = dim // num_heads
    configured_head_dim = (
        _config_value(config, "head_dim", None) if config is not None else None
    )
    if configured_head_dim is not None and configured_head_dim != head_dim:
        raise ValueError(
            "head_dim must equal dim / num_heads: "
            f"{configured_head_dim} != {head_dim}."
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


def build_patch_positions(
    grid_hw: tuple[int, int], *, device: torch.device
) -> Tensor:
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
    return torch.cat((torch.zeros(1, 2, device=device, dtype=torch.long), patch_positions))


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

        # The identity configuration intentionally creates neither blocks nor
        # a frequency computer/query parameter.  This matters for exact legacy
        # checkpoint and parameter-set compatibility in CourtHierarchicalModel.
        if self.depth in (None, 0):
            self.blocks = nn.ModuleList()
            return

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
            ffn_type="swiglu",
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
            raise ValueError("patch_valid_mask must keep at least one patch per sample.")
        return patch_valid_mask

    def forward(
        self,
        features: Tensor,
        *,
        patch_valid_mask: Tensor | None = None,
    ) -> TransformerEncoderOutput:
        """Transform valid deepest-grid tokens and return map plus pose query."""

        batch, _, height, width = self._validate_input(features)
        if not self.enabled:
            if patch_valid_mask is not None:
                raise ValueError(
                    "patch_valid_mask is not accepted by the identity Transformer."
                )
            return TransformerEncoderOutput(spatial=features, pose_query=None)

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


# Descriptive names used by different model-config revisions are aliases, not
# separate implementations.  Keeping one implementation prevents config
# spelling from creating subtly different RoPE or query behavior.
CourtIntermediateTransformerEncoder = CourtTransformerEncoder
IntermediateTransformerEncoder = CourtTransformerEncoder
PatchTransformerEncoder = CourtTransformerEncoder
TransformerEncoder = CourtTransformerEncoder


__all__ = [
    "CourtIntermediateTransformerEncoder",
    "CourtTransformerEncoder",
    "IntermediateTransformerEncoder",
    "PatchTransformerEncoder",
    "TransformerEncoder",
    "TransformerEncoderOutput",
    "build_patch_positions",
    "apply_rotary_emb",
]
