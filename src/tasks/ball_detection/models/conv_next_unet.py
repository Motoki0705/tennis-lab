"""ConvNeXt-based spatio-temporal U-Net for ball detection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from src.tasks.ball_detection.configuration import validate_model
from src.tasks.ball_detection.model_io.adapters import build_ball_model_input_spec
from src.utils.tensor_utils import flatten_time_to_batch, restore_time_from_batch

if TYPE_CHECKING:
    from omegaconf import DictConfig


class ChannelLayerNorm(nn.Module):
    """Apply LayerNorm over channels for channel-first 2D or 3D features."""

    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the channel dimension while preserving tensor layout."""
        x = x.movedim(1, -1)
        x = self.norm(x)
        return x.movedim(-1, 1)


class ConvNeXtBlock(nn.Module):
    """Apply a channel-preserving ConvNeXt block."""

    def __init__(self, dim: int, drop_path_prob: float = 0.0) -> None:
        super().__init__()
        _validate_dim(dim)
        _validate_drop_path_prob(drop_path_prob)

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
            bias=True,
        )
        self.norm = nn.LayerNorm(dim, eps=1.0e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = _build_drop_path(drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial mixing, channel mixing, and the residual connection."""
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + cast(torch.Tensor, self.drop_path(x))


class Conv2dBlock(nn.Module):
    """Apply channel-preserving ConvNeXt blocks to frame-wise features."""

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_dim(dim)
        if depth <= 0:
            raise ValueError("depth must be positive.")
        _validate_drop_path_prob(drop_path_prob)

        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock(dim=dim, drop_path_prob=drop_path_prob)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the ConvNeXt blocks without changing shape."""
        return cast(torch.Tensor, self.blocks(x))


class Conv3dBlock(nn.Module):
    """Apply pre-normalized factorized 3D convolutions with Drop Path."""

    def __init__(self, dim: int, drop_path_prob: float = 0.0) -> None:
        super().__init__()
        _validate_dim(dim)
        _validate_drop_path_prob(drop_path_prob)

        self.spatial_norm = ChannelLayerNorm(dim)
        self.spatial_conv = nn.Conv3d(
            dim,
            dim,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            bias=True,
        )
        self.temporal_norm = ChannelLayerNorm(dim)
        self.temporal_conv = nn.Conv3d(
            dim,
            dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            bias=True,
        )
        self.act = nn.GELU()
        self.drop_path = _build_drop_path(drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the factorized convolution branch and residual connection."""
        residual = x
        x = self.spatial_conv(self.spatial_norm(x))
        x = self.act(x)
        x = self.temporal_conv(self.temporal_norm(x))
        x = self.act(x)
        return residual + cast(torch.Tensor, self.drop_path(x))


class StemLayer(nn.Module):
    """Apply the ConvNeXt 4x4 patchifying stem frame by frame."""

    def __init__(self, in_channels: int, dim: int) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        _validate_dim(dim)
        self.conv = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=4,
            stride=4,
            bias=True,
        )
        self.norm = ChannelLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify every frame while preserving batch and time axes."""
        x_2d, batch_size, timesteps = flatten_time_to_batch(x)
        x_2d = self.norm(self.conv(x_2d))
        return restore_time_from_batch(x_2d, batch_size, timesteps)


class DownsamplingLayer(nn.Module):
    """Downsample frame-wise features with LayerNorm then a strided Conv2d."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        _validate_dim(dim)
        _validate_dim(out_dim)
        self.norm = ChannelLayerNorm(dim)
        self.conv = nn.Conv2d(
            dim,
            out_dim,
            kernel_size=2,
            stride=2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Halve spatial resolution without changing the time axis."""
        x_2d, batch_size, timesteps = flatten_time_to_batch(x)
        x_2d = self.conv(self.norm(x_2d))
        return restore_time_from_batch(x_2d, batch_size, timesteps)


class UpsamplingLayer(nn.Module):
    """Normalize and resize a 3D feature to the matching skip resolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        _validate_dim(dim)
        self.norm = ChannelLayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        size: tuple[int, int, int],
    ) -> torch.Tensor:
        """Apply LayerNorm followed by trilinear interpolation."""
        x = self.norm(x)
        return F.interpolate(
            x,
            size=size,
            mode="trilinear",
            align_corners=False,
        )


class EncoderBlock(nn.Module):
    """Encode one stage with frame-wise ConvNeXt and factorized Conv3d."""

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.block_2d = Conv2dBlock(dim, depth, drop_path_prob)
        self.block_3d = Conv3dBlock(dim, drop_path_prob)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the 3D feature and flattened frame-wise 2D skip feature."""
        x_2d, batch_size, timesteps = flatten_time_to_batch(x)
        skip_2d = self.block_2d(x_2d)
        x_3d = restore_time_from_batch(skip_2d, batch_size, timesteps)
        return self.block_3d(x_3d), skip_2d


class BottleneckBlock(nn.Module):
    """Process the deepest features without changing their dimension."""

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_dim = dim
        self.block_2d = Conv2dBlock(dim, depth, drop_path_prob)
        self.block_3d = Conv3dBlock(dim, drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frame-wise and spatio-temporal bottleneck processing."""
        x_2d, batch_size, timesteps = flatten_time_to_batch(x)
        x_2d = self.block_2d(x_2d)
        x = restore_time_from_batch(x_2d, batch_size, timesteps)
        return cast(torch.Tensor, self.block_3d(x))


class DecoderBlock(nn.Module):
    """Fuse 3D and 2D skip features at one decoder stage."""

    def __init__(
        self,
        dim: int,
        skip_dim: int,
        depth: int = 2,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_dim(dim)
        _validate_dim(skip_dim)

        self.fuse_3d = nn.Sequential(
            ChannelLayerNorm(dim + skip_dim),
            nn.Conv3d(dim + skip_dim, skip_dim, kernel_size=1, bias=True),
        )
        self.block_3d = Conv3dBlock(skip_dim, drop_path_prob)
        self.fuse_2d = nn.Sequential(
            ChannelLayerNorm(2 * skip_dim),
            nn.Conv2d(2 * skip_dim, skip_dim, kernel_size=1, bias=True),
        )
        self.block_2d = Conv2dBlock(skip_dim, depth, drop_path_prob)

    def forward(
        self,
        x: torch.Tensor,
        skip_3d: torch.Tensor,
        skip_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse decoder state with matching-resolution encoder features."""
        x = self.fuse_3d(torch.cat([x, skip_3d], dim=1))
        x = self.block_3d(x)

        x_2d, batch_size, timesteps = flatten_time_to_batch(x)
        x_2d = self.fuse_2d(torch.cat([x_2d, skip_2d], dim=1))
        x_2d = self.block_2d(x_2d)
        return restore_time_from_batch(x_2d, batch_size, timesteps)


class ConvNeXtUNet(nn.Module):
    """Spatio-temporal ConvNeXt U-Net producing per-frame logits.

    Input shape:
        ``(B, C, T, H, W)``

    Output shape:
        ``(B, num_classes, T, floor(H/4), floor(W/4))``
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: Sequence[int],
        depth: int,
        drop_path_prob: float,
    ) -> None:
        super().__init__()
        dims = tuple(int(dim) for dim in dims)
        self._validate_init_args(
            in_channels=in_channels,
            num_classes=num_classes,
            dims=dims,
            depth=depth,
            drop_path_prob=drop_path_prob,
        )

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.dims = dims

        self.downsampling_layers = nn.ModuleList(
            [StemLayer(self.in_channels, dims[0])]
            + [
                DownsamplingLayer(dim, out_dim)
                for dim, out_dim in zip(dims, dims[1:], strict=False)
            ]
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    dim=dim,
                    depth=depth,
                    drop_path_prob=drop_path_prob,
                )
                for dim in dims
            ]
        )
        self.bottleneck = BottleneckBlock(
            dim=dims[-1],
            depth=depth,
            drop_path_prob=drop_path_prob,
        )
        self.upsampling_layers = nn.ModuleList(
            [UpsamplingLayer(dim) for dim in reversed(dims[1:])]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    dim=dim,
                    skip_dim=skip_dim,
                    depth=depth,
                    drop_path_prob=drop_path_prob,
                )
                for dim, skip_dim in zip(
                    reversed(dims[1:]),
                    reversed(dims[:-1]),
                    strict=True,
                )
            ]
        )
        self.final_norm = ChannelLayerNorm(dims[0])
        self.final_conv = nn.Conv3d(
            dims[0],
            self.num_classes,
            kernel_size=1,
        )

    @staticmethod
    def _validate_init_args(
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, ...],
        depth: int,
        drop_path_prob: float,
    ) -> None:
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if len(dims) < 2:
            raise ValueError("dims must contain at least two stages.")
        if any(dim <= 0 for dim in dims):
            raise ValueError("All dims must be positive.")
        if depth <= 0:
            raise ValueError("depth must be positive.")
        _validate_drop_path_prob(drop_path_prob)

    @classmethod
    def from_config(cls, config: DictConfig) -> ConvNeXtUNet:
        """Create the model from a composed Hydra config."""
        model_cfg = validate_model(config)
        return cls(
            in_channels=build_ball_model_input_spec(config).in_channels,
            num_classes=int(model_cfg["num_classes"]),
            dims=tuple(int(dim) for dim in model_cfg["dims"]),
            depth=int(model_cfg["depth"]),
            drop_path_prob=float(model_cfg["drop_path_prob"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ConvNeXt spatio-temporal U-Net."""
        skip_3d_features: list[torch.Tensor] = []
        skip_2d_features: list[torch.Tensor] = []

        x = self.downsampling_layers[0](x)
        for stage_index, encoder_layer in enumerate(self.encoder_layers[:-1]):
            x, skip_2d = encoder_layer(x)
            skip_3d_features.append(x)
            skip_2d_features.append(skip_2d)
            x = self.downsampling_layers[stage_index + 1](x)
        x, _ = self.encoder_layers[-1](x)

        x = self.bottleneck(x)
        for upsampling_layer, decoder_layer, skip_3d, skip_2d in zip(
            self.upsampling_layers,
            self.decoder_layers,
            reversed(skip_3d_features),
            reversed(skip_2d_features),
            strict=True,
        ):
            x = upsampling_layer(x, size=skip_3d.shape[-3:])
            x = decoder_layer(x, skip_3d, skip_2d)

        return cast(torch.Tensor, self.final_conv(self.final_norm(x)))


def _build_drop_path(drop_path_prob: float) -> nn.Module:
    if drop_path_prob <= 0.0:
        return nn.Identity()
    return cast(nn.Module, StochasticDepth(drop_path_prob, mode="row"))


def _validate_dim(dim: int) -> None:
    if dim <= 0:
        raise ValueError("dim must be positive.")


def _validate_drop_path_prob(drop_path_prob: float) -> None:
    if not 0.0 <= drop_path_prob < 1.0:
        raise ValueError("drop_path_prob must be in [0, 1).")


__all__ = [
    "BottleneckBlock",
    "ChannelLayerNorm",
    "Conv2dBlock",
    "Conv3dBlock",
    "ConvNeXtBlock",
    "ConvNeXtUNet",
    "DecoderBlock",
    "DownsamplingLayer",
    "EncoderBlock",
    "StemLayer",
    "UpsamplingLayer",
]
