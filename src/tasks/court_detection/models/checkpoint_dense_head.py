"""Reconstruct serialized Court dense heads for checkpoint inference.

The residual head was used by the deployed pose checkpoint before its training
configuration was merged into the main configuration surface. Keeping this
small parser next to the model code lets inference restore the exact module
hierarchy and state-dict keys without weakening current training validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from torch import Tensor, nn

from src.tasks.court_detection.data.contracts import CourtTargetKind


@dataclass(frozen=True, slots=True)
class CheckpointDenseHeadBranchSpec:
    hidden_channels: int
    depth: int


class CourtCheckpointDenseResidualBlock(nn.Module):
    """Spatial residual adapter matching the deployed checkpoint hierarchy."""

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

    def forward(self, features: Tensor) -> Tensor:
        return cast("Tensor", self.activation(features + self.network(features)))


class CourtCheckpointDenseResidualHead(nn.Module):
    """Task-specific residual head matching serialized residual checkpoints."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        branch: CheckpointDenseHeadBranchSpec,
        normalization_groups: int,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("Dense head input/output channels must be positive.")
        if branch.hidden_channels % normalization_groups:
            raise ValueError(
                "Dense head hidden_channels must be divisible by normalization_groups."
            )
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                branch.hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(normalization_groups, branch.hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(
                CourtCheckpointDenseResidualBlock(
                    branch.hidden_channels,
                    normalization_groups=normalization_groups,
                )
                for _ in range(branch.depth)
            )
        )
        self.output = nn.Conv2d(
            branch.hidden_channels,
            output_channels,
            kernel_size=1,
        )

    def forward(self, features: Tensor) -> Tensor:
        return cast("Tensor", self.output(self.blocks(self.stem(features))))


def build_checkpoint_dense_heads(
    config: Mapping[str, object],
    *,
    input_channels: int,
    output_channels: Mapping[CourtTargetKind, int],
) -> nn.ModuleDict:
    """Build the exact dense-head hierarchy recorded in a Court checkpoint."""
    name = config.get("name")
    if name == "linear":
        return nn.ModuleDict(
            {
                kind: nn.Conv2d(input_channels, channels, kernel_size=1)
                for kind, channels in output_channels.items()
            }
        )
    if name != "residual":
        raise ValueError(
            "Serialized Court dense head name must be 'linear' or 'residual'."
        )

    normalization_groups = _positive_int(
        config.get("normalization_groups"),
        name="normalization_groups",
    )
    heads: dict[str, nn.Module] = {}
    for kind, channels in output_channels.items():
        raw_branch = config.get(kind)
        if not isinstance(raw_branch, Mapping):
            raise ValueError(
                f"Serialized Court dense head is missing branch {kind!r}."
            )
        branch = CheckpointDenseHeadBranchSpec(
            hidden_channels=_positive_int(
                raw_branch.get("hidden_channels"),
                name=f"{kind}.hidden_channels",
            ),
            depth=_positive_int(raw_branch.get("depth"), name=f"{kind}.depth"),
        )
        if branch.hidden_channels % normalization_groups:
            raise ValueError(
                f"Serialized Court dense head {kind!r} hidden_channels must be "
                "divisible by normalization_groups."
            )
        heads[kind] = CourtCheckpointDenseResidualHead(
            input_channels=input_channels,
            output_channels=channels,
            branch=branch,
            normalization_groups=normalization_groups,
        )
    return nn.ModuleDict(heads)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Serialized Court dense head {name} must be positive.")
    return value


__all__ = [
    "CheckpointDenseHeadBranchSpec",
    "CourtCheckpointDenseResidualBlock",
    "CourtCheckpointDenseResidualHead",
    "build_checkpoint_dense_heads",
]
