"""Raw pose10D and selected dense heads for the Court query variant."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from src.tasks.court_detection.configuration import CourtQueryHeadsConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
)
from src.tasks.court_detection.models.query_encoder.contracts import CourtPose10DRaw


class CourtPose10DHead(nn.Module):
    """Map the sole pose query to the exact raw ten-scalar contract."""

    def __init__(self, *, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or depth <= 0:
            raise ValueError("Pose head dimensions and depth must be positive.")
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth - 1):
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.GELU()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 10))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, pose_query: Tensor) -> CourtPose10DRaw:
        if pose_query.ndim != 2 or pose_query.shape[1] != self.input_dim:
            raise ValueError("Pose head input must have shape (B,input_dim).")
        return CourtPose10DRaw(self.network(pose_query))


class CourtQueryDenseHeads(nn.Module):
    """Apply one 1x1 head for each explicitly selected dense target."""

    def __init__(
        self,
        *,
        input_dim: int,
        config: CourtQueryHeadsConfig,
        target_bundle: CourtTargetBundleSpec,
    ) -> None:
        super().__init__()
        if config.dense_targets != target_bundle.kinds:
            raise ValueError(
                "Query dense-head config must exactly match the target bundle."
            )
        self.target_bundle_spec = target_bundle
        self.heads = nn.ModuleDict(
            {
                kind: nn.Conv2d(
                    input_dim,
                    spec.output_channels,
                    kernel_size=1,
                )
                for kind, spec in target_bundle.targets.items()
            }
        )

    def forward(self, features: Tensor) -> Mapping[CourtTargetKind, Tensor]:
        if features.ndim != 4:
            raise ValueError("Dense-head features must have shape (B,C,H,W).")
        return {
            kind: self.heads[kind](features)
            for kind in self.target_bundle_spec.kinds
        }


__all__ = ["CourtPose10DHead", "CourtQueryDenseHeads"]
