"""Small, task-local camera-pose output contract and regression head.

The hierarchical Court model has one output seam for all of its branches.  The
dense branches stay a mapping of logits while the optional camera branch is a
raw ten-scalar value.  Keeping this contract beside the model (rather than in
the old query implementation) lets model-I/O and training code consume pose
outputs without depending on a particular encoder implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor, nn

from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.geometry.pose import POSE10D_RAW_ORDER


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
            if kind not in {"kp", "seg", "line"}:
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


__all__ = ["CourtModelOutput", "CourtPose10DHead", "CourtRawPoseOutput"]
