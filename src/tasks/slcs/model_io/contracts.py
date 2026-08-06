"""Typed values crossing the SLCS model, loss, and inference boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from torch import Tensor


class SLCSRawOutput(TypedDict):
    """Exact tensor mapping emitted by :class:`SLCSFusionModel`."""

    player_position: Tensor
    player_rotation: Tensor
    player_position_log_b: Tensor
    player_rotation_log_b: Tensor
    ball_position: Tensor
    ball_position_log_b: Tensor


@dataclass(frozen=True, slots=True)
class SLCSDecodedOutput:
    """Validated normalized predictions returned by the model adapter."""

    player_position: Tensor
    player_rotation: Tensor
    player_position_log_b: Tensor
    player_rotation_log_b: Tensor
    ball_position: Tensor
    ball_position_log_b: Tensor

    def detached_cpu(self) -> SLCSDecodedOutput:
        """Detach every prediction and transfer it to CPU."""
        return SLCSDecodedOutput(
            player_position=self.player_position.detach().cpu(),
            player_rotation=self.player_rotation.detach().cpu(),
            player_position_log_b=self.player_position_log_b.detach().cpu(),
            player_rotation_log_b=self.player_rotation_log_b.detach().cpu(),
            ball_position=self.ball_position.detach().cpu(),
            ball_position_log_b=self.ball_position_log_b.detach().cpu(),
        )


@dataclass(frozen=True, slots=True)
class SLCSPhysicalOutput:
    """Decoded positions, yaw, and uncertainty in physical units."""

    player_position_meters: Tensor
    player_yaw_radians: Tensor
    ball_position_meters: Tensor
    player_position_sigma_m: Tensor
    player_rotation_sigma_rad: Tensor
    ball_position_sigma_m: Tensor


@dataclass(frozen=True, slots=True)
class SLCSTrainingTargets:
    """Validated targets and masks prepared before model execution."""

    target_player_position: Tensor
    target_player_rotation: Tensor
    target_ball_position: Tensor
    player_mask: Tensor
    player_weight: Tensor
    ball_mask: Tensor
    ball_weight: Tensor
    frame_mask: Tensor

    def detached_cpu(self) -> SLCSTrainingTargets:
        """Detach every prepared target and transfer it to CPU."""
        return SLCSTrainingTargets(
            target_player_position=self.target_player_position.detach().cpu(),
            target_player_rotation=self.target_player_rotation.detach().cpu(),
            target_ball_position=self.target_ball_position.detach().cpu(),
            player_mask=self.player_mask.detach().cpu(),
            player_weight=self.player_weight.detach().cpu(),
            ball_mask=self.ball_mask.detach().cpu(),
            ball_weight=self.ball_weight.detach().cpu(),
            frame_mask=self.frame_mask.detach().cpu(),
        )


@dataclass(frozen=True, slots=True)
class SLCSClipPrediction:
    """Full-timeline normalized and physical predictions for one clip view."""

    normalized: SLCSDecodedOutput
    physical: SLCSPhysicalOutput
    coverage: Tensor


__all__ = [
    "SLCSClipPrediction",
    "SLCSDecodedOutput",
    "SLCSPhysicalOutput",
    "SLCSRawOutput",
    "SLCSTrainingTargets",
]
