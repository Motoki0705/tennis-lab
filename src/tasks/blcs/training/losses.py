"""BLCS supervision in explicit physical scales and recorded-clock flight losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from src.tasks.blcs.model_io.contracts import (
    BLCSTrajectoryPrediction,
    BLCSTrajectoryTrainingBatch,
)
from src.tasks.blcs.physics.rollout import rollout_residuals, target_parameters
from src.utils.geometry.court_pose import court_position_to_world_translation
from src.utils.projection.differentiable_projection import (
    DifferentiablePinholeProjection,
)
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    denormalize_court_velocity,
)

LOSS_TERMS = (
    "position",
    "reprojection",
    "velocity",
    "rollout_position",
    "rollout_velocity",
    "parameters",
    "event",
    "reconstruction",
    "parameter_consistency",
)
PHYSICAL_TERMS = frozenset(LOSS_TERMS[3:])


def robust_mean(error: Tensor, *, scale: float, beta: float) -> Tensor:
    """Dimensionless Smooth L1: error/scale and transition beta/scale.

    An empty training population contributes differentiable zero. Metrics must
    instead report count=0 and undefined, never interpret this as zero error.
    """
    if not error.numel():
        return error.sum()
    return nn.functional.smooth_l1_loss(
        error / scale, torch.zeros_like(error), beta=beta / scale
    )


def projection_errors(
    position_norm: Tensor,
    target: BLCSTrajectoryTrainingBatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Normalized-image and pixel residuals, supported views and all clean views."""
    uv, front = DifferentiablePinholeProjection()(
        world_points=court_position_to_world_translation(position_norm.float()),
        camera_R=target.camera_R,
        camera_C=target.camera_C,
        camera_f=target.camera_f,
        camera_cx=target.camera_cx,
        camera_cy=target.camera_cy,
        camera_w=target.camera_w,
        camera_h=target.camera_h,
    )
    eligible = target.target_vis.bool() & target.loss_mask[:, None]
    supported = eligible & front.bool()
    error = uv - target.target_uv
    size = torch.stack((target.camera_w, target.camera_h), dim=-1).unsqueeze(2)
    return error, error * size, supported, eligible


class BLCSLoss(nn.Module):
    """Standard A objective plus an explicit optional parameter-head experiment."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.weights = {key: float(config[f"{key}_weight"]) for key in LOSS_TERMS}
        self.horizons = tuple(int(value) for value in config["rollout_horizons"])

    def physics_factor(self, step: int) -> float:
        start = int(self.config["physics_start_step"])
        warmup = int(self.config["physics_warmup_steps"])
        return (
            min(1.0, max(0.0, (step - start) / max(warmup, 1)))
            if warmup
            else float(step >= start)
        )

    def forward(
        self,
        prediction: BLCSTrajectoryPrediction,
        target: BLCSTrajectoryTrainingBatch,
        *,
        step: int = 0,
    ) -> dict[str, Tensor]:
        position = denormalize_court_position(prediction.position.float())
        truth = denormalize_court_position(target.position.float())
        valid = target.loss_mask
        zero = position.sum() * 0
        losses = {key: zero for key in LOSS_TERMS}
        c = self.config
        pos_args = {
            "scale": float(c["position_scale_m"]),
            "beta": float(c["position_beta_m"]),
        }
        vel_args = {
            "scale": float(c["velocity_scale_mps"]),
            "beta": float(c["velocity_beta_mps"]),
        }
        # The default scales reproduce the historical normalized position loss
        # exactly, allowing the historical strategy to remain a controlled arm.
        losses["position"] = robust_mean((position - truth)[valid], **pos_args)
        if self.weights["reprojection"]:
            uv, _, support, _ = projection_errors(prediction.position, target)
            losses["reprojection"] = robust_mean(
                uv[support], scale=1.0, beta=float(c["reprojection_beta_fraction"])
            )
        velocity = (
            None
            if prediction.velocity is None
            else denormalize_court_velocity(prediction.velocity.float())
        )
        needs_velocity = any(
            self.weights[key]
            for key in (
                "velocity",
                "rollout_position",
                "rollout_velocity",
                "reconstruction",
                "parameter_consistency",
            )
        )
        if needs_velocity and velocity is None:
            raise ValueError("Velocity/physics losses require a velocity head")
        if velocity is not None and self.weights["velocity"]:
            losses["velocity"] = robust_mean(
                (velocity - denormalize_court_velocity(target.velocity.float()))[valid],
                **vel_args,
            )
        if velocity is not None and (
            self.weights["rollout_position"] or self.weights["rollout_velocity"]
        ):
            residual = rollout_residuals(
                position, velocity, valid, target.physics, self.horizons
            )
            losses["rollout_position"] = robust_mean(residual.position, **pos_args)
            losses["rollout_velocity"] = robust_mean(residual.velocity, **vel_args)
        parameters = prediction.physics_parameters
        if any(
            self.weights[key]
            for key in (
                "parameters",
                "event",
                "reconstruction",
                "parameter_consistency",
            )
        ):
            if (
                parameters is None
                or prediction.event_logits is None
                or velocity is None
            ):
                raise ValueError(
                    "Parameter-head losses require the complete physics head"
                )
            params = parameters.float()
            scale = params.new_tensor([0.02, 5.0, 5.0, 5.0, 0.2, 0.2, 0.2])
            losses["parameters"] = robust_mean(
                ((params - target_parameters(target.physics)) / scale)[valid],
                scale=1.0,
                beta=1.0,
            )
            events = torch.zeros_like(valid)
            events[:, 1:] = (
                target.physics.segment[:, 1:] != target.physics.segment[:, :-1]
            )
            event_valid = valid.clone()
            event_valid[:, 0] = (
                False  # crop entry is an imposed reset, not an observed event
            )
            if event_valid.any():
                losses["event"] = nn.functional.binary_cross_entropy_with_logits(
                    prediction.event_logits[..., 0].float()[event_valid],
                    events.float()[event_valid],
                    pos_weight=position.new_tensor(float(c["event_positive_weight"])),
                )
            residual = rollout_residuals(
                position,
                velocity,
                valid,
                target.physics,
                self.horizons,
                parameters=params,
            )
            endpoint = residual.start + residual.horizon
            integrated = residual.position + position[residual.sample, endpoint]
            losses["reconstruction"] = robust_mean(
                integrated - truth[residual.sample, endpoint], **pos_args
            )
            losses["parameter_consistency"] = robust_mean(
                residual.position, **pos_args
            ) + robust_mean(residual.velocity, **vel_args)
        factor = self.physics_factor(step)
        losses["total"] = sum(
            (
                losses[key] * weight * (factor if key in PHYSICAL_TERMS else 1.0)
                for key, weight in self.weights.items()
            ),
            zero,
        )
        losses["physics_factor"] = position.new_tensor(factor)
        return losses
