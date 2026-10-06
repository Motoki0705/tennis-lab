"""Physics-head objectives: parameters, integrated reconstruction, consistency.

All terms are in network units.  Training uses the ground-truth segmentation of
each window; the integrated trajectory is compared with ground truth
(``reconstruction``) and with the direct coordinates (``consistency``), whose
gradient side is chosen explicitly.
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from src.tasks.ball_refiner_3d.configuration.training import PhysicsLossConfig
from src.tasks.ball_refiner_3d.model_io.adapters import normalization
from src.tasks.ball_refiner_3d.model_io.contracts import RefinerOutput
from src.tasks.ball_refiner_3d.physics.reconstruction import (
    integrate_segments,
    segment_layout,
)
from src.tasks.ball_refiner_3d.physics.targets import FlightClock
from src.tasks.ball_refiner_3d.training.losses.position import masked_l1


def physics_losses(
    output: RefinerOutput,
    batch: dict[str, Tensor],
    clock: FlightClock,
    config: PhysicsLossConfig,
) -> dict[str, Tensor]:
    """Unweighted terms; ``field`` includes the surface cross entropy."""
    physics = output.physics
    if physics is None or physics.segment_states is None:
        raise ValueError("Physics objectives require field and segment outputs")
    valid_segments = segment_layout(batch["segment"]).valid
    losses = {
        "field": F.l1_loss(physics.field, batch["field_target"])
        + F.cross_entropy(physics.surface_logits, batch["surface_target"]),
        "segment": (physics.segment_states - batch["segment_target"])
        .abs()
        .mean(dim=-1)[valid_segments]
        .mean(),
    }
    if not config.integrates:
        return losses
    valid = ~batch["padding"]
    scale = output.coordinates.new_tensor(normalization(3)[0])
    integrated = (
        integrate_segments(
            physics.segment_states, physics.field, batch["segment"], clock
        )
        / scale
    )
    direct = output.coordinates
    if config.consistency_gradient == "direct":
        reference, moved = integrated.detach(), direct
    elif config.consistency_gradient == "integrated":
        reference, moved = direct.detach(), integrated
    else:
        reference, moved = direct, integrated
    losses["reconstruction"] = masked_l1(integrated, batch["target"], valid)
    losses["consistency"] = masked_l1(moved, reference, valid)
    return losses


def weighted_physics(losses: dict[str, Tensor], config: PhysicsLossConfig) -> Tensor:
    weights = {
        "field": config.field_weight,
        "segment": config.segment_weight,
        "reconstruction": config.reconstruction_weight,
        "consistency": config.consistency_weight,
    }
    total = sum(
        (weights[name] * value for name, value in losses.items() if weights[name] > 0),
        start=next(iter(losses.values())).new_zeros(()),
    )
    return total
