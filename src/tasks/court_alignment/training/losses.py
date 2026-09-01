"""Losses for the Court Alignment KP and center-vote heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.tasks.court_alignment.models.cnn import NUM_KEYPOINTS


def validate_heatmap_pair(logits: object, target: object) -> tuple[Tensor, Tensor]:
    """Validate heatmap tensors at the training boundary."""
    if not isinstance(logits, Tensor) or logits.ndim != 4:
        raise ValueError("heatmap logits must have shape (B,14,H,W).")
    if logits.shape[1] != NUM_KEYPOINTS or any(size <= 0 for size in logits.shape):
        raise ValueError("heatmap logits must have shape (B,14,H,W).")
    if not isinstance(target, Tensor) or target.shape != logits.shape:
        raise ValueError("heatmap target must have exactly the logits shape (B,14,H,W).")
    for name, value in (("logits", logits), ("target", target)):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values.")
    if bool(torch.any((target < 0.0) | (target > 1.0))):
        raise ValueError("heatmap target values must be in [0,1].")
    return logits, target


def validate_center_vote_inputs(
    prediction: object, target: object, mask: object
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate center-vote tensors at the training boundary."""
    if not isinstance(prediction, Tensor) or prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("center-vote prediction must have shape (B,2,H,W).")
    prediction_tensor = prediction
    if not isinstance(target, Tensor) or target.shape != prediction_tensor.shape:
        raise ValueError("center-vote target must have exactly the prediction shape (B,2,H,W).")
    target_tensor = target
    if not isinstance(mask, Tensor):
        raise TypeError("center-vote mask must be a tensor.")
    mask_tensor: Tensor = mask
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.unsqueeze(1)
    if mask_tensor.shape != (prediction_tensor.shape[0], 1, *prediction_tensor.shape[-2:]):
        raise ValueError("center-vote mask must have shape (B,H,W) or (B,1,H,W).")
    if mask_tensor.dtype != torch.bool:
        raise TypeError("center-vote mask must have boolean dtype.")
    if prediction_tensor.device != target_tensor.device or mask_tensor.device != prediction_tensor.device:
        raise ValueError("center-vote prediction, target, and mask must share a device.")
    if not prediction_tensor.is_floating_point() or not target_tensor.is_floating_point():
        raise TypeError("center-vote prediction and target must be floating point.")
    if not bool(torch.isfinite(prediction_tensor).all()) or not bool(torch.isfinite(target_tensor).all()):
        raise ValueError("center-vote prediction and target must be finite.")
    return prediction_tensor, target_tensor, mask_tensor


def _centernet_focal_per_sample(
    logits: Tensor,
    target: Tensor,
    *,
    alpha: float,
    beta: float,
) -> Tensor:
    """Computation-only CenterNet loss used by module forwards."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        scores = logits.float().sigmoid().clamp(1.0e-6, 1.0 - 1.0e-6)
        target_compute = target.float()
        positive = target_compute.eq(1.0)
        negative = ~positive
        negative_weight = (1.0 - target_compute).pow(beta)
        positive_loss = -torch.log(scores) * (1.0 - scores).pow(alpha) * positive
        negative_loss = (
            -torch.log1p(-scores)
            * scores.pow(alpha)
            * negative_weight
            * negative
        )
        per_sample = (positive_loss + negative_loss).flatten(1).sum(dim=1)
        positives_per_sample = positive.flatten(1).sum(dim=1)
        return (per_sample / positives_per_sample.clamp_min(1.0)).to(logits.dtype)


def _center_vote_mean(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    beta: float,
) -> Tensor:
    """Computation-only masked vote loss used by module forwards."""
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        elementwise = F.smooth_l1_loss(
            prediction.float(),
            target.float(),
            beta=beta,
            reduction="none",
        )
        mask_compute = mask.float()
        weighted = elementwise * mask_compute
        denominator = mask_compute.sum() * prediction.shape[1]
        return (weighted.sum() / denominator.clamp_min(1.0)).to(prediction.dtype)


def centernet_focal_loss(
    logits: Tensor,
    target: Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    reduction: str = "mean",
    allow_no_positive: bool = False,
) -> Tensor:
    """Compute CenterNet's positive/negative modified focal loss.

    Gaussian targets are treated as positives only at value exactly one.  The
    surrounding Gaussian is down-weighted as a negative by ``(1-target)^beta``
    which is the standard CenterNet convention.  Normalising by the number of
    positive pixels makes sigma ablations comparable.
    """
    validate_heatmap_pair(logits, target)
    if not math.isfinite(float(alpha)) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative.")
    if not math.isfinite(float(beta)) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative.")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    if type(allow_no_positive) is not bool:
        raise TypeError("allow_no_positive must be boolean.")
    if not allow_no_positive and not bool(target.eq(1.0).any()):
        raise ValueError(
            "CenterNet focal target contains no exact positive pixel; "
            "render targets with a lattice positive or explicitly set "
            "allow_no_positive=True."
        )
    per_sample = _centernet_focal_per_sample(
        logits,
        target,
        alpha=float(alpha),
        beta=float(beta),
    )
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()


def center_vote_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    beta: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Masked Smooth-L1 loss for pixel-space center-vote offsets.

    ``mask`` may be ``(B,H,W)`` or ``(B,1,H,W)``.  A sample with no visible
    keypoint pixels contributes an explicit zero, retaining a valid autograd
    graph instead of silently changing the batch normalisation.
    """
    _, _, normalized_mask = validate_center_vote_inputs(prediction, target, mask)
    if not math.isfinite(float(beta)) or beta <= 0.0:
        raise ValueError("beta must be finite and positive.")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    if reduction == "mean":
        return _center_vote_mean(
            prediction,
            target,
            normalized_mask,
            beta=float(beta),
        )
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        elementwise = F.smooth_l1_loss(
            prediction.float(),
            target.float(),
            beta=float(beta),
            reduction="none",
        )
        weighted = elementwise * normalized_mask.float()
        if reduction == "sum":
            return weighted.sum().to(prediction.dtype)
        return weighted.to(prediction.dtype)


@dataclass(frozen=True, slots=True)
class CourtAlignmentLossOutput:
    """Decomposed loss values useful for Lightning logging."""

    total: Tensor
    heatmap: Tensor
    center_vote: Tensor


class CourtAlignmentLoss(nn.Module):
    """Combine heatmap focal and masked center-vote objectives."""

    def __init__(
        self,
        *,
        heatmap_weight: float = 1.0,
        center_vote_weight: float = 1.0,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        vote_beta: float = 1.0,
    ) -> None:
        super().__init__()
        values = (heatmap_weight, center_vote_weight, focal_alpha, focal_beta, vote_beta)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("Court alignment loss weights and hyperparameters must be finite.")
        if heatmap_weight < 0.0 or center_vote_weight < 0.0:
            raise ValueError("Court alignment loss weights must be non-negative.")
        if focal_alpha < 0.0 or focal_beta < 0.0 or vote_beta <= 0.0:
            raise ValueError("Invalid focal or center-vote hyperparameters.")
        self.heatmap_weight = float(heatmap_weight)
        self.center_vote_weight = float(center_vote_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_beta = float(focal_beta)
        self.vote_beta = float(vote_beta)

    def validate_inputs(
        self,
        heatmap_logits: Tensor,
        target_heatmaps: Tensor,
        center_votes: Tensor,
        target_center_votes: Tensor,
        target_center_vote_mask: Tensor,
    ) -> Tensor:
        """Validate the runtime contract outside the computation graph."""
        validate_heatmap_pair(heatmap_logits, target_heatmaps)
        if not bool(target_heatmaps.eq(1.0).any()):
            raise ValueError(
                "CenterNet focal target contains no exact positive pixel."
            )
        _, _, normalized_mask = validate_center_vote_inputs(
            center_votes,
            target_center_votes,
            target_center_vote_mask,
        )
        if heatmap_logits.shape[0] != center_votes.shape[0] or (
            heatmap_logits.shape[-2:] != center_votes.shape[-2:]
        ):
            raise ValueError(
                "Heatmap and center-vote tensors must share batch and spatial shape."
            )
        return normalized_mask

    def forward(
        self,
        heatmap_logits: Tensor,
        target_heatmaps: Tensor,
        center_votes: Tensor,
        target_center_votes: Tensor,
        target_center_vote_mask: Tensor,
    ) -> CourtAlignmentLossOutput:
        heatmap = _centernet_focal_per_sample(
            heatmap_logits,
            target_heatmaps,
            alpha=self.focal_alpha,
            beta=self.focal_beta,
        ).mean()
        vote = _center_vote_mean(
            center_votes,
            target_center_votes,
            target_center_vote_mask,
            beta=self.vote_beta,
        )
        total = self.heatmap_weight * heatmap + self.center_vote_weight * vote
        return CourtAlignmentLossOutput(total=total, heatmap=heatmap, center_vote=vote)


# Explicit aliases keep callers free to use the shorter vocabulary.
center_net_focal_loss = centernet_focal_loss
masked_center_vote_loss = center_vote_loss


class CenterNetFocalLoss(nn.Module):
    """Module wrapper around :func:`centernet_focal_loss`."""

    def __init__(
        self,
        *,
        alpha: float = 2.0,
        beta: float = 4.0,
        allow_no_positive: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.allow_no_positive = allow_no_positive

    def validate_inputs(self, logits: Tensor, target: Tensor) -> None:
        """Validate the runtime contract outside the computation graph."""
        validate_heatmap_pair(logits, target)
        if not self.allow_no_positive and not bool(target.eq(1.0).any()):
            raise ValueError("CenterNet focal target contains no exact positive pixel.")

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return _centernet_focal_per_sample(
            logits,
            target,
            alpha=self.alpha,
            beta=self.beta,
        ).mean()


__all__ = [
    "CenterNetFocalLoss",
    "CourtAlignmentLoss",
    "CourtAlignmentLossOutput",
    "center_net_focal_loss",
    "center_vote_loss",
    "centernet_focal_loss",
    "masked_center_vote_loss",
    "validate_center_vote_inputs",
    "validate_heatmap_pair",
]
