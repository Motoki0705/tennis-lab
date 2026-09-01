"""Losses for the Court Alignment KP and center-vote heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.tasks.court_alignment.models.cnn import NUM_KEYPOINTS


def _validate_heatmap_pair(logits: Tensor, target: Tensor) -> None:
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
    _validate_heatmap_pair(logits, target)
    if not math.isfinite(float(alpha)) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative.")
    if not math.isfinite(float(beta)) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative.")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    if type(allow_no_positive) is not bool:
        raise TypeError("allow_no_positive must be boolean.")
    # float32 math avoids overflow/underflow for the deliberately sharp sigma
    # values used by the ablation, while gradients are cast back naturally.
    compute_dtype = torch.float32 if logits.dtype in {torch.float16, torch.bfloat16} else logits.dtype
    with torch.autocast(device_type=logits.device.type, enabled=False):
        scores = logits.to(compute_dtype).sigmoid().clamp(1.0e-6, 1.0 - 1.0e-6)
        target_compute = target.to(compute_dtype)
        positive = target_compute.eq(1.0)
        if not allow_no_positive and not bool(positive.any()):
            raise ValueError(
                "CenterNet focal target contains no exact positive pixel; "
                "render targets with a lattice positive or explicitly set "
                "allow_no_positive=True."
            )
        negative = ~positive
        negative_weight = (1.0 - target_compute).pow(float(beta))
        positive_loss = -torch.log(scores) * (1.0 - scores).pow(float(alpha)) * positive
        negative_loss = -torch.log1p(-scores) * scores.pow(float(alpha)) * negative_weight * negative
        per_sample = (positive_loss + negative_loss).flatten(1).sum(dim=1)
        positives_per_sample = positive.flatten(1).sum(dim=1)
        normalised = per_sample / positives_per_sample.clamp_min(1.0)
        if reduction == "none":
            return normalised.to(logits.dtype)
        if reduction == "sum":
            return normalised.sum().to(logits.dtype)
        return normalised.mean().to(logits.dtype)


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
    if not isinstance(prediction, Tensor) or prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("center-vote prediction must have shape (B,2,H,W).")
    if not isinstance(target, Tensor) or target.shape != prediction.shape:
        raise ValueError("center-vote target must have exactly the prediction shape (B,2,H,W).")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape != (prediction.shape[0], 1, *prediction.shape[-2:]):
        raise ValueError("center-vote mask must have shape (B,H,W) or (B,1,H,W).")
    if mask.dtype != torch.bool:
        raise TypeError("center-vote mask must have boolean dtype.")
    if prediction.device != target.device or mask.device != prediction.device:
        raise ValueError("center-vote prediction, target, and mask must share a device.")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("center-vote prediction and target must be floating point.")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("center-vote prediction and target must be finite.")
    if not math.isfinite(float(beta)) or beta <= 0.0:
        raise ValueError("beta must be finite and positive.")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    compute_dtype = torch.float32 if prediction.dtype in {torch.float16, torch.bfloat16} else prediction.dtype
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        elementwise = F.smooth_l1_loss(
            prediction.to(compute_dtype),
            target.to(compute_dtype),
            beta=float(beta),
            reduction="none",
        )
        weighted = elementwise * mask.to(compute_dtype)
        if reduction == "none":
            return weighted.to(prediction.dtype)
        if reduction == "sum":
            return weighted.sum().to(prediction.dtype)
        denominator = mask.to(compute_dtype).sum() * prediction.shape[1]
        return (weighted.sum() / denominator.clamp_min(1.0)).to(prediction.dtype)


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

    def forward(
        self,
        heatmap_logits: Tensor,
        target_heatmaps: Tensor,
        center_votes: Tensor,
        target_center_votes: Tensor,
        target_center_vote_mask: Tensor,
    ) -> CourtAlignmentLossOutput:
        heatmap = centernet_focal_loss(
            heatmap_logits,
            target_heatmaps,
            alpha=self.focal_alpha,
            beta=self.focal_beta,
        )
        vote = center_vote_loss(
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

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return centernet_focal_loss(
            logits,
            target,
            alpha=self.alpha,
            beta=self.beta,
            allow_no_positive=self.allow_no_positive,
        )


__all__ = [
    "CenterNetFocalLoss",
    "CourtAlignmentLoss",
    "CourtAlignmentLossOutput",
    "center_net_focal_loss",
    "center_vote_loss",
    "centernet_focal_loss",
    "masked_center_vote_loss",
]
