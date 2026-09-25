"""Scene-local metric learning; numeric person labels never enter the model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def reid_pairs(output: dict[str, Tensor], identities: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    embeddings, valid = output["track_embedding"], output["track_valid"]
    if identities.shape != valid.shape:
        raise ValueError("Person identity teachers must have shape (B,V,P)")
    if bool((valid & identities.lt(0)).any()):
        raise ValueError("Every observed training track requires a person identity")
    b, v, p, d = embeddings.shape
    z, ids = embeddings.reshape(b, v * p, d).float(), identities.reshape(b, v * p)
    present = (valid & identities.ge(0)).reshape(b, v * p)
    views = torch.arange(v, device=z.device).repeat_interleave(p)
    cross_view = views[:, None] < views[None, :]
    mask = present[:, :, None] & present[:, None, :] & cross_view
    return z @ z.transpose(-1, -2), ids[:, :, None].eq(ids[:, None, :]), mask


def reid_loss(output: dict[str, Tensor], identities: Tensor, *, temperature: float, margin: float) -> dict[str, Tensor]:
    scores, same, mask = reid_pairs(output, identities)
    positive, negative = mask & same, mask & ~same
    pair_loss = F.binary_cross_entropy_with_logits((scores - margin) / temperature, same.float(), reduction="none")
    positive_sum, negative_sum = (pair_loss * positive).sum(), (pair_loss * negative).sum()
    positive_count, negative_count = positive.sum(), negative.sum()
    terms = (positive_count > 0).float() + (negative_count > 0).float()
    metric_loss = (positive_sum / positive_count.clamp_min(1) + negative_sum / negative_count.clamp_min(1)) / terms.clamp_min(1)
    return {"loss": metric_loss,
        "positive_loss_sum": positive_sum, "negative_loss_sum": negative_sum,
        "positive_count": positive_count, "negative_count": negative_count}


def court_side_loss(output: dict[str, Tensor], side: Tensor, padding: Tensor, reference: Tensor) -> dict[str, Tensor]:
    logits = output["side_logits"].float()
    valid = (~padding).any(-1) & torch.arange(logits.shape[1], device=logits.device)[None].ne(reference[:, None])
    total = (F.binary_cross_entropy_with_logits(logits, side.float(), reduction="none") * valid).sum()
    correct = logits.ge(0).eq(side)
    return {"loss": total / valid.sum().clamp_min(1), "side_loss_sum": total, "side_count": valid.sum(),
        "same_correct": (correct & valid & ~side).sum(), "same_count": (valid & ~side).sum(),
        "opposite_correct": (correct & valid & side).sum(), "opposite_count": (valid & side).sum()}
