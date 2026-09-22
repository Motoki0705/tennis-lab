"""Batched clip-global identity cost construction and association statistics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def association_loss(
    output: dict[str, Tensor],
    identity: Tensor,
    observed: Tensor,
    side: Tensor,
    view_valid: Tensor,
    reference: Tensor,
    *,
    identity_weight: float = 1.0,
    side_weight: float = 1.0,
    side_threshold: float = 0.5,
) -> dict[str, Tensor]:
    logits = output["object_id_logits"]
    b, _, _, _, classes = logits.shape
    k = classes - 1
    if identity.shape != logits.shape[:-1] or observed.shape != identity.shape:
        raise ValueError("Identity targets and visibility must match logits")
    # Dataset compacts teacher-only IDs once on CPU. Fixed K avoids device-side unique.
    membership = F.one_hot(identity.clamp_min(0), num_classes=k).to(torch.float32)
    membership = membership * (observed & identity.ge(0))[..., None]
    with torch.no_grad(), torch.autocast(device_type=logits.device.type, enabled=False):
        logp = logits.float().log_softmax(-1)
        cost = -torch.bmm(
            membership.reshape(b, -1, k).transpose(1, 2),
            logp[..., :k].reshape(b, -1, k),
        )
        host_cost = cost.cpu().numpy()
        permutation = np.stack([linear_sum_assignment(row)[1] for row in host_cost])
        mapping = torch.as_tensor(permutation, device=logits.device, dtype=torch.long)
        assigned = mapping.gather(1, identity.clamp_min(0).reshape(b, -1)).reshape_as(
            identity
        )
        target = torch.where(identity.ge(0), assigned, k).masked_fill(~observed, -100)
    id_sum = F.cross_entropy(
        logits.float().reshape(-1, classes),
        target.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    id_count = observed.sum()
    id_loss = id_sum / id_count.clamp_min(1)
    side_valid = view_valid & torch.arange(
        view_valid.shape[1], device=reference.device
    )[None].ne(reference[:, None])
    side_sum = (
        F.binary_cross_entropy_with_logits(
            output["side_logits"].float(), side.float(), reduction="none"
        )
        * side_valid
    ).sum()
    side_count = side_valid.sum()
    side_loss = side_sum / side_count.clamp_min(1)
    predicted = logits.argmax(-1)
    side_pred = output["side_logits"].float().sigmoid().ge(side_threshold)
    correct = ((predicted == target) & observed).sum()
    side_correct = ((side_pred == side.bool()) & side_valid).sum()
    fp = observed & identity.lt(0)
    pred_fp = observed & predicted.eq(k)
    return {
        "loss": identity_loss_value(id_loss, side_loss, identity_weight, side_weight),
        "identity_loss": id_loss,
        "side_loss": side_loss,
        "identity_accuracy": correct / id_count.clamp_min(1),
        "side_accuracy": side_correct / side_count.clamp_min(1),
        "identity_loss_sum": id_sum.detach(),
        "identity_count": id_count,
        "identity_correct": correct,
        "side_loss_sum": side_sum.detach(),
        "side_count": side_count,
        "side_correct": side_correct,
        "same_count": (side_valid & ~side.bool()).sum(),
        "same_correct": (side_valid & ~side.bool() & ~side_pred).sum(),
        "opposite_count": (side_valid & side.bool()).sum(),
        "opposite_correct": (side_valid & side.bool() & side_pred).sum(),
        "fp_count": fp.sum(),
        "fp_predicted": pred_fp.sum(),
        "fp_correct": (fp & pred_fp).sum(),
    }


def identity_loss_value(
    identity: Tensor, side: Tensor, identity_weight: float, side_weight: float
) -> Tensor:
    return identity * identity_weight + side * side_weight
