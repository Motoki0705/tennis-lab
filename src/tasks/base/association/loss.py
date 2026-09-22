"""Clip-global permutation matching for observed identities and camera sides."""

from __future__ import annotations

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
) -> dict[str, Tensor]:
    logits = output["object_id_logits"]
    k = logits.shape[-1] - 1
    if identity.shape != logits.shape[:-1] or observed.shape != identity.shape:
        raise ValueError("Identity targets/observation masks must match object logits")
    if (identity < -1).any():
        raise ValueError(
            "Teacher ID must be a physical instance or -1 for a false positive"
        )
    log_prob = logits.float().log_softmax(-1)
    targets = torch.full_like(identity, -100)
    for b in range(logits.shape[0]):
        valid = observed[b]
        ids = identity[b][valid].unique()
        ids = ids[ids >= 0]
        if len(ids) > k:
            raise ValueError(
                f"Clip has {len(ids)} identities, exceeding max_identities={k}"
            )
        if len(ids):
            # One assignment for the ENTIRE clip, never a separate view/frame permutation.
            cost = torch.stack(
                [-log_prob[b][valid & identity[b].eq(i), :k].sum(0) for i in ids]
            )
            rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
            for row, column in zip(rows, columns, strict=True):
                targets[b][valid & identity[b].eq(ids[int(row)])] = int(column)
        targets[b][valid & identity[b].eq(-1)] = k
    valid_objects = targets.ne(-100)
    id_loss = (
        F.cross_entropy(logits[valid_objects].float(), targets[valid_objects])
        if valid_objects.any()
        else logits.sum() * 0
    )
    side_logits = output["side_logits"]
    side_valid = view_valid.clone()
    side_valid.scatter_(1, reference[:, None], False)
    side_loss = (
        F.binary_cross_entropy_with_logits(
            side_logits[side_valid].float(), side[side_valid].float()
        )
        if side_valid.any()
        else side_logits.sum() * 0
    )
    identity_accuracy = (
        (logits.argmax(-1)[valid_objects] == targets[valid_objects]).float().mean()
        if valid_objects.any()
        else logits.new_tensor(0.0)
    )
    side_accuracy = (
        ((side_logits[side_valid] >= 0) == side[side_valid].bool()).float().mean()
        if side_valid.any()
        else side_logits.new_tensor(1.0)
    )
    return {
        "loss": id_loss + side_loss,
        "identity_loss": id_loss,
        "side_loss": side_loss,
        "identity_accuracy": identity_accuracy,
        "side_accuracy": side_accuracy,
    }
