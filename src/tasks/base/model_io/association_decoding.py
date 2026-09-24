"""Decode camera sides and per-view one-to-one identity assignments."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from src.tasks.base.model_io.association_contracts import (
    AssociationInferencePolicy,
    IdentityReason,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")
_no_grad: Callable[[Callable[_P, _R]], Callable[_P, _R]] = torch.no_grad()


@_no_grad
def decode_association(
    output: dict[str, Tensor],
    *,
    observed: Tensor,
    reference: Tensor,
    view_valid: Tensor,
    side_threshold: float = 0.5,
) -> dict[str, Tensor]:
    """Enforce per-view uniqueness; non-target detections may occur repeatedly."""
    logits = output["object_id_logits"]
    k = logits.shape[-1] - 1
    identities = torch.full(
        logits.shape[:-1], -1, dtype=torch.long, device=logits.device
    )
    for b in range(logits.shape[0]):
        for v in range(logits.shape[1]):
            for t in range(logits.shape[2]):
                slots = observed[b, v, t].nonzero().flatten()
                if not len(slots):
                    continue
                values = logits[b, v, t, slots].float()
                scores = torch.cat(
                    (values[:, :k], values[:, k:].expand(-1, len(slots))), dim=-1
                )
                rows, columns = linear_sum_assignment(-scores.cpu().numpy())
                for row, column in zip(rows, columns, strict=True):
                    if column < k:
                        identities[b, v, t, slots[int(row)]] = int(column)
    side = output["side_logits"].float().sigmoid().ge(side_threshold) & view_valid
    side.scatter_(1, reference[:, None], False)
    return {
        "object_ids": identities,
        "view_half_turns": side,
        "view_valid": view_valid,
        **output,
    }


@_no_grad
def decode_reliable_identities(
    logits: Tensor, observed: Tensor, *, policy: AssociationInferencePolicy
) -> dict[str, Tensor]:
    """Reject uncertain edges of a one-to-one assignment, without reassignment.

    A row's alternative excludes its assigned real-ID edge. FP columns are
    repeated only after normalizing the original classes. This rejects both
    uniform logits and symmetric collisions between otherwise confident rows.
    """
    if logits.ndim < 3 or observed.shape != logits.shape[:-1] or observed.dtype != torch.bool:
        raise ValueError("Identity logits and observation masks disagree")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Non-finite association logits")
    shape = observed.shape
    slots, classes = logits.shape[-2:]
    ids = torch.full(shape, -1, dtype=torch.int64)
    reasons = torch.full(shape, int(IdentityReason.MISSING), dtype=torch.uint8)
    probabilities = torch.zeros(shape, dtype=torch.float32)
    gaps = torch.zeros(shape, dtype=torch.float32)
    logp = logits.detach().float().cpu().log_softmax(-1).reshape(-1, slots, classes)
    masks = observed.cpu().reshape(-1, slots)
    flat_ids, flat_reasons = ids.reshape(-1, slots), reasons.reshape(-1, slots)
    flat_p, flat_gaps = probabilities.reshape(-1, slots), gaps.reshape(-1, slots)
    for frame, mask in enumerate(masks):
        selected = mask.nonzero().flatten()
        if not len(selected):
            continue
        scores = logp[frame, selected]
        cost = -torch.cat((scores[:, :-1], scores[:, -1:].expand(-1, len(selected))), -1).numpy()
        rows, columns = linear_sum_assignment(cost)
        best = float(cost[rows, columns].sum())
        for row, column in zip(rows, columns, strict=True):
            slot = int(selected[row])
            cls = min(int(column), classes - 1)
            probability = float(scores[row, cls].exp())
            flat_p[frame, slot] = probability
            flat_reasons[frame, slot] = int(IdentityReason.AMBIGUOUS)
            if cls == classes - 1:
                if probability >= policy.min_probability:
                    flat_reasons[frame, slot] = int(IdentityReason.FALSE_POSITIVE)
                continue
            alternative = cost.copy()
            alternative[row, column] = float("inf")
            alt_rows, alt_cols = linear_sum_assignment(alternative)
            gap = max(0.0, float(alternative[alt_rows, alt_cols].sum()) - best)
            flat_gaps[frame, slot] = gap
            if probability >= policy.min_probability and gap >= policy.min_assignment_gap:
                flat_ids[frame, slot] = cls
                flat_reasons[frame, slot] = int(IdentityReason.ACCEPTED)
    return {"ids": ids, "reasons": reasons, "probabilities": probabilities, "gaps": gaps}
