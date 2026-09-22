"""Decode camera sides and per-view one-to-one identity assignments."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

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
