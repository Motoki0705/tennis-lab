"""Decode side/identity predictions for downstream calibration and triangulation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from src.tasks.base.association.training import INPUT_KEYS, AssociationLightningModule

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
                values = logits[b, v, t, slots].float().log_softmax(-1)
                scores = torch.cat(
                    (values[:, :k], values[:, k:].expand(-1, len(slots))), dim=-1
                )
                rows, columns = linear_sum_assignment(-scores.cpu().numpy())
                for row, column in zip(rows, columns, strict=True):
                    if column < k:
                        identities[b, v, t, slots[int(row)]] = int(column)
    side = output["side_logits"].ge(0) & view_valid
    side.scatter_(1, reference[:, None], False)
    return {
        "object_ids": identities,
        "view_half_turns": side,
        "view_valid": view_valid,
        **output,
    }


class AssociationPredictor:
    """Requires only raw normalized observations and explicit reference index."""

    def __init__(
        self, module: AssociationLightningModule, *, device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.module = module.to(self.device).eval()

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> AssociationPredictor:
        module = AssociationLightningModule.load_from_checkpoint(
            path, map_location="cpu", weights_only=False
        )
        return cls(module, device=device)

    @_no_grad
    def predict(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        if set(observations) != set(INPUT_KEYS):
            raise ValueError(
                "Association inference requires exactly the six observation/reference tensors"
            )
        inputs = {key: value.to(self.device) for key, value in observations.items()}
        output = self.module.model(**inputs)
        return {
            key: value.cpu()
            for key, value in decode_association(
                output,
                observed=inputs["object_vis"].any(-1)
                & ~inputs["padding_mask"][..., None],
                reference=inputs["reference_view_index"],
                view_valid=(~inputs["padding_mask"]).any(-1),
            ).items()
        }


def stitch_overlap_ids(
    previous: Tensor, current: Tensor, *, next_identity: int
) -> tuple[Tensor, int]:
    """Map clip IDs once from identical overlapping (view,time,local-slot) observations.

    Inputs cover the same overlap and contain -1 for missing/non-target objects.
    Returned mapping[k] is a scene ID; callers apply it to the complete new window.
    Unmatched clip IDs receive new scene IDs. A slot must denote the same raw
    observation at both sides of the overlap, not an independently reordered row.
    """
    if previous.shape != current.shape or previous.ndim != 3:
        raise ValueError("Overlap IDs must have the same (V,T,P) shape")
    local_ids = current[current >= 0].unique()
    old_ids = previous[previous >= 0].unique()
    width = int(local_ids.max().item()) + 1 if len(local_ids) else 0
    mapping = current.new_full((width,), -1)
    if len(local_ids) and len(old_ids):
        counts = torch.stack(
            [
                torch.stack(
                    [((current == local) & (previous == old)).sum() for old in old_ids]
                )
                for local in local_ids
            ]
        )
        rows, columns = linear_sum_assignment(-counts.cpu().numpy())
        for row, column in zip(rows, columns, strict=True):
            if counts[int(row), int(column)] > 0:
                mapping[local_ids[int(row)]] = old_ids[int(column)]
    for local in local_ids:
        if mapping[local] < 0:
            mapping[local] = next_identity
            next_identity += 1
    return mapping, next_identity
