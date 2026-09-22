"""Collation for camera-local association observations and teacher-only targets."""

from __future__ import annotations

import torch
from torch import Tensor


def collate_association(
    samples: list[dict[str, Tensor]], *, pad_views_to: int | None = None
) -> dict[str, Tensor]:
    if not samples:
        raise ValueError("Association batch must not be empty")
    max_v = max(s["object_uv"].shape[0] for s in samples)
    if pad_views_to is not None:
        if max_v > pad_views_to:
            raise ValueError("Observed views exceed the configured padded view width")
        max_v = pad_views_to
    max_t = max(s["object_uv"].shape[1] for s in samples)
    output: dict[str, Tensor] = {}
    for key in samples[0]:
        if key == "reference_view_index":
            output[key] = torch.stack([s[key] for s in samples])
            continue
        rows = []
        for sample in samples:
            value = sample[key]
            shape = (
                (max_v,) if key == "side_target" else (max_v, max_t, *value.shape[2:])
            )
            fill = (
                True
                if key == "padding_mask"
                else (-1 if key == "object_id_target" else 0)
            )
            row = value.new_full(shape, fill)
            row[tuple(slice(0, n) for n in value.shape)] = value
            rows.append(row)
        output[key] = torch.stack(rows)
    return output
