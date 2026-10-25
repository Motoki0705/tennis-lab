"""Model-independent sampling and augmentation of complete shared frame packs."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from fractions import Fraction
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Sampler

from src.tasks.ball_detection.data.augmentation.video import VideoAugmenter
from src.tasks.ball_detection.training.heatmap_objective import heatmap_objective


def pack_index(seed: int, position: int, candidates: int) -> int:
    identity = hashlib.sha256(f"shared-pack-v1:{seed}:{position}".encode()).digest()
    return random.Random(int.from_bytes(identity, "little")).randrange(candidates)


class PackSampler(Sampler[int]):
    """Counter-based replacement draws; loader prefetch never changes committed position."""

    def __init__(
        self, candidates: int, *, seed: int, position: int, count: int
    ) -> None:
        self.candidates, self.seed, self.position, self.count = (
            candidates,
            seed,
            position,
            count,
        )

    def __iter__(self) -> Iterator[int]:
        for position in range(self.position, self.position + self.count):
            yield pack_index(self.seed, position, self.candidates)

    def __len__(self) -> int:
        return self.count


def augment_pack(
    batch: dict[str, Any],
    rgb: Tensor,
    augmenter: VideoAugmenter | None,
    *,
    position: int,
) -> tuple[Tensor, dict[str, Any], str]:
    offsets = tuple(batch["pack_offsets"])
    if offsets != (0, len(batch["clip_id"])):
        raise ValueError("Augmentation expects exactly one complete pack occurrence")
    children, frames = rgb.shape[:2]
    pack_rgb = rgb.reshape(1, children * frames, *rgb.shape[2:])
    pts = batch["pts"].reshape(1, -1)
    times = (
        (pts - pts[:, :1]).double() * float(Fraction(batch["time_base"][0]))
    ).float()
    pack = dict(
        uv=batch["uv"].reshape(1, -1, 2),
        position_valid=batch["position_valid"].reshape(1, -1),
        timestamps=times,
        start=[batch["pack_start"][0]],
        clip_id=[batch["clip_id"][0]],
        frame_step=[1],
    )
    audit: dict[str, Tensor] = {}
    if augmenter is not None:
        pack_rgb, pack, audit = augmenter(pack_rgb, pack, epoch=position)
    kept = pack["position_valid"].reshape(children, frames).cpu()
    removed = batch["position_valid"] & ~kept
    kinds = batch["label_kind"].clone().masked_fill(removed, 0)
    result = dict(
        batch,
        uv=pack["uv"].reshape(children, frames, 2).cpu(),
        position_valid=kept,
        heatmap_valid=kinds != 0,
        label_kind=kinds,
        augmentation_eligibility={
            k: int(v.sum()) for k, v in audit.items() if k.startswith("occlusion_")
        },
    )
    # Small transform parameters audit the model-independent draws without copying RGB off GPU.
    digest = hashlib.sha256(
        json.dumps(
            dict(position=position, pack_id=batch["pack_id"][0]), sort_keys=True
        ).encode()
    )
    for name, value in sorted(dict(audit, label_kind=kinds).items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return pack_rgb.reshape_as(rgb), result, digest.hexdigest()


def microbatch(batch: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    count = len(batch["clip_id"])
    result = {}
    for key, value in batch.items():
        if (
            isinstance(value, Tensor)
            and value.ndim
            and value.shape[0] == count
            or isinstance(value, list)
            and len(value) == count
        ):
            result[key] = value[start:stop]
    return result


def supervised_loss_sum(
    logits: Tensor,
    batch: dict[str, Any],
    *,
    sigma_ratio: float,
    gamma: float,
    coordinates: bool = False,
) -> tuple[Tensor, int]:
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite heatmap training output")
    mask = batch["position_valid"] if coordinates else batch["heatmap_valid"]
    count = int(mask.sum())
    if count == 0:
        # Still forward/backward every child. Unknown frames never become negative targets.
        return logits.float().sum() * 0, 0
    if coordinates:
        target = batch["uv"].to(logits.device)
        if logits.shape != target.shape or mask.shape != target.shape[:-1]:
            raise ValueError("Temporal coordinate shape/mask mismatch")
        selected = mask.to(logits.device)
        # Match the legacy smooth-L1 mean across x/y, then sum over real teachers.
        loss = torch.nn.functional.smooth_l1_loss(
            logits[selected].float(),
            target[selected].float(),
            beta=0.01,
            reduction="none",
        )
        return loss.mean(-1).sum(), count
    _, per_frame = heatmap_objective(
        logits, batch, sigma_ratio=sigma_ratio, gamma=gamma
    )
    return torch.where(
        batch["heatmap_valid"].to(logits.device), per_frame, 0.0
    ).sum(), count
