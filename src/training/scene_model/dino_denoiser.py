"""Minimal DINO denoiser wrapper that perturbs bbox labels for supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.datasets.dancetrack import TargetFrame


@dataclass(slots=True)
class DenoiserState:
    """Container returned by :class:`DinoDenoiser`."""

    boxes: Tensor
    labels: Tensor
    pad_size: int


class DinoDenoiser:
    """Generates noisy bbox/id pairs compatible with DINO's training recipe."""

    def __init__(self, cfg: DictConfig | Mapping[str, Any] | None) -> None:
        data = _to_dict(cfg)
        self.num_noisy_queries = int(data.get("num_noisy_queries", 0))
        self.box_noise_scale = float(data.get("box_noise_scale", 0.0))
        self.label_noise_prob = float(data.get("label_noise_scale", 0.0))

    def make_noise(self, targets: Sequence[Sequence[TargetFrame]]) -> DenoiserState:
        """Return perturbed boxes/labels for all valid frames in the batch."""

        if self.num_noisy_queries <= 0:
            return DenoiserState(
                boxes=torch.zeros((0, 0, 4), dtype=torch.float32),
                labels=torch.zeros((0, 0), dtype=torch.long),
                pad_size=0,
            )
        boxes_list: list[Tensor] = []
        labels_list: list[Tensor] = []
        for sample in targets:
            for frame in sample:
                frame_boxes, frame_labels = self._frame_noise(frame)
                boxes_list.append(frame_boxes)
                labels_list.append(frame_labels)
        if not boxes_list:
            empty_boxes = torch.zeros((0, self.num_noisy_queries, 4), dtype=torch.float32)
            empty_labels = torch.zeros((0, self.num_noisy_queries), dtype=torch.long)
            return DenoiserState(empty_boxes, empty_labels, self.num_noisy_queries)
        return DenoiserState(torch.stack(boxes_list), torch.stack(labels_list), self.num_noisy_queries)

    def _frame_noise(self, target: TargetFrame) -> tuple[Tensor, Tensor]:
        device = target.center.device
        boxes = torch.cat([target.center, target.size], dim=-1) if target.center.numel() else torch.zeros((0, 4), dtype=torch.float32, device=device)
        num = min(boxes.size(0), self.num_noisy_queries)
        padded_boxes = torch.zeros((self.num_noisy_queries, 4), dtype=torch.float32, device=device)
        if num > 0:
            noise = self.box_noise_scale * torch.randn((num, 4), dtype=torch.float32, device=device)
            padded_boxes[:num] = boxes[:num].to(torch.float32) + noise
        labels = torch.full((self.num_noisy_queries,), -1, dtype=torch.long, device=device)
        if num > 0 and target.track_ids.numel() > 0:
            base = target.track_ids[:num].to(torch.long)
            if self.label_noise_prob > 0:
                mask = torch.rand(num, device=device) < self.label_noise_prob
                if mask.any():
                    base = base.clone()
                    base[mask] = torch.randint(0, 1024, (mask.sum().item(),), device=device)
            labels[:num] = base
        return padded_boxes, labels


def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)
