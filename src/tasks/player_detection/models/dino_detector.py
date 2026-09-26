"""Official DINO 4-scale Swin-L prepared for player fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from src.submodules.models.dino.architecture import (
    build_dino,
    dino_4scale_swin_args,
    load_dino_state_dict,
)
from src.tasks.player_detection.configuration import DinoModelConfig
from src.tasks.player_detection.data.detection_dataset import PLAYER_CLASS_ID


class PlayerDinoModel(nn.Module):
    """Holds the unmodified upstream model as ``dino`` (exported verbatim)."""

    def __init__(self, dino: nn.Module) -> None:
        super().__init__()
        self.dino = dino

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = self.dino(images, targets)
        return outputs


def freeze_backbone(dino: nn.Module, keywords: tuple[str, ...]) -> int:
    """Freeze Swin parameters by name keyword; every keyword must match.

    Returns the number of frozen parameter tensors.
    """
    # Upstream Joiner(backbone, position_embedding) is an nn.Sequential.
    swin = cast(nn.Sequential, dino.backbone)[0]
    matched = dict.fromkeys(keywords, 0)
    frozen = 0
    for name, parameter in swin.named_parameters():
        hits = [keyword for keyword in keywords if keyword in name]
        for keyword in hits:
            matched[keyword] += 1
        if hits:
            parameter.requires_grad_(False)
            frozen += 1
    unmatched = sorted(keyword for keyword, count in matched.items() if count == 0)
    if unmatched:
        raise ValueError(f"backbone_freeze_keywords matched no Swin parameter: {unmatched}")
    return frozen


def build_player_dino(
    config: DinoModelConfig, *, device: torch.device | str
) -> tuple[PlayerDinoModel, nn.Module]:
    """Build the official model/criterion and load the COCO initialization."""
    args = dino_4scale_swin_args(device, use_checkpoint=config.use_checkpoint)
    dino, criterion = build_dino(config.repository, args)
    dino.load_state_dict(load_dino_state_dict(config.init_checkpoint), strict=True)
    freeze_backbone(dino, config.backbone_freeze_keywords)
    return PlayerDinoModel(dino), criterion


def backbone_parameter_names(model: PlayerDinoModel) -> frozenset[str]:
    return frozenset(
        name for name, _ in model.named_parameters() if name.startswith("dino.backbone.")
    )


@dataclass(frozen=True, slots=True)
class FrameDetections:
    """Top-k player detections of one frame in original-image pixels."""

    boxes_xyxy: torch.Tensor  # (K,4)
    scores: torch.Tensor  # (K,)


def decode_player_detections(
    outputs: Mapping[str, torch.Tensor],
    original_sizes: list[tuple[int, int]],
    *,
    max_detections: int,
) -> list[FrameDetections]:
    """Player-logit top-k per image, boxes scaled to each original frame."""
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    if logits.shape[0] != len(original_sizes) or boxes.shape[:2] != logits.shape[:2]:
        raise ValueError(
            f"DINO outputs {tuple(logits.shape)}/{tuple(boxes.shape)} do not match "
            f"{len(original_sizes)} images"
        )
    if max_detections > logits.shape[1]:
        raise ValueError(f"max_detections {max_detections} exceeds {logits.shape[1]} queries")
    scores = logits[..., PLAYER_CLASS_ID].sigmoid()
    top_scores, top_index = scores.topk(max_detections, dim=1)
    selected = boxes.gather(1, top_index.unsqueeze(-1).expand(-1, -1, 4))
    center_x, center_y, width, height = selected.unbind(-1)
    xyxy = torch.stack(
        [center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2],
        dim=-1,
    ).clamp(0.0, 1.0)
    results: list[FrameDetections] = []
    for image_index, (image_height, image_width) in enumerate(original_sizes):
        scale = xyxy.new_tensor([image_width, image_height, image_width, image_height])
        results.append(
            FrameDetections(
                boxes_xyxy=(xyxy[image_index] * scale).detach().float().cpu(),
                scores=top_scores[image_index].detach().float().cpu(),
            )
        )
    return results
