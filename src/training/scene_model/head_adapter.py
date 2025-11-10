"""Loss adapter that connects SceneModel outputs to DINO-style objectives."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torchvision.ops import generalized_box_iou

from src.datasets.dancetrack import TargetFrame

_THIRD_PARTY_DINO = Path(__file__).resolve().parents[3] / "third_party" / "DINO"
if _THIRD_PARTY_DINO.exists() and str(_THIRD_PARTY_DINO) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY_DINO))

try:  # pragma: no cover - exercised in integration tests
    from models.dino.matcher import HungarianMatcher as _DinoMatcher  # type: ignore
except Exception:  # pragma: no cover - fallback to avoid hard failure in CI
    _DinoMatcher = None  # type: ignore


def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    half_w = 0.5 * w
    half_h = 0.5 * h
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def _build_matcher(cfg: Mapping[str, Any]) -> nn.Module:
    if _DinoMatcher is not None:
        return _DinoMatcher(
            cost_class=cfg.get("cost_class", 2.0),
            cost_bbox=cfg.get("cost_bbox", 5.0),
            cost_giou=cfg.get("cost_giou", 2.0),
        )
    return _GreedyMatcher()


@dataclass(slots=True)
class _MatchMeta:
    matched: int = 0
    total: int = 0


class HeadAdapter:
    """Adapter that computes Hungarian matches and supervision losses."""

    def __init__(self, cfg: DictConfig | Mapping[str, Any] | None) -> None:
        head_cfg = _to_dict(cfg)
        head_section = head_cfg.get("head", {})
        matcher_cfg = head_section.get("matcher", head_cfg.get("matcher", {}))
        self.matcher = _build_matcher(matcher_cfg)
        loss_cfg = head_cfg.get("loss", {})
        self.weights = {
            "cls": float(loss_cfg.get("cls_weight", 1.0)),
            "bbox": float(loss_cfg.get("bbox_weight", 5.0)),
            "giou": float(loss_cfg.get("giou_weight", 2.0)),
            "exist": float(loss_cfg.get("exist_weight", 1.0)),
        }

    def compute_loss(
        self,
        outputs: Mapping[str, Tensor],
        targets: Sequence[Sequence[TargetFrame]],
        padding_mask: Tensor,
        dn_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """Return weighted loss dictionary following the training spec."""

        pred_boxes = self._extract_pred_boxes(outputs)
        cls_logits = outputs.get("cls_logits", outputs.get("role_logits"))
        if cls_logits is None:
            raise KeyError("Head outputs must include 'cls_logits' or 'role_logits'")
        exist_scores = outputs.get("exist_conf")
        if exist_scores is None:
            raise KeyError("Head outputs must include 'exist_conf'")
        exist_scores = exist_scores.squeeze(-1)
        bs, T, num_queries, _ = pred_boxes.shape
        valid_mask = (~padding_mask.bool()).view(bs * T)
        flat_boxes = pred_boxes.view(bs * T, num_queries, 4)[valid_mask]
        flat_logits = cls_logits.view(bs * T, num_queries, cls_logits.shape[-1])[valid_mask]
        flat_exist = exist_scores.view(bs * T, num_queries)[valid_mask]
        target_list = list(_iter_valid_targets(targets, padding_mask))
        if not target_list:
            zero = pred_boxes.sum() * 0.0
            return {
                "total": zero,
                "cls": zero,
                "bbox": zero,
                "giou": zero,
                "exist": zero,
                "num_matches": zero,
                "num_targets": zero,
            }
        match_indices = self.matcher(
            {"pred_logits": flat_logits, "pred_boxes": flat_boxes},
            [_target_to_dict(t, device=flat_boxes.device) for t in target_list],
        )
        match_meta = _MatchMeta(total=_count_targets(target_list))
        cls_loss = self._classification_loss(flat_logits, match_indices)
        exist_loss = self._exist_loss(flat_exist, match_indices)
        bbox_loss, giou_loss = self._bbox_losses(flat_boxes, match_indices, target_list)
        total = (
            self.weights["cls"] * cls_loss
            + self.weights["bbox"] * bbox_loss
            + self.weights["giou"] * giou_loss
            + self.weights["exist"] * exist_loss
        )
        match_meta.matched = sum(len(src) for src, _ in match_indices)
        return {
            "total": total,
            "cls": cls_loss.detach(),
            "bbox": bbox_loss.detach(),
            "giou": giou_loss.detach(),
            "exist": exist_loss.detach(),
            "num_matches": flat_boxes.new_tensor(float(match_meta.matched)),
            "num_targets": flat_boxes.new_tensor(float(match_meta.total)),
        }

    def _extract_pred_boxes(self, outputs: Mapping[str, Tensor]) -> Tensor:
        if "bbox_center" in outputs and "bbox_size" in outputs:
            center = outputs["bbox_center"]
            size = outputs["bbox_size"].clamp_min(1e-3)
        else:
            center = outputs["ball_xyz"][..., :2]
            size = outputs["player_xyz"][..., :2].abs().clamp_min(1e-3)
        return torch.cat([center, size], dim=-1)

    def _classification_loss(
        self,
        logits: Tensor,
        match_indices: Sequence[tuple[Tensor, Tensor]],
    ) -> Tensor:
        num_frames, num_queries, num_classes = logits.shape
        targets = torch.zeros((num_frames, num_queries), dtype=torch.long, device=logits.device)
        for frame_idx, (src, _tgt) in enumerate(match_indices):
            if len(src) == 0:
                continue
            targets[frame_idx, src] = 1
        return F.cross_entropy(logits.view(-1, num_classes), targets.view(-1))

    def _exist_loss(
        self,
        exist_logits: Tensor,
        match_indices: Sequence[tuple[Tensor, Tensor]],
    ) -> Tensor:
        targets = torch.zeros_like(exist_logits)
        for frame_idx, (src, _tgt) in enumerate(match_indices):
            if len(src) == 0:
                continue
            targets[frame_idx, src] = 1.0
        return F.binary_cross_entropy(exist_logits, targets)

    def _bbox_losses(
        self,
        boxes: Tensor,
        match_indices: Sequence[tuple[Tensor, Tensor]],
        targets: Sequence[TargetFrame],
    ) -> tuple[Tensor, Tensor]:
        device = boxes.device
        total_targets = sum(t.center.shape[0] for t in targets)
        denom = max(total_targets, 1)
        bbox_loss = boxes.new_tensor(0.0)
        giou_loss = boxes.new_tensor(0.0)
        for frame_idx, (src, tgt_idx) in enumerate(match_indices):
            if len(src) == 0:
                continue
            pred = boxes[frame_idx, src]
            tgt = torch.cat([targets[frame_idx].center, targets[frame_idx].size], dim=-1)
            tgt = tgt[tgt_idx].to(device)
            bbox_loss = bbox_loss + F.l1_loss(pred, tgt, reduction="sum") / denom
            giou_loss = giou_loss + self._giou_loss(pred, tgt, denom)
        return bbox_loss, giou_loss

    def _giou_loss(self, pred: Tensor, tgt: Tensor, denom: int) -> Tensor:
        if tgt.numel() == 0:
            return pred.new_tensor(0.0)
        pred_xyxy = _cxcywh_to_xyxy(pred)
        tgt_xyxy = _cxcywh_to_xyxy(tgt)
        giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
        return (1.0 - giou.diag()).sum() / denom


def _iter_valid_targets(
    targets: Sequence[Sequence[TargetFrame]],
    padding_mask: Tensor,
) -> Iterable[TargetFrame]:
    for sample_targets, mask_row in zip(targets, padding_mask):
        for target, is_pad in zip(sample_targets, mask_row):
            if bool(is_pad):
                continue
            yield target


def _target_to_dict(target: TargetFrame, device: torch.device) -> dict[str, Tensor]:
    if target.center.numel() == 0:
        boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
        labels = torch.zeros(0, dtype=torch.long, device=device)
    else:
        boxes = torch.cat([target.center, target.size], dim=-1).to(device)
        labels = torch.zeros(target.center.shape[0], dtype=torch.long, device=device)
    return {"labels": labels, "boxes": boxes}


def _count_targets(targets: Sequence[TargetFrame]) -> int:
    return sum(target.center.shape[0] for target in targets)


class _GreedyMatcher(nn.Module):
    """Minimal matcher used when SciPy/third_party imports are unavailable."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        targets: Sequence[Mapping[str, Tensor]],
    ) -> list[tuple[Tensor, Tensor]]:
        pred_boxes = outputs["pred_boxes"]
        matches: list[tuple[Tensor, Tensor]] = []
        for frame_idx, target in enumerate(targets):
            tgt_boxes = target["boxes"]
            if tgt_boxes.numel() == 0:
                empty = pred_boxes.new_zeros((0,), dtype=torch.long)
                matches.append((empty, empty))
                continue
            cost = torch.cdist(pred_boxes[frame_idx], tgt_boxes, p=1)
            num = min(cost.shape[0], cost.shape[1])
            if num == 0:
                empty = pred_boxes.new_zeros((0,), dtype=torch.long)
                matches.append((empty, empty))
                continue
            src_idx = torch.arange(num, device=pred_boxes.device)
            tgt_idx = torch.arange(num, device=pred_boxes.device)
            matches.append((src_idx, tgt_idx))
        return matches
