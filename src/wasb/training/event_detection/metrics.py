"""Metrics utilities for WASB trajectory event detection training."""

from __future__ import annotations

import torch
from torch import Tensor


def _safe_div(num: Tensor, denom: Tensor) -> Tensor:
    return num / (denom.clamp(min=1.0))


def event_metrics(
    *,
    pred: Tensor,
    target: Tensor,
    ignore_index: int,
) -> dict[str, Tensor]:
    valid = target != ignore_index
    if not valid.any():
        z = torch.zeros((), device=target.device, dtype=torch.float32)
        return {
            "acc": z,
            "event_f1": z,
            "shot_f1": z,
            "bounce_f1": z,
            "event_recall": z,
            "event_precision": z,
        }

    pred = pred[valid]
    target = target[valid]
    event_mask = (target == 1) | (target == 2)
    if event_mask.any():
        acc = (pred[event_mask] == target[event_mask]).to(torch.float32).mean()
    else:
        acc = torch.zeros((), device=target.device, dtype=torch.float32)

    pred_event = pred > 0
    target_event = target > 0
    tp = (pred_event & target_event).sum().to(torch.float32)
    fp = (pred_event & ~target_event).sum().to(torch.float32)
    fn = (~pred_event & target_event).sum().to(torch.float32)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    event_f1 = _safe_div(2 * precision * recall, precision + recall)

    def f1_for_class(cls: int) -> Tensor:
        pred_c = pred == cls
        tgt_c = target == cls
        tp_c = (pred_c & tgt_c).sum().to(torch.float32)
        fp_c = (pred_c & ~tgt_c).sum().to(torch.float32)
        fn_c = (~pred_c & tgt_c).sum().to(torch.float32)
        p_c = _safe_div(tp_c, tp_c + fp_c)
        r_c = _safe_div(tp_c, tp_c + fn_c)
        return _safe_div(2 * p_c * r_c, p_c + r_c)

    return {
        "acc": acc,
        "event_f1": event_f1,
        "shot_f1": f1_for_class(1),
        "bounce_f1": f1_for_class(2),
        "event_recall": recall,
        "event_precision": precision,
    }

