from __future__ import annotations

import torch

from src.wasb.models.event_detection import TrajectoryEventTransformer
from src.wasb.training.event_detection_lightning_module import (
    EventDetectionLightningModule,
)


def test_event_transformer_forward_shape() -> None:
    model = TrajectoryEventTransformer(d_model=32, num_layers=2, num_heads=4, num_classes=3, max_len=16)
    xy = torch.zeros((2, 8, 2), dtype=torch.float32)
    mask = torch.zeros((2, 8), dtype=torch.bool)
    logits = model(xy, key_padding_mask=mask)
    assert logits.shape == (2, 8, 3)


def test_event_metrics_acc_ignores_background() -> None:
    target = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    pred = torch.tensor([0, 0, 0, 2], dtype=torch.int64)
    metrics = EventDetectionLightningModule._event_metrics(pred, target, ignore_index=-100)
    assert float(metrics["acc"].item()) == 0.5
