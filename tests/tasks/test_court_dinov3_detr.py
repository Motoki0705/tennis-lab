from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.tasks.court_detection.models import (
    build_court_detection_model,
    dinov3_detr,
)
from src.tasks.court_detection.models.dinov3_detr import (
    DINOv3DETR,
    SinePositionEmbedding2D,
)


class FakeDINOv3Backbone(nn.Module):
    """Small patch encoder matching the DINOv3 feature API."""

    embed_dim = 32
    patch_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.patch_embed(inputs)
        return {"x_norm_patchtokens": features.flatten(2).transpose(1, 2)}


def _patch_backbone_loader(monkeypatch) -> FakeDINOv3Backbone:
    backbone = FakeDINOv3Backbone()
    monkeypatch.setattr(
        dinov3_detr,
        "load_dinov3_backbone",
        lambda **_: backbone,
    )
    return backbone


def _build_small_model(**overrides) -> DINOv3DETR:
    arguments = {
        "num_classes": 7,
        "hidden_dim": 32,
        "num_queries": 8,
        "num_decoder_layers": 2,
        "num_attention_heads": 4,
        "feedforward_dim": 64,
        "dropout": 0.0,
        "mask_dim": 16,
    }
    arguments.update(overrides)
    return DINOv3DETR(**arguments)


def test_sine_position_embedding_shape() -> None:
    features = torch.randn(2, 32, 3, 5)

    positions = SinePositionEmbedding2D(hidden_dim=32)(features)

    assert positions.shape == (2, 15, 32)
    assert torch.isfinite(positions).all()


def test_dinov3_detr_returns_dense_and_query_logits(monkeypatch) -> None:
    backbone = _patch_backbone_loader(monkeypatch)
    model = _build_small_model()
    inputs = torch.randn(2, 3, 19, 27)

    dense_logits = model(inputs)
    query_outputs = model.forward_query_outputs(inputs)

    assert dense_logits.shape == (2, 7, 19, 27)
    assert query_outputs["pred_logits"].shape == (2, 8, 8)
    assert query_outputs["pred_masks"].shape == (2, 8, 19, 27)
    torch.testing.assert_close(
        dense_logits.softmax(dim=1).sum(dim=1),
        torch.ones(2, 19, 27),
    )
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_court_model_factory_builds_dinov3_detr(monkeypatch) -> None:
    _patch_backbone_loader(monkeypatch)
    config = OmegaConf.create(
        {
            "data": {"task": "seg", "num_classes": 7},
            "model": {
                "name": "court_dinov3_detr_seg",
                "in_channels": 3,
                "num_classes": 7,
                "backbone": {"freeze": False},
                "decoder": {
                    "hidden_dim": 32,
                    "num_queries": 8,
                    "num_layers": 1,
                    "num_heads": 4,
                    "feedforward_dim": 64,
                    "dropout": 0.0,
                },
                "segmentation_head": {"mask_dim": 16},
            },
        }
    )

    model = build_court_detection_model(config)

    assert isinstance(model, DINOv3DETR)
    assert model(torch.randn(1, 3, 16, 20)).shape == (1, 7, 16, 20)
