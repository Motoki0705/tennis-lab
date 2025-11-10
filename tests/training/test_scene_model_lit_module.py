from __future__ import annotations

from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from src.datasets.collate_tracking import SceneBatch
from src.datasets.dancetrack import TargetFrame
from src.training.scene_model.lightning import SceneModelLightningModule


def _lit_cfg() -> Any:
    return OmegaConf.create(
        {
            "model": {
                "scene": {
                    "img_size": 64,
                    "patch_size": 8,
                    "vit_depth": 1,
                    "vit_heads": 2,
                    "D_model": 64,
                    "temporal_depth": 1,
                    "temporal_heads": 2,
                    "max_T_hint": 4,
                    "fps": 30.0,
                    "absK": 4,
                    "abs_Thorizon": 1.0,
                    "relQ": 4,
                    "rel_wmin": 1.0,
                    "rel_wmax": 4.0,
                    "num_queries": 4,
                    "decoder_layers": 1,
                    "num_points": 2,
                    "window_k": 1,
                    "offset_mode": "per_tau",
                    "tbptt_detach": False,
                    "smpl_param_dim": 0,
                },
                "backbone": {"freeze": False, "type": "native"},
                "head": {
                    "type": "bbox",
                    "num_classes": 2,
                    "matcher": {"cost_class": 1.0, "cost_bbox": 1.0, "cost_giou": 1.0},
                    "loss": {
                        "cls_weight": 1.0,
                        "bbox_weight": 1.0,
                        "giou_weight": 1.0,
                        "exist_weight": 1.0,
                    },
                },
            },
            "training": {
                "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
                "scheduler": {"min_lr_ratio": 0.1},
                "denoiser": {"num_noisy_queries": 0},
                "trainer": {"max_epochs": 1},
            },
            "debug": {"minimal": True},
        }
    )


def _make_batch() -> SceneBatch:
    frames = torch.randn(1, 2, 3, 64, 64)
    target = TargetFrame(
        center=torch.tensor([[10.0, 10.0]]),
        size=torch.tensor([[5.0, 5.0]]),
        track_ids=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([1.0]),
    )
    targets = [[target, TargetFrame.empty()]]
    padding_mask = torch.tensor([[False, False]])
    return SceneBatch(
        frames=frames, targets=targets, padding_mask=padding_mask, sequence_ids=["demo"]
    )


def test_lightning_module_backward_step(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SceneModelLightningModule(_lit_cfg())
    batch = _make_batch()
    loss = module.training_step(batch, 0)
    loss.backward()
    optim_conf = module.configure_optimizers()
    optimizer = optim_conf["optimizer"] if isinstance(optim_conf, dict) else optim_conf
    if hasattr(torch.cuda, "is_current_stream_capturing"):
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    optimizer.step()
    optimizer.zero_grad()
