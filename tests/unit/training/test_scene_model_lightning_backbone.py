from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from src.training.scene_model.lightning import SceneModelLightningModule


def _lit_cfg(freeze: bool = True, unfreeze_after: int | None = 1) -> DictConfig:
    base = OmegaConf.create(
        {
            "model": {
                "scene": {
                    "img_size": 32,
                    "patch_size": 8,
                    "vit_depth": 1,
                    "vit_heads": 2,
                    "D_model": 64,
                    "temporal_depth": 1,
                    "temporal_heads": 2,
                    "max_T_hint": 2,
                    "fps": 30.0,
                    "absK": 2,
                    "abs_Thorizon": 1.0,
                    "relQ": 2,
                    "rel_wmin": 1.0,
                    "rel_wmax": 2.0,
                    "num_queries": 2,
                    "decoder_layers": 1,
                    "num_points": 2,
                    "window_k": 1,
                    "offset_mode": "per_tau",
                    "tbptt_detach": False,
                    "smpl_param_dim": 0,
                },
                "backbone": {
                    "type": "native",
                    "freeze": freeze,
                    "unfreeze_after": unfreeze_after,
                },
                "head": {"type": "scene"},
                "loss": {
                    "cls_weight": 1.0,
                    "bbox_weight": 1.0,
                    "giou_weight": 1.0,
                    "exist_weight": 1.0,
                },
            },
            "training": {
                "optimizer": {"lr": 1e-4},
                "scheduler": {},
                "denoiser": {},
                "trainer": {"max_epochs": 2},
            },
            "debug": {"minimal": True},
        }
    )
    return base


def _backbone_params(module: SceneModelLightningModule) -> Iterable[torch.nn.Parameter]:
    return cast(Iterable[torch.nn.Parameter], module.model.backbone.parameters())


def test_backbone_initially_frozen() -> None:
    cfg = _lit_cfg(freeze=True, unfreeze_after=None)
    module = SceneModelLightningModule(cfg)
    assert all(not p.requires_grad for p in _backbone_params(module))


def test_backbone_unfreezes_after_epoch() -> None:
    cfg = _lit_cfg(freeze=True, unfreeze_after=1)
    module = SceneModelLightningModule(cfg)
    assert all(not p.requires_grad for p in _backbone_params(module))
    module._maybe_update_backbone_freeze(0)
    assert all(not p.requires_grad for p in _backbone_params(module)), (
        "should stay frozen before threshold"
    )
    module._maybe_update_backbone_freeze(1)
    assert any(p.requires_grad for p in _backbone_params(module)), (
        "backbone should unfreeze once epoch >= unfreeze_after"
    )


def test_backbone_stays_trainable_when_freeze_disabled() -> None:
    cfg = _lit_cfg(freeze=False, unfreeze_after=None)
    module = SceneModelLightningModule(cfg)
    assert any(p.requires_grad for p in _backbone_params(module))
