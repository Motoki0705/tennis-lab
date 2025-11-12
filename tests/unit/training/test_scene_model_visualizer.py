from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf

from src.datasets.collate_tracking import SceneBatch
from src.datasets.dancetrack import TargetFrame
from src.training.scene_model.lightning import SceneModelLightningModule


def _lit_cfg() -> DictConfig:
    return OmegaConf.create(
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
                "backbone": {"type": "native"},
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
                "trainer": {"max_epochs": 1},
            },
            "dataset": {
                "image": {
                    "normalize": {
                        "mean": [0.5, 0.5, 0.5],
                        "std": [0.5, 0.5, 0.5],
                    }
                }
            },
            "logging": {"visualizer": {"max_frames": 2, "max_boxes": 2}},
            "debug": {"minimal": True},
        }
    )


def test_build_visual_grid_returns_tensor() -> None:
    cfg = _lit_cfg()
    module = SceneModelLightningModule(cfg)
    frames = torch.rand(1, 2, 3, 32, 32)
    padding_mask = torch.zeros(1, 2, dtype=torch.bool)
    targets = [
        [TargetFrame.empty(), TargetFrame.empty()],
    ]
    batch = SceneBatch(
        frames=frames,
        targets=targets,
        padding_mask=padding_mask,
        sequence_ids=["demo"],
    )
    outputs = {
        "bbox_center": torch.full((1, 2, 3, 2), 0.5),
        "bbox_size": torch.full((1, 2, 3, 2), 0.25),
        "exist_conf": torch.ones(1, 2, 3, 1),
    }
    grid = module._build_visual_grid(batch, outputs)
    assert grid is not None
    assert grid.shape[0] == 3
    assert grid.ndim == 3
