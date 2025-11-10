from __future__ import annotations

from typing import Any

import torch
from omegaconf import OmegaConf

from src.models.scene_model.build import Dinov3BackboneAdapter, build_scene_model


def _model_cfg() -> Any:
    return OmegaConf.create(
        {
            "scene": {
                "img_size": 64,
                "patch_size": 8,
                "vit_depth": 1,
                "vit_heads": 2,
                "D_model": 32,
                "temporal_depth": 1,
                "temporal_heads": 2,
                "max_T_hint": 4,
                "fps": 30.0,
                "absK": 4,
                "abs_Thorizon": 1.0,
                "relQ": 4,
                "rel_wmin": 1.0,
                "rel_wmax": 4.0,
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
            "loss": {"cls_weight": 1.0, "bbox_weight": 1.0, "giou_weight": 1.0, "exist_weight": 1.0},
        }
    )


def test_build_scene_model_switches_head() -> None:
    cfg = _model_cfg()
    cfg.head.type = "bbox"
    cfg.head.num_classes = 3
    model = build_scene_model(cfg, debug_cfg={"minimal": False})
    dummy = torch.randn(1, 2, cfg.scene.num_queries, cfg.scene.D_model)
    outputs = model.head(dummy)
    assert "cls_logits" in outputs and outputs["cls_logits"].shape[-1] == 3
    assert "bbox_center" in outputs


class _DummyDinov3:
    embed_dim = 32
    patch_size = 16

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> "_DummyDinov3":
        self.device = device
        return self

    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1, return_class_token: bool = False, **_: Any):
        batch, _, H, W = x.shape
        tokens = H * W // (self.patch_size ** 2)
        patches = torch.zeros(batch, tokens, self.embed_dim, device=x.device)
        cls = torch.zeros(batch, self.embed_dim, device=x.device)
        if return_class_token:
            return ((patches, cls),)
        return (patches,)


def test_build_scene_model_replaces_backbone(monkeypatch, tmp_path) -> None:
    cfg = _model_cfg()
    cfg.scene.D_model = 32
    cfg.backbone.type = "dinov3_vits16"
    weights_path = tmp_path / "dinov3_fake_weights.pth"
    weights_path.write_bytes(b"torch hub mock")
    cfg.backbone.weights_path = str(weights_path)

    def _fake_load(*args, **kwargs):  # noqa: ANN001 - signature matches torch.hub.load
        return _DummyDinov3()

    monkeypatch.setattr(torch.hub, "load", _fake_load)
    model = build_scene_model(cfg, debug_cfg={"minimal": False})
    assert isinstance(model.backbone, Dinov3BackboneAdapter)
    frames = torch.randn(1, 1, 3, 64, 64)
    patch_tokens, cls_tokens = model.backbone(frames)
    assert patch_tokens.shape[0] == 1 and cls_tokens.shape[0] == 1
