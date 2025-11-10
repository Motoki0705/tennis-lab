from __future__ import annotations

import torch

from src.models import SceneConfig, SceneModel


def _make_config() -> SceneConfig:
    return SceneConfig(
        img_size=16,
        patch_size=8,
        vit_depth=1,
        vit_heads=2,
        D_model=32,
        temporal_depth=1,
        temporal_heads=2,
        max_T_hint=4,
        fps=30.0,
        absK=4,
        abs_Thorizon=2.0,
        relQ=4,
        rel_wmin=1.0,
        rel_wmax=12.0,
        num_queries=2,
        decoder_layers=1,
        num_points=2,
        window_k=1,
        offset_mode="per_tau",
        tbptt_detach=False,
        smpl_param_dim=6,
    )


def test_scene_model_forward_shapes() -> None:
    torch.manual_seed(0)
    cfg = _make_config()
    model = SceneModel(cfg)
    frames = torch.randn(2, 3, 3, cfg.img_size, cfg.img_size)
    out = model(frames)
    assert out["role_logits"].shape == (2, 3, cfg.num_queries, 2)
    assert out["exist_conf"].shape == (2, 3, cfg.num_queries, 1)
    assert out["ball_xyz"].shape == (2, 3, cfg.num_queries, 3)
    assert out["smpl"].shape == (2, 3, cfg.num_queries, cfg.smpl_param_dim)
    assert torch.all(out["exist_conf"].ge(0.0))
    assert torch.all(out["exist_conf"].le(1.0))


def test_build_kv_handles_padding() -> None:
    cfg = _make_config()
    model = SceneModel(cfg)
    patches = torch.zeros(1, 2, (cfg.img_size // cfg.patch_size) ** 2, cfg.D_model)
    feats, mask, delta = model.decoder.build_kv(patches, t_center=0)
    # First window slot corresponds to t=-1 and should be masked
    assert bool(mask[:, 0].all())
    # Middle slot should be valid
    assert not bool(mask[:, 1].any())
    # Delta seconds include negative offset for the padded frame
    assert torch.isclose(delta[0, 0, 0], torch.tensor(-1 / cfg.fps)).item()
