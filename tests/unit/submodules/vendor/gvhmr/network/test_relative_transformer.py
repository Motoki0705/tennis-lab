"""Tests for the vendored GVHMR denoiser network (small configs, CPU)."""

import torch

from src.submodules.vendor.gvhmr.network.relative_transformer import NetworkEncoderRoPE


def make_small_net(**overrides) -> NetworkEncoderRoPE:
    kwargs = dict(
        output_dim=151,
        max_len=16,
        imgseq_dim=32,
        latent_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return NetworkEncoderRoPE(**kwargs)


class TestNetworkEncoderRoPE:
    def test_forward_shapes(self):
        net = make_small_net().eval()
        B, L = 2, 8
        out = net(
            length=torch.tensor([L, L]),
            obs=torch.rand(B, L, 17, 3),
            f_cliffcam=torch.rand(B, L, 3),
            f_cam_angvel=torch.rand(B, L, 6),
            f_imgseq=torch.rand(B, L, 32),
        )
        assert out["pred_x"].shape == (B, L, 151)
        assert out["pred_cam"].shape == (B, L, 3)
        assert out["static_conf_logits"].shape == (B, L, 6)
        assert out["pred_context"].shape == (B, L, 64)

    def test_avgbeta_makes_betas_constant_over_time(self):
        net = make_small_net().eval()
        B, L = 1, 8
        out = net(
            length=torch.tensor([L]),
            obs=torch.rand(B, L, 17, 3),
            f_cliffcam=torch.rand(B, L, 3),
            f_cam_angvel=torch.rand(B, L, 6),
            f_imgseq=torch.rand(B, L, 32),
        )
        betas = out["pred_x"][..., 126:136]  # (B, L, 10)
        torch.testing.assert_close(
            betas, betas[:, :1].expand(-1, L, -1), atol=1e-6, rtol=0
        )

    def test_long_sequence_uses_windowed_attention(self):
        net = make_small_net(max_len=8).eval()
        B, L = 1, 12  # L > max_len triggers the windowed attention mask
        out = net(
            length=torch.tensor([L]),
            obs=torch.rand(B, L, 17, 3),
            f_cliffcam=torch.rand(B, L, 3),
            f_cam_angvel=torch.rand(B, L, 6),
            f_imgseq=torch.rand(B, L, 32),
        )
        assert out["pred_x"].shape == (B, L, 151)
        assert torch.isfinite(out["pred_x"]).all()

    def test_pred_cam_scale_clamped(self):
        net = make_small_net().eval()
        out = net(
            length=torch.tensor([4]),
            obs=torch.rand(1, 4, 17, 3),
            f_cliffcam=torch.rand(1, 4, 3),
            f_cam_angvel=torch.rand(1, 4, 6),
            f_imgseq=torch.rand(1, 4, 32),
        )
        assert (out["pred_cam"][..., 0] >= 0.25).all()
