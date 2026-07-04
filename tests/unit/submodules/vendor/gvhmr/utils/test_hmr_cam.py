"""Tests for vendored hmr_cam helpers (camera / bbox math)."""

import pytest
import torch

from src.submodules.vendor.gvhmr.utils.hmr_cam import (
    compute_transl_full_cam,
    estimate_K,
    get_a_pred_cam,
    get_bbx_xys_from_xyxy,
    normalize_kp2d,
)


class TestEstimateK:
    def test_diagonal_focal_and_center(self):
        K = estimate_K(1920, 1080)
        assert K.shape == (3, 3)
        expected_f = (1920**2 + 1080**2) ** 0.5
        assert K[0, 0].item() == pytest.approx(expected_f)
        assert K[1, 1].item() == pytest.approx(expected_f)
        assert K[0, 2].item() == 960.0
        assert K[1, 2].item() == 540.0
        assert K[2, 2].item() == 1.0


class TestGetBbxXysFromXyxy:
    def test_square_size_with_aspect_fit(self):
        # 100x200 box centered at (100, 150)
        bbx_xyxy = torch.tensor([[50.0, 50.0, 150.0, 250.0]])
        xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2)
        assert xys.shape == (1, 3)
        torch.testing.assert_close(xys[0], torch.tensor([100.0, 150.0, 240.0]))


class TestPredCamRoundTrip:
    def test_compute_transl_inverse_of_get_a_pred_cam(self):
        gen = torch.Generator().manual_seed(0)
        L = 6
        K = estimate_K(1280, 720).expand(L, 3, 3)
        bbx_xys = torch.stack(
            [
                torch.rand(L, generator=gen) * 1000 + 100,
                torch.rand(L, generator=gen) * 500 + 100,
                torch.rand(L, generator=gen) * 200 + 100,
            ],
            dim=-1,
        )
        transl = torch.stack(
            [
                torch.randn(L, generator=gen),
                torch.randn(L, generator=gen),
                torch.rand(L, generator=gen) * 5 + 2,  # positive depth
            ],
            dim=-1,
        )
        pred_cam = get_a_pred_cam(transl, bbx_xys, K)
        transl_rec = compute_transl_full_cam(pred_cam, bbx_xys, K)
        torch.testing.assert_close(transl_rec, transl, atol=1e-4, rtol=1e-4)


class TestNormalizeKp2d:
    def test_center_maps_to_zero_and_outside_masked(self):
        kp2d = torch.tensor(
            [[[[100.0, 150.0, 0.9], [500.0, 150.0, 0.9]]]]
        )  # (B=1, L=1, J=2, 3)
        bbx_xys = torch.tensor([[[100.0, 150.0, 200.0]]])  # (1, 1, 3)
        out = normalize_kp2d(kp2d, bbx_xys)
        assert out.shape == (1, 1, 2, 3)
        # center point -> (0, 0), confidence kept
        torch.testing.assert_close(out[0, 0, 0], torch.tensor([0.0, 0.0, 0.9]))
        # point outside the box -> confidence zeroed
        assert out[0, 0, 1, 2].item() == 0.0
