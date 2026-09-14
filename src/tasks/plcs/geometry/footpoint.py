"""Observation-only ground-plane ankle prior in aligned CourtKP14 coordinates.

Invalid homographies/ankles have explicit validity flags and zero coordinates.
An unknown prior is therefore a distinct model input, never a claimed location.
All geometric arithmetic runs in float32 even under mixed precision.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d
from src.utils.schema.court_normalization import normalize_court_position


def footpoint_prior(
    human_kp: Tensor,
    court_kp: Tensor,
    human_vis: Tensor,
    court_vis: Tensor,
    padding_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-view XY, validity (B,V,T), and fused XYZ (B,T,3).

    Fit image -> normalized ground XY with visible CourtKP14 correspondences.
    A ridge stabilizes the 8-parameter least squares system; rank, fit residual,
    projective denominator and domain bounds explicitly reject unstable fits.
    Average the visible ankles, then average valid views. Z is ground height;
    the residual head learns root height as well as horizontal displacement.
    """
    with torch.autocast(device_type=human_kp.device.type, enabled=False):
        uv = court_kp.float()[..., :14, :] - 0.5
        vis = (court_vis[..., :14] > 0) & ~padding_mask.unsqueeze(-1)
        xy = normalize_court_position(court_keypoints_3d(STANDARD_COURT_CONFIG)).to(uv)[
            :14, :2
        ]
        xy = xy.expand_as(uv)
        u, v = uv.unbind(-1)
        x, y = xy.unbind(-1)
        z, o = torch.zeros_like(u), torch.ones_like(u)
        a = torch.stack(
            (
                torch.stack((u, v, o, z, z, z, -u * x, -v * x), -1),
                torch.stack((z, z, z, u, v, o, -u * y, -v * y), -1),
            ),
            -2,
        ).flatten(-3, -2)
        b = xy.flatten(-2)
        w = vis.unsqueeze(-1).expand(*vis.shape, 2).flatten(-2).float()
        aw = a * w.unsqueeze(-1)
        ata = a.transpose(-1, -2) @ aw
        eig = torch.linalg.eigvalsh(ata)
        eye = torch.eye(8, device=uv.device)
        h = torch.linalg.solve(
            ata + 1e-7 * eye, (aw.transpose(-1, -2) @ b.unsqueeze(-1))
        ).squeeze(-1)
        residual = ((a @ h.unsqueeze(-1)).squeeze(-1) - b).square() * w
        fit_valid = (
            (vis.sum(-1) >= 4)
            & (eig[..., 0] > 1e-7)
            & (residual.sum(-1) / w.sum(-1).clamp_min(1) < 0.0025)
        )
        av = (human_vis[..., 15:17] > 0).float()
        ankle = (human_kp.float()[..., 15:17, :] * av.unsqueeze(-1)).sum(-2) / av.sum(
            -1
        ).clamp_min(1).unsqueeze(-1) - 0.5
        au, avv = ankle.unbind(-1)
        den = h[..., 6] * au + h[..., 7] * avv + 1
        safe_den = torch.where(den.abs() > 1e-4, den, torch.ones_like(den))
        ground = torch.stack(
            (
                (h[..., 0] * au + h[..., 1] * avv + h[..., 2]) / safe_den,
                (h[..., 3] * au + h[..., 4] * avv + h[..., 5]) / safe_den,
            ),
            -1,
        )
        valid = (
            fit_valid
            & (av.sum(-1) > 0)
            & (den.abs() > 1e-4)
            & (ground.abs().amax(-1) < 4)
            & ~padding_mask
        )
        ground = torch.where(valid.unsqueeze(-1), ground, torch.zeros_like(ground))
        fused_xy = ground.sum(1) / valid.sum(1).clamp_min(1).unsqueeze(-1)
        fused = torch.cat((fused_xy, torch.zeros_like(fused_xy[..., :1])), -1)
    return ground, valid, fused
