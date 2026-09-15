"""Geometry-aware tokens and observation-only ankle prior + learned XYZ residual."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.plcs.geometry.footpoint import footpoint_prior
from src.tasks.plcs.models.plcs_multiview_axial_split_model import (
    PLCSMultiViewAxialSplitModel,
)


class PLCSMultiViewAxialFootResidualModel(PLCSMultiViewAxialSplitModel):
    """Preserve axial trunks and add explicit visibility, relative pose and geometry.

    Input court UV must already be aligned to the output reference frame by the
    standard adapter. No camera calibration or target 3D enters forward.
    Invalid priors use zero with a separate validity feature. The output heads
    predict normalized XYZ offsets, including root height above the ground.
    """

    def _configure_observation_embedding(self) -> None:
        if self.num_court_tokens != 14:
            raise ValueError("Foot residual model requires aligned CourtKP14.")
        # 17 relative XY + 17 visibility + 14 visibility + ankle UV + body
        # extent XY + per-view ground XY + prior-valid + fused XYZ + disagreement XY.
        self.geometry_embed = nn.Sequential(
            nn.Linear(77, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _embed_observations(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        x, _ = super()._embed_observations(
            human_kp, court_kp, human_vis, court_vis, padding_mask
        )
        ground, valid, anchor = footpoint_prior(
            human_kp, court_kp, human_vis, court_vis, padding_mask
        )
        visible = (human_vis > 0).to(human_kp.dtype)
        ankle_vis = visible[..., 15:17]
        ankle = (human_kp[..., 15:17, :] * ankle_vis.unsqueeze(-1)).sum(
            -2
        ) / ankle_vis.sum(-1).clamp_min(1).unsqueeze(-1)
        center = (human_kp * visible.unsqueeze(-1)).sum(-2) / visible.sum(-1).clamp_min(
            1
        ).unsqueeze(-1)
        relative = (human_kp - center.unsqueeze(-2)) * visible.unsqueeze(-1)
        extent = relative.square().sum(-2).sqrt().clamp_min(0.01)
        relative = relative / extent.unsqueeze(-2)
        fused = anchor.unsqueeze(1).expand(-1, human_kp.shape[1], -1, -1)
        features = torch.cat(
            (
                relative.flatten(-2),
                visible,
                (court_vis > 0).to(visible),
                ankle,
                extent,
                ground,
                valid.unsqueeze(-1).to(visible),
                fused,
                ground - fused[..., :2],
            ),
            -1,
        )
        extra = self.geometry_embed(features)
        extra = extra * (~padding_mask).unsqueeze(-1)
        return x + extra.permute(0, 2, 1, 3), anchor
