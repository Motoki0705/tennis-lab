"""Token embedding modules for event detection models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.geometry import NUM_COURT_KP


class CourtTokenEmbedding(nn.Module):
    """Embed court keypoints as tokens.

    Args:
        dim: Output embedding dimension.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = 2 + 1  # (u, v) + visibility
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.Dropout(dropout))

    def forward(self, court_kp: Tensor, court_vis: Tensor | None) -> Tensor:
        """Embed court keypoints.

        Args:
            court_kp: Court keypoints, (B, 20, 2) or (B, 40).
            court_vis: Court visibility, (B, 20) or None.

        Returns:
            Tokens of shape (B, 20, D).
        """
        B = court_kp.shape[0]
        if court_kp.dim() == 2:
            court_kp = court_kp.view(B, NUM_COURT_KP, 2)
        vis = (
            torch.ones(B, NUM_COURT_KP, device=court_kp.device, dtype=court_kp.dtype)
            if court_vis is None
            else court_vis.to(court_kp.dtype)
        )
        x = torch.cat([court_kp, vis.unsqueeze(-1)], dim=-1)
        return self.proj(x)


class BallUVTokenEmbedding(nn.Module):
    """Embed 2D ball observations as tokens.

    Args:
        dim: Output embedding dimension.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = 2 + 1  # (u, v) + visibility
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.Dropout(dropout))

    def forward(self, ball_uv: Tensor, ball_mask: Tensor | None) -> Tensor:
        """Embed ball UV.

        Args:
            ball_uv: Ball UV, (B, T, 2).
            ball_mask: Visibility mask, (B, T) or None.

        Returns:
            Tokens of shape (B, T, D).
        """
        B, T, _ = ball_uv.shape
        vis = (
            torch.ones(B, T, device=ball_uv.device, dtype=ball_uv.dtype)
            if ball_mask is None
            else ball_mask.to(ball_uv.dtype)
        )
        x = torch.cat([ball_uv, vis.unsqueeze(-1)], dim=-1)
        return self.proj(x)


class Ball3DTokenEmbedding(nn.Module):
    """Embed 3D ball trajectory points as tokens.

    Args:
        dim: Output embedding dimension.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = 3
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.Dropout(dropout))

    def forward(self, ball_pos_world: Tensor) -> Tensor:
        """Embed ball 3D trajectory.

        Args:
            ball_pos_world: Ball positions, (B, T, 3).

        Returns:
            Tokens of shape (B, T, D).
        """
        return self.proj(ball_pos_world)


if __name__ == "__main__":
    B, T, D = 2, 16, 32
    court_kp = torch.rand(B, 20, 2)
    court_vis = torch.ones(B, 20)
    ball_uv = torch.rand(B, T, 2)
    ball_mask = torch.ones(B, T)
    ball_3d = torch.randn(B, T, 3)

    court_emb = CourtTokenEmbedding(dim=D, dropout=0.0)
    ball_uv_emb = BallUVTokenEmbedding(dim=D, dropout=0.0)
    ball_3d_emb = Ball3DTokenEmbedding(dim=D, dropout=0.0)

    assert court_emb(court_kp, court_vis).shape == (B, 20, D)
    assert ball_uv_emb(ball_uv, ball_mask).shape == (B, T, D)
    assert ball_3d_emb(ball_3d).shape == (B, T, D)
    print("embeddings smoke ok")

