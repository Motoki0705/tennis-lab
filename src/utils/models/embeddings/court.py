"""Court keypoint embeddings with invisible-token substitution."""

from __future__ import annotations

import torch
from torch import nn, Tensor

from src.utils.models.embeddings.shared import InvisibleTokenEmbedding
from src.utils.schema.court import NUM_COURT_KP


class CourtKPUVEmbedding(nn.Module):
    """Embed court keypoints with invisible-token substitution.

    Args:
        dim: Embedding dimension.
        dropout: Dropout probability.
        invisible_token: Shared invisible token module.
    """

    def __init__(
        self,
        *,
        dim: int,
        dropout: float = 0.1,
        invisible_token: InvisibleTokenEmbedding,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, int(dim)),
            nn.Dropout(float(dropout)),
        )
        self.invisible_token = invisible_token

    def forward(self, court_kp: Tensor, court_vis: Tensor | None = None) -> Tensor:
        """Embed court keypoints.

        Args:
            court_kp: Court keypoints, shape (B, N*2) or (B, N, 2).
            court_vis: Visibility flags, shape (B, N). Optional.

        Returns:
            Tensor: Embedded tokens, shape (B, N, D).
        """
        batch_size = int(court_kp.shape[0])
        if court_kp.dim() == 2:
            court_kp = court_kp.view(batch_size, -1, 2)

        feat = self.proj(court_kp)
        if court_vis is None:
            return feat

        mask = (court_vis > 0).unsqueeze(-1)
        inv = self.invisible_token().to(dtype=feat.dtype, device=feat.device)
        inv = inv.view(1, 1, -1).expand_as(feat)
        return torch.where(mask, feat, inv)


if __name__ == "__main__":
    torch.manual_seed(0)
    invisible = InvisibleTokenEmbedding(dim=8)
    embed = CourtKPUVEmbedding(dim=8, dropout=0.0, invisible_token=invisible)
    court_kp = torch.randn(2, NUM_COURT_KP, 2)
    court_vis = torch.tensor([[1] * NUM_COURT_KP, [0] * NUM_COURT_KP], dtype=torch.float32)
    out = embed(court_kp, court_vis)
    assert out.shape == (2, NUM_COURT_KP, 8)
    print("CourtKPUVEmbedding smoke ok")
