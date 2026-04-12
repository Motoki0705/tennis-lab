"""Player keypoint embeddings with invisible-token substitution."""

from __future__ import annotations

import torch
from torch import nn, Tensor

from src.utils.models.embeddings.invisible_embedding import InvisibleTokenEmbedding
from src.utils.schema.player import NUM_HUMAN_KP


class PlayerKPUVEmbedding(nn.Module):
    """Embed player keypoints with invisible-token substitution.

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

    def forward(self, human_kp: Tensor, human_vis: Tensor | None = None) -> Tensor:
        """Embed human keypoints.

        Args:
            human_kp: Human keypoints, shape (B, 34) or (B, 17, 2).
            human_vis: Visibility flags, shape (B, 17). Optional.

        Returns:
            Tensor: Embedded tokens, shape (B, NUM_HUMAN_KP, D).
        """
        batch_size = int(human_kp.shape[0])
        if human_kp.dim() == 2:
            human_kp = human_kp.view(batch_size, NUM_HUMAN_KP, 2)

        feat = self.proj(human_kp)
        if human_vis is None:
            return feat

        mask = (human_vis > 0).unsqueeze(-1)
        inv = self.invisible_token().to(dtype=feat.dtype, device=feat.device)
        inv = inv.view(1, 1, -1).expand_as(feat)
        return torch.where(mask, feat, inv)


if __name__ == "__main__":
    torch.manual_seed(0)
    invisible = InvisibleTokenEmbedding(dim=8)
    embed = PlayerKPUVEmbedding(dim=8, dropout=0.0, invisible_token=invisible)
    human_kp = torch.randn(2, NUM_HUMAN_KP, 2)
    human_vis = torch.tensor([[1] * NUM_HUMAN_KP, [0] * NUM_HUMAN_KP], dtype=torch.float32)
    out = embed(human_kp, human_vis)
    assert out.shape == (2, NUM_HUMAN_KP, 8)
    print("PlayerKPUVEmbedding smoke ok")
