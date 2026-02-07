"""PLCS keypoint-3D model implementation.

This variant keeps the same tokenization + transformer backbone as PLCSModel
but predicts per-player-token 3D keypoints directly instead of CLS-based
position/rotation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.plcs.models.components.heads import PerTokenKeypoint3DHead
from src.plcs.models.plcs_model import PLCSModel
from src.utils.geometry import NUM_HUMAN_KP


class PLCSKeypoint3DModel(PLCSModel):
    """PLCS model variant that directly regresses 3D keypoints from player tokens."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        head_dropout = float(kwargs.get("dropout", 0.1))
        super().__init__(*args, **kwargs)
        self.kp3d_head = PerTokenKeypoint3DHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=head_dropout,
        )

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass returning per-keypoint 3D predictions.

        Returns:
            dict: Dictionary with 'player_kp_3d' shape (B, 17, 3).

        """
        x, player_start_idx = self._encode_tokens(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            court_vis=court_vis,
        )
        player_tokens = x[:, player_start_idx:player_start_idx + NUM_HUMAN_KP, :]
        player_kp_3d = self.kp3d_head(player_tokens)
        return {"player_kp_3d": player_kp_3d}


if __name__ == "__main__":
    torch.manual_seed(0)

    model = PLCSKeypoint3DModel(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )

    B = 2
    human_kp = torch.randn(B, NUM_HUMAN_KP, 2)
    court_kp = torch.randn(B, 20, 2)
    human_vis = (torch.rand(B, NUM_HUMAN_KP) > 0.2).to(torch.float32)
    court_vis = (torch.rand(B, 20) > 0.1).to(torch.float32)

    with torch.no_grad():
        out = model(human_kp=human_kp, court_kp=court_kp, human_vis=human_vis, court_vis=court_vis)

    print("PLCSKeypoint3DModel:")
    for key, value in out.items():
        print(f"  {key}: {tuple(value.shape)}")
