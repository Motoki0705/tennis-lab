from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.blcs.inference.predictor import BLCSPredictor


class _LineModel(nn.Module):
    court_input_type = "line"

    def forward(
        self,
        *,
        ball_uv: Tensor,
        court_line_map: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del court_line_map, ball_vis, ball_mask
        return {"position": torch.zeros(*ball_uv.shape[:-1], 3)}


def test_line_predictor_routes_only_court_line_map() -> None:
    predictor = BLCSPredictor(_LineModel(), torch.device("cpu"))
    result = predictor.predict(
        ball_uv=torch.zeros(1, 2, 4, 2),
        court_line_map=torch.zeros(1, 2, 4, 1, 24, 40),
        ball_vis=torch.ones(1, 2, 4),
        ball_mask=torch.ones(1, 2, 4),
        denormalize=False,
    )
    assert result["position"].shape == (1, 2, 4, 3)
