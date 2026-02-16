"""Multi-task model that shares a backbone across UV/3D tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from torch import Tensor

from src.ball_multitask.models.backbone import BallMultitaskBackbone
from src.ball_multitask.models.heads.event_head import EventLogitsHeadAdapter
from src.ball_multitask.models.heads.trajectory_head import Trajectory3DHeadAdapter
from src.ball_multitask.models.heads.uv_head import UVCompletionHead

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallMultitaskModel(nn.Module):
    """Unified model for UV completion, UV→3D, and event detection."""

    def __init__(
        self,
        *,
        backbone: BallMultitaskBackbone,
        num_events: int = 2,
        uv_head_hidden_dim: int | None = None,
        uv_head_dropout: float = 0.1,
        traj_head_hidden_dim: int | None = None,
        traj_head_dropout: float = 0.1,
        traj_head_layers: int = 2,
        event_head_dropout: float = 0.1,
        event_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_events = int(num_events)
        self.event_names = event_names

        self.uv_head = UVCompletionHead(
            input_dim=int(backbone.hidden_dim),
            hidden_dim=uv_head_hidden_dim,
            dropout=uv_head_dropout,
        )
        self.traj_head = Trajectory3DHeadAdapter(
            input_dim=int(backbone.hidden_dim),
            hidden_dim=traj_head_hidden_dim,
            dropout=traj_head_dropout,
            num_layers=traj_head_layers,
        )
        self.event_head = EventLogitsHeadAdapter(
            input_dim=int(backbone.hidden_dim),
            num_events=self.num_events,
            dropout=event_head_dropout,
        )

    @classmethod
    def from_config(cls, config: DictConfig) -> "BallMultitaskModel":
        """Create model from configuration."""
        model_cfg = config.get("model", {}) or {}
        backbone = BallMultitaskBackbone.from_config(config)
        event_names = model_cfg.get("event_names")
        if event_names is not None:
            event_names = [str(name) for name in event_names]
        return cls(
            backbone=backbone,
            num_events=int(model_cfg.get("num_events", 2)),
            uv_head_hidden_dim=model_cfg.get("uv_head_hidden_dim"),
            uv_head_dropout=float(model_cfg.get("uv_head_dropout", model_cfg.get("dropout", 0.1))),
            traj_head_hidden_dim=model_cfg.get("traj_head_hidden_dim"),
            traj_head_dropout=float(model_cfg.get("traj_head_dropout", model_cfg.get("dropout", 0.1))),
            traj_head_layers=int(model_cfg.get("traj_head_layers", 2)),
            event_head_dropout=float(model_cfg.get("event_head_dropout", model_cfg.get("dropout", 0.1))),
            event_names=event_names,
        )

    def forward(
        self,
        *,
        input_type: Literal["uv", "3d"] = "uv",
        ball_uv: Tensor | None = None,
        court_kp: Tensor | None = None,
        ball_pos: Tensor | None = None,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward with explicit input type."""
        if input_type == "3d":
            if ball_pos is None:
                raise ValueError("ball_pos is required for input_type='3d'.")
            return self.forward_3d_event(ball_pos, ball_vis=ball_vis, ball_mask=ball_mask, seq_len=seq_len)

        if ball_uv is None or court_kp is None:
            raise ValueError("ball_uv and court_kp are required for input_type='uv'.")
        return self.forward_uv(
            ball_uv,
            court_kp,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            court_vis=court_vis,
            seq_len=seq_len,
        )

    def forward_uv(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        *,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass for UV inputs.

        Returns a dict with keys:
        - uv_completed
        - position_3d
        - event_logits
        """
        ball_h = self.backbone.forward_uv(
            ball_uv,
            court_kp,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            court_vis=court_vis,
            seq_len=seq_len,
        )
        return {
            "uv_completed": self.uv_head(ball_h),
            "position_3d": self.traj_head(ball_h),
            "event_logits": self.event_head(ball_h),
        }

    def forward_3d_event(
        self,
        ball_pos: Tensor,
        *,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass for 3D inputs (event-only)."""
        ball_h = self.backbone.forward_3d(
            ball_pos,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            seq_len=seq_len,
        )
        return {"event_logits": self.event_head(ball_h)}

    @torch.no_grad()
    def predict_uv(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        *,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Inference for UV inputs."""
        self.eval()
        return self.forward_uv(
            ball_uv,
            court_kp,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            court_vis=court_vis,
            seq_len=seq_len,
        )

    def get_num_params(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BallMultitaskModel(backbone=BallMultitaskBackbone(hidden_dim=32, num_ball_layers=2, num_query_layers=2, num_heads=4))
    ball_uv = torch.randn(2, 8, 2)
    court_kp = torch.randn(2, 20, 2)
    out = model.forward_uv(ball_uv, court_kp)
    assert out["uv_completed"].shape == (2, 8, 2)
    assert out["position_3d"].shape == (2, 8, 3)
    assert out["event_logits"].shape == (2, 8, 2)
    print("multitask_model smoke ok")
