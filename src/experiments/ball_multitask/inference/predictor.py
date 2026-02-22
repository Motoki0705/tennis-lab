"""Offline predictor for ball multi-task models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.base.inference.predictor import BasePredictor
from src.ball_multitask.models.multitask_model import BallMultitaskModel
from src.ball_multitask.training.lightning_module import BallMultitaskLightningModule
from src.event_detection.utils.peaks import extract_event_peaks
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class BallMultitaskPredictor(BasePredictor):
    """Offline predictor returning UV completion, 3D trajectory, and event peaks."""

    def __init__(
        self,
        model: BallMultitaskModel,
        device: torch.device,
        *,
        norm_scale_xyz: tuple[float, float, float] = COURT_COORD_SCALE_XYZ,
        event_names: list[str] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.norm_scale_xyz = norm_scale_xyz
        self.model.eval()
        if event_names is None:
            event_names = [f"event_{i}" for i in range(int(self.model.num_events))]
        self.event_names = event_names

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        _ = kwargs
        checkpoints = cls._ensure_checkpoint(checkpoint_path)
        if len(checkpoints) != 1:
            raise ValueError("BallMultitaskPredictor expects a single checkpoint path.")
        device = cls._resolve_device(device)
        lightning_module = BallMultitaskLightningModule.load_from_checkpoint(
            checkpoints[0],
            map_location=device,
        )
        model = lightning_module.model
        event_names = None
        if hasattr(lightning_module, "config"):
            model_cfg = lightning_module.config.get("model", {}) or {}
            names = model_cfg.get("event_names")
            if names:
                event_names = [str(name) for name in names]
        return cls(model=model, device=device, event_names=event_names)

    @torch.no_grad()
    def predict(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        *,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
        threshold: float = 0.5,
        min_distance: int = 1,
        top_k: int | None = None,
        denormalize: bool = True,
        in_frame_threshold: float = 0.5,
        cut_out_of_frame: bool = False,
    ) -> dict[str, Any]:
        """Run offline inference on UV inputs."""
        if ball_vis is None and ball_mask is not None:
            ball_vis, ball_mask = ball_mask, None

        if ball_uv.dim() == 2:
            ball_uv = ball_uv.unsqueeze(0)
        if court_kp.dim() == 2:
            court_kp = court_kp.unsqueeze(0)
        if ball_vis is not None and ball_vis.dim() == 1:
            ball_vis = ball_vis.unsqueeze(0)
        if ball_mask is not None and ball_mask.dim() == 1:
            ball_mask = ball_mask.unsqueeze(0)
        if court_vis is not None and court_vis.dim() == 1:
            court_vis = court_vis.unsqueeze(0)
        if seq_len is not None and seq_len.dim() == 0:
            seq_len = seq_len.unsqueeze(0)

        ball_uv, court_kp, ball_vis, court_vis = self._to_device(
            self.device, ball_uv, court_kp, ball_vis, court_vis
        )
        if ball_mask is not None:
            ball_mask = ball_mask.to(self.device)
        if seq_len is not None:
            seq_len = seq_len.to(self.device)

        outputs = self.model.predict_uv(
            ball_uv,
            court_kp,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            court_vis=court_vis,
            seq_len=seq_len,
        )

        uv_completed = outputs["uv_completed"]
        pos3d = outputs["position_3d"]
        logits = outputs["event_logits"]
        in_frame_logits = outputs["in_frame_logits"]
        in_frame_probs = torch.sigmoid(in_frame_logits)
        in_frame_pred = in_frame_probs >= float(in_frame_threshold)
        probs = torch.sigmoid(logits)

        peaks, peak_scores = extract_event_peaks(
            probs,
            seq_len,
            threshold=float(threshold),
            min_distance=int(min_distance),
            top_k=top_k,
        )

        if denormalize:
            pos3d = self._denormalize_position(pos3d)
        if cut_out_of_frame:
            uv_completed = uv_completed.clone()
            uv_completed[~in_frame_pred] = torch.nan

        return {
            "uv_completed": uv_completed.cpu(),
            "position_3d": pos3d.cpu(),
            "event_logits": logits.cpu(),
            "event_probs": probs.cpu(),
            "in_frame_logits": in_frame_logits.cpu(),
            "in_frame_probs": in_frame_probs.cpu(),
            "in_frame_pred": in_frame_pred.to(torch.float32).cpu(),
            "event_peaks": peaks,
            "event_peak_scores": peak_scores,
            "event_names": list(self.event_names),
        }

    def _denormalize_position(self, position: Tensor) -> Tensor:
        scale = torch.tensor(
            list(self.norm_scale_xyz),
            device=position.device,
            dtype=position.dtype,
        )
        return position * scale


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BallMultitaskModel.from_config({"model": {"hidden_dim": 32, "num_layers": 2, "num_heads": 4}})
    predictor = BallMultitaskPredictor(model=model, device=torch.device("cpu"))
    ball_uv = torch.rand(1, 8, 2)
    court_kp = torch.rand(1, 20, 2)
    out = predictor.predict(ball_uv, court_kp)
    assert out["uv_completed"].shape == (1, 8, 2)
    print("ball_multitask.predictor smoke ok")
