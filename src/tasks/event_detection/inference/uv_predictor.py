"""UV event detection inference predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.base.inference.predictor import BasePredictor
from src.event_detection.models.uv_event_model import UVEventModel
from src.event_detection.training.lightning_module import EventDetectionLightningModule
from src.event_detection.utils.peaks import extract_event_peaks


class UVEventPredictor(BasePredictor):
    """UV event detection inference predictor.

    Predicts per-frame event logits/probabilities and extracts peak timings.

    Args:
        model: UVEventModel instance.
        device: Inference device.
        event_names: Optional list of event labels.
    """

    def __init__(
        self,
        model: UVEventModel,
        device: torch.device,
        event_names: list[str] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
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
        """Create a UVEventPredictor from a Lightning checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Unused (for compatibility).

        Returns:
            Initialized UVEventPredictor instance.
        """
        _ = kwargs
        checkpoints = cls._ensure_checkpoint(checkpoint_path)
        if len(checkpoints) != 1:
            raise ValueError("UVEventPredictor expects a single checkpoint path.")
        device = cls._resolve_device(device)
        lightning_module = EventDetectionLightningModule.load_from_checkpoint(
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
        court_kp: Tensor | None = None,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
        *,
        threshold: float = 0.5,
        min_distance: int = 1,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Predict event logits/probabilities and extract event timings.

        Args:
            ball_uv: Ball UV trajectory. Shape (B, T, 2) or (T, 2).
            court_kp: Court keypoints. Shape (B, 20, 2) or (20, 2). Optional for nocourt model.
            ball_vis: Ball visibility flags. Shape (B, T) or (T,).
            ball_mask: Ball padding mask. Shape (B, T) or (T,).
            court_vis: Court visibility mask. Shape (B, 20) or (20,).
            seq_len: Optional sequence lengths. Shape (B,) or scalar.
            threshold: Minimum probability for peak detection.
            min_distance: Minimum distance between peaks (frames).
            top_k: Optional maximum number of peaks per event.

        Returns:
            Dictionary with:
                - event_logits: (B, T, E) logits
                - event_probs: (B, T, E) probabilities
                - event_peaks: list[B][E][N] of peak indices
                - event_peak_scores: list[B][E][N] of peak scores
                - event_names: list of event labels
        """
        if ball_vis is None and ball_mask is not None:
            ball_vis, ball_mask = ball_mask, None

        use_court_context = bool(getattr(self.model, "uses_court_context", True))

        if ball_uv.dim() == 2:
            ball_uv = ball_uv.unsqueeze(0)
        if use_court_context and court_kp is None:
            raise ValueError("court_kp is required for court-aware models.")
        if court_kp is not None and court_kp.dim() == 2:
            court_kp = court_kp.unsqueeze(0)
        if ball_vis is not None and ball_vis.dim() == 1:
            ball_vis = ball_vis.unsqueeze(0)
        if ball_mask is not None and ball_mask.dim() == 1:
            ball_mask = ball_mask.unsqueeze(0)
        if court_vis is not None and court_vis.dim() == 1:
            court_vis = court_vis.unsqueeze(0)
        if seq_len is not None and seq_len.dim() == 0:
            seq_len = seq_len.unsqueeze(0)

        if use_court_context:
            ball_uv, court_kp, ball_vis, court_vis = self._to_device(
                self.device, ball_uv, court_kp, ball_vis, court_vis
            )
        else:
            ball_uv, ball_vis = self._to_device(self.device, ball_uv, ball_vis)
        if ball_mask is not None:
            ball_mask = ball_mask.to(self.device)
        if seq_len is not None:
            seq_len = seq_len.to(self.device)

        if use_court_context:
            logits = self.model(
                ball_uv,
                court_kp,
                ball_vis=ball_vis,
                ball_mask=ball_mask,
                court_vis=court_vis,
                seq_len=seq_len,
            )
        else:
            logits = self.model(
                ball_uv,
                ball_vis=ball_vis,
                ball_mask=ball_mask,
                seq_len=seq_len,
            )
        probs = torch.sigmoid(logits)

        peaks, peak_scores = extract_event_peaks(
            probs,
            seq_len,
            threshold=threshold,
            min_distance=min_distance,
            top_k=top_k,
        )

        return {
            "event_logits": logits.cpu(),
            "event_probs": probs.cpu(),
            "event_peaks": peaks,
            "event_peak_scores": peak_scores,
            "event_names": list(self.event_names),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    model = UVEventModel(hidden_dim=32, num_layers=2, num_heads=4, max_seq_len=16, num_events=2)
    predictor = UVEventPredictor(model=model, device=torch.device("cpu"), event_names=["shot", "bounce"])
    ball_uv = torch.rand(1, 16, 2)
    court_kp = torch.rand(1, 20, 2)
    ball_vis = torch.ones(1, 16)
    ball_mask = torch.ones(1, 16)
    court_vis = torch.ones(1, 20)
    seq_len = torch.tensor([16])
    out = predictor.predict(
        ball_uv,
        court_kp,
        ball_vis=ball_vis,
        ball_mask=ball_mask,
        court_vis=court_vis,
        seq_len=seq_len,
        threshold=0.3,
        min_distance=2,
    )
    assert out["event_logits"].shape == (1, 16, 2)
    assert out["event_probs"].shape == (1, 16, 2)
    print("event_detection.uv_predictor smoke ok")
