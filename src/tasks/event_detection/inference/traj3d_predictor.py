"""3D trajectory event detection inference predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.event_detection.models.traj3d_event_model import Traj3DEventModel
from src.tasks.event_detection.training.lightning_module import EventDetectionLightningModule

from src.tasks.event_detection.utils.peaks import extract_event_peaks


class Traj3DEventPredictor(BasePredictor):
    """3D trajectory event detection inference predictor.

    Predicts per-frame event logits/probabilities and extracts peak timings.

    Args:
        model: Traj3DEventModel instance.
        device: Inference device.
        event_names: Optional list of event labels.
    """

    def __init__(
        self,
        model: Traj3DEventModel,
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
        """Create a Traj3DEventPredictor from a Lightning checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Unused (for compatibility).

        Returns:
            Initialized Traj3DEventPredictor instance.
        """
        _ = kwargs
        checkpoints = cls._ensure_checkpoint(checkpoint_path)
        if len(checkpoints) != 1:
            raise ValueError("Traj3DEventPredictor expects a single checkpoint path.")
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
        ball_pos_world: Tensor,
        seq_len: Tensor | None = None,
        *,
        threshold: float = 0.5,
        min_distance: int = 1,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Predict event logits/probabilities and extract event timings.

        Args:
            ball_pos_world: Ball 3D trajectory. Shape (B, T, 3) or (T, 3).
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
        if ball_pos_world.dim() == 2:
            ball_pos_world = ball_pos_world.unsqueeze(0)
        if seq_len is not None and seq_len.dim() == 0:
            seq_len = seq_len.unsqueeze(0)

        (ball_pos_world,) = self._to_device(self.device, ball_pos_world)
        if seq_len is not None:
            seq_len = seq_len.to(self.device)

        logits = self.model(ball_pos_world, seq_len=seq_len)
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
    model = Traj3DEventModel(hidden_dim=32, num_layers=2, num_heads=4, max_seq_len=16, num_events=2)
    predictor = Traj3DEventPredictor(model=model, device=torch.device("cpu"))
    ball_pos_world = torch.randn(1, 16, 3)
    seq_len = torch.tensor([16])
    out = predictor.predict(ball_pos_world, seq_len=seq_len, threshold=0.2, min_distance=2)
    assert out["event_logits"].shape == (1, 16, 2)
    assert out["event_probs"].shape == (1, 16, 2)
    print("event_detection.traj3d_predictor smoke ok")
