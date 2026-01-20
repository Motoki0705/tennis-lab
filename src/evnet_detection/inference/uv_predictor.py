"""UV event detection inference predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.base.api.predictor import BasePredictor
from src.evnet_detection.models.uv_event_model import UVEventModel
from src.evnet_detection.training.lightning_module import EventDetectionLightningModule


def _find_peaks_1d(
    values: Tensor,
    *,
    threshold: float,
    min_distance: int,
    top_k: int | None,
) -> tuple[list[int], list[float]]:
    """Find peak indices and scores in a 1D tensor.

    Args:
        values: 1D tensor of scores.
        threshold: Minimum score for a peak.
        min_distance: Minimum index distance between peaks.
        top_k: Optional limit on number of peaks (by score).

    Returns:
        Tuple of (indices, scores) as Python lists.
    """
    if values.numel() == 0:
        return [], []

    threshold = float(threshold)
    min_distance = max(int(min_distance), 1)
    values = values.detach().cpu()

    left = torch.cat([values[:1], values[:-1]])
    right = torch.cat([values[1:], values[-1:]])
    is_peak = (values >= left) & (values >= right) & (values >= threshold)

    idx = torch.nonzero(is_peak).flatten().tolist()
    scores = values[is_peak].tolist()

    if not idx:
        return [], []

    if min_distance > 1:
        order = sorted(range(len(idx)), key=lambda i: scores[i], reverse=True)
        selected_idx: list[int] = []
        selected_scores: list[float] = []
        for i in order:
            t = idx[i]
            if all(abs(t - s) >= min_distance for s in selected_idx):
                selected_idx.append(t)
                selected_scores.append(float(scores[i]))
        idx = selected_idx
        scores = selected_scores

    if top_k is not None and len(idx) > int(top_k):
        order = sorted(range(len(idx)), key=lambda i: scores[i], reverse=True)[: int(top_k)]
        idx = [idx[i] for i in order]
        scores = [scores[i] for i in order]

    order_time = sorted(range(len(idx)), key=lambda i: idx[i])
    idx = [idx[i] for i in order_time]
    scores = [scores[i] for i in order_time]
    return idx, scores


def _extract_event_peaks(
    probs: Tensor,
    seq_len: Tensor | None,
    *,
    threshold: float,
    min_distance: int,
    top_k: int | None,
) -> tuple[list[list[list[int]]], list[list[list[float]]]]:
    """Extract per-event peak indices and scores.

    Args:
        probs: Event probabilities of shape (B, T, E).
        seq_len: Optional sequence lengths of shape (B,).
        threshold: Minimum score for a peak.
        min_distance: Minimum index distance between peaks.
        top_k: Optional limit on number of peaks (by score).

    Returns:
        Tuple of (peaks, peak_scores), both shaped [B][E][N].
    """
    B, T, E = probs.shape
    peaks: list[list[list[int]]] = []
    peak_scores: list[list[list[float]]] = []
    seq_len_cpu = seq_len.detach().cpu() if seq_len is not None else None

    for b in range(B):
        b_peaks: list[list[int]] = []
        b_scores: list[list[float]] = []
        t_len = int(seq_len_cpu[b].item()) if seq_len_cpu is not None else T
        t_len = max(0, min(T, t_len))
        for e in range(E):
            series = probs[b, :t_len, e]
            idx, scores = _find_peaks_1d(
                series,
                threshold=threshold,
                min_distance=min_distance,
                top_k=top_k,
            )
            b_peaks.append(idx)
            b_scores.append(scores)
        peaks.append(b_peaks)
        peak_scores.append(b_scores)

    return peaks, peak_scores


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
        court_kp: Tensor,
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
            court_kp: Court keypoints. Shape (B, 20, 2) or (20, 2).
            ball_mask: Ball visibility mask. Shape (B, T) or (T,).
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
        if ball_uv.dim() == 2:
            ball_uv = ball_uv.unsqueeze(0)
        if court_kp.dim() == 2:
            court_kp = court_kp.unsqueeze(0)
        if ball_mask is not None and ball_mask.dim() == 1:
            ball_mask = ball_mask.unsqueeze(0)
        if court_vis is not None and court_vis.dim() == 1:
            court_vis = court_vis.unsqueeze(0)
        if seq_len is not None and seq_len.dim() == 0:
            seq_len = seq_len.unsqueeze(0)

        ball_uv, court_kp, ball_mask, court_vis = self._to_device(
            self.device, ball_uv, court_kp, ball_mask, court_vis
        )
        if seq_len is not None:
            seq_len = seq_len.to(self.device)

        logits = self.model(
            ball_uv,
            court_kp,
            ball_mask=ball_mask,
            court_vis=court_vis,
            seq_len=seq_len,
        )
        probs = torch.sigmoid(logits)

        peaks, peak_scores = _extract_event_peaks(
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
    ball_mask = torch.ones(1, 16)
    court_vis = torch.ones(1, 20)
    seq_len = torch.tensor([16])
    out = predictor.predict(
        ball_uv,
        court_kp,
        ball_mask=ball_mask,
        court_vis=court_vis,
        seq_len=seq_len,
        threshold=0.3,
        min_distance=2,
    )
    assert out["event_logits"].shape == (1, 16, 2)
    assert out["event_probs"].shape == (1, 16, 2)
    print("evnet_detection.uv_predictor smoke ok")
