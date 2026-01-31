"""PyTorch Lightning module for training event detection models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
import torch
from torch import Tensor
from torch.nn import functional as F
from src.base.training.lightning_module import BaseLightningModule
from src.evnet_detection.models.traj3d_event_model import Traj3DEventModel
from src.evnet_detection.models.uv_event_model import UVEventModel
from src.evnet_detection.utils.peaks import extract_event_peaks

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _make_time_mask(seq_len: Tensor, T: int) -> Tensor:
    """Create a (B, T) mask where frames < seq_len are valid."""
    B = seq_len.shape[0]
    t = torch.arange(T, device=seq_len.device)[None, :]
    return t < seq_len.to(torch.long).view(B, 1)


class EventDetectionLightningModule(BaseLightningModule):
    """Lightning module for training UV/3D event detection models."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        model_cfg = self.config.get("model", {}) or {}
        model_name = str(model_cfg.get("name", "uv_transformer"))
        self.input_type: Literal["uv", "3d"] = "3d" if "traj3d" in model_name else "uv"

        if self.input_type == "3d":
            self.model = Traj3DEventModel.from_config(self.config)
        else:
            self.model = UVEventModel.from_config(self.config)

        train_cfg = self.config.get("training", {}) or {}
        metrics_cfg = self.config.get("metrics", {}) or {}
        self.peak_threshold = float(metrics_cfg.get("peak_threshold", 0.5))
        self.match_tolerance_frames = int(metrics_cfg.get("match_tolerance_frames", 3))

        pos_weight_cfg = train_cfg.get("pos_weight", [1.0, 1.0])
        pos_weight = torch.tensor(pos_weight_cfg, dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight, persistent=False)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Forward pass producing logits.

        Args:
            batch: Batch dictionary.

        Returns:
            Logits tensor of shape (B, T, E).
        """
        if self.input_type == "3d":
            return self.model(batch["ball_pos_world"], seq_len=batch.get("seq_len"))
        return self.model(
            batch["ball_uv"],
            batch["court_kp"],
            ball_vis=batch.get("ball_vis"),
            ball_mask=batch.get("ball_mask"),
            court_vis=batch.get("court_vis"),
            seq_len=batch.get("seq_len"),
        )

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        logits = self.forward(batch)  # (B, T, E)
        targets = batch["targets"].to(logits.dtype)
        B, T, E = logits.shape

        pos_weight = self.pos_weight
        if pos_weight.numel() != E:
            raise ValueError(f"training.pos_weight has length {pos_weight.numel()} but num_events={E}")

        loss_per = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=pos_weight.to(device=logits.device, dtype=logits.dtype),
        )
        time_mask = _make_time_mask(batch["seq_len"], T).to(loss_per.dtype)  # (B, T)
        loss = (loss_per * time_mask.unsqueeze(-1)).sum() / (time_mask.sum() * E + 1e-8)

        accuracy = self._peak_match_accuracy(logits, targets, batch.get("seq_len"))

        self.log(f"{stage}/loss", loss, prog_bar=(stage != "test"))
        self.log(f"{stage}/accuracy", accuracy, prog_bar=(stage != "test"), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        _ = batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        _ = batch_idx
        self._shared_step(batch, "val")

    # configure_optimizers inherited from BaseLightningModule

    def _peak_match_accuracy(self, logits: Tensor, targets: Tensor, seq_len: Tensor | None) -> Tensor:
        probs = torch.sigmoid(logits)
        pred_peaks, _ = extract_event_peaks(
            probs,
            seq_len,
            threshold=self.peak_threshold,
            min_distance=max(self.match_tolerance_frames, 1),
            top_k=None,
        )
        gt_peaks, _ = extract_event_peaks(
            targets,
            seq_len,
            threshold=self.peak_threshold,
            min_distance=1,
            top_k=None,
        )

        B, _, E = probs.shape
        correct = 0
        total = B * E
        tolerance = max(self.match_tolerance_frames, 0)
        for b in range(B):
            for e in range(E):
                pred = pred_peaks[b][e]
                gt = gt_peaks[b][e]
                if not gt and not pred:
                    correct += 1
                    continue
                if not gt or not pred:
                    continue
                matched = False
                for g in gt:
                    if any(abs(g - p) <= tolerance for p in pred):
                        matched = True
                        break
                if matched:
                    correct += 1
        return logits.new_tensor(correct / max(total, 1))


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = {
        "model": {"name": "uv_transformer", "hidden_dim": 32, "num_layers": 2, "num_heads": 4, "max_seq_len": 16, "num_events": 2},
        "training": {"pos_weight": [1.0, 1.0]},
    }
    module = EventDetectionLightningModule(cfg)  # type: ignore[arg-type]
    batch = {
        "ball_uv": torch.rand(2, 16, 2),
        "ball_vis": torch.ones(2, 16),
        "ball_mask": torch.ones(2, 16),
        "court_kp": torch.rand(2, 20, 2),
        "court_vis": torch.ones(2, 20),
        "targets": torch.zeros(2, 16, 2),
        "seq_len": torch.tensor([16, 8]),
    }
    logits = module.forward(batch)
    assert logits.shape == (2, 16, 2)
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    print("lightning smoke ok")
