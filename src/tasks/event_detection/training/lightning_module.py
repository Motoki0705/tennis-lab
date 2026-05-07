"""PyTorch Lightning module for training event detection models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_callback import save_image_to_tensorboard
from src.tasks.event_detection.models import build_event_detection_model
from src.tasks.event_detection.utils.peaks import extract_event_peaks

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
        if model_name == "traj3d_transformer":
            self.input_type: Literal["uv", "uv_nocourt", "3d"] = "3d"
        elif model_name == "uv_transformer":
            self.input_type = "uv"
        elif model_name == "uv_transformer_nocourt":
            self.input_type = "uv_nocourt"
        else:
            raise ValueError(
                "Unknown event_detection model.name="
                f"'{model_name}'. Supported: ['uv_transformer', 'uv_transformer_nocourt', 'traj3d_transformer']"
            )
        self.model = build_event_detection_model(self.config)

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
        if self.input_type == "uv_nocourt":
            return self.model(
                batch["ball_uv"],
                ball_vis=batch.get("ball_vis"),
                ball_mask=batch.get("ball_mask"),
                seq_len=batch.get("seq_len"),
            )
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

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        _ = batch_idx
        self._shared_step(batch, "test")

    # ------------------------------------------------------------------
    # Qualitative validation logging
    # ------------------------------------------------------------------

    def render_qualitative_samples(
        self,
        batches: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        artifact_dir: Path,
        tb_writer: Any | None,
        global_step: int,
        epoch: int,
    ) -> None:
        """Render event probability timelines with GT/predicted peak markers."""
        device = next(self.parameters()).device

        for batch_idx, batch in enumerate(batches):
            batch_dev = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                logits = self.forward(batch_dev).cpu()  # (B, T, E)

            targets = batch["targets"]  # (B, T, E)
            seq_len = batch.get("seq_len")  # (B,)
            probs = torch.sigmoid(logits)

            # Render first sample
            b = 0
            T = int(seq_len[b].item()) if seq_len is not None else probs.shape[1]
            E = probs.shape[2]
            event_names = [f"event_{e}" for e in range(E)]

            fig_h = max(200 * E, 200)
            fig_w = 800
            panel = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255

            row_h = fig_h // E
            for e in range(E):
                y_off = e * row_h
                prob_e = probs[b, :T, e].numpy()
                gt_e = targets[b, :T, e].numpy()

                # Draw axes
                cv2.line(panel, (50, y_off + row_h - 30), (fig_w - 20, y_off + row_h - 30), (0, 0, 0), 1)
                cv2.putText(panel, event_names[e], (5, y_off + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Plot probability curve
                plot_h = row_h - 50
                for t in range(1, T):
                    x1 = 50 + int((t - 1) / max(T - 1, 1) * (fig_w - 70))
                    x2 = 50 + int(t / max(T - 1, 1) * (fig_w - 70))
                    y1 = y_off + row_h - 30 - int(prob_e[t - 1] * plot_h)
                    y2 = y_off + row_h - 30 - int(prob_e[t] * plot_h)
                    cv2.line(panel, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw GT peaks as green markers
                for t in range(T):
                    if gt_e[t] > 0.5:
                        x = 50 + int(t / max(T - 1, 1) * (fig_w - 70))
                        cv2.drawMarker(panel, (x, y_off + row_h - 30), (0, 180, 0), cv2.MARKER_TRIANGLE_UP, 12, 2)

                # Draw predicted peaks as red markers
                pred_peaks_list, _ = extract_event_peaks(
                    probs[b:b + 1, :T],
                    seq_len[b:b + 1] if seq_len is not None else None,
                    threshold=self.peak_threshold,
                    min_distance=max(self.match_tolerance_frames, 1),
                    top_k=None,
                )
                for pk in pred_peaks_list[0][e]:
                    x = 50 + int(pk / max(T - 1, 1) * (fig_w - 70))
                    cv2.drawMarker(panel, (x, y_off + row_h - 45), (0, 0, 255), cv2.MARKER_TRIANGLE_DOWN, 12, 2)

            path = artifact_dir / f"event_batch{batch_idx:02d}.png"
            cv2.imwrite(str(path), panel)

            save_image_to_tensorboard(
                tb_writer,
                f"qualitative/event_detection/batch{batch_idx:02d}",
                panel,
                global_step,
            )

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
    num_court_kp = 12
    cfg = {
        "model": {"name": "uv_transformer", "hidden_dim": 32, "num_layers": 2, "num_heads": 4, "max_seq_len": 16, "num_events": 2},
        "data": {"num_court_kp": num_court_kp},
        "training": {"pos_weight": [1.0, 1.0]},
    }
    module = EventDetectionLightningModule(cfg)  # type: ignore[arg-type]
    batch = {
        "ball_uv": torch.rand(2, 16, 2),
        "ball_vis": torch.ones(2, 16),
        "ball_mask": torch.ones(2, 16),
        "court_kp": torch.rand(2, num_court_kp, 2),
        "court_vis": torch.ones(2, num_court_kp),
        "targets": torch.zeros(2, 16, 2),
        "seq_len": torch.tensor([16, 8]),
    }
    logits = module.forward(batch)
    assert logits.shape == (2, 16, 2)
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    print("lightning smoke ok")
