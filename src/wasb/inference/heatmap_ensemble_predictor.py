"""Ensemble heatmap-based ball predictor for WASB models.

This module provides an inference-time ensemble for multiple WASB Lightning
checkpoints trained to predict dense ball heatmaps.

Ensemble rule (per frame):
1) Convert each model heatmap to probabilities (sigmoid by default).
2) Zero-out values below `heatmap_threshold` (e.g. 0.5).
3) Sum the thresholded heatmaps across models.
4) Take argmax of the summed heatmap as the ball position.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from src.wasb.training import WASBLightningModule


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def ensemble_heatmaps_argmax(
    heatmaps: list[Tensor],
    *,
    heatmap_threshold: float = 0.5,
    apply_sigmoid: bool = True,
) -> tuple[Tensor, Tensor]:
    """Threshold-sum heatmaps and return argmax indices + peak scores.

    Args:
        heatmaps: List of heatmaps shaped `[B, H, W]` (all same size).
            Values are assumed to be logits if `apply_sigmoid=True`.
        heatmap_threshold: Probability threshold to zero-out low-confidence
            values after sigmoid.
        apply_sigmoid: If True, apply sigmoid to each heatmap before
            thresholding.

    Returns:
        Tuple of:
          - argmax_idx: `[B]` flattened argmax index (row-major)
          - peak_score: `[B]` max value of summed heatmap
    """
    if not heatmaps:
        raise ValueError("heatmaps must be non-empty")

    thr = float(heatmap_threshold)
    summed: Tensor | None = None
    for hm in heatmaps:
        if hm.dim() != 3:
            raise ValueError(f"Expected heatmap shape [B,H,W], got {tuple(hm.shape)}")
        prob = torch.sigmoid(hm) if apply_sigmoid else hm
        prob = torch.where(prob >= thr, prob, torch.zeros_like(prob))
        summed = prob if summed is None else (summed + prob)

    assert summed is not None
    b, h, w = summed.shape
    flat = summed.view(b, h * w)
    argmax_idx = torch.argmax(flat, dim=-1)
    peak_score = torch.max(flat, dim=-1).values
    return argmax_idx, peak_score


@dataclass
class _Runner:
    module: WASBLightningModule
    frames_in: int
    resize_hw: tuple[int, int] | None
    heatmap_hw: tuple[int, int] | None
    buffer: list[Tensor]

    def reset(self) -> None:
        self.buffer = []

    def _frame_to_tensor(self, frame_rgb: np.ndarray) -> Tensor:
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB uint8 frame [H,W,3], got {frame_rgb.dtype} {frame_rgb.shape}")

        if self.resize_hw is not None:
            h, w = self.resize_hw
            frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        t = torch.from_numpy(frame_rgb).to(dtype=torch.float32)
        t = t.permute(2, 0, 1).contiguous() / 255.0
        return t

    @torch.no_grad()
    def predict_batch_heatmaps(self, frames_rgb: np.ndarray, *, device: torch.device) -> Tensor:
        """Predict per-frame heatmaps for the given batch.

        Returns:
            Heatmaps shaped `[B, H, W]` (single heatmap per input frame).
        """
        if len(frames_rgb) == 0:
            return torch.zeros((0, 1, 1), dtype=torch.float32)

        new_frames = [self._frame_to_tensor(f) for f in frames_rgb]
        all_frames = self.buffer + new_frames
        all_t = torch.stack(all_frames, dim=0)  # [L, C, H, W]

        windows: list[Tensor] = []
        prev_len = len(self.buffer)
        for i in range(len(new_frames)):
            pos = prev_len + i
            start = pos - self.frames_in + 1
            if start < 0:
                idxs = [0] * (-start) + list(range(0, pos + 1))
            else:
                idxs = list(range(start, pos + 1))
            window = all_t[idxs]  # [T, C, H, W]
            windows.append(window)

        window_batch = torch.stack(windows, dim=0).to(device=device)  # [B, T, C, H, W]
        frames_input = self.module.prepare_frames(window_batch)
        outputs = self.module.model(frames_input)
        heatmaps = self.module.extract_heatmaps(outputs)

        if heatmaps.dim() != 4:
            raise ValueError(f"Expected heatmaps [B,T,H,W], got {tuple(heatmaps.shape)}")
        heatmaps = heatmaps[:, -1]  # last target frame

        # Keep a small overlap buffer for the next call.
        keep = max(self.frames_in - 1, 0)
        self.buffer = all_frames[-keep:] if keep > 0 else []

        if self.heatmap_hw is not None:
            hh, hw = self.heatmap_hw
            if heatmaps.shape[-2:] != (hh, hw):
                heatmaps = F.interpolate(
                    heatmaps.unsqueeze(1),
                    size=(hh, hw),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        return heatmaps.detach()


class HeatmapEnsemblePredictor:
    """Ensemble predictor over multiple trained WASB Lightning checkpoints.

    Public API:
      - `load_from_checkpoint(...)` (loads multiple checkpoints)
      - `predict(frames, frame_indices=...)` (batched streaming)
      - `reset_tracker()` (clear temporal buffers)
    """

    def __init__(
        self,
        runners: list[_Runner],
        *,
        device: torch.device,
        heatmap_threshold: float = 0.5,
        apply_sigmoid: bool = True,
        output_heatmap_hw: tuple[int, int] | None = None,
    ) -> None:
        if not runners:
            raise ValueError("runners must be non-empty")
        self.runners = runners
        self.device = device
        self.heatmap_threshold = float(heatmap_threshold)
        self.apply_sigmoid = bool(apply_sigmoid)
        self.output_heatmap_hw = output_heatmap_hw

        self._expected_next_frame_index = 0

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_paths: list[str | Path] | tuple[str | Path, ...],
        *,
        device: str | torch.device = "cpu",
        heatmap_threshold: float = 0.5,
        apply_sigmoid: bool = True,
        output_heatmap_hw: tuple[int, int] | None = None,
    ) -> Self:
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            torch_device = torch.device("cpu")

        ckpts = [Path(p) for p in checkpoint_paths]
        if not ckpts:
            raise ValueError("checkpoint_paths must be non-empty")
        for p in ckpts:
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")

        runners: list[_Runner] = []
        for ckpt in ckpts:
            module = WASBLightningModule.load_from_checkpoint(
                str(ckpt), map_location=torch_device
            )
            module.eval()

            cfg = getattr(module, "config", {}) or {}
            model_cfg = _cfg_get(cfg, "model", {})
            data_cfg = _cfg_get(cfg, "data", {})

            frames_in = int(_cfg_get(model_cfg, "frames_in", _cfg_get(data_cfg, "frames_in", 1)))

            resize_hw = None
            inp_h = _cfg_get(model_cfg, "inp_height", None)
            inp_w = _cfg_get(model_cfg, "inp_width", None)
            if inp_h is not None and inp_w is not None:
                resize_hw = (int(inp_h), int(inp_w))
            else:
                rhw = _cfg_get(data_cfg, "resize_hw", None)
                if rhw is not None:
                    resize_hw = (int(rhw[0]), int(rhw[1]))

            heatmap_hw = None
            hhw = _cfg_get(data_cfg, "heatmap_hw", None)
            if hhw is not None:
                heatmap_hw = (int(hhw[0]), int(hhw[1]))

            runners.append(
                _Runner(
                    module=module,
                    frames_in=max(frames_in, 1),
                    resize_hw=resize_hw,
                    heatmap_hw=heatmap_hw,
                    buffer=[],
                )
            )

        return cls(
            runners=runners,
            device=torch_device,
            heatmap_threshold=heatmap_threshold,
            apply_sigmoid=apply_sigmoid,
            output_heatmap_hw=output_heatmap_hw,
        )

    def reset_tracker(self) -> None:
        for r in self.runners:
            r.reset()
        self._expected_next_frame_index = 0

    @torch.no_grad()
    def predict(
        self,
        frames: np.ndarray,
        *,
        frame_indices: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict ball position for a batch of consecutive frames.

        Args:
            frames: RGB uint8 array shaped `[B, H, W, 3]`.
            frame_indices: Optional frame indices for bookkeeping.

        Returns:
            Dict with:
              - `ball_uv`: `[B,2]` normalized coords in [0,1]
              - `ball_xy_px`: `[B,2]` pixel coords in original frame space
              - `visibility`: `[B]` boolean (True when peak_score>0)
              - `score`: `[B]` peak score of summed heatmap
              - `frame_indices`: `[B]` int64
        """
        if frames.size == 0:
            return {
                "ball_uv": np.zeros((0, 2), dtype=np.float32),
                "ball_xy_px": np.zeros((0, 2), dtype=np.float32),
                "visibility": np.zeros((0,), dtype=bool),
                "score": np.zeros((0,), dtype=np.float32),
                "frame_indices": np.zeros((0,), dtype=np.int64),
            }

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected frames [B,H,W,3], got {frames.shape}")

        b, h0, w0, _ = frames.shape
        if frame_indices is None:
            start = self._expected_next_frame_index
            frame_indices = list(range(start, start + b))

        if len(frame_indices) != b:
            raise ValueError("frame_indices length must match number of frames")

        # If indices jump (e.g. a new clip), drop temporal context.
        if frame_indices and frame_indices[0] != self._expected_next_frame_index:
            for r in self.runners:
                r.reset()
        self._expected_next_frame_index = frame_indices[-1] + 1

        per_model_heatmaps: list[Tensor] = []
        for r in self.runners:
            hm = r.predict_batch_heatmaps(frames, device=self.device)  # [B,H,W]
            if self.output_heatmap_hw is not None and hm.shape[-2:] != self.output_heatmap_hw:
                oh, ow = self.output_heatmap_hw
                hm = F.interpolate(
                    hm.unsqueeze(1), size=(oh, ow), mode="bilinear", align_corners=False
                ).squeeze(1)
            per_model_heatmaps.append(hm)

        # Ensure all heatmaps are the same spatial size for ensembling.
        target_hw = per_model_heatmaps[0].shape[-2:]
        aligned: list[Tensor] = []
        for hm in per_model_heatmaps:
            if hm.shape[-2:] != target_hw:
                hm = F.interpolate(
                    hm.unsqueeze(1),
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            aligned.append(hm)

        argmax_idx, peak_score = ensemble_heatmaps_argmax(
            aligned,
            heatmap_threshold=self.heatmap_threshold,
            apply_sigmoid=self.apply_sigmoid,
        )

        hh, hw = target_hw
        y = (argmax_idx // hw).to(dtype=torch.float32)
        x = (argmax_idx % hw).to(dtype=torch.float32)
        denom_w = float(max(hw - 1, 1))
        denom_h = float(max(hh - 1, 1))
        uv = torch.stack((x / denom_w, y / denom_h), dim=-1).clamp(0.0, 1.0)  # [B,2]

        # Map normalized coordinates to original frame size.
        ball_xy_px = torch.stack(
            (
                uv[:, 0] * float(max(w0 - 1, 1)),
                uv[:, 1] * float(max(h0 - 1, 1)),
            ),
            dim=-1,
        )

        visibility = (peak_score > 0).to(dtype=torch.bool)

        return {
            "ball_uv": uv.detach().cpu().numpy().astype(np.float32, copy=False),
            "ball_xy_px": ball_xy_px.detach().cpu().numpy().astype(np.float32, copy=False),
            "visibility": visibility.detach().cpu().numpy().astype(bool, copy=False),
            "score": peak_score.detach().cpu().numpy().astype(np.float32, copy=False),
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        }
