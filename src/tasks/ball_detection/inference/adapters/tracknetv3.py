"""TrackNetV3 temporal input adapter."""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor

from src.ball_detection.inference.adapters.base import ModelInputAdapter


class TrackNetV3InputAdapter(ModelInputAdapter):
    """Adapter for TrackNetV3 fixed-length windows."""

    def __init__(self, *, seq_len: int) -> None:
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError("TrackNetV3 adapter requires positive seq_len.")
        self._buffer: deque[Tensor] = deque(maxlen=self.seq_len)

    def reset(self) -> None:
        self._buffer.clear()

    def step_input(self, frame_chw: Tensor) -> Tensor:
        if frame_chw.dim() != 3 or frame_chw.shape[0] != 3:
            raise ValueError(f"Expected frame shape [3,H,W], got {tuple(frame_chw.shape)}")
        frame_cpu = frame_chw.detach().cpu()
        self._buffer.append(frame_cpu)
        window = list(self._buffer)
        if not window:
            raise RuntimeError("TrackNetV3 adapter buffer is unexpectedly empty.")
        if len(window) < self.seq_len:
            pad = [window[0]] * (self.seq_len - len(window))
            window = pad + window
        return torch.stack(window, dim=0).unsqueeze(0)

    def extract_current_logits(self, logits: Tensor) -> Tensor:
        if logits.dim() == 4:
            return logits[:, -1]
        if logits.dim() == 3:
            return logits
        raise ValueError(
            "TrackNetV3 logits must have shape [B,T,H,W] or [B,H,W], "
            f"got {tuple(logits.shape)}"
        )
