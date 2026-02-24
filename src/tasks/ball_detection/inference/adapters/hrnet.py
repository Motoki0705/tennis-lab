"""WASB-HRNet temporal context input adapter."""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor

from src.tasks.ball_detection.inference.adapters.base import ModelInputAdapter


class HRNetContextInputAdapter(ModelInputAdapter):
    """Adapter for WASB-HRNet that stacks context frames along channels."""

    def __init__(self, *, context_frames: int) -> None:
        self.context_frames = int(context_frames)
        if self.context_frames <= 0:
            raise ValueError("HRNet adapter requires positive context_frames.")
        self._buffer: deque[Tensor] = deque(maxlen=self.context_frames)

    def reset(self) -> None:
        self._buffer.clear()

    def step_input(self, frame_chw: Tensor) -> Tensor:
        if frame_chw.dim() != 3 or frame_chw.shape[0] != 3:
            raise ValueError(f"Expected frame shape [3,H,W], got {tuple(frame_chw.shape)}")
        frame_cpu = frame_chw.detach().cpu()
        self._buffer.append(frame_cpu)
        context = list(self._buffer)
        if not context:
            raise RuntimeError("HRNet adapter buffer is unexpectedly empty.")
        if len(context) < self.context_frames:
            pad = [context[0]] * (self.context_frames - len(context))
            context = pad + context
        return torch.cat(context, dim=0).unsqueeze(0)

    def extract_current_logits(self, logits: Tensor) -> Tensor:
        if logits.dim() == 4:
            return logits[:, -1]
        if logits.dim() == 3:
            return logits
        raise ValueError(
            "HRNet logits must have shape [B,T,H,W] or [B,H,W], "
            f"got {tuple(logits.shape)}"
        )
