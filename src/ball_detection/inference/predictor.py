"""Inference predictor for local ball_detection checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.base.inference.predictor import BasePredictor
from src.ball_detection.models import build_model


class BallPredictor(BasePredictor):
    """Checkpoint-based predictor for `BallDetectorModel`."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        _ = kwargs
        ckpt = cls._ensure_checkpoint(checkpoint_path)
        device_obj = cls._resolve_device(device)
        state = torch.load(ckpt[0], map_location=device_obj)

        model = build_model({})
        state_dict = state.get("state_dict", state)
        remapped = {
            k.replace("model.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("model.") or k.startswith("encoder")
        }
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys in checkpoint: {unexpected}")
        if missing:
            # Missing keys are tolerated to support lightweight checkpoints.
            pass
        return cls(model=model, device=device_obj)

    @torch.no_grad()
    def predict(self, frames: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        _ = kwargs
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        frames = frames.to(self.device)
        out = self.model(frames)
        xy = out["xy"].detach().cpu()
        vis_logit = out["visibility_logit"].detach().cpu()
        vis_prob = torch.sigmoid(vis_logit)
        return {
            "ball_uv": xy,
            "visibility": (vis_prob >= 0.5).to(torch.float32),
            "score": vis_prob,
        }
