"""Inference predictor for local ball_detection checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.base.inference.predictor import BasePredictor
from src.ball_detection.models.heatmap_utils import decode_heatmap_logits
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
        state = torch.load(ckpt[0], map_location=device_obj, weights_only=False)

        cfg: dict[str, Any] = {}
        hyper_params = state.get("hyper_parameters", {}) if isinstance(state, dict) else {}
        if isinstance(hyper_params, dict):
            cfg_candidate = hyper_params.get("config", {})
            if isinstance(cfg_candidate, DictConfig):
                cfg = OmegaConf.to_container(cfg_candidate, resolve=True)  # type: ignore[assignment]
            if isinstance(cfg_candidate, dict):
                cfg = cfg_candidate

        model = build_model(cfg)
        state_dict = state.get("state_dict", state)
        target_keys = set(model.state_dict().keys())
        remapped: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            if key in target_keys:
                remapped[key] = value
                continue

            candidate = key
            while candidate.startswith("model."):
                candidate = candidate.removeprefix("model.")
                if candidate in target_keys:
                    remapped[candidate] = value
                    break
            else:
                remapped[candidate] = value

        if len(remapped) == 0:
            raise ValueError(f"No loadable keys found in checkpoint: {ckpt[0]}")

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if unexpected:
            # Unexpected keys are tolerated to support checkpoints with extra metadata.
            pass
        if missing:
            # Missing keys are tolerated to support lightweight checkpoints.
            pass
        return cls(model=model, device=device_obj)

    @torch.no_grad()
    def predict_heatmap_logits(self, frames: Tensor, **kwargs: Any) -> Tensor:
        """Predict heatmap logits from a heatmap-based checkpoint model.

        Args:
            frames: Input tensor consumed by the underlying model.
            **kwargs: Reserved for future compatibility.

        Returns:
            Heatmap logits with shape [B, H, W] or [B, T, H, W].
        """
        _ = kwargs
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        frames = frames.to(self.device)
        out = self.model(frames)
        heatmap_logits = out.get("heatmap_logits")
        if heatmap_logits is None:
            raise ValueError("Loaded checkpoint model does not provide heatmap_logits.")
        if heatmap_logits.dim() not in (3, 4):
            raise ValueError(
                "heatmap_logits must have shape [B, H, W] or [B, T, H, W], "
                f"got {tuple(heatmap_logits.shape)}"
            )
        return heatmap_logits.detach().cpu()

    @torch.no_grad()
    def predict(self, frames: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        _ = kwargs
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        frames = frames.to(self.device)
        out = self.model(frames)
        if "heatmap_logits" in out:
            xy, vis_logit = decode_heatmap_logits(out["heatmap_logits"])
        else:
            xy = out["xy"]
            vis_logit = out["visibility_logit"]
        xy = xy.detach().cpu()
        vis_logit = vis_logit.detach().cpu()
        vis_prob = torch.sigmoid(vis_logit)
        return {
            "ball_uv": xy,
            "visibility": (vis_prob >= 0.5).to(torch.float32),
            "score": vis_prob,
        }
