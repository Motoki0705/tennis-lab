"""Ensemble predictor combining local and WASB-origin models."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from src.ball_detection.inference.predictor import BallPredictor
from src.ball_detection.models.heatmap_utils import decode_heatmap_logits


class ModelInputAdapter:
    """Model-specific frame adapter that builds per-timestep model inputs."""

    def reset(self) -> None:
        """Reset internal temporal buffers."""
        raise NotImplementedError

    def step_input(self, frame_chw: Tensor) -> Tensor:
        """Consume one frame and return model input for this timestep."""
        raise NotImplementedError

    def extract_current_logits(self, logits: Tensor) -> Tensor:
        """Extract current-frame logits as shape [B, H, W]."""
        raise NotImplementedError


class TrackNetV3InputAdapter(ModelInputAdapter):
    """Adapter for TrackNetV3 that requires a fixed-length frame window."""

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


class BallEnsemblePredictor:
    """Heatmap-fusion ensemble over multiple predictors with aligned output contract."""

    def __init__(
        self,
        predictors: list[BallPredictor],
        weights: list[float] | None = None,
        *,
        visibility_threshold: float = 0.5,
        adapters: list[ModelInputAdapter] | None = None,
    ) -> None:
        if not predictors:
            raise ValueError("predictors must be non-empty")
        self.predictors = predictors
        self.visibility_threshold = float(visibility_threshold)
        if weights is None:
            self.weights = [1.0 / len(predictors)] * len(predictors)
        else:
            if len(weights) != len(predictors):
                raise ValueError("weights length must match predictors length")
            s = sum(weights)
            if s <= 0:
                raise ValueError("weights sum must be positive")
            self.weights = [float(w) / s for w in weights]
        if adapters is None:
            self.adapters = [self._build_default_adapter(p) for p in self.predictors]
        else:
            if len(adapters) != len(predictors):
                raise ValueError("adapters length must match predictors length")
            self.adapters = adapters

    @staticmethod
    def _build_default_adapter(predictor: BallPredictor) -> ModelInputAdapter:
        model = predictor.model
        seq_len = getattr(model, "seq_len", None)
        if seq_len is not None:
            return TrackNetV3InputAdapter(seq_len=int(seq_len))

        backbone = getattr(model, "backbone", None)
        input_channels = getattr(backbone, "input_channels", None)
        if input_channels is not None:
            channels = int(input_channels)
            if channels % 3 != 0:
                raise ValueError(f"HRNet input channels must be divisible by 3, got {channels}")
            return HRNetContextInputAdapter(context_frames=channels // 3)

        raise ValueError(
            "Could not infer input adapter for predictor model. "
            "Pass adapters explicitly to BallEnsemblePredictor."
        )

    def reset(self) -> None:
        """Reset all model adapters (clears temporal buffers)."""
        for adapter in self.adapters:
            adapter.reset()

    @staticmethod
    def _ensure_logits_bhw(logits: Tensor) -> Tensor:
        if logits.dim() == 3:
            return logits
        if logits.dim() == 4:
            return logits[:, -1]
        raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")

    @staticmethod
    def _align_hw(prob: Tensor, target_hw: tuple[int, int]) -> Tensor:
        if prob.shape[-2:] == target_hw:
            return prob
        resized = F.interpolate(prob.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False)
        return resized[:, 0]

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: list[str | Path],
        *,
        device: str = "cpu",
        weights: list[float] | None = None,
        visibility_threshold: float = 0.5,
    ) -> "BallEnsemblePredictor":
        predictors = [BallPredictor.load_from_checkpoint(path, device=device) for path in checkpoint_paths]
        return cls(
            predictors=predictors,
            weights=weights,
            visibility_threshold=visibility_threshold,
        )

    @torch.no_grad()
    def predict(self, frames: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        _ = kwargs
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        if frames.dim() != 4:
            raise ValueError(f"frames must have shape [T,3,H,W] or [3,H,W], got {tuple(frames.shape)}")
        if frames.shape[1] != 3:
            raise ValueError(f"Expected RGB frames with C=3, got C={frames.shape[1]}")

        self.reset()
        frames_cpu = frames.detach().cpu()

        uv_seq: list[Tensor] = []
        score_seq: list[Tensor] = []
        eps = 1e-6

        for t in range(frames_cpu.shape[0]):
            frame_t = frames_cpu[t]
            prob_fused: Tensor | None = None
            target_hw: tuple[int, int] | None = None

            for weight, predictor, adapter in zip(
                self.weights,
                self.predictors,
                self.adapters,
                strict=True,
            ):
                model_input = adapter.step_input(frame_t)
                logits = predictor.predict_heatmap_logits(model_input)
                logits_bhw = self._ensure_logits_bhw(adapter.extract_current_logits(logits))

                if target_hw is None:
                    target_hw = (int(logits_bhw.shape[-2]), int(logits_bhw.shape[-1]))
                prob = torch.sigmoid(logits_bhw)
                prob = self._align_hw(prob, target_hw)

                if prob_fused is None:
                    prob_fused = prob * float(weight)
                else:
                    prob_fused = prob_fused + prob * float(weight)

            if prob_fused is None:
                raise RuntimeError("No ensemble members produced heatmap logits.")
            fused_logits = torch.logit(torch.clamp(prob_fused, min=eps, max=1.0 - eps))
            xy_t, vis_logit_t = decode_heatmap_logits(fused_logits)
            uv_seq.append(xy_t[0])
            score_seq.append(torch.sigmoid(vis_logit_t[0]))

        xy = torch.stack(uv_seq, dim=0).to(torch.float32)
        score = torch.stack(score_seq, dim=0).to(torch.float32)
        visibility = (score >= self.visibility_threshold).to(torch.float32)
        return {
            "ball_uv": xy,
            "visibility": visibility,
            "score": score,
        }
