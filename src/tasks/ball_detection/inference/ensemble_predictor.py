"""Ensemble predictor combining local heatmap models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from src.ball_detection.inference.adapters import ModelInputAdapter, build_adapter_for_model
from src.ball_detection.inference.predictor import BallPredictor
from src.ball_detection.models.heatmap_utils import decode_heatmap_logits


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
            self.adapters = [build_adapter_for_model(p.model) for p in self.predictors]
        else:
            if len(adapters) != len(predictors):
                raise ValueError("adapters length must match predictors length")
            self.adapters = adapters

    def reset(self) -> None:
        """Reset all model adapters (clears temporal buffers)."""
        for adapter in self.adapters:
            adapter.reset()

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
        model_config_paths: list[str | Path | None] | None = None,
        visibility_threshold: float = 0.5,
    ) -> "BallEnsemblePredictor":
        if model_config_paths is not None and len(model_config_paths) != len(checkpoint_paths):
            raise ValueError("model_config_paths length must match checkpoint_paths length")
        predictors: list[BallPredictor] = []
        for idx, path in enumerate(checkpoint_paths):
            fallback_model_cfg_path = None
            if model_config_paths is not None:
                fallback_model_cfg_path = model_config_paths[idx]
            predictors.append(
                BallPredictor.load_from_checkpoint(
                    path,
                    device=device,
                    fallback_model_cfg_path=fallback_model_cfg_path,
                )
            )
        return cls(
            predictors=predictors,
            weights=weights,
            visibility_threshold=visibility_threshold,
        )

    @torch.no_grad()
    def predict(self, frames: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        reset_state = bool(kwargs.get("reset_state", True))
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        if frames.dim() != 4:
            raise ValueError(f"frames must have shape [T,3,H,W] or [3,H,W], got {tuple(frames.shape)}")
        if frames.shape[1] != 3:
            raise ValueError(f"Expected RGB frames with C=3, got C={frames.shape[1]}")

        if reset_state:
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
                logits_bhw = adapter.extract_current_logits(logits)

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
