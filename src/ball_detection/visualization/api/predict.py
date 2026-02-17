"""Prediction API for ball_detection visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from src.ball_detection.inference import (
    BallEnsemblePredictor,
    BallPredictor,
    ModelInputAdapter,
    build_adapter_for_model,
)
from src.ball_detection.inference.types import InferenceConfig, InferenceResult
from src.ball_detection.models.heatmap_utils import decode_heatmap_logits
from src.ball_detection.visualization.adapters.predict_inputs import PredictionClip


class _PredictRunner(Protocol):
    def reset(self) -> None:
        """Reset internal temporal state."""

    def predict_frames(self, frames_t: torch.Tensor, *, reset_state: bool) -> dict[str, torch.Tensor]:
        """Predict per-frame tensors for one clip."""


@dataclass
class _SingleModelRunner:
    predictor: BallPredictor
    adapter: ModelInputAdapter
    visibility_threshold: float

    def reset(self) -> None:
        self.adapter.reset()

    @torch.no_grad()
    def predict_frames(self, frames_t: torch.Tensor, *, reset_state: bool) -> dict[str, torch.Tensor]:
        if reset_state:
            self.reset()

        uv_seq: list[torch.Tensor] = []
        score_seq: list[torch.Tensor] = []
        for t in range(frames_t.shape[0]):
            model_input = self.adapter.step_input(frames_t[t])
            logits = self.predictor.predict_heatmap_logits(model_input)
            logits_bhw = self.adapter.extract_current_logits(logits)
            xy_t, vis_logit_t = decode_heatmap_logits(logits_bhw)
            uv_seq.append(xy_t[0])
            score_seq.append(torch.sigmoid(vis_logit_t[0]))

        score = torch.stack(score_seq, dim=0).to(torch.float32)
        return {
            "ball_uv": torch.stack(uv_seq, dim=0).to(torch.float32),
            "score": score,
            "visibility": (score >= self.visibility_threshold).to(torch.float32),
        }


@dataclass
class _EnsembleModelRunner:
    predictor: BallEnsemblePredictor

    def reset(self) -> None:
        self.predictor.reset()

    @torch.no_grad()
    def predict_frames(self, frames_t: torch.Tensor, *, reset_state: bool) -> dict[str, torch.Tensor]:
        return self.predictor.predict(frames_t, reset_state=reset_state)


@dataclass
class PredictorRuntime:
    """Stateful predictor runtime reused across clip predictions."""

    inference_config: InferenceConfig
    runner: _PredictRunner

    @staticmethod
    def _frames_to_tensor(
        frames_rgb: NDArray[np.uint8],
        *,
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:
        tensors: list[torch.Tensor] = []
        for frame in frames_rgb:
            resized = cv2.resize(frame, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float() / 255.0
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    @staticmethod
    def _sanitize_scores(scores: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)

    @staticmethod
    def _uv_to_xy_px(
        ball_uv: NDArray[np.float32],
        *,
        width: int,
        height: int,
    ) -> NDArray[np.float32]:
        xy = np.asarray(ball_uv, dtype=np.float32).copy()
        if xy.size == 0:
            return xy
        xy[:, 0] *= float(max(width - 1, 1))
        xy[:, 1] *= float(max(height - 1, 1))
        return xy

    def reset(self) -> None:
        self.runner.reset()

    def predict_clip(
        self,
        *,
        clip: PredictionClip,
        reset_state: bool,
        output_width: int,
        output_height: int,
    ) -> InferenceResult:
        if clip.frames_rgb.ndim != 4 or clip.frames_rgb.shape[-1] != 3:
            raise ValueError(f"Expected clip frames shape [T,H,W,3], got {tuple(clip.frames_rgb.shape)}")
        if clip.frames_rgb.shape[0] != clip.frame_indices.shape[0]:
            raise ValueError("clip frame count and frame_indices length must match")
        if clip.frames_rgb.shape[0] == 0:
            empty_i64 = np.zeros((0,), dtype=np.int64)
            empty_xy = np.zeros((0, 2), dtype=np.float32)
            empty_f32 = np.zeros((0,), dtype=np.float32)
            return InferenceResult(
                frame_indices=empty_i64,
                ball_uv=empty_xy,
                ball_xy_px=empty_xy.copy(),
                visibility=np.zeros((0,), dtype=bool),
                score=empty_f32,
            )

        frames_t = self._frames_to_tensor(
            clip.frames_rgb,
            image_h=int(self.inference_config.image_h),
            image_w=int(self.inference_config.image_w),
        )
        outputs = self.runner.predict_frames(frames_t, reset_state=reset_state)

        ball_uv = outputs["ball_uv"].detach().cpu().numpy().astype(np.float32, copy=False)
        score = outputs["score"].detach().cpu().numpy().astype(np.float32, copy=False)
        score = self._sanitize_scores(score)
        ball_uv = np.clip(ball_uv, 0.0, 1.0).astype(np.float32, copy=False)
        visibility = (score >= float(self.inference_config.visibility_threshold)).astype(bool)
        ball_xy_px = self._uv_to_xy_px(ball_uv, width=output_width, height=output_height)

        return InferenceResult(
            frame_indices=clip.frame_indices.astype(np.int64, copy=False),
            ball_uv=ball_uv,
            ball_xy_px=ball_xy_px,
            visibility=visibility,
            score=score,
        )


def _validate_ball_detection_member(backend: str) -> None:
    normalized = str(backend).strip().lower()
    if normalized != "ball_detection":
        raise ValueError(
            "This visualization predictor API supports only backend='ball_detection'. "
            f"Got backend='{backend}'."
        )


def build_predictor_runtime(*, inference_config: InferenceConfig) -> PredictorRuntime:
    """Build reusable predictor runtime from inference config."""
    strategy = str(inference_config.strategy).strip().lower()

    if strategy == "single":
        _validate_ball_detection_member(inference_config.single_member.backend)
        predictor = BallPredictor.load_from_checkpoint(
            inference_config.single_member.checkpoint,
            device=inference_config.device,
            fallback_model_cfg_path=inference_config.single_member.model_config_path,
        )
        adapter = build_adapter_for_model(predictor.model)
        runner: _PredictRunner = _SingleModelRunner(
            predictor=predictor,
            adapter=adapter,
            visibility_threshold=float(inference_config.visibility_threshold),
        )
        return PredictorRuntime(inference_config=inference_config, runner=runner)

    if strategy == "ensemble":
        if not inference_config.ensemble_members:
            raise ValueError("inference.ensemble.members must be non-empty for strategy=ensemble")
        for member in inference_config.ensemble_members:
            _validate_ball_detection_member(member.backend)

        checkpoints = [member.checkpoint for member in inference_config.ensemble_members]
        weights = [float(member.weight) for member in inference_config.ensemble_members]
        model_config_paths = [member.model_config_path for member in inference_config.ensemble_members]
        predictor = BallEnsemblePredictor.from_checkpoints(
            checkpoints,
            device=inference_config.device,
            weights=weights,
            model_config_paths=model_config_paths,
            visibility_threshold=float(inference_config.visibility_threshold),
        )
        runner = _EnsembleModelRunner(predictor=predictor)
        return PredictorRuntime(inference_config=inference_config, runner=runner)

    raise ValueError(
        f"Unknown inference.strategy '{inference_config.strategy}'. Expected 'single' or 'ensemble'."
    )
