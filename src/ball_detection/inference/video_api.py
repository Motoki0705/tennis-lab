"""Video inference API for ball_detection with single/ensemble strategy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig

from src.ball_detection.inference.predictor import BallPredictor
from src.wasb.inference import WASBPredictor

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

_DEFAULT_TRACKNET_CHECKPOINT = Path(
    "outputs/ball_detection/tracknetv3_wbce_full_e30/logs/version_0/checkpoints/last.ckpt"
)
_DEFAULT_WASB_CHECKPOINT = Path("checkpoints/wasb/wasb_tennis_best.pth.tar")


@dataclass(frozen=True)
class VideoInferenceMemberConfig:
    """One predictor member for single/ensemble inference."""

    backend: str
    checkpoint: Path
    weight: float
    score_threshold: float


@dataclass(frozen=True)
class VideoInferenceConfig:
    """Resolved runtime config for video inference."""

    strategy: str
    device: str
    image_h: int
    image_w: int
    batch_size: int
    max_frames: int | None
    window_size: int | None
    visibility_threshold: float
    single_member: VideoInferenceMemberConfig
    ensemble_members: tuple[VideoInferenceMemberConfig, ...]


@dataclass(frozen=True)
class VideoInferenceResult:
    """Per-frame 2D ball predictions."""

    frame_indices: IntArray
    ball_uv: FloatArray
    ball_xy_px: FloatArray
    visibility: BoolArray
    score: FloatArray


class _VideoBackend(Protocol):
    def predict(self, frames_rgb: NDArray[np.uint8], frame_indices: IntArray) -> VideoInferenceResult:
        """Predict per-frame ball positions for one backend."""


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


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Expected positive integer or null, got: {value}")
    return parsed


def _parse_member(
    raw: Any,
    *,
    default_backend: str,
    default_checkpoint: Path,
    default_weight: float,
    default_score_threshold: float,
) -> VideoInferenceMemberConfig:
    backend = str(_cfg_get(raw, "backend", default_backend)).strip().lower()
    checkpoint_raw = _cfg_get(raw, "checkpoint", _cfg_get(raw, "path", default_checkpoint))
    if checkpoint_raw is None or str(checkpoint_raw).strip() == "":
        raise ValueError("checkpoint must be provided for inference member")

    checkpoint = Path(str(checkpoint_raw)).expanduser()
    return VideoInferenceMemberConfig(
        backend=backend,
        checkpoint=checkpoint,
        weight=float(_cfg_get(raw, "weight", default_weight)),
        score_threshold=float(_cfg_get(raw, "score_threshold", default_score_threshold)),
    )


def _normalize_weights(weights: list[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    return [float(w) / total for w in weights]


def _empty_result() -> VideoInferenceResult:
    empty_i64: IntArray = np.zeros((0,), dtype=np.int64)
    empty_xy: FloatArray = np.zeros((0, 2), dtype=np.float32)
    empty_f32: FloatArray = np.zeros((0,), dtype=np.float32)
    empty_bool: BoolArray = np.zeros((0,), dtype=bool)
    return VideoInferenceResult(
        frame_indices=empty_i64,
        ball_uv=empty_xy,
        ball_xy_px=empty_xy.copy(),
        visibility=empty_bool,
        score=empty_f32,
    )


def _uv_to_xy_px(ball_uv: FloatArray, *, width: int, height: int) -> FloatArray:
    ball_xy_px = np.asarray(ball_uv, dtype=np.float32).copy()
    if ball_xy_px.size == 0:
        return ball_xy_px
    ball_xy_px[:, 0] *= float(max(width - 1, 1))
    ball_xy_px[:, 1] *= float(max(height - 1, 1))
    return ball_xy_px


def _sanitize_scores(scores: FloatArray) -> FloatArray:
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


class _BallDetectionVideoBackend:
    """Video inference backend using local ball_detection checkpoints."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        device: str,
        image_h: int,
        image_w: int,
        batch_size: int,
        window_size: int | None,
        score_threshold: float,
    ) -> None:
        self.predictor = BallPredictor.load_from_checkpoint(checkpoint, device=device)
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.batch_size = max(1, int(batch_size))
        self.window_size = window_size
        self.score_threshold = float(score_threshold)

    def _resolve_window_size(self) -> int:
        model_seq_len = getattr(self.predictor.model, "seq_len", None)
        if model_seq_len is not None:
            seq_len = int(model_seq_len)
            if self.window_size is not None and int(self.window_size) != seq_len:
                raise ValueError(
                    "window_size mismatch: "
                    f"config window_size={self.window_size}, model seq_len={seq_len}"
                )
            return seq_len
        if self.window_size is not None:
            return int(self.window_size)
        return 8

    def _preprocess_frames(self, frames_rgb: NDArray[np.uint8]) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        for frame in frames_rgb:
            resized = cv2.resize(frame, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float() / 255.0
            tensors.append(tensor)
        return tensors

    @staticmethod
    def _extract_uv_last(ball_uv: torch.Tensor) -> torch.Tensor:
        if ball_uv.dim() == 2 and ball_uv.shape[-1] == 2:
            return ball_uv
        if ball_uv.dim() == 3 and ball_uv.shape[-1] == 2:
            return ball_uv[:, -1, :]
        raise ValueError(f"Unsupported ball_uv shape: {tuple(ball_uv.shape)}")

    @staticmethod
    def _extract_scalar_last(values: torch.Tensor, *, name: str) -> torch.Tensor:
        if values.dim() == 1:
            return values
        if values.dim() == 2:
            return values[:, -1]
        raise ValueError(f"Unsupported {name} shape: {tuple(values.shape)}")

    def predict(self, frames_rgb: NDArray[np.uint8], frame_indices: IntArray) -> VideoInferenceResult:
        if len(frames_rgb) == 0:
            return _empty_result()

        frame_tensors = self._preprocess_frames(frames_rgb)
        seq_len = self._resolve_window_size()
        if seq_len <= 0:
            raise ValueError(f"Invalid sequence length: {seq_len}")

        all_uv: list[FloatArray] = []
        all_score: list[FloatArray] = []

        for start in range(0, len(frame_tensors), self.batch_size):
            end = min(start + self.batch_size, len(frame_tensors))
            windows: list[torch.Tensor] = []
            for frame_idx in range(start, end):
                start_idx = frame_idx - seq_len + 1
                indices = [max(0, idx) for idx in range(start_idx, frame_idx + 1)]
                windows.append(torch.stack([frame_tensors[idx] for idx in indices], dim=0))

            batch_frames = torch.stack(windows, dim=0)
            outputs = self.predictor.predict(batch_frames)

            uv_last = self._extract_uv_last(outputs["ball_uv"]).numpy().astype(np.float32, copy=False)
            score_last = (
                self._extract_scalar_last(outputs["score"], name="score")
                .numpy()
                .astype(np.float32, copy=False)
            )

            all_uv.append(uv_last)
            all_score.append(score_last)

        ball_uv = np.concatenate(all_uv, axis=0)
        ball_uv = np.clip(ball_uv, 0.0, 1.0).astype(np.float32, copy=False)
        score = np.concatenate(all_score, axis=0).astype(np.float32, copy=False)
        score = _sanitize_scores(score)

        height = int(frames_rgb.shape[1])
        width = int(frames_rgb.shape[2])
        ball_xy_px = _uv_to_xy_px(ball_uv, width=width, height=height)
        visibility = (score >= self.score_threshold).astype(bool)

        return VideoInferenceResult(
            frame_indices=frame_indices.copy(),
            ball_uv=ball_uv,
            ball_xy_px=ball_xy_px,
            visibility=visibility,
            score=score,
        )


class _WASBVideoBackend:
    """Video inference backend using WASB predictor checkpoints."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        device: str,
        batch_size: int,
        score_threshold: float,
    ) -> None:
        self.predictor = WASBPredictor.load_from_checkpoint(
            checkpoint,
            device=device,
            score_threshold=score_threshold,
        )
        self.batch_size = max(1, int(batch_size))
        self.score_threshold = float(score_threshold)

    def predict(self, frames_rgb: NDArray[np.uint8], frame_indices: IntArray) -> VideoInferenceResult:
        if len(frames_rgb) == 0:
            return _empty_result()

        self.predictor.reset_tracker()

        uv_list: list[FloatArray] = []
        xy_list: list[FloatArray] = []
        score_list: list[FloatArray] = []
        idx_list: list[IntArray] = []

        for start in range(0, len(frames_rgb), self.batch_size):
            end = min(start + self.batch_size, len(frames_rgb))
            chunk = np.ascontiguousarray(frames_rgb[start:end], dtype=np.uint8)
            chunk_indices = frame_indices[start:end].tolist()
            outputs = self.predictor.predict(chunk, frame_indices=chunk_indices)

            uv_list.append(np.asarray(outputs["ball_uv"], dtype=np.float32))
            xy_list.append(np.asarray(outputs["ball_xy_px"], dtype=np.float32))
            score_list.append(np.asarray(outputs["score"], dtype=np.float32))
            idx_list.append(np.asarray(outputs["frame_indices"], dtype=np.int64))

        ball_uv = np.concatenate(uv_list, axis=0)
        ball_uv = np.clip(ball_uv, 0.0, 1.0).astype(np.float32, copy=False)
        ball_xy_px = np.concatenate(xy_list, axis=0).astype(np.float32, copy=False)
        score = np.concatenate(score_list, axis=0).astype(np.float32, copy=False)
        score = _sanitize_scores(score)
        out_indices = np.concatenate(idx_list, axis=0).astype(np.int64, copy=False)

        if out_indices.shape != frame_indices.shape or not np.array_equal(out_indices, frame_indices):
            raise ValueError("WASB predictor returned frame indices inconsistent with input sequence.")

        visibility = (score >= self.score_threshold).astype(bool)
        return VideoInferenceResult(
            frame_indices=out_indices,
            ball_uv=ball_uv,
            ball_xy_px=ball_xy_px,
            visibility=visibility,
            score=score,
        )


def _build_backend(
    member: VideoInferenceMemberConfig,
    *,
    config: VideoInferenceConfig,
) -> _VideoBackend:
    backend = member.backend.strip().lower()
    if backend == "ball_detection":
        return _BallDetectionVideoBackend(
            checkpoint=member.checkpoint,
            device=config.device,
            image_h=config.image_h,
            image_w=config.image_w,
            batch_size=config.batch_size,
            window_size=config.window_size,
            score_threshold=member.score_threshold,
        )
    if backend == "wasb":
        return _WASBVideoBackend(
            checkpoint=member.checkpoint,
            device=config.device,
            batch_size=config.batch_size,
            score_threshold=member.score_threshold,
        )

    raise ValueError(
        f"Unknown inference backend: {member.backend}. Expected one of ['ball_detection', 'wasb']."
    )


def _fuse_ensemble_results(
    results: list[VideoInferenceResult],
    *,
    weights: list[float],
    visibility_threshold: float,
    width: int,
    height: int,
) -> VideoInferenceResult:
    if not results:
        raise ValueError("No results to ensemble.")

    base_idx = results[0].frame_indices
    for idx, res in enumerate(results[1:], start=1):
        if res.frame_indices.shape != base_idx.shape or not np.array_equal(res.frame_indices, base_idx):
            raise ValueError(f"Ensemble member {idx} produced mismatched frame indices.")

    weights_norm = _normalize_weights(weights)

    ball_uv = np.zeros_like(results[0].ball_uv, dtype=np.float32)
    score = np.zeros_like(results[0].score, dtype=np.float32)
    for weight, res in zip(weights_norm, results, strict=True):
        ball_uv += res.ball_uv * float(weight)
        score += _sanitize_scores(res.score) * float(weight)

    ball_uv = np.clip(ball_uv, 0.0, 1.0).astype(np.float32, copy=False)
    score = _sanitize_scores(score)
    ball_xy_px = _uv_to_xy_px(ball_uv, width=width, height=height)
    visibility = (score >= float(visibility_threshold)).astype(bool)

    return VideoInferenceResult(
        frame_indices=base_idx.copy(),
        ball_uv=ball_uv,
        ball_xy_px=ball_xy_px,
        visibility=visibility,
        score=score,
    )


def build_video_inference_config(cfg: DictConfig) -> VideoInferenceConfig:
    """Build video inference config from Hydra-composed config."""
    inf = cfg.get("inference", {}) or {}
    run = cfg.get("run", {}) or {}

    strategy = str(_cfg_get(inf, "strategy", "single")).strip().lower()
    run_device = str(_cfg_get(run, "device", _cfg_get(inf, "device", "auto")))

    default_score_threshold = float(_cfg_get(inf, "visibility_threshold", 0.5))

    single_cfg = _cfg_get(inf, "single", {})
    single_member = _parse_member(
        single_cfg,
        default_backend="ball_detection",
        default_checkpoint=_DEFAULT_TRACKNET_CHECKPOINT,
        default_weight=1.0,
        default_score_threshold=default_score_threshold,
    )

    ensemble_cfg = _cfg_get(inf, "ensemble", {})
    members_raw = list(_cfg_get(ensemble_cfg, "members", []))
    if not members_raw:
        members_raw = [
            {
                "backend": "ball_detection",
                "checkpoint": str(_DEFAULT_TRACKNET_CHECKPOINT),
                "weight": 0.5,
                "score_threshold": default_score_threshold,
            },
            {
                "backend": "wasb",
                "checkpoint": str(_DEFAULT_WASB_CHECKPOINT),
                "weight": 0.5,
                "score_threshold": default_score_threshold,
            },
        ]

    ensemble_members = tuple(
        _parse_member(
            member,
            default_backend="ball_detection",
            default_checkpoint=_DEFAULT_TRACKNET_CHECKPOINT,
            default_weight=1.0,
            default_score_threshold=default_score_threshold,
        )
        for member in members_raw
    )

    return VideoInferenceConfig(
        strategy=strategy,
        device=_resolve_device(run_device),
        image_h=int(_cfg_get(inf, "image_h", 288)),
        image_w=int(_cfg_get(inf, "image_w", 512)),
        batch_size=max(1, int(_cfg_get(inf, "batch_size", 16))),
        max_frames=_parse_optional_int(_cfg_get(inf, "max_frames", None)),
        window_size=_parse_optional_int(_cfg_get(inf, "window_size", None)),
        visibility_threshold=float(default_score_threshold),
        single_member=single_member,
        ensemble_members=ensemble_members,
    )


def run_video_inference(
    *,
    frames_rgb: NDArray[np.uint8],
    config: VideoInferenceConfig,
) -> VideoInferenceResult:
    """Run per-frame video inference according to strategy in ``config``."""
    frames = np.asarray(frames_rgb)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames_rgb must have shape [T, H, W, 3], got {tuple(frames.shape)}")

    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    if config.max_frames is not None:
        frames = frames[: config.max_frames]

    if frames.shape[0] == 0:
        return _empty_result()

    frame_indices: IntArray = np.arange(frames.shape[0], dtype=np.int64)
    strategy = config.strategy.strip().lower()

    if strategy == "single":
        backend = _build_backend(config.single_member, config=config)
        return backend.predict(frames, frame_indices)

    if strategy == "ensemble":
        if len(config.ensemble_members) == 0:
            raise ValueError("inference.ensemble.members must be non-empty for strategy=ensemble")

        results: list[VideoInferenceResult] = []
        weights: list[float] = []
        for member in config.ensemble_members:
            backend = _build_backend(member, config=config)
            results.append(backend.predict(frames, frame_indices))
            weights.append(float(member.weight))

        return _fuse_ensemble_results(
            results,
            weights=weights,
            visibility_threshold=config.visibility_threshold,
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
        )

    raise ValueError(
        f"Unknown inference.strategy '{config.strategy}'. Expected 'single' or 'ensemble'."
    )
