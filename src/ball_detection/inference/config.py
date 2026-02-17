"""Inference config parser for ball_detection visualization/runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from src.ball_detection.inference.types import InferenceConfig, InferenceMemberConfig

_DEFAULT_TRACKNET_CHECKPOINT = Path(
    "outputs/ball_detection/tracknetv3_wbce_full_e30/logs/version_0/checkpoints/last.ckpt"
)


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
) -> InferenceMemberConfig:
    backend = str(_cfg_get(raw, "backend", default_backend)).strip().lower()
    checkpoint_raw = _cfg_get(raw, "checkpoint", _cfg_get(raw, "path", default_checkpoint))
    if checkpoint_raw is None or str(checkpoint_raw).strip() == "":
        raise ValueError("checkpoint must be provided for inference member")

    return InferenceMemberConfig(
        backend=backend,
        checkpoint=Path(str(checkpoint_raw)).expanduser(),
        weight=float(_cfg_get(raw, "weight", default_weight)),
        score_threshold=float(_cfg_get(raw, "score_threshold", default_score_threshold)),
    )


def build_inference_config(cfg: DictConfig) -> InferenceConfig:
    """Build inference config from Hydra-composed config."""
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
                "weight": 1.0,
                "score_threshold": default_score_threshold,
            }
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

    return InferenceConfig(
        strategy=strategy,
        device=_resolve_device(run_device),
        image_h=int(_cfg_get(inf, "image_h", 288)),
        image_w=int(_cfg_get(inf, "image_w", 512)),
        batch_size=max(1, int(_cfg_get(inf, "batch_size", 16))),
        max_frames=_parse_optional_int(_cfg_get(inf, "max_frames", None)),
        window_size=_parse_optional_int(_cfg_get(inf, "window_size", None)),
        clip_frames=_parse_optional_int(_cfg_get(inf, "clip_frames", None)),
        clip_stride=_parse_optional_int(_cfg_get(inf, "clip_stride", None)),
        visibility_threshold=float(default_score_threshold),
        single_member=single_member,
        ensemble_members=ensemble_members,
    )
