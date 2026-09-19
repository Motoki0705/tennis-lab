"""Execute the existing ball detector with immutable inputs and audited caches."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.configuration import build_ball_detection_config
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionConfig,
    BallDetectionModule,
    BallDetectionResult,
)
from src.utils.checksum import FileIntegrityError, dual_sha256
from src.utils.configuration import PathResolver
from src.utils.io import save_json_atomic
from src.utils.video import probe_video_info

_MANAGED = {"enabled", "source", "save_result", "output_path", "load_path"}


@dataclass(frozen=True)
class DetectorBallSettings:
    config: BallDetectionConfig
    checkpoint_sha256: str

    @classmethod
    def from_config(
        cls, value: DictConfig, resolver: PathResolver, *, device: str
    ) -> DetectorBallSettings:
        if not isinstance(value, DictConfig):
            raise TypeError("ball_detector must be a configuration mapping")
        raw = cast(dict[str, object], OmegaConf.to_container(value, resolve=True))
        if _MANAGED.intersection(raw):
            raise ValueError(
                "ball_detector execution and output fields are managed by the builder"
            )
        digest = raw.pop("checkpoint_sha256", None)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                "ball_detector.checkpoint_sha256 requires 64 lowercase hex digits"
            )
        config = build_ball_detection_config(
            {
                **raw,
                "enabled": True,
                "source": "execute",
                "save_result": True,
                "output_path": "ball_detection_result.json",
                "load_path": None,
            },
            resolver,
            device=device,
        )
        if not math.isfinite(config.score_threshold):
            raise ValueError("ball_detector.score_threshold must be finite")
        return cls(config, digest)

    def identity(self) -> dict[str, object]:
        result = {
            key: value
            for key, value in asdict(self.config).items()
            if key not in {"resolver", "output_path", "load_path"}
        }
        result["checkpoint"] = str(self.config.checkpoint)
        result["image_size"] = list(self.config.image_size)
        return result


def _input_identity(
    clip: ClipManifest, settings: DetectorBallSettings
) -> dict[str, object]:
    if ClipManifest.load(clip.clip_dir) != clip:
        raise ValueError("Detector clip manifest changed")
    actual = dual_sha256(settings.config.checkpoint)
    if actual != settings.checkpoint_sha256:
        raise FileIntegrityError(
            "Ball detector checkpoint SHA-256 mismatch",
            details={
                "path": str(settings.config.checkpoint),
                "actual": actual,
                "expected": settings.checkpoint_sha256,
            },
        )
    if not math.isfinite(clip.fps) or clip.fps <= 0:
        raise ValueError("Detector clip FPS must be finite and positive")
    for camera in clip.camera_ids:
        info = probe_video_info(clip.media_path(camera))
        if (info.width, info.height, info.frame_count) != (
            clip.width,
            clip.height,
            clip.num_frames,
        ) or not math.isclose(info.fps, clip.fps, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"Detector media shape/FPS disagrees with manifest: {camera}"
            )
    return {
        "schema_version": 1,
        "source": "detector",
        "producer": "src.tennis_scene.pipeline.components.ball_detection.BallDetectionModule",
        "is_ground_truth": False,
        "clip_id": clip.clip_id,
        "manifest_sha256": clip.digest(),
        "camera_ids": list(clip.camera_ids),
        "video_sha256": {
            name: dual_sha256(clip.clip_dir / name) for name in clip.video_paths
        },
        "checkpoint_sha256": actual,
        "expected_checkpoint_sha256": settings.checkpoint_sha256,
        "settings": settings.identity(),
    }


def _validated_result(path: Path, clip: ClipManifest) -> BallDetectionResult:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {
        "ball_uv",
        "ball_uv_px",
        "visibility",
        "score",
    }:
        raise ValueError("Invalid detector result fields")
    visibility = np.asarray(raw["visibility"])
    if visibility.dtype != np.bool_:
        raise ValueError("Detector visibility must contain booleans")
    ball = BallDetectionResult.from_dict(raw)
    valid, errors = ball.validate()
    if not valid or ball.ball_uv.shape != (len(clip.camera_ids), clip.num_frames, 2):
        raise ValueError(f"Invalid detector camera/frame result: {errors}")
    if np.any(ball.score > 1) or np.any(ball.ball_uv < 0) or np.any(ball.ball_uv > 1):
        raise ValueError("Detector scores and normalized coordinates must be in [0,1]")
    scale = np.array([max(clip.width - 1, 1), max(clip.height - 1, 1)])
    if np.any(ball.ball_uv_px < 0) or np.any(ball.ball_uv_px > scale):
        raise ValueError("Detector pixel coordinates must be inside the image")
    if not np.allclose(ball.ball_uv_px, ball.ball_uv * scale, rtol=1e-6, atol=1e-4):
        raise ValueError(
            "Detector pixel coordinates disagree with normalized coordinates"
        )
    return ball


def observe_detector_ball(
    clip: ClipManifest, output: Path, settings: DetectorBallSettings
) -> None:
    """Reuse only a complete matching receipt; never overwrite a stale cache."""
    if output.resolve().is_relative_to(clip.clip_dir.resolve()):
        raise ValueError("Detector observations must be outside the source clip")
    identity = _input_identity(clip, settings)
    result_path = output / "ball_detection_result.json"
    receipt_path = output / "ball_import.metadata.json"
    if result_path.exists() or receipt_path.exists():
        if not result_path.is_file() or not receipt_path.is_file():
            raise ValueError("Partial detector ball cache")
        receipt = json.loads(receipt_path.read_text())
        expected = {**identity, "result_sha256": dual_sha256(result_path)}
        if receipt != expected:
            raise ValueError("Detector ball cache identity/result mismatch")
        _validated_result(result_path, clip)
        if _input_identity(clip, settings) != identity:
            raise ValueError("Detector inputs changed during cache validation")
        return
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ball-detector-", dir=output) as temporary:
        staged = Path(temporary) / result_path.name
        config = replace(settings.config, save_result=True, output_path=staged)
        result = BallDetectionModule(config).process(
            [clip.media_path(camera) for camera in clip.camera_ids],
            image_width=clip.width,
            image_height=clip.height,
        )
        saved = _validated_result(staged, clip)
        if saved.to_dict() != result.to_dict():
            raise ValueError("Detector saved result differs from returned result")
        if _input_identity(clip, settings) != identity:
            raise ValueError("Detector inputs changed during inference")
        receipt = {**identity, "result_sha256": dual_sha256(staged)}
        staged.replace(result_path)
        save_json_atomic(receipt, receipt_path)
