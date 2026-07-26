"""
Run 14-keypoint court inference on provider fit groups and publish the results.

Usage:
    python -m src.synthetic_data_generation.scripts.infer_fit_view_courts
    python -m src.synthetic_data_generation.scripts.infer_fit_view_courts device=cuda:0

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/infer_fit_view_courts.yaml`.
    - Holdout groups are partitioned before image decode and never inferred.
    - Publication is atomic, content-addressed, and refuses replacement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from hydra.utils import to_absolute_path
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.alignment.fit_view_detection import (
    CourtKeypointPrediction,
    FitViewDetectionSettings,
    infer_fit_view_court_detections,
    publish_fit_view_court_detections,
)
from src.synthetic_data_generation.provider.bundle import (
    load_scene_provider_bundle,
    sha256_file,
)
from src.tasks.court_detection.evaluation.contracts import (
    HomographyEvaluationCriteria,
)
from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor
from src.utils.hydra import hydra_main

LOGGER = logging.getLogger(__name__)


class _PredictorAdapter:
    """Convert the task predictor's tensor dictionary to the alignment port."""

    def __init__(self, predictor: CourtKeypointPredictor) -> None:
        self._predictor = predictor

    def predict_rgb(
        self,
        image_rgb: NDArray[np.uint8],
    ) -> CourtKeypointPrediction:
        result = self._predictor.predict(image_rgb)
        return CourtKeypointPrediction(
            keypoints_xy=result["keypoints"].numpy().astype(np.float32),
            peak_scores=result["scores"].numpy().astype(np.float32),
        )


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="infer_fit_view_courts",
)
def main(cfg: DictConfig) -> int:
    """Run deterministic fit-only court inference and publish its artifact."""
    repo_root = Path(to_absolute_path(".")).resolve()
    provider_path = _path(cfg.provider_bundle)
    checkpoint_path = _path(cfg.checkpoint)
    output_dir = _path(cfg.output_dir)
    expected_checkpoint_sha256 = str(cfg.checkpoint_sha256)
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "Court checkpoint hash mismatch: "
            f"declared {expected_checkpoint_sha256}, "
            f"computed {actual_checkpoint_sha256}."
        )

    device = str(cfg.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} was requested but CUDA is unavailable."
        )
    seed = int(cfg.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    LOGGER.info("Loading and verifying provider bundle: %s", provider_path)
    bundle = load_scene_provider_bundle(
        provider_path,
        verify_files=bool(cfg.verify_provider_files),
    )
    LOGGER.info("Loading court detector: %s", checkpoint_path)
    predictor = CourtKeypointPredictor.load_from_checkpoint(
        checkpoint_path,
        device=device,
        weights_only=False,
        subpixel_refine=bool(cfg.subpixel_refine),
    )
    expected_short_side = int(cfg.expected_short_side)
    if predictor.short_side != expected_short_side:
        raise ValueError(
            "Checkpoint preprocessing mismatch: "
            f"expected short_side={expected_short_side}, "
            f"loaded {predictor.short_side}."
        )
    if device.startswith("cuda") and predictor.device.type != "cuda":
        raise RuntimeError(
            f"Predictor silently resolved requested {device!r} to {predictor.device}."
        )

    homography_raw = OmegaConf.to_container(cfg.homography, resolve=True)
    if not isinstance(homography_raw, dict):
        raise TypeError("homography config must be a mapping.")
    settings = FitViewDetectionSettings(
        artifact_id=str(cfg.artifact_id),
        holdout_group_ids=tuple(int(value) for value in cfg.holdout_group_ids),
        min_peak_score=float(cfg.min_peak_score),
        min_confident_keypoints=int(cfg.min_confident_keypoints),
        homography=HomographyEvaluationCriteria(**cast(dict[str, Any], homography_raw)),
    )
    code_files = (
        repo_root / "src/synthetic_data_generation/alignment/fit_view_detection.py",
        repo_root / "src/synthetic_data_generation/scripts/infer_fit_view_courts.py",
        repo_root / "src/synthetic_data_generation/configs/infer_fit_view_courts.yaml",
        repo_root / "src/synthetic_data_generation/provider/bundle.py",
        repo_root / "src/tasks/court_detection/inference/predictor.py",
        repo_root / "src/tasks/court_detection/inference/preprocess.py",
        repo_root / "src/tasks/court_detection/evaluation/homography_quality.py",
        repo_root / "src/tasks/court_detection/evaluation/image_evidence.py",
    )
    created_at = datetime.now(UTC).isoformat()
    LOGGER.info(
        "Inferring %d fit cameras; holdout groups %s remain quarantined.",
        sum(
            camera.group_id not in set(settings.holdout_group_ids)
            for camera in bundle.manifest.cameras
        ),
        settings.holdout_group_ids,
    )
    artifact = infer_fit_view_court_detections(
        bundle,
        _PredictorAdapter(predictor),
        settings=settings,
        detector={
            "implementation": (
                "src.tasks.court_detection.inference.predictor.CourtKeypointPredictor"
            ),
            "checkpoint": _relative_or_absolute(checkpoint_path, root=repo_root),
            "checkpoint_sha256": actual_checkpoint_sha256,
            "num_keypoints": 14,
            "short_side": predictor.short_side,
            "resize_alignment": 8,
            "resize_interpolation": "PIL.Image.Resampling.BILINEAR",
            "normalization": "ImageNet mean/std",
            "subpixel_refine": predictor.subpixel_refine,
            "output_rescale": "x*(original_width-1),y*(original_height-1)",
        },
        provenance={
            "seed": seed,
            "git_revision": _git(repo_root, "rev-parse", "HEAD"),
            "git_dirty": bool(_git(repo_root, "status", "--porcelain=v1")),
            "code_files": [
                {
                    "path": _relative_or_absolute(path, root=repo_root),
                    "sha256": sha256_file(path),
                }
                for path in code_files
            ],
            "code_sha256": _code_fingerprint(code_files, root=repo_root),
            "command": shlex.join(
                [
                    sys.executable,
                    "-m",
                    "src.synthetic_data_generation.scripts.infer_fit_view_courts",
                    *sys.argv[1:],
                ]
            ),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "opencv_version": cv2.__version__,
            "device": str(predictor.device),
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(predictor.device)
                if predictor.device.type == "cuda"
                else None
            ),
        },
        created_at_utc=created_at,
    )
    artifact_path = publish_fit_view_court_detections(
        artifact,
        output_dir=output_dir,
    )
    summary = cast(dict[str, Any], artifact["summary"])
    LOGGER.info(
        "Published %s: accepted=%s rejected=%s fingerprint=%s",
        artifact_path,
        summary["accepted_count"],
        summary["rejected_count"],
        artifact["artifact_fingerprint"],
    )
    print(artifact_path)
    return 0


def _path(value: object) -> Path:
    return Path(to_absolute_path(str(value))).resolve()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _relative_or_absolute(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _code_fingerprint(paths: tuple[Path, ...], *, root: Path) -> str:
    inventory = [
        {
            "path": _relative_or_absolute(path, root=root),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    encoded = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    cast(Any, main)()
