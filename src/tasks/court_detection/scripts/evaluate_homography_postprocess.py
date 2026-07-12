"""Evaluate court keypoint subpixel and homography post-processing.

Usage:
    CUDA_VISIBLE_DEVICES='' python -m src.tasks.court_detection.scripts.evaluate_homography_postprocess
    CUDA_VISIBLE_DEVICES='' python -m src.tasks.court_detection.scripts.evaluate_homography_postprocess eval.val_max_samples=80

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/evaluate_homography_postprocess.yaml`.
    - The script forces CPU execution and does not allocate CUDA tensors.
    - It writes the same report printed to stdout as JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.geometry import refine_court_keypoints_with_homography
from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic
from src.utils.hydra import hydra_main
from src.utils.io import load_json, save_json
from src.utils.schema.court import COURT_KP_NAMES
from src.utils.video import probe_video_info, read_video_frame

VARIANT_NAMES = ("argmax", "subpixel", "subpixel_homography")


@dataclass
class DistanceStats:
    """Accumulate per-keypoint pixel distances."""

    sum_per_kp: NDArray[np.float64]
    count_per_kp: NDArray[np.int64]

    @classmethod
    def create(cls, num_keypoints: int) -> DistanceStats:
        return cls(
            sum_per_kp=np.zeros(num_keypoints, dtype=np.float64),
            count_per_kp=np.zeros(num_keypoints, dtype=np.int64),
        )

    def update(
        self,
        predicted: NDArray[np.float32],
        target: NDArray[np.float32],
        visible: NDArray[np.bool_],
    ) -> None:
        distances = np.linalg.norm(predicted - target, axis=1)
        valid = visible.astype(bool)
        self.sum_per_kp[valid] += distances[valid]
        self.count_per_kp[valid] += 1

    def to_dict(self, keypoint_names: list[str]) -> dict[str, Any]:
        total_count = int(self.count_per_kp.sum())
        total_sum = float(self.sum_per_kp.sum())
        per_kp: list[dict[str, Any]] = []
        for index, name in enumerate(keypoint_names):
            count = int(self.count_per_kp[index])
            mean = (
                float(self.sum_per_kp[index] / count)
                if count > 0
                else None
            )
            per_kp.append(
                {
                    "index": index,
                    "name": name,
                    "mean_dist_px": mean,
                    "count": count,
                }
            )
        return {
            "mean_dist_px": total_sum / total_count if total_count > 0 else None,
            "count": total_count,
            "per_kp": per_kp,
        }


@hydra_main(
    config_path="../configs",
    config_name="evaluate_homography_postprocess",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    torch.set_grad_enabled(False)
    keypoint_names = list(COURT_KP_NAMES[:14])
    checkpoint_path = Path(str(cfg.eval.checkpoint)).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    model = _load_model(checkpoint_path, cfg)
    val_report = _evaluate_val_subset(model, cfg, keypoint_names)
    clip_report = _evaluate_clip(checkpoint_path, cfg, keypoint_names)
    report = {
        "checkpoint": str(checkpoint_path),
        "val": val_report,
        "tennis_clip": clip_report,
    }

    print(json.dumps(report, indent=2))
    output_path = Path(str(cfg.eval.output_json)).expanduser()
    save_json(report, output_path)
    print(f"saved: {output_path}")
    return 0


def _load_model(
    checkpoint_path: Path,
    cfg: DictConfig,
) -> CourtDetectionLightningModule:
    cfg_resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    module = CourtDetectionLightningModule(cfg_resolved)
    module.load_state_dict(state_dict, strict=True)
    module.to(torch.device("cpu"))
    module.eval()
    return module


def _evaluate_val_subset(
    module: CourtDetectionLightningModule,
    cfg: DictConfig,
    keypoint_names: list[str],
) -> dict[str, Any]:
    datamodule = CourtDetectionDataModule(cfg)
    datamodule.setup("validate")
    max_samples = int(cfg.eval.val_max_samples)
    if max_samples <= 0:
        raise ValueError(f"eval.val_max_samples must be positive, got {max_samples}")

    stats = _stats_by_variant(num_keypoints=len(keypoint_names))
    homography_success = 0
    homography_failure_reasons: dict[str, int] = {}
    processed = 0

    for batch in datamodule.val_dataloader():
        remaining = max_samples - processed
        if remaining <= 0:
            break
        batch = _slice_batch(batch, min(remaining, int(batch["image"].shape[0])))
        images = cast(Tensor, batch["image"]).to(torch.device("cpu"))
        logits = module.model(images)
        heatmaps = torch.sigmoid(logits)
        batch_size = int(images.shape[0])

        for sample_index in range(batch_size):
            image_size = cast(Tensor, batch["image_size"])[sample_index]
            height = int(image_size[0].item())
            width = int(image_size[1].item())
            sample_heatmaps = heatmaps[
                sample_index : sample_index + 1,
                :,
                :height,
                :width,
            ]
            argmax_px, scores = _decode_pixel_keypoints(
                sample_heatmaps,
                subpixel_refine=False,
            )
            subpixel_px, _ = _decode_pixel_keypoints(
                sample_heatmaps,
                subpixel_refine=True,
            )
            postprocessed = refine_court_keypoints_with_homography(
                subpixel_px[None, ...],
                scores[None, ...],
                min_score=float(cfg.eval.postprocess.min_score),
                ransac_reproj_threshold=float(
                    cfg.eval.postprocess.ransac_reproj_threshold
                ),
                temporal_median_window=0,
            )
            if bool(postprocessed.diagnostics["frames"][0]["success"]):
                homography_success += 1
            else:
                reason = str(postprocessed.diagnostics["frames"][0]["reason"])
                homography_failure_reasons[reason] = (
                    homography_failure_reasons.get(reason, 0) + 1
                )

            target = _tensor_to_numpy(cast(Tensor, batch["keypoints"])[sample_index])
            visible = _tensor_to_numpy_bool(
                cast(Tensor, batch["kp_visible"])[sample_index]
            )
            stats["argmax"].update(argmax_px, target, visible)
            stats["subpixel"].update(subpixel_px, target, visible)
            stats["subpixel_homography"].update(
                postprocessed.keypoints[0],
                target,
                visible,
            )
            processed += 1

    return {
        "num_samples": processed,
        "postprocess_success_frames": homography_success,
        "postprocess_failure_reasons": homography_failure_reasons,
        "variants": _stats_to_dict(stats, keypoint_names),
    }


def _evaluate_clip(
    checkpoint_path: Path,
    cfg: DictConfig,
    keypoint_names: list[str],
) -> dict[str, Any]:
    video_path = Path(str(cfg.eval.clip_video_path)).expanduser()
    gt_path = Path(str(cfg.eval.clip_gt_path)).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"clip video not found: {video_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"clip GT not found: {gt_path}")

    predictor = CourtKeypointPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device="cpu",
        subpixel_refine=True,
        weights_only=False,
    )
    video_info = probe_video_info(video_path)
    gt = load_json(gt_path)
    gt_keypoints = np.asarray(gt["keypoints"], dtype=np.float32)
    frame_positions = {
        int(frame_index): position
        for position, frame_index in enumerate(gt["frame_indices"])
    }
    frame_reports: list[dict[str, Any]] = []
    stats = _stats_by_variant(num_keypoints=len(keypoint_names))
    frame_indices = [int(value) for value in cfg.eval.clip_frame_indices]

    for frame_index in frame_indices:
        if frame_index not in frame_positions:
            raise ValueError(f"frame {frame_index} is missing from clip GT")
        packet = read_video_frame(video_path, frame_index)
        frame_rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
        argmax = predictor.predict(frame_rgb, subpixel_refine=False)
        subpixel = predictor.predict(frame_rgb, subpixel_refine=True)
        argmax_px = _tensor_to_numpy(argmax["keypoints"])
        subpixel_px = _tensor_to_numpy(subpixel["keypoints"])
        scores = _tensor_to_numpy(subpixel["scores"])
        postprocessed = refine_court_keypoints_with_homography(
            subpixel_px[None, ...],
            scores[None, ...],
            min_score=float(cfg.eval.postprocess.min_score),
            ransac_reproj_threshold=float(cfg.eval.postprocess.ransac_reproj_threshold),
            temporal_median_window=0,
        )
        gt_px = _denormalize_gt_keypoints(
            gt_keypoints[0, frame_positions[frame_index]],
            width=video_info.width,
            height=video_info.height,
        )
        visible: NDArray[np.bool_] = np.ones(len(keypoint_names), dtype=bool)
        stats["argmax"].update(argmax_px, gt_px, visible)
        stats["subpixel"].update(subpixel_px, gt_px, visible)
        stats["subpixel_homography"].update(postprocessed.keypoints[0], gt_px, visible)
        frame_reports.append(
            {
                "frame_index": frame_index,
                "postprocess_success": bool(
                    postprocessed.diagnostics["frames"][0]["success"]
                ),
                "postprocess_reason": postprocessed.diagnostics["frames"][0]["reason"],
                "variants": {
                    "argmax": _per_kp_errors(argmax_px, gt_px, keypoint_names),
                    "subpixel": _per_kp_errors(subpixel_px, gt_px, keypoint_names),
                    "subpixel_homography": _per_kp_errors(
                        postprocessed.keypoints[0],
                        gt_px,
                        keypoint_names,
                    ),
                },
            }
        )

    return {
        "video_path": str(video_path),
        "gt_path": str(gt_path),
        "video_size": [int(video_info.width), int(video_info.height)],
        "frames": frame_reports,
        "aggregate": _stats_to_dict(stats, keypoint_names),
    }


def _decode_pixel_keypoints(
    heatmaps: Tensor,
    *,
    subpixel_refine: bool,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    coords, scores = heatmaps_to_argmax(heatmaps)
    if subpixel_refine:
        coords = refine_peaks_log_parabolic(heatmaps, coords)
    height, width = heatmaps.shape[-2:]
    scale = coords.new_tensor(
        [
            float(width - 1) if width > 1 else 0.0,
            float(height - 1) if height > 1 else 0.0,
        ]
    )
    coords_px = (coords[0] * scale).detach().cpu().numpy().astype(np.float32)
    scores_np = scores[0].detach().cpu().numpy().astype(np.float32)
    return coords_px, scores_np


def _stats_by_variant(num_keypoints: int) -> dict[str, DistanceStats]:
    return {
        variant: DistanceStats.create(num_keypoints)
        for variant in VARIANT_NAMES
    }


def _stats_to_dict(
    stats: dict[str, DistanceStats],
    keypoint_names: list[str],
) -> dict[str, Any]:
    return {
        variant: accumulator.to_dict(keypoint_names)
        for variant, accumulator in stats.items()
    }


def _per_kp_errors(
    predicted: NDArray[np.float32],
    target: NDArray[np.float32],
    keypoint_names: list[str],
) -> list[dict[str, Any]]:
    distances = np.linalg.norm(predicted - target, axis=1)
    return [
        {
            "index": index,
            "name": name,
            "error_px": float(distances[index]),
        }
        for index, name in enumerate(keypoint_names)
    ]


def _denormalize_gt_keypoints(
    normalized: NDArray[np.float32],
    *,
    width: int,
    height: int,
) -> NDArray[np.float32]:
    keypoints = np.array(normalized, copy=True, dtype=np.float32)
    keypoints[..., 0] *= max(width - 1, 1)
    keypoints[..., 1] *= max(height - 1, 1)
    return keypoints.astype(np.float32)


def _tensor_to_numpy(value: Tensor) -> NDArray[np.float32]:
    return value.detach().cpu().numpy().astype(np.float32)


def _tensor_to_numpy_bool(value: Tensor) -> NDArray[np.bool_]:
    return value.detach().cpu().numpy().astype(bool)


def _slice_batch(batch: dict[str, Any], count: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, (Tensor, list)):
            sliced[key] = value[:count]
        else:
            sliced[key] = value
    return sliced


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
