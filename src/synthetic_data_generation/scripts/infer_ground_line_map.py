"""
Estimate the B00 ground plane and aggregate fit-view court-line predictions.

Usage:
    python -m src.synthetic_data_generation.scripts.infer_ground_line_map
    python -m src.synthetic_data_generation.scripts.infer_ground_line_map device=cuda:0

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/infer_ground_line_map.yaml`.
    - The line checkpoint's Colab backbone path is replaced only after the
      configured local DINOv3 file name and SHA-256 are verified.
    - Holdout groups are partitioned before image decode and are never inferred.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytorch_lightning
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.alignment.ground_line_map import (
    GROUND_LINE_MAP_SCHEMA,
    GroundLineAccumulator,
    GroundLineMapSettings,
    expanded_plane_bounds,
    publish_ground_line_map_artifact,
)
from src.synthetic_data_generation.alignment.ground_plane import (
    GroundPlaneFitSettings,
    estimate_ground_plane,
)
from src.synthetic_data_generation.alignment.line_inference import (
    infer_line_projection,
    load_verified_line_detector,
)
from src.synthetic_data_generation.alignment.view_inputs import (
    load_provider_rgb_image,
    partition_fit_and_holdout_cameras,
)
from src.synthetic_data_generation.provider.bundle import (
    load_scene_provider_bundle,
    sha256_file,
)
from src.utils.hydra import hydra_main

LOGGER = logging.getLogger(__name__)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="infer_ground_line_map",
)
def main(cfg: DictConfig) -> int:
    """Run fit-only ground-plane estimation and court-line aggregation."""
    repo_root = Path(to_absolute_path(".")).resolve()
    provider_path = _path(cfg.provider_bundle)
    line_checkpoint = _path(cfg.line_checkpoint)
    backbone_repository = _path(cfg.backbone_repository)
    backbone_checkpoint = _path(cfg.backbone_checkpoint)
    output_dir = _path(cfg.output_dir)
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
    holdout_groups = tuple(int(value) for value in cfg.holdout_group_ids)
    fit_cameras, holdout_cameras = partition_fit_and_holdout_cameras(
        bundle.manifest.cameras,
        holdout_group_ids=holdout_groups,
    )
    if not fit_cameras or not holdout_cameras:
        raise ValueError("Fit and holdout camera partitions must both be non-empty.")

    plane_raw = OmegaConf.to_container(cfg.ground_plane, resolve=True)
    if not isinstance(plane_raw, dict):
        raise TypeError("ground_plane config must be a mapping.")
    plane_settings = GroundPlaneFitSettings(**cast(dict[str, Any], plane_raw))
    points_scene = np.load(
        bundle.point_cloud_path(),
        allow_pickle=False,
    )
    LOGGER.info(
        "Estimating ground plane from %d points and %d fit cameras.",
        len(points_scene),
        len(fit_cameras),
    )
    plane = estimate_ground_plane(
        points_scene,
        fit_cameras,
        settings=plane_settings,
    )
    LOGGER.info(
        "Accepted ground plane: support=%s RMS=%.6f normal/up=%.6f "
        "camera_height=[%.6f, %.6f].",
        plane.metrics["support_point_count"],
        plane.metrics["support_residual_rms"],
        plane.metrics["normal_up_cosine"],
        plane.metrics["camera_height_min"],
        plane.metrics["camera_height_max"],
    )

    projection_raw = OmegaConf.to_container(cfg.line_projection, resolve=True)
    if not isinstance(projection_raw, dict):
        raise TypeError("line_projection config must be a mapping.")
    projection_settings = GroundLineMapSettings(**cast(dict[str, Any], projection_raw))
    bounds = expanded_plane_bounds(
        plane,
        margin=projection_settings.bounds_margin,
    )
    accumulator = GroundLineAccumulator(
        bounds=bounds,
        grid_spacing=projection_settings.grid_spacing,
    )

    LOGGER.info("Loading line detector with verified local DINOv3 backbone.")
    detector = load_verified_line_detector(
        line_checkpoint,
        checkpoint_sha256=str(cfg.line_checkpoint_sha256),
        backbone_repository=backbone_repository,
        backbone_checkpoint=backbone_checkpoint,
        backbone_checkpoint_sha256=str(cfg.backbone_checkpoint_sha256),
        device=device,
        expected_short_side=int(cfg.expected_short_side),
    )

    image_files = {image.camera_id: image for image in bundle.manifest.images}
    records: list[dict[str, Any]] = []
    for index, camera in enumerate(fit_cameras):
        image_rgb = load_provider_rgb_image(bundle.image_path(camera.camera_id))
        observation = infer_line_projection(
            image_rgb,
            camera,
            detector=detector,
            plane=plane,
            bounds=bounds,
            settings=projection_settings,
        )
        projection = observation.projection
        reasons: list[str] = []
        if len(projection.points_uv) < projection_settings.min_projected_pixels:
            reasons.append("insufficient_projected_line_pixels")
            raster_cell_count = 0
        else:
            raster_cell_count = accumulator.add_view(projection)
        pose = np.asarray(camera.camera_to_scene, dtype=np.float64).reshape(4, 4)
        camera_center = pose[:3, 3]
        camera_height = float(plane.signed_distance(camera_center[None, :])[0])
        camera_foot = camera_center - camera_height * np.asarray(plane.normal)
        camera_ground_uv = plane.to_uv(camera_foot[None, :])[0]
        records.append(
            {
                "camera_id": camera.camera_id,
                "source_frame_index": camera.source_frame_index,
                "group_id": camera.group_id,
                "image": {
                    "relative_path": image_files[camera.camera_id].file.relative_path,
                    "sha256": image_files[camera.camera_id].file.sha256,
                    "width": camera.width,
                    "height": camera.height,
                },
                "line_output_width": observation.output_width,
                "line_output_height": observation.output_height,
                "selected_line_pixel_count": observation.selected_line_pixel_count,
                "projected_line_pixel_count": len(projection.points_uv),
                "raster_cell_count": raster_cell_count,
                "accepted": not reasons,
                "rejection_reasons": reasons,
                "camera_height_scene": camera_height,
                "camera_ground_uv": camera_ground_uv.astype(float).tolist(),
                "projection_rejections": {
                    "parallel": projection.invalid_parallel_count,
                    "behind_camera": projection.invalid_behind_count,
                    "beyond_max_range": projection.invalid_range_count,
                    "outside_bounds": projection.invalid_bounds_count,
                },
                "projected_probability_mean": _mean_or_none(projection.probabilities),
                "camera_range_median_scene": _quantile_or_none(
                    projection.camera_ranges,
                    0.5,
                ),
                "proximity_weight_median": _quantile_or_none(
                    projection.proximity_weights,
                    0.5,
                ),
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(fit_cameras):
            LOGGER.info(
                "Line projection progress: %d/%d fit views.",
                index + 1,
                len(fit_cameras),
            )

    arrays = accumulator.arrays()
    accepted_records = [record for record in records if record["accepted"]]
    rejection_counts = Counter(
        reason for record in records for reason in record["rejection_reasons"]
    )
    positive_view_counts = arrays["view_count"][arrays["view_count"] > 0]
    positive_evidence = arrays["evidence_sum"][arrays["evidence_sum"] > 0.0]
    summary = {
        "input_fit_view_count": len(records),
        "accepted_view_count": len(accepted_records),
        "rejected_view_count": len(records) - len(accepted_records),
        "rejection_reasons": dict(rejection_counts),
        "accepted_count_by_group": {
            str(group): sum(
                record["accepted"] and record["group_id"] == group for record in records
            )
            for group in sorted({record["group_id"] for record in records})
        },
        "selected_line_pixel_count": sum(
            int(record["selected_line_pixel_count"]) for record in records
        ),
        "projected_line_pixel_count": sum(
            int(record["projected_line_pixel_count"]) for record in records
        ),
        "raster_width": accumulator.width,
        "raster_height": accumulator.height,
        "raster_nonzero_cell_count": int(np.count_nonzero(arrays["view_count"])),
        "view_count_max": int(np.max(arrays["view_count"])),
        "view_count_q50_positive": _quantile_or_none(positive_view_counts, 0.5),
        "view_count_q95_positive": _quantile_or_none(positive_view_counts, 0.95),
        "evidence_sum_max": float(np.max(arrays["evidence_sum"])),
        "evidence_sum_q95_positive": _quantile_or_none(positive_evidence, 0.95),
        "evidence_sum_q99_positive": _quantile_or_none(positive_evidence, 0.99),
    }
    code_files = (
        repo_root / "src/synthetic_data_generation/alignment/ground_plane.py",
        repo_root / "src/synthetic_data_generation/alignment/ground_line_map.py",
        repo_root / "src/synthetic_data_generation/scripts/infer_ground_line_map.py",
        repo_root / "src/synthetic_data_generation/configs/infer_ground_line_map.yaml",
        repo_root / "src/synthetic_data_generation/provider/bundle.py",
        repo_root / "src/tasks/court_detection/inference/mask_predictor.py",
        repo_root / "src/tasks/court_detection/inference/preprocess.py",
    )
    artifact_payload = {
        "schema": GROUND_LINE_MAP_SCHEMA,
        "artifact_id": str(cfg.artifact_id),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provider": {
            "bundle_id": bundle.manifest.bundle_id,
            "bundle_fingerprint": bundle.manifest.bundle_fingerprint,
            "scene_fingerprint": bundle.manifest.scene_fingerprint,
            "camera_array_sha256": bundle.manifest.camera_array_sha256,
            "shared_intrinsics_sha256": bundle.manifest.shared_intrinsics_sha256,
            "image_set_sha256": bundle.manifest.image_set_sha256,
        },
        "split": {
            "group_definition": "provider SceneCamera.group_id",
            "fit_group_ids": sorted({camera.group_id for camera in fit_cameras}),
            "holdout_group_ids": list(holdout_groups),
            "fit_camera_ids": [camera.camera_id for camera in fit_cameras],
            "holdout_camera_ids": [camera.camera_id for camera in holdout_cameras],
            "holdout_inference_status": "not_run",
        },
        "detector": {
            "implementation": (
                "src.tasks.court_detection.inference.mask_predictor.CourtLinePredictor"
            ),
            "checkpoint": _relative_or_absolute(line_checkpoint, root=repo_root),
            "checkpoint_sha256": str(cfg.line_checkpoint_sha256),
            "embedded_backbone_path": detector.embedded_backbone_path,
            "backbone_repository": _relative_or_absolute(
                backbone_repository,
                root=repo_root,
            ),
            "backbone_checkpoint": _relative_or_absolute(
                backbone_checkpoint,
                root=repo_root,
            ),
            "backbone_checkpoint_sha256": detector.backbone_checkpoint_sha256,
            "backbone_override": "explicit verified local path",
            "short_side": detector.predictor.short_side,
            "resize_alignment": 8,
            "input_color_space": "srgb8-rgb",
            "normalization": "ImageNet mean/std",
        },
        "ground_plane": {
            "estimate": plane.to_dict(),
            "fit_settings": asdict(plane_settings),
            "fit_camera_scope": "fit groups only",
        },
        "projection": {
            "settings": asdict(projection_settings),
            "bounds_uv": list(bounds),
            "pixel_coordinates": bundle.manifest.pixel_coordinates,
            "ray_model": "K^-1 pixel ray transformed by camera_to_scene",
            "weight_model": "1/(1+(camera_range/proximity_scale)^power)",
            "raster_reducer": "per-view cell max then weighted global sum",
        },
        "records": records,
        "summary": summary,
        "provenance": {
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
                    "src.synthetic_data_generation.scripts.infer_ground_line_map",
                    *sys.argv[1:],
                ]
            ),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "pytorch_lightning_version": pytorch_lightning.__version__,
            "opencv_version": cv2.__version__,
            "device": str(detector.predictor.device),
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(detector.predictor.device)
                if detector.predictor.device.type == "cuda"
                else None
            ),
        },
    }
    published = publish_ground_line_map_artifact(
        artifact_payload,
        arrays=arrays,
        output_dir=output_dir,
    )
    LOGGER.info(
        "Published ground-line map: %s; accepted=%d/%d, nonzero cells=%d.",
        published,
        summary["accepted_view_count"],
        summary["input_fit_view_count"],
        summary["raster_nonzero_cell_count"],
    )
    print(published)
    return 0


def _mean_or_none(values: np.ndarray[Any, Any]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _quantile_or_none(
    values: np.ndarray[Any, Any],
    quantile: float,
) -> float | None:
    return float(np.quantile(values, quantile)) if len(values) else None


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
