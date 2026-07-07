"""Measure keypoint visibility statistics for court-detection KP augmentation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from PIL import Image

from src.tasks.court_detection.data.augmentation import (
    build_kp_transforms,
    kp_in_bounds_mask,
)
from src.utils.hydra import hydra_main
from src.utils.io import find_existing_file, load_json


def _image_size(images_dir: Path, image_id: str) -> tuple[int, int]:
    """Read ``(width, height)`` from the image header."""
    path = find_existing_file(images_dir, image_id, (".png", ".jpg"))
    if path is None:
        raise FileNotFoundError(f"Image not found for id {image_id!r} in {images_dir}")
    with Image.open(path) as img:
        width, height = img.size
    return int(width), int(height)


def _stage_names(transforms: list) -> list[str]:
    return [type(transform).__name__ for transform in transforms]


def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    random.seed(int(cfg.measure.seed))
    np.random.seed(int(cfg.measure.seed))

    data_dir = Path(str(cfg.data.data_dir)).expanduser()
    entries = load_json(data_dir / f"data_{cfg.measure.split}.json")
    rng = random.Random(int(cfg.measure.seed))
    if int(cfg.measure.num_samples) < len(entries):
        entries = rng.sample(list(entries), int(cfg.measure.num_samples))

    aug = cfg.data.augmentation
    pipeline, _ = build_kp_transforms(
        is_train=True,
        train_scales=list(aug.train_scales),
        crop_scale=tuple(aug.crop_scale),
        crop_ratio=tuple(aug.crop_ratio),
        hflip_prob=float(aug.hflip_prob),
        swap_pairs=[tuple(pair) for pair in cfg.data.hflip_swap_pairs],
        affine_degrees=float(aug.affine_degrees),
        affine_translate=tuple(aug.affine_translate),
        affine_scale=tuple(aug.affine_scale),
        affine_shear=float(aug.affine_shear),
        perspective_distortion=float(aug.perspective_distortion),
        perspective_prob=float(aug.perspective_prob),
        min_visible_kp=int(aug.min_visible_kp),
        visibility_max_retries=int(aug.visibility_max_retries),
    )

    num_kp = int(cfg.data.num_keypoints)
    reps = int(cfg.measure.reps_per_image)
    stages = _stage_names(pipeline.transforms)

    orig_visible_counts: np.ndarray = np.zeros(num_kp, dtype=np.int64)
    final_visible_counts: np.ndarray = np.zeros(num_kp, dtype=np.int64)
    first_loss_by_stage = dict.fromkeys(stages, 0)
    total = 0
    all_survived = 0
    lost_sum = 0.0
    attempts_sum = 0
    constraint_missed = 0

    for entry in entries:
        kps = np.asarray(entry["kps"], dtype=np.float32)
        width, height = _image_size(data_dir / "images", str(entry["id"]))
        orig_mask = kp_in_bounds_mask(kps, width, height)
        target = min(pipeline.min_visible_kp, int(orig_mask.sum()))

        for _ in range(reps):
            chain, _, mask, attempts = pipeline.draw_params(width, height, kps)
            total += 1
            attempts_sum += attempts
            orig_visible_counts += orig_mask
            final_visible_counts += mask
            visible = int(mask.sum())
            if visible < target:
                constraint_missed += 1
            lost = int(orig_mask.sum()) - visible
            lost_sum += lost
            if lost == 0:
                all_survived += 1

            current_kps = kps
            current_mask = orig_mask.copy()
            current_w, current_h = width, height
            for transform, params, name in zip(
                pipeline.transforms, chain, stages, strict=True,
            ):
                current_kps = transform.apply_to_kps(current_kps, params)
                current_w, current_h = transform.out_size(
                    params, current_w, current_h,
                )
                permuted = transform.apply_to_mask(current_mask, params)
                new_mask = permuted & kp_in_bounds_mask(
                    current_kps, current_w, current_h,
                )
                first_loss_by_stage[name] += int((permuted & ~new_mask).sum())
                current_mask = new_mask

    report = {
        "config": {
            "split": str(cfg.measure.split),
            "num_images": len(entries),
            "reps_per_image": reps,
            "seed": int(cfg.measure.seed),
            "min_visible_kp": int(aug.min_visible_kp),
            "visibility_max_retries": int(aug.visibility_max_retries),
            "crop_scale": list(aug.crop_scale),
            "crop_ratio": list(aug.crop_ratio),
        },
        "per_kp_visibility_rate": (final_visible_counts / total).round(4).tolist(),
        "per_kp_original_in_bounds_rate": (
            orig_visible_counts / total
        ).round(4).tolist(),
        "all_originally_visible_survived_rate": round(all_survived / total, 4),
        "mean_originally_visible_kps_lost": round(lost_sum / total, 4),
        "mean_draw_attempts": round(attempts_sum / total, 3),
        "constraint_missed_rate": round(constraint_missed / total, 4),
        "first_loss_total_by_stage": first_loss_by_stage,
        "total_samples": total,
    }

    print(json.dumps(report, indent=2))
    output_dir = Path(str(cfg.measure.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"minvis{int(aug.min_visible_kp)}"
    out_path = output_dir / f"kp_visibility_{cfg.measure.split}_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    entry = hydra_main(
        config_path="../configs",
        config_name="measure_kp_visibility",
        version_base="1.3",
    )(main)
    raise SystemExit(entry())
