"""Save a single WASB dataset sample as images.

This script instantiates ``BallDetectionDataModule`` using the data config at
``src/tasks/wasb/configs/data/default.yaml`` (via Hydra defaults), then saves:

- Original frame (no data augmentation)
- Augmented frame (data augmentation enabled)
- Original frame with target heatmap overlay

Example:

`uv run --no-sync python -m src.tasks.wasb.scripts.save_one_sample_visuals \\
  data.root_dir=data/tennis data.train_matches=[game1] sample_index=0`

Useful overrides:

- `split=train|val|test`
- `sample_index=...`
- `target_index=...` (0..data.frames_out-1)
- `overlay_alpha=...`
- `num_samples=...` (default: 1)
- `sample_indices=[...]` (list of indices)
"""

# mypy: disable-error-code=index
# mypy: disable-error-code=misc

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Literal

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torchvision.utils import save_image

from src.tasks.wasb.data.ball_detection_datamodule import BallDetectionDataModule

Split = Literal["train", "val", "test"]


def _force_cpu_only() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _clone_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _make_datamodule(cfg: DictConfig, *, augment_enabled: bool) -> BallDetectionDataModule:
    cfg_copy = _clone_config(cfg)
    data_cfg = getattr(cfg_copy, "data", None)
    if data_cfg is None:
        raise ValueError(
            "Config must contain a 'data' section (defaults: - data: default)."
        )
    if getattr(data_cfg, "augment", None) is None:
        data_cfg.augment = {}
    data_cfg.augment.enabled = bool(augment_enabled)

    datamodule = BallDetectionDataModule(cfg_copy)
    datamodule.num_workers = 0
    datamodule.pin_memory = False
    return datamodule


def _get_dataset(datamodule: BallDetectionDataModule, split: Split) -> Any:
    if split in ("train", "val"):
        datamodule.setup(stage="fit")
        if split == "train":
            return datamodule.train_dataset
        return datamodule.val_dataset

    datamodule.setup(stage="test")
    return datamodule.test_dataset


def _save_visuals(
    *,
    original_sample: dict,
    augmented_sample: dict,
    target_index: int,
    overlay_alpha: float,
    file_stem: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_orig = original_sample["frames"]  # [T, C, H, W]
    frames_aug = augmented_sample["frames"]  # [T, C, H, W]
    heatmaps = original_sample["target_heatmaps"]  # [T_out, Hh, Wh]

    if frames_orig.shape[0] != frames_aug.shape[0]:
        raise RuntimeError(
            "Original/Augmented samples have different sequence lengths."
        )
    if not (0 <= target_index < heatmaps.shape[0]):
        raise ValueError(
            f"target_index must be in [0, {heatmaps.shape[0] - 1}] "
            f"(got {target_index})."
        )

    frame_start = max(int(frames_orig.shape[0]) - int(heatmaps.shape[0]), 0)
    frame_idx = min(frame_start + target_index, frames_orig.shape[0] - 1)

    img_orig = frames_orig[frame_idx]
    img_aug = frames_aug[frame_idx]

    save_image(img_orig, out_dir / f"{file_stem}_original.png")
    save_image(img_aug, out_dir / f"{file_stem}_augmented.png")

    hm = heatmaps[target_index]
    hm_up = F.interpolate(
        hm.unsqueeze(0).unsqueeze(0),
        size=img_orig.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # [1, H, W]
    hm_min, hm_max = hm_up.min(), hm_up.max()
    hm_norm = (hm_up - hm_min) / (hm_max - hm_min + 1e-6)

    overlay = torch.clamp(
        img_orig + hm_norm.repeat(3, 1, 1) * float(overlay_alpha),
        0.0,
        1.0,
    )
    save_image(overlay, out_dir / f"{file_stem}_overlay.png")

    meta = {
        "match": original_sample.get("match"),
        "clip": original_sample.get("clip"),
        "frame_path": original_sample.get("frame_paths", [None])[frame_idx],
        "frame_idx": int(frame_idx),
        "target_index": int(target_index),
        "targets_px": (
            original_sample.get("targets_px")[target_index].tolist()
            if isinstance(original_sample.get("targets_px"), torch.Tensor)
            else None
        ),
        "visibility": (
            int(original_sample.get("visibility")[target_index].item())
            if isinstance(original_sample.get("visibility"), torch.Tensor)
            else None
        ),
    }
    (out_dir / f"{file_stem}_meta.yaml").write_text(OmegaConf.to_yaml(meta))


@hydra.main(
    config_path="../../configs",
    config_name="save_one_sample_visuals",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    _force_cpu_only()
    seed = int(getattr(cfg, "seed", 42))
    _seed_everything(seed)

    split: Split = str(getattr(cfg, "split", "train"))  # type: ignore[assignment]
    target_index = int(getattr(cfg, "target_index", 0))
    overlay_alpha = float(getattr(cfg, "overlay_alpha", 0.5))
    num_samples = int(getattr(cfg, "num_samples", 1))
    sample_indices_cfg = getattr(cfg, "sample_indices", None)

    runtime_out = Path(HydraConfig.get().runtime.output_dir)
    out_dir = Path(to_absolute_path(str(runtime_out)))

    dm_orig = _make_datamodule(cfg, augment_enabled=False)
    dm_aug = _make_datamodule(cfg, augment_enabled=True)

    ds_orig = _get_dataset(dm_orig, split)
    ds_aug = _get_dataset(dm_aug, split)
    if ds_orig is None or ds_aug is None:
        raise RuntimeError(
            "Dataset is not initialized; check split and call setup()."
        )

    if len(ds_orig) != len(ds_aug):
        raise RuntimeError(
            "Original/Augmented datasets have different lengths; check data config."
        )

    if sample_indices_cfg is not None:
        sample_indices = [int(x) for x in list(sample_indices_cfg)]
    else:
        sample_index = int(getattr(cfg, "sample_index", 0))
        if num_samples <= 1:
            sample_indices = [sample_index]
        else:
            if num_samples > len(ds_orig):
                raise ValueError(
                    f"num_samples must be <= dataset length (len={len(ds_orig)}), "
                    f"got {num_samples}."
                )
            sample_indices = random.sample(range(len(ds_orig)), k=num_samples)

    for sample_index in sample_indices:
        if not (0 <= sample_index < len(ds_orig)):
            raise ValueError(
                f"sample_index must be within dataset bounds (len={len(ds_orig)}), "
                f"got {sample_index}."
            )

        original_sample = ds_orig[sample_index]

        # Re-seed per sample before fetching the augmented sample for reproducibility.
        _seed_everything(seed + int(sample_index))
        augmented_sample = ds_aug[sample_index]

        _save_visuals(
            original_sample=original_sample,
            augmented_sample=augmented_sample,
            target_index=target_index,
            overlay_alpha=overlay_alpha,
            file_stem=f"{split}_idx{sample_index:06d}",
            out_dir=out_dir,
        )

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
