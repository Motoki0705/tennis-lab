"""Encode tennis frames into DINOv3 ViT patch tokens and save to disk.

This script loads a pretrained DINOv3 heatmap model, extracts its ViT
backbone, and encodes every frame in game1-10 into patch tokens. Outputs
are saved under `data/tennis/patch_embeddings` with one file per clip
and per augmentation pass (e.g., `Clip3.pt`, `Clip3_aug01.pt`, ...).

Example:
    uv run python -m src.wasb.scripts.tools.encode_dinov3_patch_tokens \
      model_checkpoint=outputs/dinov3_heatmap/logs/version_0/checkpoints/last.ckpt \
      output_dir=data/tennis/embed

Hydra parameters:
    - model_checkpoint: Lightning checkpoint path for dinov3_heatmap (optional).
    - output_dir: Base directory to store patch tokens.
    - num_augments: Number of augmentation passes per clip.
    - matches: List of matches to process (defaults to data.train/val/test).
    - preprocess.resize_hw: Resize (H, W) before encoding.
    - preprocess.normalize: Whether to apply mean/std normalization.
    - save_dtype: Data type for saved tokens (float32/float16/bfloat16).
    - overwrite: Whether to overwrite existing token files.
    - batch_size / num_workers / device: Data loading and inference settings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.wasb.models import DinoV3FPNHeatmap
from src.wasb.training.ball_detection.lightning_module import WASBLightningModule

LOGGER = logging.getLogger(__name__)


class ClipFramesDataset(Dataset):
    """Dataset for per-clip frame encoding."""

    def __init__(self, frame_paths: Sequence[Path], transform: transforms.Compose) -> None:
        self._frame_paths = list(frame_paths)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._frame_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame_path = self._frame_paths[idx]
        with Image.open(frame_path) as img:
            img = img.convert("RGB")
            tensor = self._transform(img)
        return tensor


def _resolve_matches(
    *,
    root_dir: Path,
    train_matches: Sequence[str],
    val_matches: Sequence[str],
    test_matches: Sequence[str],
    override: Sequence[str] | None,
) -> list[str]:
    if override is not None and len(override) > 0:
        return list(override)

    ordered: list[str] = []
    seen: set[str] = set()
    for group in (train_matches, val_matches, test_matches):
        for match in group:
            if match not in seen:
                ordered.append(match)
                seen.add(match)
    if ordered:
        return ordered

    return sorted([p.name for p in root_dir.iterdir() if p.is_dir()])


def _list_frames(clip_dir: Path, image_ext: str) -> list[Path]:
    ext = image_ext.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    return sorted([p for p in clip_dir.iterdir() if p.suffix.lower() == ext])


def _build_transform(cfg: DictConfig) -> transforms.Compose:
    ops: list = []
    resize_hw = cfg.preprocess.get("resize_hw", None)
    if resize_hw is not None:
        ops.append(transforms.Resize((int(resize_hw[0]), int(resize_hw[1]))))

    data_aug = cfg.data.get("augment", {}) if hasattr(cfg, "data") else {}
    enabled = bool(data_aug.get("enabled", False))
    if enabled:
        cj = data_aug.get("color_jitter", {})
        cj_prob = float(cj.get("prob", 0.0))
        if cj_prob > 0:
            ops.append(
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=float(cj.get("brightness", 0.0)),
                            contrast=float(cj.get("contrast", 0.0)),
                            saturation=float(cj.get("saturation", 0.0)),
                            hue=float(cj.get("hue", 0.0)),
                        )
                    ],
                    p=cj_prob,
                )
            )

        gs = data_aug.get("random_grayscale", {})
        gs_prob = float(gs.get("prob", 0.0))
        if gs_prob > 0:
            ops.append(transforms.RandomGrayscale(p=gs_prob))

        gb = data_aug.get("gaussian_blur", {})
        gb_prob = float(gb.get("prob", 0.0))
        if gb_prob > 0:
            kernel_size = int(gb.get("kernel_size", 3))
            sigma_min = float(gb.get("sigma_min", 0.1))
            sigma_max = float(gb.get("sigma_max", 2.0))
            ops.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=kernel_size,
                            sigma=(sigma_min, sigma_max),
                        )
                    ],
                    p=gb_prob,
                )
            )

    ops.append(transforms.ToTensor())
    if enabled:
        re_cfg = data_aug.get("random_erasing", {})
        re_prob = float(re_cfg.get("prob", 0.0))
        if re_prob > 0:
            scale = re_cfg.get("scale", [0.02, 0.2])
            ratio = re_cfg.get("ratio", [0.3, 3.3])
            value = re_cfg.get("value", 0)
            ops.append(
                transforms.RandomErasing(
                    p=re_prob,
                    scale=(float(scale[0]), float(scale[1])),
                    ratio=(float(ratio[0]), float(ratio[1])),
                    value=value,
                )
            )

    if bool(cfg.preprocess.get("normalize", False)):
        mean = cfg.preprocess.get("mean", [0.485, 0.456, 0.406])
        std = cfg.preprocess.get("std", [0.229, 0.224, 0.225])
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def _resolve_dtype(name: str) -> torch.dtype:
    key = name.lower().strip()
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported save_dtype: {name}")
    return mapping[key]


def _load_backbone(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    ckpt_path = cfg.get("model_checkpoint", None)
    if ckpt_path:
        ckpt_path = Path(to_absolute_path(str(ckpt_path)))
        module = WASBLightningModule.load_from_checkpoint(
            str(ckpt_path), map_location=device
        )
        model = getattr(module, "model", None)
        if model is None:
            raise RuntimeError("Loaded checkpoint does not contain a model.")
    else:
        model_cfg = cfg.get("model", {})
        model = DinoV3FPNHeatmap(model_cfg)
        backbone_ckpt = None
        if hasattr(model_cfg, "get"):
            backbone_ckpt = model_cfg.get("backbone_checkpoint")
        if backbone_ckpt:
            model.load_backbone_checkpoint(backbone_ckpt)

    if hasattr(model, "module"):
        model = model.module

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise AttributeError("Model does not expose a DINOv3 backbone.")
    backbone.to(device)
    backbone.eval()
    return backbone


@torch.no_grad()
def _encode_clip(
    *,
    loader: DataLoader,
    backbone: torch.nn.Module,
    device: torch.device,
    save_dtype: torch.dtype,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for frames in loader:
        frames = frames.to(device)
        outputs = backbone.get_intermediate_layers(
            frames,
            n=1,
            reshape=False,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        tokens = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
        if tokens.dim() != 3:
            raise RuntimeError(f"Expected patch tokens [B, N, C], got {tuple(tokens.shape)}")
        chunks.append(tokens.to(device="cpu", dtype=save_dtype))
    if not chunks:
        raise RuntimeError("No frames encoded for clip.")
    return torch.cat(chunks, dim=0)


def _clip_output_path(
    output_dir: Path, match: str, clip: str, aug_idx: int, num_augments: int
) -> Path:
    suffix = "" if num_augments <= 1 else f"_aug{aug_idx:02d}"
    return output_dir / match / f"{clip}{suffix}.pt"


@hydra.main(config_path="../../configs", config_name="encode_dinov3_tokens", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    root_dir = Path(to_absolute_path(str(cfg.data.get("root_dir", "data/tennis"))))
    output_dir = Path(
        to_absolute_path(str(cfg.get("output_dir", "data/tennis/patch_embeddings")))
    )
    image_ext = str(cfg.data.get("image_ext", ".jpg"))

    device = torch.device(str(cfg.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")

    matches_override = cfg.get("matches", None)
    matches = _resolve_matches(
        root_dir=root_dir,
        train_matches=cfg.data.get("train_matches", []),
        val_matches=cfg.data.get("val_matches", []),
        test_matches=cfg.data.get("test_matches", []),
        override=matches_override,
    )
    if not matches:
        raise RuntimeError(f"No matches found under {root_dir}")

    transform = _build_transform(cfg)
    save_dtype = _resolve_dtype(str(cfg.get("save_dtype", "float32")))
    backbone = _load_backbone(cfg, device)
    patch_size = getattr(backbone, "patch_size", None)
    resize_hw = cfg.preprocess.get("resize_hw", None)
    if patch_size and resize_hw is not None:
        if int(resize_hw[0]) % int(patch_size) != 0 or int(resize_hw[1]) % int(patch_size) != 0:
            LOGGER.warning(
                "resize_hw %s is not divisible by patch_size=%s; token grid may be cropped.",
                resize_hw,
                patch_size,
            )

    num_augments = int(cfg.get("num_augments", 1))
    total_clips = 0
    for match in matches:
        match_dir = root_dir / match
        if not match_dir.exists():
            LOGGER.warning("Match directory missing, skipping: %s", match_dir)
            continue
        for clip_dir in sorted(match_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            frame_paths = _list_frames(clip_dir, image_ext)
            if not frame_paths:
                continue

            for aug_idx in range(num_augments):
                out_path = _clip_output_path(
                    output_dir=output_dir,
                    match=match,
                    clip=clip_dir.name,
                    aug_idx=aug_idx,
                    num_augments=num_augments,
                )
                if out_path.exists() and not bool(cfg.get("overwrite", False)):
                    continue

                dataset = ClipFramesDataset(frame_paths, transform)
                loader = DataLoader(
                    dataset,
                    batch_size=int(cfg.get("batch_size", 32)),
                    num_workers=int(cfg.get("num_workers", 4)),
                    pin_memory=bool(cfg.get("pin_memory", True)),
                    shuffle=False,
                )
                tokens = _encode_clip(
                    loader=loader,
                    backbone=backbone,
                    device=device,
                    save_dtype=save_dtype,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(tokens, out_path)
            total_clips += 1
            if total_clips % 25 == 0:
                LOGGER.info("Encoded %d clips", total_clips)

    LOGGER.info("Finished encoding %d clips", total_clips)


if __name__ == "__main__":
    main()
