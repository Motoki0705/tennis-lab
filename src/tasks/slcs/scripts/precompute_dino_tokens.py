"""
Precompute DINOv3 patch tokens for every clip of an issue #634 dataset.

Usage:
    python -m src.tasks.slcs.scripts.precompute_dino_tokens
    python -m src.tasks.slcs.scripts.precompute_dino_tokens data.dataset_root=tennis_scene_dataset
    python -m src.tasks.slcs.scripts.precompute_dino_tokens precompute.overwrite=true precompute.device=cuda

Notes:
    - Configuration is loaded from `src/tasks/slcs/configs/precompute_dino_tokens.yaml`;
      the token spec (backbone, input size, stride) comes from `data.dino` so
      training and precompute cannot diverge.
    - Dataset, checkpoint, and DINO repository paths are resolved from their
      declared runtime roots before the encoder is loaded.
    - Tokens are written to `annotations/dino_v3/` per clip with a completion
      marker written last; completed clips are skipped unless overwrite=true.
    - Per-clip failures are reported at the end and the exit code is non-zero
      if any clip failed.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig

from src.tasks.slcs.configuration import SLCSPrecomputeConfig
from src.tasks.slcs.data.dino_precompute import FrameEncoder, run_precompute
from src.utils.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.utils.hydra import hydra_main
from src.utils.models.loading.dinov3 import load_dinov3_backbone


def _build_encoder(config: SLCSPrecomputeConfig) -> tuple[FrameEncoder, int, int]:
    """Load the DINOv3 backbone and wrap it as a FrameEncoder."""
    device = torch.device(config.device)
    adapter = load_dinov3_backbone(
        repository_path=config.repository_path,
        checkpoint_path=config.checkpoint_path,
        backbone_name=config.data.pipeline.dino_spec.backbone,
        strict=config.strict,
    )
    adapter = adapter.to(device)
    adapter.eval()

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def encoder(frames: NDArray[np.uint8]) -> NDArray[np.float16]:
        with torch.no_grad():
            x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            tokens = adapter.forward_features(x)["x_norm_patchtokens"]
            return np.asarray(tokens.to(torch.float16).cpu().numpy(), dtype=np.float16)

    return encoder, adapter.embed_dim, adapter.patch_size


def run(config: DictConfig) -> int:
    """Execute precompute; returns a process exit code."""
    runtime = SLCSPrecomputeConfig.from_config(config)
    spec = runtime.data.pipeline.dino_spec
    encoder, embed_dim, patch_size = _build_encoder(runtime)
    if embed_dim != spec.embed_dim or patch_size != spec.patch_size:
        raise ValueError(
            f"Configured dino spec (embed_dim={spec.embed_dim}, patch_size="
            f"{spec.patch_size}) does not match the loaded backbone "
            f"(embed_dim={embed_dim}, patch_size={patch_size})."
        )

    report = run_precompute(
        runtime.data.dataset_root,
        encoder,
        spec,
        batch_size=runtime.batch_size,
        overwrite=runtime.overwrite,
        generator={
            "script": "src/tasks/slcs/scripts/precompute_dino_tokens.py",
            "backbone": spec.backbone,
        },
    )
    print(
        f"processed={len(report.processed)} skipped_existing={len(report.skipped_existing)} "
        f"failed={len(report.failed)}"
    )
    for clip_id, error in report.failed.items():
        print(f"FAILED {clip_id}: {error}", file=sys.stderr)
    return 0 if report.ok else 1


@hydra_main(
    config_path="../configs",
    config_name="precompute_dino_tokens",
    version_base="1.3",
    validation_boundary="slcs.precompute_dino_tokens",
)
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for DINOv3 token precompute."""
    exit_code = run(config)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
