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

from omegaconf import DictConfig

from src.tasks.slcs.configuration import SLCSPrecomputeConfig
from src.tasks.slcs.data.dino_precompute import run_precompute
from src.tasks.slcs.model_io.factory import create_slcs_frame_token_encoder
from src.utils.hydra import hydra_main


def run(config: DictConfig) -> int:
    """Execute precompute; returns a process exit code."""
    runtime = SLCSPrecomputeConfig.from_config(config)
    spec = runtime.data.pipeline.dino_spec
    encoder = create_slcs_frame_token_encoder(runtime)

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
