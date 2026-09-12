"""Create a synchronized Meiji three-camera prediction video on CPU.

Example:
    .venv/bin/python -m src.tasks.ball_detection.scripts.visualize_meiji_multiview \
      --clip-dir data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_002/clips/clip_023 \
      --checkpoint outputs/colab_import/meiji-l4-bmp-acc4-20260911-r2/checkpoints/best-val-epoch12.ckpt \
      --output outputs/inference_visualizations/video_002__clip_023.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.tasks.ball_detection.inference import BallDetectionPredictor
from src.tasks.ball_detection.visualization.multiview_video import (
    default_cpu_threads,
    run_cpu_visualization,
)
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=default_cpu_threads())
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--peak-threshold", type=float, default=0.5)
    return parser


def _checkpoint_resolver(checkpoint_path: Path) -> PathResolver:
    checkpoint_root = checkpoint_path.resolve().parent
    return PathResolver(
        RuntimePathRoots.from_mapping(
            {
                "project_root": str(PROJECT_ROOT),
                "data_root": "data",
                "checkpoint_root": str(checkpoint_root),
                "artifact_root": "assets",
                "output_root": "outputs",
                "cache_root": ".cache",
                "external_asset_root": "third_party",
            },
            repository_root=PROJECT_ROOT,
        )
    )


def main() -> int:
    """Load the trained model on CPU and render one synchronized clip."""
    args = _parser().parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("--output must have an .mp4 suffix.")

    predictor = BallDetectionPredictor.load_from_checkpoint(
        checkpoint,
        resolver=_checkpoint_resolver(checkpoint),
        device=torch.device("cpu"),
        subpixel_refine=True,
        strict=True,
        weights_only=False,
    )
    if predictor.configured_frames != 8:
        raise ValueError(
            f"This Meiji recipe requires an 8-frame checkpoint, got "
            f"{predictor.configured_frames}."
        )
    result = run_cpu_visualization(
        predictor=predictor,
        checkpoint_path=checkpoint,
        clip_dir=args.clip_dir,
        output_path=args.output,
        image_size_hw=(288, 512),
        sequence_length=8,
        window_stride=args.window_stride,
        inference_batch_size=args.inference_batch_size,
        peak_threshold=args.peak_threshold,
        cpu_threads=args.cpu_threads,
    )
    print(f"video={result.video_path}")
    print(f"preview={result.preview_path}")
    print(f"metadata={result.metadata_path}")
    print(f"inference_seconds={result.inference_seconds:.3f}")
    print(f"render_seconds={result.render_seconds:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
