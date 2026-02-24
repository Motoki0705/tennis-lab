"""Compare two TrackNetV3 checkpoints on the same labeled subset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader, Subset

from src.tasks.ball_detection.data.datamodule import collate_ball_sequences
from src.tasks.ball_detection.data.labeled_dataset import LabeledBallDataset
from src.tasks.ball_detection.inference.predictor import BallPredictor


@dataclass(frozen=True)
class Metrics:
    """Evaluation metrics for one checkpoint."""

    checkpoint: str
    windows: int
    visible_frames: int
    acc_at_4px: float
    mae_px: float


def evaluate_checkpoint(
    *,
    checkpoint: str,
    model_config_path: str,
    root_dir: str,
    games: list[str],
    max_windows: int,
    batch_size: int,
    device: str,
) -> Metrics:
    """Evaluate one checkpoint on a fixed labeled subset."""
    dataset_full = LabeledBallDataset(
        root_dir=root_dir,
        games=games,
        image_size_hw=(288, 512),
        window_size=8,
        window_stride=8,
        min_window_size=8,
        context_frames=1,
    )
    windows = min(max_windows, len(dataset_full))
    dataset = Subset(dataset_full, list(range(windows)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_ball_sequences,
    )

    predictor = BallPredictor.load_from_checkpoint(
        checkpoint,
        device=device,
        fallback_model_cfg_path=model_config_path,
    )

    total_visible = 0
    total_correct = 0
    dist_sum = 0.0

    processed = 0
    for batch in loader:
        outputs = predictor.predict(batch["frames"])
        pred_xy = outputs["ball_uv"]

        target_xy = batch["target_xy"]
        target_vis = batch["target_vis"]
        frame_mask = batch["frame_mask"]

        valid = (frame_mask > 0) & (target_vis > 0)

        dx = (pred_xy[..., 0] - target_xy[..., 0]) * 511.0
        dy = (pred_xy[..., 1] - target_xy[..., 1]) * 287.0
        dist = torch.sqrt(dx * dx + dy * dy)

        total_visible += int(valid.sum().item())
        total_correct += int(((dist <= 4.0) & valid).sum().item())
        dist_sum += float((dist * valid).sum().item())

        processed += int(pred_xy.shape[0])
        if processed % max(batch_size * 8, 1) == 0:
            print(f"[{checkpoint}] processed_windows={processed}/{windows}", flush=True)

    return Metrics(
        checkpoint=checkpoint,
        windows=windows,
        visible_frames=total_visible,
        acc_at_4px=(total_correct / max(total_visible, 1)),
        mae_px=(dist_sum / max(total_visible, 1)),
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare two TrackNetV3 checkpoints.")
    parser.add_argument("--ckpt-a", required=True, help="First checkpoint path.")
    parser.add_argument("--ckpt-b", required=True, help="Second checkpoint path.")
    parser.add_argument(
        "--model-config",
        default="src/tasks/ball_detection/configs/model/tracknetv3.yaml",
        help="Fallback model config path for checkpoint loading.",
    )
    parser.add_argument("--root-dir", default="data/tennis")
    parser.add_argument("--games", nargs="+", default=["game10"])
    parser.add_argument("--max-windows", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Run checkpoint comparison and print JSON summary."""
    args = parse_args()
    if args.max_windows <= 0:
        raise ValueError("--max-windows must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    metrics_a = evaluate_checkpoint(
        checkpoint=args.ckpt_a,
        model_config_path=args.model_config,
        root_dir=args.root_dir,
        games=list(args.games),
        max_windows=int(args.max_windows),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    metrics_b = evaluate_checkpoint(
        checkpoint=args.ckpt_b,
        model_config_path=args.model_config,
        root_dir=args.root_dir,
        games=list(args.games),
        max_windows=int(args.max_windows),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )

    acc_delta = metrics_b.acc_at_4px - metrics_a.acc_at_4px
    mae_delta = metrics_b.mae_px - metrics_a.mae_px

    payload = {
        "a": asdict(metrics_a),
        "b": asdict(metrics_b),
        "delta": {
            "acc_at_4px_b_minus_a": acc_delta,
            "mae_px_b_minus_a": mae_delta,
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
