"""
Example use:
```bash
uv run python src/wasb/scripts/visualize_trajectory.py \
  --config src/wasb/configs/trajectory.yaml \
  --checkpoint outputs/trajectory/logs/version_2/checkpoints/last.ckpt \
  --split test \
  --num-samples 8 \
  --output-dir outputs/trajectory/vis \
  --gpus 1
```
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.wasb.data.trajectory_datamodule import TrajectoryDataModule
from src.wasb.training.trajectory_lightning_module import (
    TrajectoryLightningModule,
)
from src.wasb.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize trajectory completion results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            Path(__file__).parents[1] / "configs" / "trajectory.yaml"
        ),
        help="Path to YAML config used for data paths and parameters",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt) of the trained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/trajectory/vis",
        help="Directory to save visualization images",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Which split to visualize",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of windows to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling/selection",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs to use (0 for CPU)",
    )
    return parser.parse_args()


def load_datamodule(config: OmegaConf, split: str) -> TrajectoryDataModule:
    datamodule = TrajectoryDataModule(config)
    if split in ("train", "val"):
        datamodule.setup(stage="fit")
    else:
        datamodule.setup(stage="test")
    return datamodule


def get_dataloader(datamodule: TrajectoryDataModule, split: str):  # type: ignore[no-untyped-def]
    if split == "train":
        return datamodule.train_dataloader()
    if split == "val":
        return datamodule.val_dataloader()
    return datamodule.test_dataloader()


def load_model(checkpoint_path: Path, config: OmegaConf, device: torch.device) -> TrajectoryLightningModule:
    module = TrajectoryLightningModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        config=config,
        steps_per_epoch=None,
        map_location=device,
    )
    module.to(device)
    module.eval()
    return module


def visualize_batch(
    batch,  # type: ignore[no-untyped-def]
    module: TrajectoryLightningModule,
    device: torch.device,
    output_dir: Path,
    split: str,
    start_index: int,
    max_samples: int,
) -> int:
    xy_input_norm = batch["xy_input_norm"].to(device)
    target_xy_norm = batch["target_xy_norm"].to(device)
    loss_mask_block = batch["loss_mask_block"].to(device)
    loss_mask_sparse = batch["loss_mask_sparse"].to(device)
    loss_mask_noise = batch["loss_mask_noise"].to(device)
    orig_visibility = batch["orig_visibility"].to(device)

    with torch.no_grad():
        pred_norm = module(xy_input_norm)

    scale = torch.tensor([1920.0, 1080.0], dtype=torch.float32, device=device)
    xy_input_px = xy_input_norm * scale
    target_px = target_xy_norm * scale
    pred_px = pred_norm * scale

    bsz, T, _ = target_px.shape
    num_saved = 0

    for b in range(bsz):
        if start_index + num_saved >= max_samples:
            break

        tgt = target_px[b].detach().cpu().numpy()
        inp = xy_input_px[b].detach().cpu().numpy()
        pred = pred_px[b].detach().cpu().numpy()

        vis = orig_visibility[b].detach().cpu().numpy()
        mask_block = loss_mask_block[b].detach().cpu().numpy() > 0.5
        mask_sparse = loss_mask_sparse[b].detach().cpu().numpy() > 0.5
        mask_noise = loss_mask_noise[b].detach().cpu().numpy() > 0.5

        valid = vis > 0
        observed_clean = valid & ~mask_block & ~mask_sparse & ~mask_noise

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(tgt[:, 0], tgt[:, 1], color="lightgray", linewidth=2, label="gt")

        if observed_clean.any():
            ax.scatter(
                inp[observed_clean, 0],
                inp[observed_clean, 1],
                c="blue",
                s=20,
                marker="o",
                label="input clean",
            )

        if mask_noise.any():
            ax.scatter(
                inp[mask_noise, 0],
                inp[mask_noise, 1],
                c="orange",
                s=20,
                marker="o",
                label="input noisy",
            )

        if mask_block.any():
            ax.scatter(
                pred[mask_block, 0],
                pred[mask_block, 1],
                c="red",
                s=30,
                marker="x",
                label="pred block",
            )

        if mask_sparse.any():
            ax.scatter(
                pred[mask_sparse, 0],
                pred[mask_sparse, 1],
                c="green",
                s=30,
                marker="x",
                label="pred sparse",
            )

        ax.set_title(f"{split} sample {start_index + num_saved}")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()

        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="best")

        out_path = output_dir / f"{split}_sample_{start_index + num_saved:03d}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        num_saved += 1

    return num_saved


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)

    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = load_datamodule(config, args.split)
    dataloader = get_dataloader(datamodule, args.split)

    device = torch.device(
        "cuda" if args.gpus > 0 and torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    module = load_model(checkpoint_path, config, device)

    num_saved = 0
    for batch in dataloader:
        if num_saved >= args.num_samples:
            break
        saved_now = visualize_batch(
            batch=batch,
            module=module,
            device=device,
            output_dir=output_dir,
            split=args.split,
            start_index=num_saved,
            max_samples=args.num_samples,
        )
        num_saved += saved_now

    print(f"Saved {num_saved} samples to {output_dir}")


if __name__ == "__main__":
    main()
