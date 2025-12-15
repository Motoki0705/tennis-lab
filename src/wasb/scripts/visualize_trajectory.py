"""Visualize WASB trajectory completion results (Hydra-based).

Example commands:
    `uv run python -m src.wasb.scripts.visualize_trajectory visualization.checkpoint=outputs/trajectory/logs/version_0/checkpoints/last.ckpt`

Config entry point: `src/wasb/configs/visualize_trajectory.yaml`
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.wasb.data.trajectory_datamodule import TrajectoryDataModule
from src.wasb.training.trajectory_lightning_module import TrajectoryLightningModule


def _resolve_device(gpus: int) -> torch.device:
    if gpus > 0 and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_datamodule(config: DictConfig, split: str) -> TrajectoryDataModule:
    """Build and set up a trajectory datamodule for the requested split."""
    datamodule = TrajectoryDataModule(config)
    if split in ("train", "val"):
        datamodule.setup(stage="fit")
    else:
        datamodule.setup(stage="test")
    return datamodule


def get_dataloader(datamodule: TrajectoryDataModule, split: str):  # type: ignore[no-untyped-def]
    """Return the dataloader matching the requested split."""
    if split == "train":
        return datamodule.train_dataloader()
    if split == "val":
        return datamodule.val_dataloader()
    return datamodule.test_dataloader()


def load_model(
    checkpoint_path: Path,
    config: DictConfig,
    device: torch.device,
) -> TrajectoryLightningModule:
    """Load a trained lightning module from checkpoint."""
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
    """Render up to `max_samples` windows from a single batch."""
    xy_input_norm = batch["xy_input_norm"].to(device)
    target_xy_norm = batch["target_xy_norm"].to(device)
    loss_mask_block = batch["loss_mask_block"].to(device)
    loss_mask_sparse = batch["loss_mask_sparse"].to(device)
    loss_mask_noise = batch["loss_mask_noise"].to(device)
    orig_visibility = batch["orig_visibility"].to(device)

    with torch.no_grad():
        if getattr(module, "is_iterative", False):
            xk = xy_input_norm
            num_steps = int(getattr(module, "num_steps", 1))
            for _ in range(max(num_steps, 1)):
                delta = module(xk)
                xk = xk + delta
            pred_norm = xk
        else:
            pred_norm = module(xy_input_norm)

    scale = torch.tensor([1920.0, 1080.0], dtype=torch.float32, device=device)
    xy_input_px = xy_input_norm * scale
    target_px = target_xy_norm * scale
    pred_px = pred_norm * scale

    bsz, _t, _ = target_px.shape
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
            ax.scatter(
                pred[mask_noise, 0],
                pred[mask_noise, 1],
                c="purple",
                s=30,
                marker="x",
                label="pred noise",
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
        uniq = dict(zip(labels, handles, strict=False))
        ax.legend(uniq.values(), uniq.keys(), loc="best")

        out_path = output_dir / f"{split}_sample_{start_index + num_saved:03d}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        num_saved += 1

    return num_saved


@hydra.main(config_path="../configs", config_name="visualize_trajectory", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    seed = int(cfg.run.seed)
    pl.seed_everything(seed)

    if cfg.visualization.checkpoint is None:
        raise ValueError("visualization.checkpoint is required")

    checkpoint = Path(to_absolute_path(str(cfg.visualization.checkpoint)))
    output_dir = Path(to_absolute_path(str(cfg.visualization.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    split = str(cfg.visualization.split)
    max_samples = int(cfg.visualization.num_samples)
    device = _resolve_device(int(cfg.run.gpus))

    datamodule = load_datamodule(cfg, split=split)
    dataloader = get_dataloader(datamodule, split=split)
    module = load_model(checkpoint, cfg, device=device)

    num_done = 0
    for batch in dataloader:
        num_done += visualize_batch(
            batch,
            module=module,
            device=device,
            output_dir=output_dir,
            split=split,
            start_index=num_done,
            max_samples=max_samples,
        )
        if num_done >= max_samples:
            break


if __name__ == "__main__":
    main()
