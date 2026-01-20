"""Train a 3D-trajectory event detection model using Hydra-managed configuration.

Example commands:
    `uv run python -m src.evnet_detection.scripts.train_3d`
    `uv run python -m src.evnet_detection.scripts.train_3d run.dry_run=true`
    `uv run python -m src.evnet_detection.scripts.train_3d data.scene_dir=data/blcs run.dry_run=false`

Config entry point: `src/evnet_detection/configs/train_3d.yaml`
"""

# mypy: disable-error-code=misc

from __future__ import annotations

import logging
import os
import types
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.evnet_detection.data.datamodule import EventDetectionDataModule
from src.evnet_detection.training.lightning_module import EventDetectionLightningModule

LOGGER = logging.getLogger(__name__)


def _select_devices(gpus: int) -> tuple[str, int]:
    if gpus > 0 and torch.cuda.is_available():
        return "gpu", gpus
    return "cpu", 1


def _force_cpu_for_dry_run() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.cuda.is_available = types.MethodType(lambda *_a, **_k: False, torch.cuda)
    torch.cuda.device_count = types.MethodType(lambda *_a, **_k: 0, torch.cuda)
    torch.cuda.current_device = types.MethodType(lambda *_a, **_k: 0, torch.cuda)


def run_training(cfg: DictConfig) -> None:
    """Run training or a dry run."""
    pl.seed_everything(int(cfg.run.seed))

    output_dir = Path(to_absolute_path(str(cfg.run.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    if bool(cfg.run.dry_run):
        _force_cpu_for_dry_run()

    data_module = EventDetectionDataModule(cfg)
    model = EventDetectionLightningModule(cfg)

    if bool(cfg.run.dry_run):
        data_module._resolved = data_module._resolved.__class__(  # type: ignore[attr-defined]
            scene_dir=data_module._resolved.scene_dir,
            batch_size=2,
            num_workers=0,
            input_type=data_module._resolved.input_type,
            allow_dummy=True,
        )
        data_module.setup("fit")
        batch = next(iter(data_module.train_dataloader()))
        logits = model.forward(batch)
        LOGGER.info("Dry-run batch logits: %s", tuple(logits.shape))

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, data_module)
        LOGGER.info("Dry run complete: %s", output_dir)
        return

    accelerator, devices = _select_devices(int(cfg.run.gpus))
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=float(cfg.training.gradient_clip_val),
        logger=tb_logger,
        precision="16-mixed" if accelerator == "gpu" else 32,
        log_every_n_steps=50,
        deterministic=True,
    )
    trainer.fit(model, data_module)


@hydra.main(config_path="../configs", config_name="train_3d", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    run_training(cfg)


if __name__ == "__main__":
    main()
