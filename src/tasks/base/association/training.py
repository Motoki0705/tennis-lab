"""Training entry shared by PLCS and BLCS association-only models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any, cast

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torch.utils.data import Dataset

from src.tasks.base.association.data import AssociationDataModule
from src.tasks.base.association.loss import association_loss
from src.tasks.base.association.model import (
    AssociationModelConfig,
    ViewAssociationModel,
)
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.configuration.paths import PathRole
from src.utils.paths import PROJECT_ROOT

ASSOCIATION_CONTRACT = "camera_local_temporal_view_query_association_v1"
INPUT_KEYS = (
    "object_uv",
    "object_vis",
    "court_kp",
    "court_vis",
    "padding_mask",
    "reference_view_index",
)


class AssociationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict[str, Any],
        num_keypoints: int,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = ViewAssociationModel(
            AssociationModelConfig.from_mapping(model_config),
            num_keypoints=num_keypoints,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, **observations: Tensor) -> dict[str, Tensor]:
        return cast(dict[str, Tensor], self.model(**observations))

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        output = self.model(**{key: batch[key] for key in INPUT_KEYS})
        observed = batch["object_vis"].any(-1) & ~batch["padding_mask"][..., None]
        losses = association_loss(
            output,
            batch["object_id_target"],
            observed,
            batch["side_target"],
            (~batch["padding_mask"]).any(-1),
            batch["reference_view_index"],
        )
        self.log_dict(
            {f"{stage}/{key}": value for key, value in losses.items()},
            batch_size=batch["object_uv"].shape[0],
            on_step=stage == "train",
            on_epoch=True,
        )
        return losses["loss"]

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        self._step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["association_contract"] = ASSOCIATION_CONTRACT

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("association_contract") != ASSOCIATION_CONTRACT:
            raise ValueError(
                "Not a camera-local association checkpoint; retraining is required"
            )


def run_association_training(
    config: DictConfig, *, task: str, dataset_factory: Callable[..., Dataset]
) -> None:
    if task not in ("plcs", "blcs"):
        raise ValueError("Unknown association task")
    spec = OmegaConf.to_container(config.association, resolve=True)
    if not isinstance(spec, dict) or set(spec) != {
        "model",
        "limit_train_batches",
        "limit_val_batches",
    }:
        raise ValueError(
            "association requires model, limit_train_batches, limit_val_batches"
        )
    model_config = AssociationModelConfig.from_mapping(dict(spec["model"]))
    # The mature task dataset keeps its strict tracking-input/augmentation schema.
    # Only this task owns the association-only model, labels and loss.
    data_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    del data_config["association"]
    if (
        data_config.court_keypoints.selector != "camera_view_v2"
        or int(data_config.model.num_queries) != model_config.num_slots
    ):
        raise ValueError(
            "Association requires raw camera_view_v2 and matching local slot capacity"
        )
    pl.seed_everything(int(config.run.seed), workers=True)
    torch.set_float32_matmul_precision("high")
    module = AssociationLightningModule(
        asdict(model_config),
        17 if task == "plcs" else 1,
        float(config.training.learning_rate),
        float(config.training.weight_decay),
    )
    datamodule = AssociationDataModule(
        data_config, task=task, dataset_factory=dataset_factory
    )
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(dict(config.paths), repository_root=PROJECT_ROOT)
    )
    output_dir = resolver.resolve(PathRole.OUTPUT, str(config.run.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")
    callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        monitor="val/loss",
        mode="min",
        save_top_k=2,
        save_last=True,
        filename="association-{epoch:03d}",
    )
    trainer_config = config.training.trainer
    trainer = pl.Trainer(
        accelerator="gpu" if int(config.run.gpus) else "cpu",
        devices=1,
        default_root_dir=output_dir,
        logger=CSVLogger(str(output_dir), name="metrics"),
        callbacks=[callback],
        max_epochs=int(trainer_config.max_epochs),
        precision=str(trainer_config.precision),
        gradient_clip_val=float(trainer_config.gradient_clip_val),
        accumulate_grad_batches=int(trainer_config.accumulate_grad_batches),
        check_val_every_n_epoch=int(trainer_config.check_val_every_n_epoch),
        log_every_n_steps=int(trainer_config.log_every_n_steps),
        deterministic=bool(trainer_config.deterministic),
        limit_train_batches=spec["limit_train_batches"],
        limit_val_batches=spec["limit_val_batches"],
        fast_dev_run=bool(config.run.fast_dev_run),
        enable_progress_bar=bool(trainer_config.enable_progress_bar),
    )
    if config.run.dry_run:
        datamodule.setup("fit")
        return
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.run.resume)
