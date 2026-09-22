"""Shared loss/statistics lifecycle for task-owned association Lightning modules."""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.model_io.association_contracts import (
    ASSOCIATION_CONTRACT,
    validate_association_checkpoint,
)
from src.tasks.base.training.association_losses import association_loss
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.utils.configuration import PathRole

_COUNT_KEYS = (
    "identity_loss_sum",
    "identity_count",
    "identity_correct",
    "side_loss_sum",
    "side_count",
    "side_correct",
    "same_count",
    "same_correct",
    "opposite_count",
    "opposite_correct",
    "fp_count",
    "fp_predicted",
    "fp_correct",
)


class AssociationLightningBase(BaseLightningModule):
    model_io: Any

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._statistics: dict[str, dict[str, Tensor]] = {}

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        call = self.model_io.build_call(batch)
        output = self.model_io.execute_call(call)
        values = association_loss(
            output,
            batch["object_id_target"],
            batch["object_vis"].any(-1) & ~batch["padding_mask"][..., None],
            batch["side_target"],
            (~batch["padding_mask"]).any(-1),
            batch["reference_view_index"],
            identity_weight=float(self.config.loss.identity_weight),
            side_weight=float(self.config.loss.side_weight),
            side_threshold=float(self.config.metrics.side_threshold),
        )
        state = self._statistics.setdefault(
            stage,
            {key: torch.zeros((), device=values["loss"].device) for key in _COUNT_KEYS},
        )
        for key in _COUNT_KEYS:
            state[key] += values[key].detach()
        if stage == "train":
            self.log(
                "train/step_loss",
                values["loss"],
                on_step=True,
                on_epoch=False,
                batch_size=batch["object_uv"].shape[0],
            )
        return values["loss"]

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        self._step(batch, "test")

    def _finish_epoch(self, stage: str) -> None:
        state = self._statistics.pop(stage, {})
        if not state:
            return

        def ratio(numerator: str, denominator: str) -> Tensor:
            return state[numerator] / state[denominator].clamp_min(1)

        metrics = {
            "identity_loss": ratio("identity_loss_sum", "identity_count"),
            "side_loss": ratio("side_loss_sum", "side_count"),
            "identity_accuracy": ratio("identity_correct", "identity_count"),
            "side_accuracy": ratio("side_correct", "side_count"),
            "same_recall": ratio("same_correct", "same_count"),
            "opposite_recall": ratio("opposite_correct", "opposite_count"),
            "fp_precision": ratio("fp_correct", "fp_predicted"),
            "fp_recall": ratio("fp_correct", "fp_count"),
        }
        metrics["loss"] = (
            metrics["identity_loss"] * self.config.loss.identity_weight
            + metrics["side_loss"] * self.config.loss.side_weight
        )
        metrics["side_balanced_accuracy"] = (
            metrics["same_recall"] + metrics["opposite_recall"]
        ) / 2
        self.log_dict(
            {f"{stage}/{key}": value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
        )
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            output = self.path_resolver.resolve(
                PathRole.OUTPUT,
                str(self.config.run.output_dir),
                "association_metrics.jsonl",
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "stage": stage,
                "epoch": int(self.current_epoch),
                "step": int(self.global_step),
                **{key: float(value.cpu()) for key, value in metrics.items()},
            }
            with output.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

    def on_train_epoch_end(self) -> None:
        self._finish_epoch("train")

    def on_validation_epoch_end(self) -> None:
        self._finish_epoch("val")

    def on_test_epoch_end(self) -> None:
        self._finish_epoch("test")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["association_contract"] = ASSOCIATION_CONTRACT
        checkpoint["association_model"] = str(self.config.model.name)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        validate_association_checkpoint(
            checkpoint, model_name=str(self.config.model.name)
        )
