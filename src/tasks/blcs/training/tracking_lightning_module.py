"""Lightning module for the multi-ball track-query baseline."""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any, cast

import numpy as np
import torch

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.blcs.models import build_blcs_model
from src.tasks.blcs.training.tracking_losses import BLCSTrackingLoss
from src.tasks.blcs.training.tracking_metrics import blcs_tracking_metrics

logger = logging.getLogger(__name__)


class BLCSTrackingLightningModule(BaseLightningModule):
    """Train and evaluate multi-ball clip-local slots."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model = build_blcs_model(config)
        self.criterion = BLCSTrackingLoss(config.loss)
        root = as_config_mapping(config, path="configuration")
        self.tracking_metrics = TrackingMetricConfig.from_mapping(
            require_config_mapping(root, "tracking_metrics", path="configuration")
        )

    @staticmethod
    def _migrate_legacy_group_embedding_keys(
        state_dict: MutableMapping[str, torch.Tensor],
    ) -> None:
        """Rename the temporary ``group_encoder`` checkpoint prefix."""
        legacy_prefix = "model.group_encoder."
        current_prefix = "model.group_embed."
        legacy_keys = [key for key in state_dict if key.startswith(legacy_prefix)]
        if not legacy_keys:
            return

        collisions = [
            current_prefix + key.removeprefix(legacy_prefix)
            for key in legacy_keys
            if current_prefix + key.removeprefix(legacy_prefix) in state_dict
        ]
        if collisions:
            raise RuntimeError(
                "Checkpoint contains both legacy group_encoder and current "
                f"group_embed keys: {collisions}."
            )

        for legacy_key in legacy_keys:
            current_key = current_prefix + legacy_key.removeprefix(legacy_prefix)
            state_dict[current_key] = state_dict.pop(legacy_key)
        logger.info(
            "Migrated %d legacy group_encoder checkpoint keys to group_embed.",
            len(legacy_keys),
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Apply explicit state-dict migrations before Lightning loads weights."""
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, MutableMapping):
            raise TypeError("Tracking checkpoint must contain a state_dict mapping.")
        self._migrate_legacy_group_embedding_keys(state_dict)

    @staticmethod
    def _model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        keys = (
            "ball_uv",
            "ball_visible",
            "court_kp",
            "court_vis",
            "frame_mask",
            "view_mask",
        )
        return {key: batch[key] for key in keys}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return cast(dict[str, torch.Tensor], self.model(**self._model_inputs(batch)))

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = self(batch)
        losses, assignments = self.criterion(prediction, batch)
        self.log(
            f"{stage}/loss", losses["total"], on_step=stage == "train", on_epoch=True
        )
        for name, value in losses.items():
            if name != "total":
                self.log(f"{stage}/loss_{name}", value, on_step=False, on_epoch=True)
        if stage != "train":
            for name, value in blcs_tracking_metrics(
                prediction,
                batch,
                assignments,
                config=self.tracking_metrics,
            ).items():
                self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True)
        return losses["total"], prediction

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        loss, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        _, prediction = self._shared_step(batch, "val")
        return prediction

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        _, prediction = self._shared_step(batch, "test")
        self.collect_test_predictions(batch, prediction)
        return prediction

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return {
            "pred_position": self._to_numpy(result["position"]),
            "pred_presence_logits": self._to_numpy(result["presence_logits"]),
            "target_position": self._to_numpy(batch["target_position"]),
            "target_presence": self._to_numpy(batch["target_presence"]),
            "target_instance_id": self._to_numpy(batch["target_instance_id"]),
            "frame_mask": self._to_numpy(batch["frame_mask"]),
        }

    def on_test_epoch_end(self) -> None:
        metrics = {
            key.removeprefix("test/"): float(value.detach().cpu())
            for key, value in self.trainer.callback_metrics.items()
            if key.startswith("test/") and isinstance(value, torch.Tensor)
        }
        self.save_test_predictions(metrics)
