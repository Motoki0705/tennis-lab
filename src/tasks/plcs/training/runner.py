"""PLCS training runner using a unified data/model I/O pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.model_io import (
    resolve_plcs_track_query_reference_contract,
    validate_plcs_checkpoint_court_keypoints,
    validate_plcs_checkpoint_track_query_reference,
)
from src.tasks.plcs.training.composition import (
    build_plcs_datamodule,
    build_plcs_lightning_module,
)
from src.tasks.plcs.training.tracking_lightning_module import (
    _require_presence_competition,
)
from src.utils.schema.court_normalization import validate_court_coordinate_normalization


class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS."""

    def prepare_config(self, config: Any) -> None:
        PLCSTrainingConfig.from_config(config)
        super().prepare_config(config)

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return build_plcs_datamodule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return build_plcs_lightning_module(config)

    def maybe_load_init_weights(
        self,
        config: Any,
        lightning_module: pl.LightningModule,
    ) -> None:
        """Validate artifact contracts before weight-only initialization."""
        init_path = config.run.init_weights
        if init_path is not None:
            checkpoint = torch.load(
                init_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Invalid PLCS init_weights checkpoint: {init_path}.")
            runtime = PLCSTrainingConfig.from_config(lightning_module.config)
            validate_court_coordinate_normalization(
                checkpoint,
                artifact="PLCS init_weights checkpoint",
            )
            validate_plcs_checkpoint_court_keypoints(
                checkpoint,
                runtime.court_keypoint_contract,
            )
            if runtime.model.name in {
                "plcs_track_query",
                "plcs_track_query_ablation",
                "plcs_track_query_reference",
                "plcs_track_query_reference_ablation",
            }:
                validate_plcs_checkpoint_track_query_reference(
                    checkpoint,
                    resolve_plcs_track_query_reference_contract(
                        runtime.model,
                        runtime.court_keypoint_contract,
                    ),
                )
            if (
                runtime.model.track_query_presence_competition
                in {"deepsets", "deepsets_centered"}
                and runtime.fine_tune_mode != "presence_competition"
            ):
                source_state = checkpoint.get("state_dict")
                if not isinstance(source_state, Mapping):
                    raise TypeError(
                        "Enabled PLCS presence-competition init_weights "
                        "checkpoint requires a mapping 'state_dict'."
                    )
                if any(not isinstance(key, str) for key in source_state):
                    raise TypeError(
                        "Enabled PLCS presence-competition init_weights "
                        "state_dict keys must all be strings."
                    )
                target_keys = set(lightning_module.state_dict())
                source_keys = set(source_state)
                missing_enabled_keys = target_keys - source_keys
                unexpected_enabled_keys = source_keys - target_keys
                if missing_enabled_keys or unexpected_enabled_keys:
                    raise RuntimeError(
                        "Enabled PLCS presence-competition init_weights must "
                        "match exactly outside fine_tune_mode="
                        "'presence_competition'; "
                        f"missing={sorted(missing_enabled_keys)[:5]}, "
                        f"unexpected={sorted(unexpected_enabled_keys)[:5]}."
                    )
                lightning_module.load_state_dict(source_state, strict=True)
                return
            if runtime.fine_tune_mode == "presence_head":
                source_state = checkpoint.get("state_dict")
                if not isinstance(source_state, Mapping):
                    raise TypeError(
                        "PLCS presence-head init_weights checkpoint requires a "
                        "mapping 'state_dict'."
                    )
                target_model_keys = {
                    key
                    for key in lightning_module.state_dict()
                    if key.startswith("model.")
                }
                source_keys = {
                    key for key in source_state if isinstance(key, str)
                }
                missing_keys = sorted(target_model_keys - source_keys)
                if missing_keys:
                    raise RuntimeError(
                        "PLCS presence-head init_weights must initialize every "
                        "model parameter and buffer before freezing; checkpoint "
                        f"{init_path} is missing {len(missing_keys)} required "
                        f"target key(s): {missing_keys[:5]}."
                    )
            if runtime.fine_tune_mode == "presence_competition":
                source_state = checkpoint.get("state_dict")
                if not isinstance(source_state, Mapping):
                    raise TypeError(
                        "PLCS presence-competition init_weights checkpoint "
                        "requires a mapping 'state_dict'."
                    )
                if any(not isinstance(key, str) for key in source_state):
                    raise TypeError(
                        "PLCS presence-competition init_weights state_dict keys "
                        "must all be strings."
                    )
                target_state = lightning_module.state_dict()
                target_keys = set(target_state)
                source_keys = set(source_state)
                branch_prefix = "model.presence_competition."
                branch_keys = {
                    key for key in target_keys if key.startswith(branch_prefix)
                }
                if not branch_keys:
                    raise RuntimeError(
                        "PLCS presence-competition fine-tuning resolved no "
                        "competition state keys."
                    )
                competition_missing_keys = target_keys - source_keys
                competition_unexpected_keys = source_keys - target_keys
                if competition_missing_keys not in (set(), branch_keys):
                    disallowed_missing = sorted(
                        competition_missing_keys - branch_keys
                    )
                    raise RuntimeError(
                        "PLCS presence-competition init_weights may omit exactly "
                        "all competition branch keys for legacy migration, but "
                        f"found disallowed/partial missing keys: "
                        f"{(disallowed_missing or sorted(competition_missing_keys))[:5]}."
                    )
                if competition_unexpected_keys:
                    raise RuntimeError(
                        "PLCS presence-competition init_weights rejects "
                        "unexpected state keys: "
                        f"{sorted(competition_unexpected_keys)[:5]}."
                    )
                if competition_missing_keys == branch_keys:
                    _require_presence_competition(
                        lightning_module.model
                    ).reset_output_projection()
                result = lightning_module.load_state_dict(
                    source_state,
                    strict=False,
                )
                if (
                    set(result.missing_keys) != competition_missing_keys
                    or result.unexpected_keys
                ):
                    raise RuntimeError(
                        "PLCS presence-competition init_weights load result "
                        "violated the validated key allowlist."
                    )
                return
        super().maybe_load_init_weights(config, lightning_module)

    def callbacks_extra(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        logger: TensorBoardLogger,
    ) -> list[Any]:
        extras: list[Any] = super().callbacks_extra(config, datamodule, logger)

        runtime = PLCSTrainingConfig.from_config(config)
        if runtime.data.backend != "chunked":
            return extras

        from src.tasks.base.training.chunk_rotation_callback import (
            ChunkRotationCallback,
        )

        extras.append(ChunkRotationCallback())
        return extras
