"""PyTorch Lightning module for composable Court detection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_saving import save_qualitative_clip
from src.tasks.court_detection.configuration import (
    CourtQueryLossConfig,
    CourtQueryModelConfig,
    CourtTrainingConfig,
)
from src.tasks.court_detection.data.bundle_state import (
    deserialize_query_checkpoint_state,
    deserialize_target_bundle,
    serialize_query_checkpoint_state,
    serialize_target_bundle,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
)
from src.tasks.court_detection.model_io.adapters import (
    CourtModelIOAdapter,
    CourtQueryModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtLogits,
    CourtQueryRawOutput,
    CourtQueryTrainingResult,
    CourtTrainingResult,
)
from src.tasks.court_detection.model_io.factory import build_court_detection_pair
from src.tasks.court_detection.training.metrics import (
    CourtDetectionMetrics,
    CourtPoseMetrics,
)
from src.tasks.court_detection.visualization.adapters.render_inputs import (
    CourtQualitativeRenderer,
    build_court_qualitative_renderer,
)


class CourtDetectionLightningModule(BaseLightningModule):
    """Train one shared Court trunk against any valid non-empty target bundle."""

    def __init__(
        self,
        config: object,
        *,
        target_bundle: CourtTargetBundleSpec | None = None,
        target_bundle_state: Mapping[str, object] | None = None,
        query_checkpoint_state: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(config)
        runtime = CourtTrainingConfig.from_config(config)
        query_variant = isinstance(runtime.model, CourtQueryModelConfig)
        if target_bundle is None:
            if query_variant:
                if query_checkpoint_state is None:
                    raise ValueError(
                        "Court query Lightning construction requires a target bundle "
                        "or its versioned query checkpoint snapshot."
                    )
                resolved_bundle = deserialize_query_checkpoint_state(
                    query_checkpoint_state
                ).target_bundle
            else:
                if target_bundle_state is None:
                    raise ValueError(
                        "Court Lightning construction requires a target bundle "
                        "or its checkpoint snapshot."
                    )
                resolved_bundle = deserialize_target_bundle(target_bundle_state)
        else:
            resolved_bundle = target_bundle
            if query_variant:
                if target_bundle_state is not None:
                    raise ValueError(
                        "Court query checkpoints cannot use the legacy dense-only snapshot."
                    )
                if query_checkpoint_state is not None:
                    restored = deserialize_query_checkpoint_state(
                        query_checkpoint_state
                    )
                    if restored.target_bundle != resolved_bundle:
                        raise ValueError(
                            "Court query target bundle disagrees with its checkpoint snapshot."
                        )
            elif (
                target_bundle_state is not None
                and deserialize_target_bundle(target_bundle_state) != resolved_bundle
            ):
                raise ValueError(
                    "Court target bundle disagrees with its checkpoint snapshot."
                )
        if query_variant:
            if not isinstance(runtime.loss, CourtQueryLossConfig):  # pragma: no cover
                raise TypeError("Court query runtime requires CourtQueryLossConfig.")
            query_snapshot = serialize_query_checkpoint_state(
                resolved_bundle,
                loss_config_name=runtime.loss.name,
                pose_supervision=runtime.loss.pose.enabled,
            )
            if query_checkpoint_state is not None:
                restored = deserialize_query_checkpoint_state(query_checkpoint_state)
                expected = deserialize_query_checkpoint_state(query_snapshot)
                if restored != expected:
                    raise ValueError(
                        "Court query supervision identity disagrees with checkpoint."
                    )
            self.save_hyperparameters({"query_checkpoint_state": query_snapshot})
        else:
            bundle_snapshot = serialize_target_bundle(resolved_bundle)
            self.save_hyperparameters({"target_bundle_state": bundle_snapshot})
        self.target_bundle = resolved_bundle
        self.query_variant = query_variant
        self.qualitative_fps = runtime.qualitative_fps
        self.qualitative_style = runtime.render_style

        model_pair = build_court_detection_pair(
            self.config,
            target_bundle=resolved_bundle,
        )
        self.model = model_pair.model
        self.model_io = cast(
            "CourtModelIOAdapter | CourtQueryModelIOAdapter",
            model_pair.adapter,
        )
        self.qualitative_renderers: dict[
            CourtTargetKind, CourtQualitativeRenderer
        ] = (
            {}
            if query_variant
            else {
                kind: build_court_qualitative_renderer(
                    cast(CourtModelIOAdapter, self.model_io),
                    kind=kind,
                )
                for kind in resolved_bundle.kinds
            }
        )
        self._stage_metrics: dict[
            str, dict[CourtTargetKind, CourtDetectionMetrics]
        ] = {
            stage: {
                kind: CourtDetectionMetrics(
                    kind,
                    spec.output_channels,
                    singleton_kp=query_variant and kind == "kp",
                )
                for kind, spec in resolved_bundle.targets.items()
            }
            for stage in ("train", "val", "test")
        }
        self._pose_metrics = {
            stage: CourtPoseMetrics() for stage in ("train", "val", "test")
        }

    def forward(
        self, *model_args: Tensor
    ) -> Mapping[CourtTargetKind, Tensor] | CourtQueryRawOutput:
        return cast(
            "Mapping[CourtTargetKind, Tensor] | CourtQueryRawOutput",
            self.model(*model_args),
        )

    def _shared_step(
        self,
        batch: Mapping[str, object],
        stage: str,
    ) -> CourtTrainingResult | CourtQueryTrainingResult:
        if isinstance(self.model_io, CourtQueryModelIOAdapter):
            query_call = self.model_io.prepare_training_batch(batch)
            output = cast(
                CourtQueryRawOutput,
                self.model(*query_call.model_call.model_args),
            )
            query_result = self.model_io.training_result(output, query_call)
            self._log_training_result(stage, query_result)
            image_size = batch.get("image_size")
            if not isinstance(image_size, Tensor):
                raise ValueError("Court batch image_size must be a Tensor.")
            for kind in self.target_bundle.kinds:
                self._stage_metrics[stage][kind].update(
                    query_result.output.dense_logits[kind],
                    query_call.dense_targets[kind],
                    image_size=image_size,
                )
            self._pose_metrics[stage].update(
                query_result.decoded_pose,
                query_call.pose_target,
            )
            return query_result
        legacy_call = self.model_io.prepare_training_batch(batch)
        logits = cast(CourtLogits, self.model(*legacy_call.model_call.model_args))
        result = self.model_io.training_result(logits, legacy_call)
        self.log(
            f"{stage}/loss",
            result.loss,
            prog_bar=True,
            sync_dist=True,
        )
        for kind, loss in result.losses.items():
            self.log(
                f"{stage}/loss_{kind}",
                loss,
                prog_bar=False,
                sync_dist=True,
            )
        image_size = batch.get("image_size")
        if not isinstance(image_size, Tensor):
            raise ValueError("Court batch image_size must be a Tensor.")
        for kind in self.target_bundle.kinds:
            self._stage_metrics[stage][kind].update(
                result.logits[kind],
                legacy_call.targets[kind],
                image_size=image_size,
            )
        return result

    def _log_training_result(
        self,
        stage: str,
        result: CourtQueryTrainingResult,
    ) -> None:
        self.log(f"{stage}/loss", result.loss, prog_bar=True, sync_dist=True)
        for kind, loss in result.dense_losses.items():
            self.log(
                f"{stage}/loss_{kind}",
                loss,
                prog_bar=False,
                sync_dist=True,
            )
        for name, loss in result.pose_losses.items():
            self.log(
                f"{stage}/loss_{name}",
                loss,
                prog_bar=False,
                sync_dist=True,
            )

    def _flush_stage_metrics(self, stage: str) -> dict[str, float]:
        flattened: dict[str, float] = {}
        for kind in self.target_bundle.kinds:
            tracker = self._stage_metrics[stage][kind]
            for metric_name, value in tracker.compute().items():
                name = f"{kind}_{metric_name}"
                flattened[name] = value
                self.log(
                    f"{stage}/{name}",
                    value,
                    prog_bar=(
                        stage == "val"
                        and metric_name in {"miou", "mean_dist", "dice"}
                    ),
                    sync_dist=False,
                )
            tracker.reset()
        if self.query_variant:
            pose_tracker = self._pose_metrics[stage]
            for name, value in pose_tracker.compute().items():
                flattened[f"pose_{name}"] = value
                self.log(
                    f"{stage}/pose_{name}",
                    value,
                    prog_bar=False,
                    sync_dist=False,
                )
            pose_tracker.reset()
        return flattened

    def training_step(
        self,
        batch: dict[str, object],
        batch_idx: int,
    ) -> Tensor:
        _ = batch_idx
        return self._shared_step(batch, "train").loss

    def on_train_epoch_end(self) -> None:
        self._flush_stage_metrics("train")

    def validation_step(
        self,
        batch: dict[str, object],
        batch_idx: int,
    ) -> None:
        _ = batch_idx
        self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._flush_stage_metrics("val")

    def on_test_epoch_start(self) -> None:
        self._reset_test_prediction_buffer()

    def test_step(
        self,
        batch: dict[str, object],
        batch_idx: int,
    ) -> None:
        _ = batch_idx
        result = self._shared_step(batch, "test")
        collected: object = (
            result.output
            if isinstance(result, CourtQueryTrainingResult)
            else result.logits
        )
        self.collect_test_predictions(
            batch,
            {"output": collected},
        )

    def on_test_epoch_end(self) -> None:
        metrics = self._flush_stage_metrics("test")
        saved = self.save_test_predictions(metrics=metrics)
        if saved is not None:
            print(f"[test] saved Court predictions -> {saved}")

    def test_prediction_payload(
        self,
        batch: dict[str, object],
        result: dict[str, object],
    ) -> dict[str, np.ndarray]:
        raw_output = result.get("output", result.get("logits"))
        payload: dict[str, np.ndarray] = {}
        image_size = batch.get("image_size")
        if isinstance(image_size, Tensor):
            payload["image_size"] = self._to_numpy(image_size)
        if isinstance(self.model_io, CourtQueryModelIOAdapter):
            if not isinstance(raw_output, CourtQueryRawOutput):
                raise ValueError("Court query test result requires typed raw output.")
            prediction = self.model_io.test_payload(batch, raw_output)
            payload["pose_translation_m"] = self._to_numpy(
                prediction.pose.translation_m
            )
            payload["pose_rotation"] = self._to_numpy(prediction.pose.rotation)
            payload["pose_focal_px"] = self._to_numpy(prediction.pose.focal_px)
            payload["pose_log_focal"] = self._to_numpy(prediction.pose.log_focal)
            predictions: Mapping[object, object] = cast(
                Mapping[object, object],
                prediction.dense,
            )
        else:
            if not isinstance(raw_output, Mapping):
                raise ValueError("Court test result requires a logits mapping.")
            decoded = self.model_io.test_payload(
                batch,
                cast(CourtLogits, raw_output),
            )
            raw_predictions = decoded["predictions"]
            if not isinstance(raw_predictions, Mapping):
                raise ValueError("Court test payload predictions must be a mapping.")
            predictions = raw_predictions
        if not isinstance(predictions, Mapping):
            raise ValueError("Court test payload predictions must be a mapping.")
        for kind, value in predictions.items():
            if not isinstance(value, Mapping):
                raise ValueError("Court head test payload must be a mapping.")
            for name, tensor in value.items():
                if isinstance(tensor, Tensor):
                    payload[f"{kind}_{name}"] = self._to_numpy(tensor)
        return payload

    def render_qualitative_samples(
        self,
        batches: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        artifact_dir: Path,
        tb_writer: Any | None,
        global_step: int,
        epoch: int,
    ) -> None:
        _ = (outputs, epoch)
        if self.query_variant:
            return
        model_io = cast(CourtModelIOAdapter, self.model_io)
        device = next(self.parameters()).device
        style = self.qualitative_style.build()
        for batch_index, cpu_batch in enumerate(batches):
            batch = cast(
                dict[str, Any],
                _move_to_device(cpu_batch, device=device),
            )
            with torch.no_grad():
                call = model_io.prepare_training_batch(batch)
                logits = cast(
                    CourtLogits,
                    self.model(*call.model_call.model_args),
                )
            model_io.validate_logits(logits, call.model_call)
            for kind in self.target_bundle.kinds:
                frames_rgb = self.qualitative_renderers[kind].render(
                    batch=batch,
                    logits=logits[kind].cpu(),
                    style=style,
                    clip_label=f"court_{kind}",
                )
                save_qualitative_clip(
                    frames_rgb=frames_rgb,
                    artifact_dir=artifact_dir,
                    name=f"court_{kind}_batch{batch_index:02d}",
                    tb_writer=tb_writer,
                    tag=(
                        f"qualitative/court_detection/{kind}/"
                        f"batch{batch_index:02d}"
                    ),
                    global_step=global_step,
                    fps=self.qualitative_fps,
                )


def _move_to_device(value: object, *, device: torch.device) -> object:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {
            key: _move_to_device(item, device=device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_to_device(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device=device) for item in value)
    return value


__all__ = ["CourtDetectionLightningModule"]
