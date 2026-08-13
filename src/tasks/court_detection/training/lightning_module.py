"""PyTorch Lightning module for composable Court detection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_saving import save_qualitative_clip
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.bundle_state import (
    deserialize_target_bundle,
    serialize_target_bundle,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
)
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtLogits,
    CourtTrainingResult,
)
from src.tasks.court_detection.model_io.factory import build_court_detection_pair
from src.tasks.court_detection.training.metrics import CourtDetectionMetrics
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
    ) -> None:
        runtime = CourtTrainingConfig.from_config(config)
        if target_bundle is None:
            if target_bundle_state is None:
                raise ValueError(
                    "Court Lightning construction requires a target bundle "
                    "or its checkpoint snapshot."
                )
            resolved_bundle = deserialize_target_bundle(target_bundle_state)
        else:
            resolved_bundle = target_bundle
            if (
                target_bundle_state is not None
                and deserialize_target_bundle(target_bundle_state) != resolved_bundle
            ):
                raise ValueError(
                    "Court target bundle disagrees with its checkpoint snapshot."
                )
        bundle_snapshot = serialize_target_bundle(resolved_bundle)
        super().__init__(
            config,
            shared_runtime=runtime.shared,
            checkpoint_hyperparameters={
                "config": config,
                "target_bundle_state": bundle_snapshot,
            },
        )
        self.target_bundle = resolved_bundle
        self.qualitative_fps = runtime.qualitative_fps
        self.qualitative_style = runtime.render_style

        model_pair = build_court_detection_pair(
            self.config,
            target_bundle=resolved_bundle,
        )
        self.model = model_pair.model
        self.model_io = cast(CourtModelIOAdapter, model_pair.adapter)
        self.qualitative_renderers: dict[
            CourtTargetKind, CourtQualitativeRenderer
        ] = {
            kind: build_court_qualitative_renderer(
                self.model_io,
                kind=kind,
            )
            for kind in resolved_bundle.kinds
        }
        self._stage_metrics: dict[
            str, dict[CourtTargetKind, CourtDetectionMetrics]
        ] = {
            stage: {
                kind: CourtDetectionMetrics(kind, spec.output_channels)
                for kind, spec in resolved_bundle.targets.items()
            }
            for stage in ("train", "val", "test")
        }

    def forward(self, *model_args: Tensor) -> dict[CourtTargetKind, Tensor]:
        return cast(dict[CourtTargetKind, Tensor], self.model(*model_args))

    def _shared_step(
        self,
        batch: Mapping[str, object],
        stage: str,
    ) -> CourtTrainingResult:
        call = self.model_io.prepare_training_batch(batch)
        logits = cast(CourtLogits, self.model(*call.model_call.model_args))
        result = self.model_io.training_result(logits, call)
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
                call.targets[kind],
                image_size=image_size,
            )
        return result

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
        self.collect_test_predictions(
            batch,
            {"logits": result.logits},
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
    ) -> dict[str, object]:
        logits = result.get("logits")
        if not isinstance(logits, Mapping):
            raise ValueError("Court test result requires a logits mapping.")
        decoded = self.model_io.test_payload(batch, cast(CourtLogits, logits))
        payload: dict[str, object] = {}
        sample_ids = decoded["sample_id"]
        if isinstance(sample_ids, (list, tuple)):
            payload["sample_id"] = sample_ids
        image_size = decoded["image_size"]
        if isinstance(image_size, Tensor):
            payload["image_size"] = image_size
        predictions = decoded["predictions"]
        if not isinstance(predictions, Mapping):
            raise ValueError("Court test payload predictions must be a mapping.")
        for kind, value in predictions.items():
            if not isinstance(value, Mapping):
                raise ValueError("Court head test payload must be a mapping.")
            for name, tensor in value.items():
                if isinstance(tensor, Tensor):
                    payload[f"{kind}_{name}"] = tensor

        targets = batch.get("targets")
        if not isinstance(targets, Mapping):
            raise ValueError("Court test batch targets must be a mapping.")
        for kind in self.target_bundle.kinds:
            target = targets.get(kind)
            if kind == "kp":
                if not isinstance(target, Mapping):
                    raise ValueError("Court KP test target must be a mapping.")
                for name in ("points_xy", "point_visible"):
                    tensor = target.get(name)
                    if not isinstance(tensor, Tensor):
                        raise ValueError(
                            f"Court KP test target {name!r} must be a Tensor."
                        )
                    payload[f"kp_target_{name}"] = tensor
            elif isinstance(target, Tensor):
                payload[f"{kind}_target"] = target
            else:
                raise ValueError(f"Court {kind} test target must be a Tensor.")
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
        device = next(self.parameters()).device
        style = self.qualitative_style.build()
        for batch_index, cpu_batch in enumerate(batches):
            batch = cast(
                dict[str, Any],
                _move_to_device(cpu_batch, device=device),
            )
            with torch.no_grad():
                call = self.model_io.prepare_training_batch(batch)
                logits = cast(
                    CourtLogits,
                    self.model(*call.model_call.model_args),
                )
            self.model_io.validate_logits(logits, call.model_call)
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
