"""PyTorch Lightning module for composable Court detection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_saving import save_qualitative_clip
from src.tasks.court_detection.configuration import (
    CourtTrainingConfig,
)
from src.tasks.court_detection.data.bundle_state import (
    deserialize_target_bundle,
    serialize_target_bundle,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
)
from src.tasks.court_detection.model_io.adapters import (
    CourtPoseModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtLogits,
    CourtModelOutput,
    CourtPoseTargetBatch,
    CourtPoseTrainingResult,
    CourtTrainingResult,
)
from src.tasks.court_detection.model_io.factory import build_court_detection_pair
from src.tasks.court_detection.training.metrics import (
    CourtDetectionMetrics,
    CourtPoseGeometryMetrics,
    CourtPoseMetrics,
    gradient_finite_status,
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
    ) -> None:
        super().__init__(config)
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
            if target_bundle_state is not None and deserialize_target_bundle(target_bundle_state) != resolved_bundle:
                raise ValueError(
                    "Court target bundle disagrees with its checkpoint snapshot."
                )
        bundle_snapshot = serialize_target_bundle(resolved_bundle)
        self.save_hyperparameters({"target_bundle_state": bundle_snapshot})
        self.target_bundle = resolved_bundle
        self.pose_variant = bool(getattr(getattr(runtime.loss, "pose", None), "enabled", False))
        self.qualitative_fps = runtime.qualitative_fps
        self.qualitative_style = runtime.render_style

        model_pair = build_court_detection_pair(
            self.config,
            target_bundle=resolved_bundle,
        )
        self.model = model_pair.model
        self.model_io = model_pair.adapter
        self.consistency_instrumented = (
            isinstance(self.model_io, CourtPoseModelIOAdapter)
            and self.model_io.consistency_instrumented
        )
        self.qualitative_renderers: dict[
            CourtTargetKind, CourtQualitativeRenderer
        ] = (
            {}
            if self.pose_variant
            else {
                kind: build_court_qualitative_renderer(
                    self.model_io,
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
                    singleton_kp=self.pose_variant and kind == "kp",
                )
                for kind, spec in resolved_bundle.targets.items()
            }
            for stage in ("train", "val", "test")
        }
        self._pose_metrics = {
            stage: CourtPoseMetrics() for stage in ("train", "val", "test")
        }
        self._pose_geometry_metrics: dict[str, CourtPoseGeometryMetrics] = {}
        if isinstance(self.model_io, CourtPoseModelIOAdapter):
            self._pose_geometry_metrics = {
                stage: CourtPoseGeometryMetrics(
                    min_depth_m=self.model_io.pose_loss_config.consistency.min_depth_m
                )
                for stage in ("train", "val", "test")
            }
        self._train_batch_started_at: float | None = None
        self._matrix_manifest_path = self._matrix_manifest_path_from_environment()
        self._matrix_loss_sums: dict[str, float] = {}
        self._matrix_loss_counts: dict[str, int] = {}
        self._matrix_gradient_finite: dict[str, bool] = {}
        self._matrix_step_time_ms = 0.0
        self._matrix_step_count = 0
        self._matrix_peak_memory_bytes = 0
        self._matrix_active_gradient_branches = self._active_gradient_branches()

    def forward(
        self, *model_args: Tensor
    ) -> Mapping[CourtTargetKind, Tensor] | CourtModelOutput:
        return cast(
            "Mapping[CourtTargetKind, Tensor] | CourtModelOutput",
            self.model(*model_args),
        )

    def _shared_step(
        self,
        batch: Mapping[str, object],
        stage: str,
    ) -> CourtTrainingResult | CourtPoseTrainingResult:
        if isinstance(self.model_io, CourtPoseModelIOAdapter):
            pose_call = self.model_io.prepare_training_batch(batch)
            output = cast(CourtModelOutput, self.model(*pose_call.model_call.model_args))
            progress_fraction = (
                self._progress_fraction(stage)
                if self.model_io.consistency_instrumented
                else None
            )
            pose_result = self.model_io.training_result(
                output,
                pose_call,
                progress_fraction=progress_fraction,
            )
            self._log_training_result(stage, pose_result)
            if stage == "train":
                self._record_matrix_loss_result(pose_result)
            image_size = batch.get("image_size")
            if not isinstance(image_size, Tensor):
                raise ValueError("Court batch image_size must be a Tensor.")
            for kind in self.target_bundle.kinds:
                self._stage_metrics[stage][kind].update(
                    pose_result.output.dense_logits[kind],
                    pose_call.targets[kind],
                    image_size=image_size,
                )
            self._pose_metrics[stage].update(
                pose_result.decoded_pose,
                cast(CourtPoseTargetBatch, pose_call.targets["pose"]),
            )
            geometry_tracker = self._pose_geometry_metrics.get(stage)
            if geometry_tracker is not None:
                kp_target = pose_call.targets.get("kp")
                if not isinstance(kp_target, Mapping):
                    raise ValueError(
                        "Pose metrics require the canonical singleton KP target."
                    )
                target_pose = cast(CourtPoseTargetBatch, pose_call.targets["pose"])
                image_size = cast(Tensor, pose_call.targets["image_size"])
                ground_truth_points = cast(Tensor, kp_target["points_xy"]).squeeze(2)
                point_visible = cast(Tensor, kp_target["point_visible"]).squeeze(2)
                if pose_result.consistency is not None:
                    geometry_tracker.update(
                        pose_result.consistency,
                        ground_truth_points_normalized=ground_truth_points,
                        point_visible=point_visible,
                        image_size=image_size,
                    )
                else:
                    geometry_tracker.update_pose_prediction(
                        pose_result.decoded_pose,
                        target_pose,
                        ground_truth_points_normalized=ground_truth_points,
                        point_visible=point_visible,
                        image_size=image_size,
                    )
            return pose_result
        legacy_call = self.model_io.prepare_training_batch(batch)
        logits = cast(CourtLogits, self.model(*legacy_call.model_call.model_args))
        result = cast(
            CourtTrainingResult,
            self.model_io.training_result(logits, legacy_call),
        )
        self.log(
            f"{stage}/loss",
            result.loss,
            prog_bar=True,
            sync_dist=True,
        )
        for kind, raw_loss in result.raw_losses.items():
            self.log(
                f"{stage}/loss_{kind}_raw",
                raw_loss,
                prog_bar=False,
                sync_dist=True,
            )
        for kind, loss in result.losses.items():
            self.log(
                f"{stage}/loss_{kind}",
                loss,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                f"{stage}/loss_{kind}_weighted",
                result.weighted_losses[kind],
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                f"{stage}/{kind}_configured_weight",
                result.configured_weights[kind],
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                f"{stage}/{kind}_effective_weight",
                result.effective_weights[kind],
                prog_bar=False,
                sync_dist=True,
            )
        if stage == "train":
            self._record_matrix_loss_result(result)
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
        result: CourtPoseTrainingResult,
    ) -> None:
        self.log(f"{stage}/loss", result.loss, prog_bar=True, sync_dist=True)
        self.log(
            f"{stage}/loss_direct_dense_raw",
            result.raw_dense_loss,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            f"{stage}/loss_direct_dense",
            result.direct_dense_loss,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            f"{stage}/loss_direct_pose",
            result.direct_pose_loss,
            prog_bar=False,
            sync_dist=True,
        )
        for kind, raw_loss in result.raw_dense_losses.items():
            self.log(
                f"{stage}/loss_{kind}_raw",
                raw_loss,
                prog_bar=False,
                sync_dist=True,
            )
        for kind, loss in result.dense_losses.items():
            self.log(
                f"{stage}/loss_{kind}",
                loss,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                f"{stage}/loss_{kind}_weighted",
                result.weighted_dense_losses[kind],
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                f"{stage}/{kind}_configured_weight",
                result.dense_configured_weights[kind],
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                f"{stage}/{kind}_effective_weight",
                result.dense_effective_weights[kind],
                prog_bar=False,
                sync_dist=True,
            )
        for pose_name, loss in result.pose_losses.items():
            self.log(
                f"{stage}/loss_{pose_name}",
                loss,
                prog_bar=False,
                sync_dist=True,
            )
        for pose_name, weighted_loss in result.weighted_pose_losses.items():
            self.log(
                f"{stage}/loss_{pose_name}_weighted",
                weighted_loss,
                prog_bar=False,
                sync_dist=True,
            )
        for pose_name, configured_weight in result.pose_configured_weights.items():
            self.log(
                f"{stage}/{pose_name}_configured_weight",
                configured_weight,
                prog_bar=False,
                sync_dist=True,
            )
        for pose_name, effective_weight in result.pose_effective_weights.items():
            self.log(
                f"{stage}/{pose_name}_effective_weight",
                effective_weight,
                prog_bar=False,
                sync_dist=True,
            )
        consistency = result.consistency
        if consistency is not None:
            values = {
                "loss_kp_pose_coordinate": consistency.coordinate_loss,
                "loss_kp_pose_cheirality": consistency.cheirality_loss,
                "loss_kp_pose_auxiliary_unweighted": consistency.auxiliary_loss,
                "loss_kp_pose_auxiliary_weighted": (
                    consistency.weighted_auxiliary_loss
                ),
                "kp_pose_configured_weight": consistency.configured_weight,
                "kp_pose_effective_weight": consistency.effective_weight,
                "kp_pose_visible_point_count": consistency.visible_point_count,
                "kp_pose_consistency_distance_px": consistency.mean_distance_px,
                "kp_pose_invalid_depth_rate": consistency.invalid_depth_rate,
            }
            for name, value in values.items():
                self.log(
                    f"{stage}/{name}",
                    value,
                    prog_bar=False,
                    sync_dist=True,
                )

    @staticmethod
    def _matrix_manifest_path_from_environment() -> Path | None:
        raw_path = os.environ.get("TENNIS_COURT_MATRIX_MANIFEST_PATH")
        if raw_path is None:
            return None
        if not raw_path:
            raise ValueError(
                "TENNIS_COURT_MATRIX_MANIFEST_PATH must be non-empty when set."
            )
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError(
                "TENNIS_COURT_MATRIX_MANIFEST_PATH must be an absolute path."
            )
        return path

    def _matrix_evidence_enabled(self) -> bool:
        return getattr(self, "_matrix_manifest_path", None) is not None

    def _active_gradient_branches(self) -> frozenset[str]:
        dense_config = self.model_io.loss_config
        active: set[str] = {
            kind
            for kind in self.target_bundle.kinds
            if float(dense_config.dense_weights.get(kind, 1.0)) > 0.0
        }
        if (
            isinstance(self.model_io, CourtPoseModelIOAdapter)
            and self.model_io.pose_loss_config.pose.enabled
        ):
            active.add("pose")
        return frozenset(active)

    def _record_matrix_loss_result(
        self,
        result: CourtTrainingResult | CourtPoseTrainingResult,
    ) -> None:
        if not self._matrix_evidence_enabled():
            return
        terms: dict[str, Tensor] = {"weighted_total": result.loss}
        if isinstance(result, CourtPoseTrainingResult):
            for kind, raw_loss in result.raw_dense_losses.items():
                if float(result.dense_effective_weights[kind]) <= 0.0:
                    continue
                terms[f"{kind}_direct"] = raw_loss
                terms[f"{kind}_configured_weight"] = (
                    result.dense_configured_weights[kind]
                )
                terms[f"{kind}_effective_weight"] = (
                    result.dense_effective_weights[kind]
                )
                terms[f"{kind}_weighted"] = result.weighted_dense_losses[kind]
            for name, raw_loss in result.pose_losses.items():
                terms[f"{name}_direct"] = raw_loss
                terms[f"{name}_configured_weight"] = (
                    result.pose_configured_weights[name]
                )
                terms[f"{name}_effective_weight"] = result.pose_effective_weights[
                    name
                ]
                terms[f"{name}_weighted"] = result.weighted_pose_losses[name]
            consistency = result.consistency
            if consistency is not None:
                terms.update(
                    {
                        "consistency_coordinate": consistency.coordinate_loss,
                        "consistency_cheirality": consistency.cheirality_loss,
                        "consistency_auxiliary_unweighted": (
                            consistency.auxiliary_loss
                        ),
                        "consistency_configured_weight": (
                            consistency.configured_weight
                        ),
                        "consistency_effective_weight": (
                            consistency.effective_weight
                        ),
                        "consistency_auxiliary_weighted": (
                            consistency.weighted_auxiliary_loss
                        ),
                    }
                )
        else:
            for kind, raw_loss in result.raw_losses.items():
                if float(result.effective_weights[kind]) <= 0.0:
                    continue
                terms[f"{kind}_direct"] = raw_loss
                terms[f"{kind}_configured_weight"] = result.configured_weights[
                    kind
                ]
                terms[f"{kind}_effective_weight"] = result.effective_weights[kind]
                terms[f"{kind}_weighted"] = result.weighted_losses[kind]
        for term_name, tensor in terms.items():
            value = float(tensor.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Court matrix loss term {term_name!r} became non-finite."
                )
            self._matrix_loss_sums[term_name] = (
                self._matrix_loss_sums.get(term_name, 0.0) + value
            )
            self._matrix_loss_counts[term_name] = (
                self._matrix_loss_counts.get(term_name, 0) + 1
            )

    def _progress_fraction(self, stage: str) -> float:
        if stage != "train":
            return 1.0
        total_steps = self._estimate_total_steps()
        completed_step = int(self.global_step) + 1
        return float(min(max(completed_step / total_steps, 0.0), 1.0))

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
        if self.pose_variant:
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
            geometry_tracker = self._pose_geometry_metrics.get(stage)
            if geometry_tracker is not None:
                for name, value in geometry_tracker.compute().items():
                    flattened[name] = value
                    self.log(
                        f"{stage}/{name}",
                        value,
                        prog_bar=False,
                        sync_dist=False,
                    )
                geometry_tracker.reset()
        return flattened

    def on_train_batch_start(self, batch: object, batch_idx: int) -> None:
        super().on_train_batch_start(batch, batch_idx)
        if not (
            self.consistency_instrumented or self._matrix_evidence_enabled()
        ):
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._train_batch_started_at = time.perf_counter()

    def on_train_batch_end(
        self,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        _ = (outputs, batch, batch_idx)
        if (
            not (
                self.consistency_instrumented or self._matrix_evidence_enabled()
            )
            or self._train_batch_started_at is None
        ):
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - self._train_batch_started_at) * 1000.0
        self.log(
            "train/train_step_time_ms",
            elapsed_ms,
            prog_bar=False,
            sync_dist=False,
        )
        if self.device.type == "cuda":
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(self.device))
            self.log(
                "train/cuda_peak_memory_bytes",
                float(peak_memory_bytes),
                prog_bar=False,
                sync_dist=False,
            )
        else:
            peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            peak_memory_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
        if self._matrix_evidence_enabled():
            self._matrix_step_time_ms += elapsed_ms
            self._matrix_step_count += 1
            self._matrix_peak_memory_bytes = max(
                self._matrix_peak_memory_bytes,
                peak_memory_bytes,
            )
        self._train_batch_started_at = None

    def on_after_backward(self) -> None:
        if not (
            self.consistency_instrumented or self._matrix_evidence_enabled()
        ):
            return
        model = self.model
        branch_parameters = {
            str(kind): tuple(head.parameters())
            for kind, head in model.heads.items()
        }
        if hasattr(model, "pose_head"):
            branch_parameters["pose"] = tuple(model.pose_head.parameters())
        for branch, parameters in branch_parameters.items():
            status = gradient_finite_status(parameters)
            self.log(
                f"train/{branch}_gradient_finite",
                status,
                prog_bar=False,
                sync_dist=False,
            )
            if (
                self._matrix_evidence_enabled()
                and branch in self._matrix_active_gradient_branches
            ):
                previous = self._matrix_gradient_finite.get(branch, True)
                self._matrix_gradient_finite[branch] = previous and status == 1.0

    @staticmethod
    def _canonical_json_sha256(value: object) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _matrix_repro_dir(self) -> Path:
        raw_repro_dir = os.environ.get("TENNIS_REPRO_DIR")
        if raw_repro_dir is None or not raw_repro_dir:
            raise RuntimeError(
                "Court matrix evidence requires a non-empty TENNIS_REPRO_DIR."
            )
        repro_dir = Path(raw_repro_dir)
        if not repro_dir.is_absolute():
            raise RuntimeError(
                "Court matrix evidence requires an absolute TENNIS_REPRO_DIR."
            )
        return repro_dir

    def _matrix_evidence_identity(self) -> tuple[str, str, str, str]:
        manifest_path = self._matrix_manifest_path
        if manifest_path is None:
            raise RuntimeError("Court matrix evidence is not enabled for this run.")
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Court matrix manifest is missing or invalid: {manifest_path}"
            ) from error
        if not isinstance(raw_manifest, dict) or any(
            not isinstance(key, str) for key in raw_manifest
        ):
            raise RuntimeError("Court matrix manifest must be a JSON object.")
        manifest = cast(dict[str, object], raw_manifest)
        manifest_sha256 = manifest.get("manifest_sha256")
        schema = manifest.get("run_evidence_schema")
        phase = manifest.get("phase")
        entries = manifest.get("entries")
        if (
            not isinstance(manifest_sha256, str)
            or not manifest_sha256
            or not isinstance(schema, str)
            or not schema
            or not isinstance(phase, str)
            or not phase
            or not isinstance(entries, list)
        ):
            raise RuntimeError("Court matrix manifest evidence identity is incomplete.")
        manifest_body = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
        if self._canonical_json_sha256(manifest_body) != manifest_sha256:
            raise RuntimeError("Court matrix manifest SHA-256 is invalid.")

        repro_dir = self._matrix_repro_dir()
        run_path = repro_dir / "run.json"
        try:
            raw_run = json.loads(run_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Court matrix queue metadata is missing or invalid: {run_path}"
            ) from error
        if not isinstance(raw_run, dict):
            raise RuntimeError("Court matrix queue metadata must be a JSON object.")
        run = cast(dict[str, object], raw_run)
        command = run.get("command")
        matches = [
            cast(dict[str, object], entry)
            for entry in entries
            if isinstance(entry, dict) and entry.get("command") == command
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "Court matrix queue command must match exactly one manifest entry."
            )
        entry = matches[0]
        entry_id = entry.get("entry_id")
        queue_name = entry.get("queue_name")
        if (
            not isinstance(entry_id, str)
            or not entry_id
            or not isinstance(queue_name, str)
            or run.get("name") != queue_name
            or str(run.get("issue")) != str(manifest.get("issue"))
        ):
            raise RuntimeError(
                "Court matrix queue metadata is not bound to the manifest entry."
            )
        return schema, phase, manifest_sha256, entry_id

    def _write_matrix_evidence(self) -> Path | None:
        if not self._matrix_evidence_enabled():
            return None
        trainer = self._safe_trainer()
        if trainer is not None and not bool(trainer.is_global_zero):
            return None
        if not self._matrix_loss_sums or set(self._matrix_loss_sums) != set(
            self._matrix_loss_counts
        ):
            raise RuntimeError("Court matrix loss evidence is missing or incomplete.")
        loss_terms = {
            name: total / self._matrix_loss_counts[name]
            for name, total in sorted(self._matrix_loss_sums.items())
            if self._matrix_loss_counts[name] > 0
        }
        if set(loss_terms) != set(self._matrix_loss_sums):
            raise RuntimeError("Court matrix loss evidence has an empty term.")
        active_branches = self._matrix_active_gradient_branches
        if set(self._matrix_gradient_finite) != set(active_branches) or not all(
            self._matrix_gradient_finite.values()
        ):
            raise RuntimeError(
                "Court matrix active-branch gradients are missing or non-finite."
            )
        if self._matrix_step_count <= 0 or self._matrix_peak_memory_bytes <= 0:
            raise RuntimeError("Court matrix timing or memory evidence is missing.")
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        if parameter_count <= 0:
            raise RuntimeError("Court matrix model parameter count must be positive.")
        schema, phase, manifest_sha256, entry_id = self._matrix_evidence_identity()
        evidence = {
            "schema": schema,
            "phase": phase,
            "manifest_sha256": manifest_sha256,
            "entry_id": entry_id,
            "complete": True,
            "loss_terms": loss_terms,
            "diagnostics": {
                "gradient_finite": dict(sorted(self._matrix_gradient_finite.items())),
                "parameter_count": parameter_count,
                "train_step_time_ms": (
                    self._matrix_step_time_ms / self._matrix_step_count
                ),
                "peak_memory_bytes": self._matrix_peak_memory_bytes,
            },
        }
        payload = json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        repro_dir = self._matrix_repro_dir()
        evidence_path = repro_dir / "court_matrix_evidence.json"
        with tempfile.NamedTemporaryFile(
            dir=repro_dir,
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, evidence_path)
        return evidence_path

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
            if isinstance(result, CourtPoseTrainingResult)
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
        if self._matrix_evidence_enabled():
            if saved is None:
                raise RuntimeError(
                    "Court matrix run completed testing without saved predictions."
                )
            evidence_path = self._write_matrix_evidence()
            if evidence_path is not None:
                print(f"[test] saved Court matrix evidence -> {evidence_path}")

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
        if isinstance(self.model_io, CourtPoseModelIOAdapter):
            if not isinstance(raw_output, CourtModelOutput):
                raise ValueError("Court pose test result requires typed raw output.")
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
                raw_output,
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
        if self.pose_variant:
            return
        model_io = self.model_io
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
