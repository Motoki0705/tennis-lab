"""Checkpoint-authoritative, source-separated Court ablation evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from src.tasks.court_detection.configuration import (
    SyntheticCourtDatasetSubsetSourceConfig,
    SyntheticCourtDatasetVersionSourceConfig,
    SyntheticCourtSourceConfig,
    SyntheticCourtStoresSourceConfig,
    TennisCourtEvaluationSourceConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtSampleRecord,
    CourtSourceSplit,
    CourtTargetBundleSpec,
    CourtTargetKind,
)
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.dataset import CourtDetectionDataset
from src.tasks.court_detection.data.inputs.synthetic_court_dataset_subset import (
    SyntheticCourtDatasetSubsetInput,
    read_subset,
)
from src.tasks.court_detection.geometry.pose import (
    canonical_semantic_court_points_batched,
    decode_pose10d_strict,
    project_predicted_canonical_points,
)
from src.tasks.court_detection.inference.checkpoint import file_sha256
from src.tasks.court_detection.model_io.contracts import CourtPoseTrainingResult
from src.tasks.court_detection.models.dinov3_dpt import CourtModelOutput
from src.tasks.court_detection.scaling.provenance import data_condition
from src.tasks.court_detection.scaling.schedule import has_step_schedule
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
    _move_to_device,
)
from src.tasks.court_detection.training.metrics import (
    COURT_RESULT_METRIC_NAMES,
    CourtDetectionMetrics,
)
from src.tasks.court_detection.visualization.adapters.render_inputs import (
    batch_to_court_frame,
)

EVALUATION_SPLITS: tuple[tuple[str, CourtSourceSplit], ...] = (
    ("synthetic_court", "test"),
    ("tennis_court_detector", "val"),
)
DENSE_METRICS = (
    "kp_mean_distance_px",
    "kp_median_distance_px",
    "seg_miou",
    "line_dice",
    "semantic_line_miou",
)


def sample_identity(dataset: CourtDetectionDataset) -> dict[str, Any]:
    return _ordered_identity([record.sample_id for record in dataset.records])


def _ordered_identity(ids: list[str]) -> dict[str, Any]:
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation sample IDs must be unique within a source/split")
    encoded = json.dumps(ids, separators=(",", ":")).encode()
    return {
        "count": len(ids),
        "sample_ids": ids,
        "ordered_sample_ids_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _synthetic_evaluation_records(
    manifest: Path,
    *,
    source_manifest_sha256: str,
    pool_records: tuple[CourtSampleRecord, ...],
    training_records: tuple[CourtSampleRecord, ...],
) -> tuple[CourtSampleRecord, ...]:
    document, _ = read_subset(manifest, "val")
    if document["source_manifest_sha256"] != source_manifest_sha256:
        raise ValueError(
            "Synthetic evaluation must use the checkpoint's verified pool."
        )
    lookup = {record.sample_id: record for record in pool_records}
    training_ids = {record.sample_id for record in training_records}
    training_groups = {
        (str(record.payload["scene_id"]), str(record.payload["trajectory_group_id"]))
        for record in training_records
    }
    selected = []
    seen: set[str] = set()
    for row in document["samples"]:
        identity = f"{row['scene_id']}:{row['sample_id']}"
        if identity in seen or identity not in lookup:
            raise ValueError(
                f"Duplicate or unknown synthetic evaluation sample: {identity}"
            )
        if identity in training_ids:
            raise ValueError(
                f"Synthetic evaluation overlaps training sample: {identity}"
            )
        record = lookup[identity]
        group = (
            str(record.payload["scene_id"]),
            str(record.payload["trajectory_group_id"]),
        )
        if group in training_groups:
            raise ValueError(
                f"Synthetic evaluation overlaps training trajectory: {group}"
            )
        seen.add(identity)
        selected.append(replace(record, split="val"))
    return tuple(selected)


def _synthetic_evaluation_loader(
    dm: CourtDetectionDataModule, manifest: Path
) -> DataLoader[Any]:
    training_input = dm._train_pipelines["synthetic_court"].input_layer
    if not isinstance(training_input, SyntheticCourtDatasetSubsetInput):
        raise ValueError(
            "An evaluation manifest requires checkpoint dataset subset provenance."
        )
    records = _synthetic_evaluation_records(
        manifest,
        source_manifest_sha256=training_input.version_manifest_sha256,
        pool_records=training_input.source.records("train"),
        training_records=training_input.records("train"),
    )
    dataset = CourtDetectionDataset(
        records, pipeline=dm._eval_pipelines["synthetic_court"]
    )
    return dm._eval_loader(dataset, stage="synthetic_court/val", rng_stage="val")


class _SceneMetrics:
    """Partition the original forward's predictions without changing batch padding."""

    def __init__(
        self, records: tuple[CourtSampleRecord, ...], bundle: CourtTargetBundleSpec
    ) -> None:
        self.identities: dict[str, dict[str, Any]] = {}
        self.sample_scenes = {
            record.sample_id: str(record.payload["scene_id"]) for record in records
        }
        self.expected_ids = [record.sample_id for record in records]
        self.observed_ids: list[str] = []
        _ordered_identity(self.expected_ids)
        for scene in dict.fromkeys(self.sample_scenes.values()):
            self.identities[scene] = _ordered_identity(
                [
                    record.sample_id
                    for record in records
                    if self.sample_scenes[record.sample_id] == scene
                ]
            )
        self.trackers = {
            scene: {
                kind: CourtDetectionMetrics(kind, spec.output_channels)
                for kind, spec in bundle.targets.items()
            }
            for scene in self.identities
        }

    def update(
        self,
        ids: list[str],
        logits: Mapping[CourtTargetKind, torch.Tensor],
        targets: Mapping[CourtTargetKind, object],
        image_size: torch.Tensor,
    ) -> None:
        indices: dict[str, list[int]] = {}
        for index, identity in enumerate(ids):
            indices.setdefault(self.sample_scenes[identity], []).append(index)
        self.observed_ids.extend(ids)
        for scene, selected in indices.items():
            for kind, tracker in self.trackers[scene].items():
                target = targets[kind]
                sliced = (
                    {
                        name: value[selected]
                        for name, value in cast(
                            Mapping[str, torch.Tensor], target
                        ).items()
                    }
                    if isinstance(target, Mapping)
                    else cast(torch.Tensor, target)[selected]
                )
                tracker.update(
                    logits[kind][selected], sliced, image_size=image_size[selected]
                )

    def compute(self) -> dict[str, Any]:
        if self.observed_ids != self.expected_ids:
            raise ValueError(
                "Synthetic evaluation did not consume the manifest's exact ID order."
            )
        scenes = {}
        for scene, trackers in self.trackers.items():
            values = {
                f"{kind}_{name}": value
                for kind, tracker in trackers.items()
                for name, value in tracker.compute().items()
            }
            scenes[scene] = {
                **self.identities[scene],
                "metrics": select_metrics(values, has_pose=False),
            }
        return {
            "scene_order": list(scenes),
            "scenes": scenes,
            "scene_macro_mean": {
                name: sum(scene["metrics"][name] for scene in scenes.values())
                / len(scenes)
                for name in DENSE_METRICS
            },
        }


class _SceneEvaluationModule(CourtDetectionLightningModule):
    scene_metrics: _SceneMetrics | None = None

    def validation_step(self, batch: dict[str, object], batch_idx: int) -> None:
        if self.scene_metrics is None:
            super().validation_step(batch, batch_idx)
            return
        result = self._shared_step(batch, "val")
        logits = (
            result.output.dense_logits
            if isinstance(result, CourtPoseTrainingResult)
            else result.logits
        )
        self.scene_metrics.update(
            cast(list[str], batch["sample_id"]),
            logits,
            cast(Mapping[CourtTargetKind, object], batch["targets"]),
            cast(torch.Tensor, batch["image_size"]),
        )


def select_metrics(values: dict[str, float], *, has_pose: bool) -> dict[str, float]:
    names = COURT_RESULT_METRIC_NAMES if has_pose else DENSE_METRICS
    # Consistency is intentionally disabled in this experiment, so that optional
    # diagnostic is absent. Every required supervised accuracy metric must exist.
    required = set(names) - {"kp_pose_consistency_distance_px"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"Evaluation omitted required metrics: {sorted(missing)}")
    return {name: values[name] for name in names if name in values}


def parameter_counts(
    module: CourtDetectionLightningModule,
) -> dict[str, dict[str, int]]:
    parts = {
        "backbone": module.model.encoder,
        "projection": module.model.feature_projections,
        "transformer": module.model.transformer_encoder,
        "dpt": module.model.decoder,
        "pose_head": module.model.pose_head,
        "dense_heads": module.model.heads,
    }
    return {
        name: {
            "total": sum(parameter.numel() for parameter in part.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in part.parameters()
                if parameter.requires_grad
            ),
        }
        for name, part in parts.items()
    }


def render_pose_overlay(batch: dict[str, Any], output: CourtModelOutput) -> Image.Image:
    """Show GT (green) and front-facing predicted pose (red) in model pixels."""
    target = batch["pose_target"]
    if output.pose is None:
        raise ValueError("Pose overlay requires a pose prediction")
    pose = decode_pose10d_strict(output.pose.values)
    canonical = canonical_semantic_court_points_batched(
        target["semantic_to_physical"],
        dtype=pose.translation_m.dtype,
        device=pose.translation_m.device,
    )
    projection = project_predicted_canonical_points(
        pose, canonical, target["intrinsics"][:, :2, 2]
    )
    frame = batch_to_court_frame(batch)
    image = Image.fromarray(frame.rgb)
    draw = ImageDraw.Draw(image)
    kp = batch["targets"]["kp"]
    height, width = (int(value) for value in batch["image_size"][0].tolist())
    scale = torch.tensor([width - 1, height - 1], device=kp["points_xy"].device)
    truth = (kp["points_xy"][0, :, 0] * scale).detach().cpu().numpy()
    prediction = projection.points_xy[0].detach().cpu().numpy()
    visible = kp["point_visible"][0, :, 0].detach().cpu().numpy()
    front = (projection.depth_m[0] > 0).detach().cpu().numpy()
    for points, accepted, color in (
        (truth, visible, "lime"),
        (prediction, front, "red"),
    ):
        for index, ((x, y), valid) in enumerate(zip(points, accepted, strict=True)):
            if (
                valid
                and np.isfinite([x, y]).all()
                and 0 <= x < width
                and 0 <= y < height
            ):
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
                draw.text((x + 4, y), str(index), fill=color)
    draw.text(
        (5, 5),
        "GT: green / predicted pose: red",
        fill="white",
        stroke_fill="black",
        stroke_width=1,
    )
    return image


def _qualitative(
    module: CourtDetectionLightningModule,
    loader: DataLoader[Any],
    directory: Path,
    *,
    device: torch.device,
    has_pose: bool,
) -> list[dict[str, Any]]:
    dataset = cast(CourtDetectionDataset, loader.dataset)
    indices: list[int] = np.linspace(
        0, len(dataset) - 1, num=min(4, len(dataset)), dtype=int
    ).tolist()
    selected = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=loader.collate_fn,
    )
    module.to(device).eval()
    directory.mkdir()
    manifest = []
    for index, raw in enumerate(selected):
        batch = cast(dict[str, Any], _move_to_device(raw, device=device))
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            call = module.model_io.prepare_training_batch(batch)
            output = module.model(*call.model_call.model_args)
        if not isinstance(output, CourtModelOutput):
            raise ValueError("Ablation requires the shared dense + pose architecture")
        files = {}
        for kind in module.target_bundle.kinds:
            frames = module.qualitative_renderers[kind].render(
                batch=batch,
                logits=output.dense_logits[kind].detach().float().cpu(),
                style=module.qualitative_style.build(),
                clip_label=batch["sample_id"][0],
            )
            path = directory / f"{index:02d}-{kind}.png"
            Image.fromarray(frames[0]).save(path)
            files[kind] = path.name
        if has_pose:
            path = directory / f"{index:02d}-pose.png"
            render_pose_overlay(batch, output).save(path)
            files["pose"] = path.name
        manifest.append(
            {
                "sample_id": batch["sample_id"][0],
                "dataset_index": indices[index],
                "files": files,
            }
        )
    return manifest


def evaluate_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    device: str = "cuda",
    path_overrides: dict[str, str] | None = None,
    expected_step: int | None = None,
    real_split: str | None = None,
    series: str | None = None,
    include_synthetic_val: bool = False,
    synthetic_eval_manifest: Path | None = None,
) -> dict[str, Any]:
    """Score both complete splits with the checkpoint's configuration and targets."""
    if include_synthetic_val and synthetic_eval_manifest is not None:
        raise ValueError(
            "Choose either include_synthetic_val or synthetic_eval_manifest."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but no GPU is available")
    digest = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    hyper = payload["hyper_parameters"]
    cfg = (
        OmegaConf.create(OmegaConf.to_container(hyper["config"], resolve=True))
        if isinstance(hyper["config"], DictConfig)
        else OmegaConf.create(hyper["config"])
    )
    scaling = has_step_schedule(cfg)
    synthetic_validation = include_synthetic_val or synthetic_eval_manifest is not None
    if synthetic_validation and (not scaling or real_split != "val"):
        raise ValueError(
            "Synthetic validation requires step-based real validation evaluation."
        )
    if scaling:
        if (
            type(expected_step) is not int
            or expected_step <= 0
            or payload.get("global_step") != expected_step
        ):
            raise ValueError(
                "Scaling evaluation requires an exact positive --expected-step matching checkpoint global_step."
            )
        if real_split not in {"val", "test"} or not series:
            raise ValueError(
                "Scaling evaluation requires explicit --split val|test and --series."
            )
    elif expected_step is not None or real_split is not None or series is not None:
        raise ValueError(
            "Step, split and series overrides require a data scaling checkpoint."
        )
    for key, value in (path_overrides or {}).items():
        if key not in {
            "data_root",
            "checkpoint_root",
            "external_asset_root",
            "artifact_root",
        }:
            raise ValueError(f"Unsupported evaluation path override: {key}")
        cfg.paths[key] = value
    pl.seed_everything(int(cfg.run.seed), workers=True)
    dm = CourtDetectionDataModule(cfg)
    condition = data_condition(dm) if scaling else None
    if scaling and condition != hyper.get("data_condition"):
        raise ValueError(
            "Training data condition or fixed evaluation registry changed since checkpoint publication."
        )
    synthetic_loader = None
    synthetic_digest = None
    if synthetic_eval_manifest is not None:
        synthetic_digest = file_sha256(synthetic_eval_manifest)
        synthetic_loader = _synthetic_evaluation_loader(dm, synthetic_eval_manifest)
    module_type = (
        _SceneEvaluationModule
        if synthetic_eval_manifest is not None
        else CourtDetectionLightningModule
    )
    module = module_type(cfg, target_bundle_state=hyper["target_bundle_state"])
    if dm.target_bundle_spec != module.target_bundle:
        raise ValueError(
            "Checkpoint target schema disagrees with the current evaluation data"
        )
    module.load_state_dict(payload["state_dict"], strict=True)
    output.mkdir(parents=True, exist_ok=False)
    plain = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    result: dict[str, Any] = {
        "schema": (
            "court_data_scaling_evaluation_v2"
            if synthetic_eval_manifest is not None
            else "court_data_scaling_evaluation_v1"
        )
        if scaling
        else "court_ablation_evaluation_v1",
        "checkpoint_sha256": digest,
        "checkpoint": str(checkpoint.resolve()),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "config": plain,
        "target_bundle": hyper["target_bundle_state"],
        "parameters": parameter_counts(module),
        "precision": str(cfg.training.trainer.precision)
        if device == "cuda"
        else "32-true",
        "qualitative_batch_size": 1,
        "evaluation_compile": False,
        "splits": {},
    }
    if scaling:
        result["data_condition"] = condition
        result["series"] = series
    if synthetic_eval_manifest is not None:
        result["synthetic_evaluation_manifest"] = {
            "path": str(synthetic_eval_manifest.resolve()),
            "sha256": synthetic_digest,
        }
    manifests = {}
    for source_config in dm.mixed_config.sources.values():
        if isinstance(source_config, SyntheticCourtSourceConfig):
            for scene in source_config.scene_ids:
                manifests[f"synthetic_court/{scene}"] = file_sha256(
                    source_config.workspace_root / scene / "datasets/court/dataset.json"
                )
        elif isinstance(source_config, SyntheticCourtDatasetSubsetSourceConfig):
            manifests["synthetic_court/dataset_subset"] = file_sha256(
                source_config.manifest
            )
            manifests["synthetic_court/validation_subset"] = file_sha256(
                source_config.validation_manifest
            )
        elif isinstance(source_config, SyntheticCourtDatasetVersionSourceConfig):
            manifests["synthetic_court/dataset_version"] = file_sha256(
                source_config.manifest
            )
        elif isinstance(source_config, SyntheticCourtStoresSourceConfig):
            for index, root in enumerate(source_config.store_roots):
                manifests[f"synthetic_court/store-{index}"] = file_sha256(
                    root / "dataset.json"
                )
        else:
            if isinstance(source_config, TennisCourtEvaluationSourceConfig):
                manifests["evaluation_registry"] = file_sha256(source_config.registry)
            manifests["tennis_court_detector"] = file_sha256(
                source_config.root / "dataset.json"
            )
    result["data_manifest_sha256"] = manifests
    if synthetic_digest is not None:
        manifests["synthetic_court/evaluation_subset"] = synthetic_digest
    del payload
    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        precision=result["precision"],
        deterministic="warn",
    )
    splits: tuple[tuple[str, CourtSourceSplit], ...] = (
        (("tennis_court_detector", cast(CourtSourceSplit, real_split)),)
        if scaling
        else EVALUATION_SPLITS
    )
    if synthetic_validation:
        splits = (("synthetic_court", "val"), *splits)
    for source, split in splits:
        loader = (
            synthetic_loader
            if source == "synthetic_court" and synthetic_loader is not None
            else dm.source_eval_dataloader(source, split)
        )
        identity = sample_identity(cast(CourtDetectionDataset, loader.dataset))
        if isinstance(module, _SceneEvaluationModule):
            module.scene_metrics = (
                _SceneMetrics(
                    cast(CourtDetectionDataset, loader.dataset).records,
                    module.target_bundle,
                )
                if source == "synthetic_court"
                else None
            )
        # validate uses the existing whole-split accumulators without buffering
        # every dense prediction. The receipt retains the true source/split name.
        measured = trainer.validate(module, dataloaders=loader, verbose=False)[0]
        values = {
            name.removeprefix("val/"): float(value) for name, value in measured.items()
        }
        has_pose = source == "synthetic_court"
        key = f"{source}-{split}"
        result["splits"][key] = {
            **identity,
            "source": source,
            "split": split,
            "batch_size": dm.batch_size,
            "pose_supervised": has_pose,
            "metrics": select_metrics(values, has_pose=has_pose),
            "qualitative": _qualitative(
                module,
                loader,
                output / key,
                device=torch.device(device),
                has_pose=has_pose,
            ),
        }
        if (
            isinstance(module, _SceneEvaluationModule)
            and module.scene_metrics is not None
        ):
            result["splits"][key].update(module.scene_metrics.compute())
    if file_sha256(checkpoint) != digest:
        raise ValueError("Checkpoint changed during evaluation")
    if (
        synthetic_eval_manifest is not None
        and file_sha256(synthetic_eval_manifest) != synthetic_digest
    ):
        raise ValueError("Synthetic evaluation manifest changed during evaluation")
    (output / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return result
