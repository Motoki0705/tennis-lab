"""Checkpoint-authoritative, source-separated Court ablation evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from src.tasks.court_detection.configuration import SyntheticCourtSourceConfig
from src.tasks.court_detection.data.contracts import CourtSourceSplit
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.dataset import CourtDetectionDataset
from src.tasks.court_detection.geometry.pose import (
    canonical_semantic_court_points_batched,
    decode_pose10d_strict,
    project_predicted_canonical_points,
)
from src.tasks.court_detection.inference.checkpoint import file_sha256
from src.tasks.court_detection.models.dinov3_dpt import CourtModelOutput
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
    _move_to_device,
)
from src.tasks.court_detection.training.metrics import COURT_RESULT_METRIC_NAMES
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
    ids = [record.sample_id for record in dataset.records]
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation sample IDs must be unique within a source/split")
    encoded = json.dumps(ids, separators=(",", ":")).encode()
    return {
        "count": len(ids),
        "sample_ids": ids,
        "ordered_sample_ids_sha256": hashlib.sha256(encoded).hexdigest(),
    }


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
    indices = np.linspace(
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
) -> dict[str, Any]:
    """Score both complete splits with the checkpoint's configuration and targets."""
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
    for key, value in (path_overrides or {}).items():
        if key not in {"data_root", "checkpoint_root", "external_asset_root"}:
            raise ValueError(f"Unsupported evaluation path override: {key}")
        cfg.paths[key] = value
    pl.seed_everything(int(cfg.run.seed), workers=True)
    dm = CourtDetectionDataModule(cfg)
    module = CourtDetectionLightningModule(
        cfg, target_bundle_state=hyper["target_bundle_state"]
    )
    if dm.target_bundle_spec != module.target_bundle:
        raise ValueError(
            "Checkpoint target schema disagrees with the current evaluation data"
        )
    module.load_state_dict(payload["state_dict"], strict=True)
    output.mkdir(parents=True, exist_ok=False)
    plain = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    result: dict[str, Any] = {
        "schema": "court_ablation_evaluation_v1",
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
    manifests = {}
    for source in dm.mixed_config.sources.values():
        if isinstance(source, SyntheticCourtSourceConfig):
            for scene in source.scene_ids:
                manifests[f"synthetic_court/{scene}"] = file_sha256(
                    source.workspace_root / scene / "datasets/court/dataset.json"
                )
        else:
            manifests["tennis_court_detector"] = file_sha256(
                source.root / "dataset.json"
            )
    result["data_manifest_sha256"] = manifests
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
    for source, split in EVALUATION_SPLITS:
        loader = dm.source_eval_dataloader(source, split)
        identity = sample_identity(cast(CourtDetectionDataset, loader.dataset))
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
    if file_sha256(checkpoint) != digest:
        raise ValueError("Checkpoint changed during evaluation")
    (output / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return result
