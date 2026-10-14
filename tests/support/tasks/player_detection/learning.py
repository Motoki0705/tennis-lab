"""Synthetic publication + CPU Lightning with tiny DINO substitutes; no real GPU."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
from omegaconf import DictConfig

from src.submodules.models.dino.architecture import dino_4scale_swin_args
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    file_reference,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Document,
    ModelRef,
    ProductOutcome,
    PublicationReceipt,
    content_digest,
)
from src.tasks.base.annotation.execution.contracts import (
    QueueConfiguration,
    StageLaunch,
)
from src.tasks.base.annotation.learning import Learning
from src.tasks.base.annotation.learning.contracts import (
    CampaignCompletion,
    FixedEvaluation,
    LearningIntent,
    LearningRuntime,
    StageCost,
    TaskProducts,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, file_digest, save_plan
from src.tasks.player_detection.annotation.contracts import DinoCheckpointSpec
from src.tasks.player_detection.configuration import EvaluateConfig
from src.tasks.player_detection.data.detection_dataset import (
    PlayerDetectionDataset,
    select_detection_frames,
)
from src.tasks.player_detection.data.snapshots.contracts import (
    SelectionPolicy,
    SourceBinding,
)
from src.tasks.player_detection.data.snapshots.partitions import freeze_partition
from src.tasks.player_detection.data.snapshots.store import (
    freeze_existing_store,
    load_snapshot,
)
from src.tasks.player_detection.data.store import (
    INDEX_FILE,
    METADATA_FILE,
    SHARDS_DIR,
    PlayerFrameStore,
    shard_name,
)
from src.tasks.player_detection.evaluation.metrics import PlayerDetectionMetrics
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from src.tasks.player_detection.model_export import EXPORT_FORMAT
from src.tasks.player_detection.models.dino_detector import decode_player_detections
from src.tasks.player_detection.training.annotation.adapter import PlayerLearningAdapter
from src.tasks.player_detection.training.annotation.contracts import (
    Augmentation,
    InputSize,
    OperatingPoint,
    PlayerLearningRecipe,
    PlayerSelectionPolicy,
)
from src.tasks.player_detection.training.annotation.runner import AnnotationPlayerRunner
from src.tasks.player_detection.training.annotation.stages import execute
from tests.support.tasks.ball_detection.learning_pipeline import CPUExecutor
from tests.support.tasks.player_detection.chat_root import (
    SyntheticRoot,
    player,
    publish_player_annotation,
    write_clip,
)

ROOT = Path(__file__).resolve().parents[4]


def save(layout: AnnotationLayout, value: Document) -> ArtifactRef:
    result: ArtifactRef = write_document(
        layout, layout.path("models", "fixture", value.sha256 + ".json"), value
    )
    return result


class TinyDino(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.backbone = torch.nn.Sequential(torch.nn.Identity())

    def forward(
        self, images: list[torch.Tensor], targets: Any = None
    ) -> dict[str, torch.Tensor]:
        del targets
        logits = torch.full((len(images), 2, 91), -20.0)
        logits[:, 0, 1] = self.weight
        boxes = torch.tensor(
            [[20 / 96, 25 / 64, 20 / 96, 30 / 64], [0.9, 0.9, 0.1, 0.1]]
        )
        return {
            "pred_logits": logits,
            "pred_boxes": boxes[None].expand(len(images), -1, -1),
            "signal": self.weight,
        }


class TinyCriterion(torch.nn.Module):
    weight_dict = {"loss_ce": 1.0}

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: Any
    ) -> dict[str, torch.Tensor]:
        del targets
        loss = (outputs["signal"] - 1.0) ** 2
        return {
            name: loss if name == "loss_ce" else loss * 0
            for name in (
                "loss_ce",
                "loss_bbox",
                "loss_giou",
                "loss_ce_dn",
                "loss_bbox_dn",
                "loss_giou_dn",
            )
        }


def use_cpu_dino(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.tasks.player_detection.models.dino_detector.build_dino",
        lambda *_args: (TinyDino(), TinyCriterion()),
    )
    monkeypatch.setattr(
        "src.tasks.player_detection.model_export.build_dino",
        lambda *_args: (TinyDino(), TinyCriterion()),
    )
    original = AnnotationPlayerRunner.build_lightning_module

    def build(
        self: AnnotationPlayerRunner,
        config: DictConfig,
        data: Any,
        *,
        steps_per_epoch: int | None = None,
    ) -> Any:
        # Only construction's CUDA requirement is substituted. Trainer, optimizer,
        # scheduler, strict parent load, epoch checkpoints, resume and export run.
        with patch("torch.cuda.is_available", return_value=True):
            return original(self, config, data, steps_per_epoch=steps_per_epoch)

    monkeypatch.setattr(AnnotationPlayerRunner, "build_lightning_module", build)
    monkeypatch.setattr(
        AnnotationPlayerRunner, "select_devices", lambda *_args: ("cpu", 1)
    )
    monkeypatch.setattr(
        "src.tasks.player_detection.training.annotation.evaluation.evaluate_checkpoints",
        tiny_evaluation,
    )


def tiny_evaluation(config: EvaluateConfig) -> dict[str, dict[str, float]]:
    """Synthetic execution evidence; native readers/decoding/metrics remain real."""
    store = PlayerFrameStore(config.dataset_dir)
    selection = select_detection_frames(
        store, "val", config.selection, frame_stride=config.frame_stride
    )
    dataset = PlayerDetectionDataset(
        store, selection, input_size=config.input_size, augmentation=None
    )
    clips = sorted({store.clip_of(int(row)).index for row in selection.frames})
    files = [
        INDEX_FILE,
        METADATA_FILE,
        *(f"{SHARDS_DIR}/{shard_name(index)}" for index in clips),
    ]
    config.output_dir.mkdir(parents=True)
    provenance = {
        "schema": "player_detection_evaluation_v1",
        "checkpoints": {
            name: {"path": str(path), "sha256": file_digest(path)}
            for name, path in config.checkpoints.items()
        },
        "dataset": {
            "path": str(config.dataset_dir),
            "sha256": {name: file_digest(config.dataset_dir / name) for name in files},
        },
        "input_size": asdict(config.input_size),
        "selection": asdict(config.selection),
        "evaluation": asdict(config.evaluation),
        "split": "val",
        "frame_stride": config.frame_stride,
        "device": config.device,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    (config.output_dir / "provenance.json").write_text(json.dumps(provenance))
    summaries = {}
    for name, path in config.checkpoints.items():
        model = TinyDino().eval()
        model.load_state_dict(
            torch.load(path, weights_only=False, map_location="cpu")["model"],
            strict=True,
        )
        metrics = PlayerDetectionMetrics(config.evaluation)
        predictions = []
        for sample in dataset:
            with torch.no_grad():
                prediction = decode_player_detections(
                    model([sample.image]),
                    [sample.original_size],
                    max_detections=config.evaluation.max_detections,
                )[0]
            metrics.update([prediction], [sample.boxes_xyxy_px])
            predictions.append(prediction)
        headline, diagnostics = metrics.compute()
        summaries[name] = headline
        output = config.output_dir / name
        output.mkdir()
        np.savez_compressed(
            output / "predictions.npz",
            boxes_xyxy=torch.stack([item.boxes_xyxy for item in predictions]).numpy(),
            scores=torch.stack([item.scores for item in predictions]).numpy(),
            frame_keys=np.asarray(dataset.prediction_ids),
        )
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(path),
                    "headline": headline,
                    "diagnostics": diagnostics,
                }
            )
        )
    (config.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset": str(config.dataset_dir),
                "split": "val",
                "frame_stride": config.frame_stride,
                "selection_stats": selection.stats,
                "metrics": summaries,
            }
        )
    )
    return summaries


class PlayerExecutor(CPUExecutor):
    def finish(self, reference: ArtifactRef) -> ArtifactRef:
        launch = read_document(self.layout, reference, StageLaunch)
        assert launch.input is not None
        with patch.dict(
            os.environ, {"TENNIS_ANNOTATION_STAGE_OPERATION": launch.input.digest}
        ):
            result: ArtifactRef = execute(
                self.layout, launch.input, Path(launch.command[-1])
            )
        self.record_exit(reference)
        return result


class Environment:
    def __init__(
        self, source: SyntheticRoot, *, policy: PlayerSelectionPolicy | None = None
    ) -> None:
        for name in ("src_c", "src_d"):
            manifest = write_clip(source.root, name, "f020-026")
            publish_player_annotation(
                source.root, manifest, {1: [player("p1", [10.0, 10.0, 30.0, 40.0])]}
            )
        directory = build_dataset(source.config)
        metadata = json.loads((directory / METADATA_FILE).read_bytes())
        splits = {"src_a": "train", "src_b": "train", "src_c": "val", "src_d": "test"}
        for item in metadata["clips"]:
            item["split"] = splits[item["source_id"]]
        for key, field in (
            ("clips_by_split", None),
            ("frames_by_split", "stored_frames"),
        ):
            metadata["counts"][key] = {
                split: sum(
                    item[field] if field else 1
                    for item in metadata["clips"]
                    if item["split"] == split
                )
                for split in ("train", "val", "test")
            }
        (directory / METADATA_FILE).write_text(json.dumps(metadata))
        self.layout = AnnotationLayout.testing(directory.parents[2] / "annotation")
        self.snapshot = freeze_existing_store(
            self.layout,
            directory,
            tuple(
                SourceBinding(store_source_id=name, source_id=content_digest(name))
                for name in sorted(splits)
            ),
        )
        snapshot, _ = load_snapshot(self.layout, self.snapshot)
        selection = SelectionPolicy(
            require_reviewed=True,
            require_located_players=True,
            min_visible_box_px=4.0,
            frame_stride=1,
        )
        self.fixed = FixedEvaluation(
            task="player",
            validation=freeze_partition(self.layout, self.snapshot, "val", selection),
            test=freeze_partition(self.layout, self.snapshot, "test", selection),
        )
        repository = self.layout.repository / "third_party/DINO"
        repository.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--allow-empty",
                "-qm",
                "fixture",
            ],
            cwd=repository,
            check=True,
        )
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        ckpt = self.layout.repository / "ckpt/player_detection"
        ckpt.mkdir(parents=True)
        torch.save({"epoch": 0, "global_step": 1}, ckpt / "parent.ckpt")
        training_ref = file_reference(
            self.layout, ckpt / "parent.ckpt", role="checkpoint"
        )
        torch.save(
            {
                "model": TinyDino().state_dict(),
                "args": dino_4scale_swin_args("cpu", use_checkpoint=False),
                "tennis_lab": {
                    "format": EXPORT_FORMAT,
                    "task": "player_detection",
                    "class_id": 1,
                    "source_checkpoint_sha256": training_ref.digest,
                    "epoch": 0,
                    "global_step": 1,
                },
            },
            ckpt / "parent.pth",
        )
        self.parent = ModelRef(
            task="player",
            version="parent",
            training_weights=training_ref,
            inference_weights=file_reference(
                self.layout, ckpt / "parent.pth", role="checkpoint"
            ),
            configuration=save(
                self.layout,
                DinoCheckpointSpec(repository_commit=revision, epoch=0, global_step=1),
            ),
        )
        settings = yaml.safe_load(
            (
                ROOT / "src/tasks/player_detection/configs/training/default.yaml"
            ).read_bytes()
        )
        settings["optimizer"].pop(
            "name"
        )  # Shared optimizer contract is AdamW; no ignored name field.
        settings["trainer"].update(
            max_epochs=2,
            accumulate_grad_batches=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
        )
        settings["early_stopping"]["enabled"] = False
        settings["qualitative_logging"]["enabled"] = False
        settings["warmup_steps"] = 0
        settings["learning_rate"] = 0.01
        settings_ref = write_asset(
            self.layout,
            self.layout.path("models", "settings.json"),
            json.dumps(settings).encode(),
        )
        self.recipe = PlayerLearningRecipe(
            training=settings_ref,
            selection=selection,
            input_size=InputSize(short_side=32, max_long_side=48),
            augmentation=Augmentation(
                hflip_prob=0.0,
                short_side_choices=(32,),
                brightness=0.0,
                contrast=0.0,
                saturation=0.0,
            ),
            operating_point=OperatingPoint(
                score_threshold=0.3, iou_threshold=0.5, max_detections=2
            ),
            batch_size=2,
            evaluation_batch_size=2,
            train_samples_per_epoch=4,
            num_workers=0,
            pin_memory=False,
            seed=42,
            use_checkpoint=False,
            backbone_freeze_keywords=(),
            backbone_lr_scale=0.1,
        )
        blob = write_asset(
            self.layout,
            self.layout.path("models", "fixture.txt"),
            b"synthetic policy and adoption evidence",
        )
        self.blob = blob
        receipt = PublicationReceipt(
            task="player",
            product="boxes",
            request_id="1" * 64,
            base_snapshot=None,
            snapshot=self.snapshot,
            adoptions=(blob,),
            added_clips=tuple(
                item.clip_id for item in snapshot.clips if item.split == "train"
            ),
            reused_clips=tuple(
                item.clip_id for item in snapshot.clips if item.split != "train"
            ),
        )
        self.publication = save(self.layout, receipt)
        plan = CampaignPlan.model_validate_json(
            json.dumps(
                {
                    "campaign_id": "fixture",
                    "phase": "production",
                    "annotation_root": str(self.layout.root),
                    "counts": {
                        "channels": 1,
                        "videos_per_channel": 1,
                        "max_clips_per_video": 1,
                    },
                    "selection": blob.model_dump(),
                    "preparation": blob.model_dump(),
                    "code_commit": "a" * 40,
                    "code_diff_digest": "b" * 64,
                    "agent_model": "gpt-6.1-sol",
                    "agent_effort": "high",
                    "resources": {
                        "codex_workers": 1,
                        "additional_storage_bytes": 1000000,
                        "minimum_free_bytes": 1,
                        "total_pss_bytes": 1000000,
                        "gpu_seconds": 100,
                        "stages": [
                            {
                                "stage": name,
                                "kind": "gpu",
                                "parallel": 1,
                                "threads_per_process": 1,
                            }
                            for name in ("train", "evaluate")
                        ],
                    },
                    "tasks": [
                        {
                            "task": "player",
                            "model": self.parent.model_dump(),
                            "inference": blob.model_dump(),
                            "review": {
                                key: blob.model_dump()
                                for key in ("policy", "preview_evidence", "decision")
                            },
                            "learning": {
                                "recipe": save(self.layout, self.recipe).model_dump(),
                                "validation": save(
                                    self.layout, self.fixed
                                ).model_dump(),
                                "selection_policy": save(
                                    self.layout, policy
                                ).model_dump()
                                if policy
                                else None,
                            },
                            "products": [
                                {
                                    "product": "boxes",
                                    "dataset_task": "player_detection",
                                    "dataset_version": "fixture",
                                    "component": "frames",
                                    "base_snapshot": None,
                                    "generation_policy": blob.model_dump(),
                                }
                            ],
                        }
                    ],
                }
            )
        )
        self.plan_ref = annotation_reference(self.layout, save_plan(self.layout, plan))
        completion = save(
            self.layout,
            CampaignCompletion(
                plan=self.plan_ref,
                tasks=(
                    TaskProducts(
                        task="player",
                        products=(
                            ProductOutcome(
                                product="boxes",
                                status="published",
                                publication=self.publication,
                                reason=None,
                            ),
                        ),
                    ),
                ),
            ),
        )
        self.adapter = PlayerLearningAdapter()
        self.executor = PlayerExecutor(self.layout)
        self.learning = Learning(self.layout, executor=self.executor)
        self.learning.models.bootstrap(self.parent, blob)
        registration = self.learning.register(completion, self.adapter)
        assert registration.intent is not None
        self.intent_ref = registration.intent
        self.intent = read_document(self.layout, self.intent_ref, LearningIntent)
        self.runtime = LearningRuntime(
            scope=blob,
            queue=QueueConfiguration(
                script="/fixture/queue.sh",
                script_digest=blob.digest,
                directory="/fixture/queue",
                session="fixture",
                issue=1137,
            ),
            train=StageCost(
                stage="train",
                pss_bytes=1,
                storage_bytes=1,
                gpu_seconds=60.0,
                timeout_seconds=40.0,
            ),
            evaluate=StageCost(
                stage="evaluate",
                pss_bytes=1,
                storage_bytes=1,
                gpu_seconds=60.0,
                timeout_seconds=40.0,
            ),
        )

    def tick(self):
        return self.learning.tick(self.intent_ref, self.adapter, self.runtime)
