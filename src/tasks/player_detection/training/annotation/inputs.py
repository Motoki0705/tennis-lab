"""Resolve cumulative teachers and fixed validation before any training side effect."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.contracts import ArtifactRef, PublicationReceipt
from src.tasks.base.annotation.learning.contracts import (
    CampaignCompletion,
    DatasetPartition,
    FixedEvaluation,
    LearningIntent,
)
from src.tasks.base.annotation.learning.validation import (
    check_completion,
    check_partitions,
    verify_inputs,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.configuration import BaseTrainingConfig
from src.tasks.player_detection.annotation.contracts import DinoCheckpointSpec
from src.tasks.player_detection.configuration import PlayerTrainingConfig
from src.tasks.player_detection.data.datamodule import PlayerDetectionDataModule
from src.tasks.player_detection.data.detection_dataset import (
    PlayerDetectionDataset,
    select_detection_frames,
)
from src.tasks.player_detection.data.snapshots.contracts import PlayerPartition
from src.tasks.player_detection.data.snapshots.partitions import (
    describe_partition,
    freeze_partition,
    validate_partition,
)
from src.tasks.player_detection.data.snapshots.store import load_snapshot
from src.tasks.player_detection.data.store import Split
from src.tasks.player_detection.inference.checkpoint import inspect_player_checkpoint

from .contracts import PlayerLearningRecipe, PlayerSelectionPolicy
from .precision import precision_contract


def training_partition(
    layout: AnnotationLayout, snapshot: ArtifactRef
) -> DatasetPartition:
    pinned, _ = load_snapshot(layout, snapshot)
    clips = [item for item in pinned.clips if item.split == "train"]
    return DatasetPartition(
        manifest=snapshot,
        split="train",
        sources=tuple(sorted({item.source_id for item in clips})),
        clips=tuple(sorted(item.clip_id for item in clips)),
    )


def run_root(layout: AnnotationLayout, intent: LearningIntent) -> Path:
    result: Path = layout.artifact_path(
        ArtifactRef(
            role="artifact",
            path=f"player_detection/train/annotation/{intent.sha256}",
            digest="0" * 64,
            size_bytes=0,
        )
    )
    return result


class FrozenPlayerDataModule(PlayerDetectionDataModule):
    """The training pool may grow; validation is a separately pinned dataset."""

    def __init__(
        self, runtime: PlayerTrainingConfig, validation: PlayerDetectionDataset
    ) -> None:
        if (
            runtime.shared.run.test_after_fit
            or runtime.shared.training.qualitative_logging.enabled
        ):
            raise ValueError(
                "fixed player learning excludes test and unpinned qualitative media"
            )
        super().__init__(runtime.data)
        self._datasets["val"] = validation
        self.selection_stats["val"] = validation.selection.stats

    def _dataset(self, split: Split) -> PlayerDetectionDataset:
        if split == "test":
            raise ValueError("test data is not part of annotation model selection")
        return super()._dataset(split)


@dataclass(frozen=True)
class Inputs:
    recipe: PlayerLearningRecipe
    settings: dict[str, Any]
    training: BaseTrainingConfig
    train: DatasetPartition
    train_support: PlayerPartition
    fixed: FixedEvaluation
    validation: PlayerPartition
    spec: DinoCheckpointSpec

    @property
    def total_updates(self) -> int:
        batches = self.recipe.train_samples_per_epoch // self.recipe.batch_size
        result: int = self.training.trainer.max_epochs * math.ceil(
            batches / self.training.trainer.accumulate_grad_batches
        )
        return result

    def configuration(
        self,
        layout: AnnotationLayout,
        intent: LearningIntent,
        output: Path,
        resume: ArtifactRef | None,
    ) -> DictConfig:
        if intent.parent.model.inference_weights.role != "checkpoint":
            raise ValueError(
                "player initialization requires a checkpoint-role DINO export"
            )
        snapshot, _ = load_snapshot(layout, intent.train.manifest)
        paths = dict(layout.resolver.roots.as_mapping())
        paths["output_root"] = str(layout.repository / "outputs")
        config = OmegaConf.create(
            {
                "paths": paths,
                "run": {
                    "output_dir": output.relative_to(
                        layout.repository / "outputs"
                    ).as_posix(),
                    "seed": self.recipe.seed,
                    "gpus": 1,
                    "resume": None
                    if resume is None
                    else {"role": resume.role, "path": resume.path},
                    "init_weights": None,
                    "fast_dev_run": False,
                    "dry_run": False,
                    "test_after_fit": False,
                    "artifact_store": {
                        "mode": "local",
                        "remote": None,
                        "remote_root": None,
                        "sync_interval_seconds": None,
                    },
                },
                "data": {
                    "dataset_dir": str(Path(snapshot.index.path).parent),
                    "batch_size": self.recipe.batch_size,
                    "eval_batch_size": self.recipe.evaluation_batch_size,
                    "num_workers": self.recipe.num_workers,
                    "pin_memory": self.recipe.pin_memory,
                    "train_samples_per_epoch": self.recipe.train_samples_per_epoch,
                    "val_frame_stride": self.validation.policy.frame_stride,
                    "test_frame_stride": 1,  # never read; the explicit data module rejects test
                    "selection": self.recipe.selection.model_dump(
                        exclude={"frame_stride"}
                    ),
                    "input_size": self.recipe.input_size.model_dump(),
                    "augmentation": self.recipe.augmentation.model_dump(mode="json"),
                },
                "model": {
                    "name": "dino_4scale_swin_l",
                    "repository": "DINO",
                    "init_checkpoint": intent.parent.model.inference_weights.path,
                    "use_checkpoint": self.recipe.use_checkpoint,
                    "backbone_freeze_keywords": list(
                        self.recipe.backbone_freeze_keywords
                    ),
                    "backbone_lr_scale": self.recipe.backbone_lr_scale,
                },
                "training": self.settings,
                "evaluation": self.recipe.operating_point.model_dump(),
            }
        )
        if not isinstance(config, DictConfig):
            raise TypeError("player training configuration must be a mapping")
        PlayerTrainingConfig.from_config(config)
        return config

    def datamodule(
        self, layout: AnnotationLayout, config: DictConfig
    ) -> FrozenPlayerDataModule:
        runtime = PlayerTrainingConfig.from_config(config)
        validate_partition(layout, self.fixed.validation)
        _, store = load_snapshot(layout, self.validation.snapshot)
        selected = select_detection_frames(
            store,
            "val",
            self.validation.policy.selection(),
            frame_stride=self.validation.policy.frame_stride,
        )
        dataset = PlayerDetectionDataset(
            store, selected, input_size=runtime.data.input_size, augmentation=None
        )
        return FrozenPlayerDataModule(runtime, dataset)


def resolve_inputs(layout: AnnotationLayout, intent: LearningIntent) -> Inputs:
    if intent.task != "player":
        raise ValueError("player learning cannot execute another task")
    verify_inputs(layout, intent)
    completion = read_document(layout, intent.completion, CampaignCompletion)
    plan = check_completion(layout, completion)
    if not completion.complete:
        raise ValueError("all annotation products must complete before player learning")
    task = next(item for item in plan.tasks if item.task == "player")
    if task.learning is None or (
        task.model,
        task.learning.recipe,
        task.learning.validation,
        task.learning.selection_policy,
    ) != (
        intent.parent.model,
        intent.recipe,
        intent.fixed_evaluation,
        intent.selection_policy,
    ):
        raise ValueError("player learning differs from its frozen production plan")
    outcomes = next(item.products for item in completion.tasks if item.task == "player")
    if (
        next(item.publication for item in outcomes if item.product == "boxes")
        != intent.publication
    ):
        raise ValueError("player publication differs from completed annotations")
    publication = read_document(layout, intent.publication, PublicationReceipt)
    if (
        publication.task != "player"
        or publication.product != "boxes"
        or training_partition(layout, publication.snapshot) != intent.train
    ):
        raise ValueError(
            "player training membership differs from the published snapshot"
        )
    recipe = read_document(layout, intent.recipe, PlayerLearningRecipe)
    fixed = read_document(layout, intent.fixed_evaluation, FixedEvaluation)
    if fixed.task != "player":
        raise ValueError("player fixed evaluation belongs to another task")
    check_partitions(
        intent.train, fixed.validation, *((fixed.test,) if fixed.test else ())
    )
    validation = validate_partition(layout, fixed.validation)
    if fixed.test is not None:
        validate_partition(layout, fixed.test)
    _, train = describe_partition(
        layout, publication.snapshot, "train", recipe.selection
    )
    if not layout.artifact_path(train.manifest).exists():
        if layout.path(
            "campaigns", plan.campaign_id, "learning", "player", "intent.json"
        ).exists():
            raise ValueError(
                "registered player learning intent has lost its pinned training support"
            )
        train = freeze_partition(
            layout, publication.snapshot, "train", recipe.selection
        )
    support = validate_partition(layout, train)
    if recipe.train_samples_per_epoch > support.selected_frames:
        raise ValueError(
            "player training sample budget exceeds the pinned eligible pool"
        )
    snapshot, store = load_snapshot(layout, publication.snapshot)
    selected = select_detection_frames(
        store, "train", recipe.selection.selection(), frame_stride=1
    )
    available = {
        snapshot.clips[store.clip_of(int(row)).index].clip_id for row in selected.frames
    }
    if not available.intersection(publication.added_clips):
        raise ValueError(
            "no_additional_player_teachers: no added clip supplies eligible train frames"
        )
    if set(publication.added_clips + publication.reused_clips) != {
        item.clip_id for item in snapshot.clips
    }:
        raise ValueError("publication does not account for every snapshot clip")
    settings: dict[str, Any] = json.loads(
        verify_reference(layout, recipe.training).read_bytes()
    )
    training = BaseTrainingConfig.from_mapping(settings)
    precision_contract(training)
    checkpoint = training.checkpoint
    if not checkpoint.enabled or (
        checkpoint.monitor,
        checkpoint.mode,
        checkpoint.save_top_k,
        checkpoint.save_last,
    ) != ("val/map", "max", 1, True):
        raise ValueError(
            "player learning requires val/map top-1 and separate last checkpoints"
        )
    if (
        training.early_stopping.enabled
        or training.qualitative_logging.enabled
        or training.gan.enabled
        or training.compile.enabled
        or training.trainer.check_val_every_n_epoch != 1
        or training.trainer.precision != "32-true"
        or training.optimizer.steps_per_epoch is not None
    ):
        raise ValueError(
            "player learning requires full-budget fp32 validation each epoch, without early stopping, GAN, compile, or unpinned qualitative media"
        )
    spec = read_document(layout, intent.parent.model.configuration, DinoCheckpointSpec)
    info = inspect_player_checkpoint(
        verify_reference(layout, intent.parent.model.inference_weights)
    )
    if (info.source_checkpoint_sha256, info.epoch, info.global_step) != (
        intent.parent.model.training_weights.digest,
        spec.epoch,
        spec.global_step,
    ):
        raise ValueError(
            "parent player export differs from its training checkpoint/specification"
        )
    repository = layout.repository / "third_party/DINO"
    if repository.resolve() != repository or not repository.is_dir():
        raise ValueError("DINO source must be the nonsymlink shared repository")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        revision != spec.repository_commit
        or subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], cwd=repository, check=False
        ).returncode
        != 0
    ):
        raise ValueError("DINO source differs from the pinned model revision")
    if intent.selection_policy is not None:
        read_document(layout, intent.selection_policy, PlayerSelectionPolicy)
    inputs = Inputs(recipe, settings, training, train, support, fixed, validation, spec)
    if (
        training.optimizer.warmup_steps is not None
        and training.optimizer.warmup_steps >= inputs.total_updates
    ):
        raise ValueError("warmup must end before the total update budget")
    return inputs
