"""Resolve immutable learning intent inputs and the exact new-run recipe."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.annotation.contracts import MDDCheckpointSpec
from src.tasks.ball_detection.checkpoints.initialization import initialize_full_model
from src.tasks.ball_detection.checkpoints.posttraining import SCHEMA
from src.tasks.ball_detection.data.augmentation.video import AugmentationConfig
from src.tasks.ball_detection.data.windows.learning.contracts import (
    HoldoutWindows,
    TrainingWindows,
)
from src.tasks.ball_detection.data.windows.learning.holdout import validate_holdout
from src.tasks.ball_detection.data.windows.learning.preparation import (
    prepare_training_windows,
    require_additional_teachers,
    training_partition,
    validate_training_windows,
)
from src.tasks.ball_detection.models.mdd import DeepMDDQueryDetector
from src.tasks.ball_detection.training.coordinates.joint import finetuning_recipe
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    file_reference,
    read_document,
    verify_reference,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ModelRef,
    PublicationReceipt,
    canonical_bytes,
)
from src.tasks.base.annotation.learning.contracts import (
    CampaignCompletion,
    FixedEvaluation,
    LearningIntent,
)
from src.tasks.base.annotation.learning.validation import (
    check_completion,
    check_evaluation_binding,
    check_partitions,
    verify_inputs,
)
from src.tasks.base.annotation.paths import AnnotationLayout

from .contracts import BalancedSelectionPolicy, BallLearningRecipe
from .selection import read_policy


def output_reference(layout: AnnotationLayout, path: Path) -> ArtifactRef:
    return file_reference(layout, path, role="artifact")


def run_root(layout: AnnotationLayout, intent: LearningIntent) -> Path:
    result: Path = layout.artifact_path(
        ArtifactRef(
            role="artifact",
            path=f"ball_detection/train/annotation/{intent.sha256}",
            digest="0" * 64,
            size_bytes=0,
        )
    )
    return result


def initialize_parent(
    layout: AnnotationLayout, intent: LearningIntent, recipe: BallLearningRecipe
) -> tuple[DeepMDDQueryDetector, dict[str, Any]]:
    spec = read_document(layout, intent.parent.model.configuration, MDDCheckpointSpec)
    parent = intent.parent.model
    if (
        parent.training_weights != parent.inference_weights
        or spec.checkpoint_schema != SCHEMA
    ):
        raise ValueError(
            "ball learning requires one complete query checkpoint for training and inference"
        )
    model, provenance = initialize_full_model(
        verify_reference(layout, parent.training_weights),
        expected_sha256=parent.training_weights.digest,
        jpeg_decoder=recipe.runtime.jpeg_decoder,
    )
    if (
        canonical_bytes(asdict(model.config)) != canonical_bytes(spec.architecture)
        or canonical_bytes(model.mdd.input_contract())
        != canonical_bytes(spec.input_contract)
        or provenance["parent_manifest_sha256"] != spec.training_manifest_digest
    ):
        raise ValueError("adopted model differs from its frozen MDD specification")
    provenance.update(
        learning_intent=intent.sha256,
        learning_recipe=intent.recipe.digest,
        training_snapshot=intent.train.manifest.digest,
        parent_model=parent.sha256,
    )
    return model, provenance


@dataclass(frozen=True)
class Inputs:
    recipe: BallLearningRecipe
    prepared_ref: ArtifactRef
    prepared: TrainingWindows
    holdout: HoldoutWindows
    fixed: FixedEvaluation
    test_holdout: HoldoutWindows | None = None

    def arguments(
        self, layout: AnnotationLayout, output: Path, resume: Path | None
    ) -> argparse.Namespace:
        values = self.recipe.model_dump(
            exclude={"schema_version", "windows", "augmentation", "runtime", "replay"}
        )
        values.update(self.recipe.runtime.model_dump())
        return argparse.Namespace(
            **values,
            manifest=verify_reference(
                layout, require_additional_teachers(self.prepared)
            ),
            augmentation_config=verify_reference(layout, self.recipe.augmentation),
            output=output,
            resume=resume,
            stop_after_epoch=None,
            replay_manifest=(verify_reference(layout, self.recipe.replay.previously_trained_manifest)
                             if self.recipe.replay else None),
            untrained_fraction=self.recipe.replay.untrained_fraction if self.recipe.replay else None,
            selection_scope="balanced_existing_new" if self.fixed.recipe is not None else "common",
        )

    def expected_recipe(
        self, layout: AnnotationLayout, intent: LearningIntent
    ) -> dict[str, Any]:
        model, parent = initialize_parent(layout, intent, self.recipe)
        result: dict[str, Any] = finetuning_recipe(
            self.arguments(layout, run_root(layout, intent), None),
            parent=parent,
            model_config=asdict(model.config),
            runtime=self.recipe.runtime.coordinate_runtime(),
            augmentation=AugmentationConfig.load(
                verify_reference(layout, self.recipe.augmentation)
            ),
            manifest_sha256=require_additional_teachers(self.prepared).digest,
            validation_manifest_sha256=self.holdout.manifest.digest,
            initial_state_sha256=parent["state_dict_sha256"],
            input_contract=model.mdd.input_contract(),
        )
        return result


def resolve_inputs(
    layout: AnnotationLayout, intent: LearningIntent, *, require_teachers: bool = True
) -> Inputs:
    if intent.task != "ball":
        raise ValueError("ball adapter cannot execute another task")
    verify_inputs(layout, intent)
    completion = read_document(layout, intent.completion, CampaignCompletion)
    plan = check_completion(layout, completion)
    if not completion.complete:
        raise ValueError(
            "all planned annotation products must finish before ball learning"
        )
    task = next(task for task in plan.tasks if task.task == "ball")
    if task.learning is None or (
        task.model,
        task.learning.recipe,
        task.learning.selection_policy,
    ) != (
        intent.parent.model,
        intent.recipe,
        intent.selection_policy,
    ):
        raise ValueError("ball learning intent differs from the frozen production plan")
    check_evaluation_binding(layout, task.learning, intent)
    outcomes = next(t.products for t in completion.tasks if t.task == "ball")
    if (
        next(p.publication for p in outcomes if p.product == "frames")
        != intent.publication
    ):
        raise ValueError("ball intent publication differs from completed annotations")
    publication = read_document(layout, intent.publication, PublicationReceipt)
    if intent.train != training_partition(layout, publication.snapshot):
        raise ValueError("ball train partition differs from published membership")
    recipe = read_document(layout, intent.recipe, BallLearningRecipe)
    if recipe.replay is not None:
        verify_reference(layout, recipe.replay.previously_trained_manifest)
        parent_spec = read_document(layout, intent.parent.model.configuration, MDDCheckpointSpec)
        if parent_spec.training_manifest_digest != recipe.replay.previously_trained_manifest.digest:
            raise ValueError("Replay manifest differs from the adopted model's training data")
    fixed = read_document(layout, intent.fixed_evaluation, FixedEvaluation)
    if fixed.task != "ball":
        raise ValueError("fixed evaluation task mismatch")
    check_partitions(
        intent.train, fixed.validation, *((fixed.test,) if fixed.test else ())
    )
    holdout = validate_holdout(layout, fixed.validation)
    test_holdout = validate_holdout(layout, fixed.test) if fixed.test is not None else None
    if fixed.recipe is not None:
        from src.tasks.ball_detection.data.windows.learning.evaluation import (
            resolve_evaluation,
        )

        if resolve_evaluation(layout, task.learning, intent.publication) != intent.fixed_evaluation:
            raise ValueError("Resolved holdouts differ from the publication and pinned recipe")
    AugmentationConfig.load(verify_reference(layout, recipe.augmentation))
    if intent.selection_policy is not None:
        selection = read_policy(layout, intent.selection_policy)
        if isinstance(selection, BalancedSelectionPolicy) != (fixed.recipe is not None):
            raise ValueError("Selection policy must agree with the fixed evaluation cohorts")
    directory = layout.path(
        "campaigns", plan.campaign_id, "learning", "ball", "inputs", intent.sha256
    )
    if (directory / "coverage.json").exists():
        prepared_ref = annotation_reference(layout, directory / "coverage.json")
        prepared = validate_training_windows(layout, prepared_ref)
        if (prepared.publication, prepared.policy) != (
            intent.publication,
            recipe.windows,
        ):
            raise ValueError("prepared windows differ from the learning intent")
    else:
        if (directory.parent.parent / "intent.json").exists():
            raise ValueError("registered learning intent has lost its prepared windows")
        prepared_ref = prepare_training_windows(
            layout, intent.publication, recipe.windows, directory
        )
        prepared = read_document(layout, prepared_ref, TrainingWindows)
    if require_teachers:
        require_additional_teachers(prepared)
    return Inputs(recipe, prepared_ref, prepared, holdout, fixed, test_holdout)


def check_model_spec(layout: AnnotationLayout, model: ModelRef) -> MDDCheckpointSpec:
    if model.task != "ball" or model.training_weights != model.inference_weights:
        raise ValueError(
            "ball candidate must use the same full checkpoint for training and inference"
        )
    spec: MDDCheckpointSpec = read_document(
        layout, model.configuration, MDDCheckpointSpec
    )
    if spec.checkpoint_schema != SCHEMA:
        raise ValueError("ball candidate schema is not query posttraining v1")
    return spec
