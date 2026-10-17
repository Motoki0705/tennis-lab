"""Materialize planned old/new holdouts after annotation publication."""

from __future__ import annotations

import json

from src.tasks.ball_detection.data.windows.manifest import build_coordinate_manifest
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    load_snapshot,
)
from src.tasks.base.annotation.artifacts import (
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    PublicationReceipt,
    canonical_bytes,
)
from src.tasks.base.annotation.learning.contracts import FixedEvaluation
from src.tasks.base.annotation.learning.validation import check_partitions
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LearningPlan

from .contracts import BallEvaluationRecipe, HoldoutWindows
from .holdout import bind_cohorts, freeze_holdout, validate_holdout
from .preparation import training_partition


def resolve_evaluation(layout: AnnotationLayout, plan: LearningPlan, publication_ref: ArtifactRef) -> ArtifactRef:
    if plan.evaluation_recipe is None:
        return plan.validation
    recipe = read_document(layout, plan.evaluation_recipe, BallEvaluationRecipe)
    original = read_document(layout, plan.validation, FixedEvaluation)
    publication = read_document(layout, publication_ref, PublicationReceipt)
    if original.task != "ball" or original.test is None or (publication.task, publication.product) != ("ball", "frames"):
        raise ValueError("Ball cohort evaluation requires original validation/test and a ball publication")
    snapshot, store = load_snapshot(layout, publication.snapshot)
    directory = layout.path("models", "ball", "evaluation", "resolved", plan.evaluation_recipe.digest,
                            publication_ref.digest)
    partitions = []
    for partition in (original.validation, original.test):
        old = validate_holdout(layout, partition)
        original_raw = json.loads(verify_reference(layout, old.manifest).read_bytes())
        raw = build_coordinate_manifest(store, recipe.windows.play(), recipe.windows.sampling(), input_kind="mdd_only",
            common_reference=original_raw["common_evaluation"],
            shard_hashes={c.store_clip_id: c.shard.digest for c in snapshot.clips})
        binding = HoldoutWindows(snapshot=publication.snapshot, manifest=old.manifest,
                                 split=old.split, existing=partition.manifest)
        bind_cohorts(layout, binding, raw)
        manifest = write_asset(layout, directory / f"{old.split}-windows.json", canonical_bytes(raw) + b"\n")
        partitions.append(freeze_holdout(layout, binding.model_copy(update={"manifest": manifest})))
    check_partitions(training_partition(layout, publication.snapshot), *partitions)
    fixed = FixedEvaluation(task="ball", validation=partitions[0], test=partitions[1],
                            recipe=plan.evaluation_recipe, original=plan.validation)
    return write_document(layout, directory / "fixed-evaluation.json", fixed)
