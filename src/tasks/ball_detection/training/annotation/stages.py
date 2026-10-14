"""Stage-owned execution; these functions never enqueue or adopt a model."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.tasks.ball_detection.annotation.contracts import MDDCheckpointSpec
from src.tasks.ball_detection.checkpoints.posttraining import SCHEMA
from src.tasks.ball_detection.evaluation.fixed_validation import compare_full_models
from src.tasks.ball_detection.training.coordinates.joint import run_finetuning
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, ModelRef, canonical_bytes
from src.tasks.base.annotation.learning.contracts import (
    EvaluationReceipt,
    LearningIntent,
    StageInput,
    TrainingReceipt,
)
from src.tasks.base.annotation.learning.execution import (
    stage_request as shared_stage_request,
)
from src.tasks.base.annotation.learning.validation import check_training
from src.tasks.base.annotation.paths import AnnotationLayout

from .contracts import BallEvaluationMetrics, BallTrainingProvenance
from .evidence import (
    checkpoint_value,
    validate_evaluation,
    validate_resume,
    validate_training,
)
from .inputs import initialize_parent, output_reference, resolve_inputs, run_root


def stage_request(
    layout: AnnotationLayout, reference: ArtifactRef, result: Path
) -> tuple[StageInput, LearningIntent]:
    return shared_stage_request(layout, reference, result, task="ball")


def execute(
    layout: AnnotationLayout, reference: ArtifactRef, result: Path
) -> ArtifactRef:
    request, intent = stage_request(layout, reference, result)
    with artifact_lock(layout, "ball-learning-" + intent.sha256, blocking=False):
        if request.stage == "train":
            return train(layout, reference, request, intent, result)
        return evaluate(layout, reference, request, intent, result)


def train(
    layout: AnnotationLayout,
    reference: ArtifactRef,
    request: StageInput,
    intent: LearningIntent,
    result: Path,
) -> ArtifactRef:
    if result.exists():
        existing = annotation_reference(layout, result)
        prior = read_document(layout, existing, TrainingReceipt)
        if prior.request != reference:
            raise ValueError("existing training receipt belongs to another attempt")
        validate_training(layout, intent, prior)
        return existing
    inputs = resolve_inputs(layout, intent)
    output = run_root(layout, intent) / f"attempt-{request.attempt:03d}"
    resume = None
    if request.resume_checkpoint is not None:
        validate_resume(layout, intent, request.resume_checkpoint)
        resume = verify_reference(layout, request.resume_checkpoint)
        output = resume.parent
        if int(output.name.removeprefix("attempt-")) >= request.attempt:
            raise ValueError("an explicit retry can only resume an earlier attempt")
    if not (output / "COMPLETED.json").exists():
        run_finetuning(
            inputs.arguments(layout, output, resume),
            validation_manifest=verify_reference(layout, inputs.holdout.manifest),
            initialize=lambda: initialize_parent(layout, intent, inputs.recipe),
        )
    latest_ref = output_reference(
        layout, output / f"epoch-{inputs.recipe.epochs - 1:03d}.pt"
    )
    latest = checkpoint_value(layout, intent, inputs, latest_ref)
    selected_epoch = latest["best_epoch"]
    if (
        type(selected_epoch) is not int
        or not 0 <= selected_epoch < inputs.recipe.epochs
    ):
        raise ValueError("completed run has an invalid selected epoch")
    selected_ref = output_reference(layout, output / f"epoch-{selected_epoch:03d}.pt")
    selected = checkpoint_value(layout, intent, inputs, selected_ref)
    spec = MDDCheckpointSpec(
        checkpoint_schema=SCHEMA,
        architecture=selected["model_config"],
        input_contract=selected["recipe"]["input_contract"],
        training_manifest_digest=selected["recipe"]["manifest_sha256"],
    )
    spec_ref = write_document(
        layout,
        layout.path(
            "models", "ball", "candidates", intent.sha256, spec.sha256 + ".json"
        ),
        spec,
    )
    candidate = ModelRef(
        task="ball",
        version="annotation-" + intent.sha256,
        training_weights=selected_ref,
        inference_weights=selected_ref,
        configuration=spec_ref,
    )
    provenance = BallTrainingProvenance(
        request=reference,
        intent=request.intent,
        prepared=inputs.prepared_ref,
        validation_manifest=inputs.holdout.manifest,
        run_config=output_reference(layout, output / "config.json"),
        latest_checkpoint=latest_ref,
        selected_checkpoint=selected_ref,
        best_record=output_reference(layout, output / "best.json"),
        completed=output_reference(layout, output / "COMPLETED.json"),
        initial_state_sha256=selected["recipe"]["initial_state_sha256"],
        stress=output_reference(layout, output / "stress.json")
        if inputs.recipe.stress_evaluation
        else None,
    )
    provenance_ref = write_document(
        layout, result.parent / "provenance.json", provenance
    )
    receipt = TrainingReceipt(
        request=reference, candidate=candidate, provenance=provenance_ref
    )
    validate_training(layout, intent, receipt)
    return write_document(layout, result, receipt)


def evaluate(
    layout: AnnotationLayout,
    reference: ArtifactRef,
    request: StageInput,
    intent: LearningIntent,
    result: Path,
) -> ArtifactRef:
    if result.exists():
        existing = annotation_reference(layout, result)
        prior = read_document(layout, existing, EvaluationReceipt)
        if prior.request != reference:
            raise ValueError("existing evaluation receipt belongs to another attempt")
        validate_evaluation(layout, intent, prior)
        return existing
    if request.training is None:
        raise ValueError("evaluation requires a completed training stage")
    training_request, training = check_training(layout, request.training)
    if training_request.intent != request.intent:
        raise ValueError("completed training belongs to another intent")
    validate_training(layout, intent, training)
    inputs = resolve_inputs(layout, intent)
    comparison_path = result.parent / "comparison.json"
    if comparison_path.exists():
        # The previous stage may have stopped after persisting metrics. The
        # receipt validator below verifies all identities before recovering it.
        comparison_ref = annotation_reference(layout, comparison_path)
        json.loads(verify_reference(layout, comparison_ref).read_bytes())
    else:
        comparison = compare_full_models(
            verify_reference(layout, intent.parent.model.inference_weights),
            verify_reference(layout, training.candidate.inference_weights),
            verify_reference(layout, inputs.holdout.manifest),
            baseline_sha256=intent.parent.model.inference_weights.digest,
            candidate_sha256=training.candidate.inference_weights.digest,
            validation_sha256=inputs.holdout.manifest.digest,
            runtime=inputs.recipe.runtime.coordinate_runtime(),
            device=torch.device(inputs.recipe.runtime.device),
            batch_size=inputs.recipe.batch_size,
        )
        comparison_ref = write_asset(
            layout, comparison_path, canonical_bytes(comparison) + b"\n"
        )
    metrics = BallEvaluationMetrics(
        request=reference,
        validation_manifest=inputs.holdout.manifest,
        comparison=comparison_ref,
    )
    metrics_ref = write_document(layout, result.parent / "metrics.json", metrics)
    receipt = EvaluationReceipt(
        request=reference,
        training=request.training,
        candidate=training.candidate,
        baseline=intent.parent,
        validation=inputs.fixed.validation,
        metrics=metrics_ref,
    )
    validate_evaluation(layout, intent, receipt)
    return write_document(layout, result, receipt)
