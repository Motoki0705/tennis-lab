"""Validate task outputs before shared lifecycle state can advance."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from typing import Any

import torch

from src.tasks.ball_detection.checkpoints.initialization import initialize_full_model
from src.tasks.ball_detection.checkpoints.posttraining import SCHEMA
from src.tasks.ball_detection.checkpoints.provenance import source_identity
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.evaluation.coordinates import selection_score
from src.tasks.ball_detection.evaluation.fixed_validation import (
    expected_support,
    verify_support,
)
from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.contracts import ArtifactRef, canonical_bytes
from src.tasks.base.annotation.learning.contracts import (
    EvaluationReceipt,
    LearningIntent,
    StageInput,
    TrainingReceipt,
)
from src.tasks.base.annotation.learning.validation import check_training, verify_inputs
from src.tasks.base.annotation.paths import AnnotationLayout

from .contracts import BallEvaluationMetrics, BallTrainingProvenance
from .inputs import Inputs, check_model_spec, output_reference, resolve_inputs, run_root


def checkpoint_value(
    layout: AnnotationLayout,
    intent: LearningIntent,
    inputs: Inputs,
    reference: ArtifactRef,
) -> dict[str, Any]:
    path = verify_reference(layout, reference)
    root = run_root(layout, intent)
    if (
        not path.is_relative_to(root)
        or len(path.relative_to(root).parts) != 2
        or not re.fullmatch(r"attempt-[0-9]+", path.parent.name)
        or int(path.parent.name.removeprefix("attempt-")) < 1
    ):
        raise ValueError("checkpoint does not belong to this ball learning intent")
    saved: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if (
        path.parent.name
        != f"attempt-{int(path.parent.name.removeprefix('attempt-')):03d}"
    ):
        raise ValueError("checkpoint attempt directory is not canonical")
    expected = inputs.expected_recipe(layout, intent)
    code = source_identity()
    if (
        saved.get("schema") != SCHEMA
        or saved.get("stage") != "query_posttraining"
        or saved["recipe"] != expected
        or saved["model_config"] != expected["model"]
    ):
        raise ValueError(
            "checkpoint initialization, snapshot or recipe differs from this intent"
        )
    if any(
        saved["code"][key] != code[key] for key in ("source_sha256", "torch_version")
    ):
        raise ValueError("checkpoint implementation/runtime changed")
    epoch = saved["epoch"]
    if (
        type(epoch) is not int
        or not 0 <= epoch < inputs.recipe.epochs
        or path.name != f"epoch-{epoch:03d}.pt"
        or saved["global_step"]
        != (epoch + 1)
        * math.ceil(inputs.recipe.windows_per_epoch / inputs.recipe.batch_size)
    ):
        raise ValueError("checkpoint epoch/update budget does not match its recipe")
    return saved


def validate_resume(
    layout: AnnotationLayout, intent: LearningIntent, reference: ArtifactRef
) -> None:
    inputs = resolve_inputs(layout, intent)
    saved = checkpoint_value(layout, intent, inputs, reference)
    path = verify_reference(layout, reference)
    if any(
        int(p.stem.removeprefix("epoch-")) > saved["epoch"]
        for p in path.parent.glob("epoch-*.pt")
    ):
        raise ValueError("resume requires the latest completed epoch")
    initialize_full_model(
        path,
        expected_sha256=reference.digest,
        jpeg_decoder=inputs.recipe.runtime.jpeg_decoder,
    )


def validate_training(
    layout: AnnotationLayout, intent: LearningIntent, receipt: TrainingReceipt
) -> None:
    inputs = resolve_inputs(layout, intent)
    verify_inputs(layout, receipt)
    request = read_document(layout, receipt.request, StageInput)
    if (
        request.stage != "train"
        or read_document(layout, request.intent, LearningIntent) != intent
    ):
        raise ValueError("training result belongs to a different stage/intent")
    provenance = read_document(layout, receipt.provenance, BallTrainingProvenance)
    verify_inputs(layout, provenance)
    if (
        provenance.request,
        provenance.intent,
        provenance.prepared,
        provenance.validation_manifest,
    ) != (
        receipt.request,
        request.intent,
        inputs.prepared_ref,
        inputs.holdout.manifest,
    ):
        raise ValueError("training provenance differs from the pinned stage inputs")
    if (
        receipt.candidate.version != "annotation-" + intent.sha256
        or receipt.candidate.training_weights != provenance.selected_checkpoint
    ):
        raise ValueError(
            "candidate does not identify this intent's selected checkpoint"
        )
    spec = check_model_spec(layout, receipt.candidate)
    selected = checkpoint_value(layout, intent, inputs, provenance.selected_checkpoint)
    latest = checkpoint_value(layout, intent, inputs, provenance.latest_checkpoint)
    run = verify_reference(layout, provenance.latest_checkpoint).parent
    run_attempt = int(run.name.removeprefix("attempt-"))
    if (
        run_attempt > request.attempt
        or (run_attempt != request.attempt and request.resume_checkpoint is None)
        or (
            request.resume_checkpoint is not None
            and verify_reference(layout, request.resume_checkpoint).parent != run
        )
    ):
        raise ValueError(
            "training run is not owned by this attempt or its explicit resume"
        )
    if (
        verify_reference(layout, provenance.selected_checkpoint).parent != run
        or provenance.run_config != output_reference(layout, run / "config.json")
        or provenance.best_record != output_reference(layout, run / "best.json")
        or provenance.completed != output_reference(layout, run / "COMPLETED.json")
    ):
        raise ValueError("training provenance mixes run outputs")
    completed = json.loads(verify_reference(layout, provenance.completed).read_bytes())
    best = json.loads(verify_reference(layout, provenance.best_record).read_bytes())
    config = json.loads(verify_reference(layout, provenance.run_config).read_bytes())
    expected = inputs.expected_recipe(layout, intent)
    if (
        canonical_bytes(config["recipe"]) != canonical_bytes(expected)
        or canonical_bytes(config["code"]) != canonical_bytes(latest["code"])
        or provenance.initial_state_sha256 != expected["initial_state_sha256"]
    ):
        raise ValueError("training config/initial weights provenance differs")
    if (
        latest["epoch"] != inputs.recipe.epochs - 1
        or latest["global_step"] != inputs.recipe.total_updates
        or completed
        != dict(
            stage="query_posttraining",
            global_step=inputs.recipe.total_updates,
            best_epoch=latest["best_epoch"],
            best_error_px=latest["best_score"],
            pretrained_checkpoint=intent.parent.model.training_weights.digest,
            initial_state_sha256=expected["initial_state_sha256"],
        )
    ):
        raise ValueError("training did not finish its declared update budget")
    if (
        best["checkpoint"] != f"epoch-{selected['epoch']:03d}.pt"
        or best["sha256"] != provenance.selected_checkpoint.digest
        or best["epoch"] != latest["best_epoch"]
        or best["global_step"] != selected["global_step"]
        or best["selection_error_px"] != latest["best_score"]
        or selection_score(selected["validation"], expected["selection_scope"]) != latest["best_score"]
    ):
        raise ValueError(
            "selected checkpoint differs from completed clean validation selection"
        )
    model, _ = initialize_full_model(
        verify_reference(layout, provenance.selected_checkpoint),
        expected_sha256=provenance.selected_checkpoint.digest,
        jpeg_decoder=inputs.recipe.runtime.jpeg_decoder,
    )
    if (
        canonical_bytes(asdict(model.config)) != canonical_bytes(spec.architecture)
        or canonical_bytes(model.mdd.input_contract())
        != canonical_bytes(spec.input_contract)
        or spec.training_manifest_digest != expected["manifest_sha256"]
    ):
        raise ValueError(
            "candidate inference specification differs from its full training checkpoint"
        )
    if inputs.recipe.stress_evaluation != (provenance.stress is not None):
        raise ValueError("stress evaluation evidence differs from the recipe")
    if provenance.stress is not None:
        if provenance.stress != output_reference(layout, run / "stress.json"):
            raise ValueError("stress evidence belongs to another run")
        if set(
            json.loads(verify_reference(layout, provenance.stress).read_bytes())
        ) != {"camera", "occlusion", "combined"}:
            raise ValueError("stress evaluation is incomplete")


def validate_evaluation(
    layout: AnnotationLayout, intent: LearningIntent, receipt: EvaluationReceipt
) -> dict[str, Any]:
    inputs = resolve_inputs(layout, intent)
    final_test = inputs.fixed.recipe is not None
    holdout = inputs.test_holdout if final_test else inputs.holdout
    if holdout is None:
        raise ValueError("Final test holdout is missing")
    verify_inputs(layout, receipt)
    request = read_document(layout, receipt.request, StageInput)
    if (
        request.stage != "evaluate"
        or read_document(layout, request.intent, LearningIntent) != intent
        or request.training != receipt.training
        or receipt.baseline != intent.parent
        or receipt.validation != inputs.fixed.validation
        or receipt.test != (inputs.fixed.test if final_test else None)
    ):
        raise ValueError("evaluation baseline, stage or fixed validation changed")
    training_request, training = check_training(layout, receipt.training)
    if (
        training_request.intent != request.intent
        or training.candidate != receipt.candidate
    ):
        raise ValueError("evaluation substituted a different completed candidate")
    validate_training(layout, intent, training)
    metrics = read_document(layout, receipt.metrics, BallEvaluationMetrics)
    if (
        metrics.request != receipt.request
        or metrics.validation_manifest != inputs.holdout.manifest
        or metrics.test_manifest != (holdout.manifest if final_test else None)
    ):
        raise ValueError("evaluation metrics refer to different fixed inputs")
    raw: dict[str, Any] = json.loads(
        verify_reference(layout, metrics.comparison).read_bytes()
    )
    runtime = inputs.recipe.runtime
    if (
        raw["schema"],
        raw["split"],
        raw["profile"],
        raw["test_usage"],
        raw["validation_manifest_sha256"],
    ) != (
        "ball.fixed-validation-comparison.v1",
        "test" if final_test else "val",
        "clean",
        "final_comparison" if final_test else "none",
        holdout.manifest.digest,
    ):
        raise ValueError("evaluation substituted its fixed validation contract")
    if (
        raw["runtime"] != asdict(runtime.coordinate_runtime())
        or raw["device"] != runtime.device
        or raw["batch_size"] != inputs.recipe.batch_size
    ):
        raise ValueError("evaluation runtime differs from the pinned recipe")
    data = FrozenBallWindowDataset(
        verify_reference(layout, holdout.manifest),
        split="test" if final_test else "val",
        requires_pose=False,
        jpeg_decoder=runtime.jpeg_decoder,
    )
    support = expected_support(data)
    if raw["support"] != support:
        raise ValueError("evaluation support differs from fixed teachers")
    scopes = ("full", "existing", "new", "balanced_existing_new") if final_test else ("common", "full")
    for name, model in (
        ("baseline", intent.parent.model),
        ("candidate", receipt.candidate),
    ):
        spec = check_model_spec(layout, model)
        row = raw[name]
        _, identity = initialize_full_model(
            verify_reference(layout, model.inference_weights),
            expected_sha256=model.inference_weights.digest,
            jpeg_decoder=runtime.jpeg_decoder,
        )
        if (
            row["checkpoint_sha256"] != model.inference_weights.digest
            or row["initial_state_sha256"] != identity["state_dict_sha256"]
            or row["evaluation"]["precision"] != runtime.precision
            or row["training_manifest_sha256"] != spec.training_manifest_digest
            or canonical_bytes(raw["architecture"])
            != canonical_bytes(spec.architecture)
            or canonical_bytes(raw["input_contract"])
            != canonical_bytes(spec.input_contract)
        ):
            raise ValueError("evaluation model identities differ")
        verify_support(row["evaluation"], support)
        for scope in scopes:
            if not math.isfinite(selection_score(row["evaluation"], scope)):
                raise ValueError("evaluation contains nonfinite selection metrics")
    expected_delta = {
        scope: selection_score(raw["candidate"]["evaluation"], scope)
        - selection_score(raw["baseline"]["evaluation"], scope)
        for scope in scopes
    }
    if raw["candidate_minus_baseline_px"] != expected_delta:
        raise ValueError("evaluation differences are inconsistent")
    return raw
