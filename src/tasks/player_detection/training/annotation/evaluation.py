"""Same fixed validation, native prediction rows, and explicit negative-frame FP."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, ModelRef, canonical_bytes
from src.tasks.base.annotation.learning.contracts import (
    EvaluationReceipt,
    LearningIntent,
    StageInput,
)
from src.tasks.base.annotation.learning.validation import check_training
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.player_detection.configuration import EvaluateConfig
from src.tasks.player_detection.data.detection_dataset import select_detection_frames
from src.tasks.player_detection.data.snapshots.store import load_snapshot
from src.tasks.player_detection.evaluation.evaluator import evaluate_checkpoints
from src.tasks.player_detection.evaluation.metrics import (
    HEADLINE_METRICS,
    PlayerDetectionMetrics,
)
from src.tasks.player_detection.models.dino_detector import FrameDetections

from .checkpoints import validate_training
from .contracts import (
    ModelEvaluation,
    PlayerComparison,
    PlayerEvaluationMetrics,
    PlayerSelectionPolicy,
)
from .inputs import Inputs, resolve_inputs


def evaluation_config(
    layout: AnnotationLayout,
    inputs: Inputs,
    baseline: ModelRef,
    candidate: ModelRef,
    output: Path,
) -> EvaluateConfig:
    snapshot, _ = load_snapshot(layout, inputs.validation.snapshot)
    return EvaluateConfig(
        checkpoints={
            "baseline": verify_reference(layout, baseline.inference_weights),
            "candidate": verify_reference(layout, candidate.inference_weights),
        },
        repository=layout.repository / "third_party/DINO",
        dataset_dir=verify_reference(layout, snapshot.index).parent,
        split="val",
        frame_stride=inputs.validation.policy.frame_stride,
        selection=inputs.validation.policy.selection(),
        input_size=inputs.recipe.input_size.configuration(),
        evaluation=inputs.recipe.operating_point.configuration(),
        num_workers=inputs.recipe.num_workers,
        overlay_frames=0,
        output_dir=output,
        device=inputs.recipe.device,
    )


def _read_json(layout: AnnotationLayout, reference: ArtifactRef) -> dict[str, Any]:
    value = json.loads(verify_reference(layout, reference).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("player evaluation evidence must be a JSON mapping")
    return value


def verify_model_evaluation(
    layout: AnnotationLayout, inputs: Inputs, evidence: ModelEvaluation
) -> tuple[dict[str, float], dict[str, float]]:
    _, store = load_snapshot(layout, inputs.validation.snapshot)
    selection = select_detection_frames(
        store,
        "val",
        inputs.validation.policy.selection(),
        frame_stride=inputs.validation.policy.frame_stride,
    )
    expected_keys = [store.frame_key(int(row)) for row in selection.frames]
    count = len(expected_keys)
    with np.load(
        verify_reference(layout, evidence.predictions), allow_pickle=False
    ) as arrays:
        if set(arrays.files) != {"boxes_xyxy", "scores", "frame_keys"}:
            raise ValueError("player validation prediction columns differ")
        boxes, scores, keys = (
            arrays["boxes_xyxy"],
            arrays["scores"],
            arrays["frame_keys"],
        )
    if (
        keys.tolist() != expected_keys
        or boxes.shape != (count, inputs.recipe.operating_point.max_detections, 4)
        or scores.shape != boxes.shape[:2]
        or not np.issubdtype(boxes.dtype, np.floating)
        or not np.issubdtype(scores.dtype, np.floating)
        or not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
        or np.any(boxes < 0)
        or np.any(boxes[..., 2:] < boxes[..., :2])
    ):
        raise ValueError(
            "player validation predictions have different support or invalid native values"
        )
    metrics = PlayerDetectionMetrics(inputs.recipe.operating_point.configuration())
    for index, row in enumerate(selection.frames):
        clip = store.clip_of(int(row))
        if np.any(boxes[index, :, (0, 2)] > clip.width) or np.any(
            boxes[index, :, (1, 3)] > clip.height
        ):
            raise ValueError("player validation boxes exceed native image dimensions")
        metrics.update(
            [
                FrameDetections(
                    torch.from_numpy(boxes[index]), torch.from_numpy(scores[index])
                )
            ],
            [torch.from_numpy(selection.boxes_xyxy[index])],
        )
    headline, diagnostics = metrics.compute()
    saved = _read_json(layout, evidence.metrics)
    if saved.get("checkpoint") != str(
        verify_reference(layout, evidence.model.inference_weights)
    ) or set(saved) != {"checkpoint", "headline", "diagnostics"}:
        raise ValueError("player validation metrics belong to another model")
    for section, expected in (("headline", headline), ("diagnostics", diagnostics)):
        actual = saved[section]
        if (
            not isinstance(actual, dict)
            or set(actual) != set(expected)
            or any(
                type(actual[name]) not in (int, float)
                or not math.isfinite(actual[name])
                or not math.isclose(actual[name], value, rel_tol=1e-7, abs_tol=1e-8)
                for name, value in expected.items()
            )
        ):
            raise ValueError(
                "player validation metrics differ from the saved native predictions"
            )
    if any(not 0 <= headline[name] <= 1 for name in HEADLINE_METRICS) or (
        diagnostics["images"],
        diagnostics["target_instances"],
        diagnostics["negative_images"],
    ) != (
        inputs.validation.selected_frames,
        inputs.validation.selected_instances,
        inputs.validation.negative_frames,
    ):
        raise ValueError(
            "player validation metrics are undefined or have changed support"
        )
    return headline, diagnostics


def validate_evaluation(
    layout: AnnotationLayout, intent: LearningIntent, receipt: EvaluationReceipt
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    inputs = resolve_inputs(layout, intent)
    request = read_document(layout, receipt.request, StageInput)
    training_request, training = check_training(layout, receipt.training)
    if (
        request.stage != "evaluate"
        or request.training != receipt.training
        or training_request.intent != request.intent
        or read_document(layout, request.intent, LearningIntent) != intent
        or receipt.baseline != intent.parent
        or receipt.candidate != training.candidate
        or receipt.validation != inputs.fixed.validation
    ):
        raise ValueError(
            "player evaluation is not bound to the intended training and fixed validation"
        )
    validate_training(layout, intent, training)
    metrics = read_document(layout, receipt.metrics, PlayerEvaluationMetrics)
    comparison = read_document(layout, metrics.comparison, PlayerComparison)
    if (
        metrics.request != receipt.request
        or comparison.request != receipt.request
        or comparison.validation_partition != receipt.validation.manifest
        or comparison.support_digest != inputs.validation.support_digest
        or comparison.baseline.model != intent.parent.model
        or comparison.candidate.model != training.candidate
    ):
        raise ValueError(
            "player comparison identities differ from the evaluation receipt"
        )
    config = evaluation_config(
        layout,
        inputs,
        receipt.baseline.model,
        receipt.candidate,
        verify_reference(layout, comparison.provenance).parent,
    )
    snapshot, store = load_snapshot(layout, inputs.validation.snapshot)
    selected = select_detection_frames(
        store, "val", config.selection, frame_stride=config.frame_stride
    )
    selected_ids = {store.clip_of(int(row)).index for row in selected.frames}
    files = (
        snapshot.index,
        snapshot.metadata,
        *(snapshot.clips[index].shard for index in sorted(selected_ids)),
    )
    provenance = _read_json(layout, comparison.provenance)
    expected = {
        "schema": "player_detection_evaluation_v1",
        "checkpoints": {
            name: {"path": str(path), "sha256": model.inference_weights.digest}
            for (name, path), model in zip(
                config.checkpoints.items(),
                (intent.parent.model, training.candidate),
                strict=True,
            )
        },
        "dataset": {
            "path": str(config.dataset_dir),
            "sha256": {
                str(
                    verify_reference(layout, item).relative_to(config.dataset_dir)
                ): item.digest
                for item in files
            },
        },
        "input_size": asdict(config.input_size),
        "selection": asdict(config.selection),
        "evaluation": asdict(config.evaluation),
        "split": "val",
        "frame_stride": config.frame_stride,
        "device": inputs.recipe.device,
        "matmul_precision": inputs.training.matmul_precision,
        "cuda_matmul_allow_tf32": inputs.training.allow_tf32,
        "cudnn_allow_tf32": inputs.training.allow_tf32,
    }
    if canonical_bytes(provenance) != canonical_bytes(expected):
        raise ValueError(
            "player comparison provenance/runtime or dataset inputs changed"
        )
    before, before_support = verify_model_evaluation(
        layout, inputs, comparison.baseline
    )
    after, support = verify_model_evaluation(layout, inputs, comparison.candidate)
    for field in (
        "images",
        "target_instances",
        "negative_images",
        "score_threshold",
        "iou_threshold",
    ):
        if before_support[field] != support[field]:
            raise ValueError(
                "player baseline and candidate have different evaluation support"
            )
    expected_summary = {
        "dataset": str(config.dataset_dir),
        "split": "val",
        "frame_stride": config.frame_stride,
        "selection_stats": selected.stats,
        "metrics": {"baseline": before, "candidate": after},
    }
    if canonical_bytes(_read_json(layout, comparison.summary)) != canonical_bytes(
        expected_summary
    ):
        raise ValueError("player comparison summary differs from verified predictions")
    return before, after, support


def rejection_reasons(
    before: dict[str, float],
    after: dict[str, float],
    support: dict[str, float],
    policy: PlayerSelectionPolicy,
) -> tuple[str, ...]:
    reasons = []
    if (
        support["images"] < policy.minimum_validation_frames
        or support["target_instances"] < policy.minimum_target_instances
    ):
        reasons.append("fixed validation support is below the agreed minimum")
    if (
        after["map"] < policy.minimum_map
        or after["map"] - before["map"] < policy.minimum_map_improvement
    ):
        reasons.append("validation mAP or improvement is below the agreed minimum")
    for metric, limit in (
        ("precision", policy.maximum_precision_regression),
        ("recall", policy.maximum_recall_regression),
    ):
        if before[metric] - after[metric] > limit:
            reasons.append(f"validation {metric} regression exceeds the agreed maximum")
    if support["negative_images"] < policy.minimum_negative_frames:
        reasons.append(
            "confirmed-absence validation support is below the agreed minimum"
        )
    if policy.maximum_negative_fp_per_image is not None:
        if support["negative_images"] == 0:
            reasons.append("confirmed-absence FP rate is unmeasured")
        elif (
            support["negative_false_positives"] / support["negative_images"]
            > policy.maximum_negative_fp_per_image
        ):
            reasons.append("confirmed-absence FP rate exceeds the agreed maximum")
    return tuple(reasons)


def evaluate(
    layout: AnnotationLayout,
    reference: ArtifactRef,
    request: StageInput,
    intent: LearningIntent,
    result: Path,
) -> ArtifactRef:
    if result.exists():
        existing = annotation_reference(layout, result)
        receipt = read_document(layout, existing, EvaluationReceipt)
        if receipt.request != reference:
            raise ValueError("player evaluation belongs to another attempt")
        validate_evaluation(layout, intent, receipt)
        return existing
    if request.training is None:
        raise ValueError("player evaluation requires completed training")
    training_request, training = check_training(layout, request.training)
    if training_request.intent != request.intent:
        raise ValueError("player evaluation cannot use another intent's training")
    validate_training(layout, intent, training)
    inputs = resolve_inputs(layout, intent)
    output = result.parent / "validation"
    config = evaluation_config(
        layout, inputs, intent.parent.model, training.candidate, output
    )
    if not (output / "summary.json").exists():
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                "partial player evaluation requires a new explicit attempt"
            )
        torch.set_float32_matmul_precision(inputs.training.matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = inputs.training.allow_tf32
        torch.backends.cudnn.allow_tf32 = inputs.training.allow_tf32
        evaluate_checkpoints(config)
    comparison = PlayerComparison(
        request=reference,
        validation_partition=inputs.fixed.validation.manifest,
        support_digest=inputs.validation.support_digest,
        provenance=annotation_reference(layout, output / "provenance.json"),
        summary=annotation_reference(layout, output / "summary.json"),
        baseline=ModelEvaluation(
            model=intent.parent.model,
            predictions=annotation_reference(
                layout, output / "baseline/predictions.npz"
            ),
            metrics=annotation_reference(layout, output / "baseline/metrics.json"),
        ),
        candidate=ModelEvaluation(
            model=training.candidate,
            predictions=annotation_reference(
                layout, output / "candidate/predictions.npz"
            ),
            metrics=annotation_reference(layout, output / "candidate/metrics.json"),
        ),
    )
    comparison_ref = write_document(
        layout, result.parent / "comparison.json", comparison
    )
    metrics_ref = write_document(
        layout,
        result.parent / "metrics.json",
        PlayerEvaluationMetrics(request=reference, comparison=comparison_ref),
    )
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
