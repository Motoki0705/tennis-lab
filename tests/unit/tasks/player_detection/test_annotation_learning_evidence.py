from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    file_reference,
    read_document,
)
from src.tasks.base.annotation.execution.contracts import StageLaunch
from src.tasks.base.annotation.learning.contracts import TrainingReceipt
from src.tasks.base.annotation.learning.execution import stage_request
from src.tasks.player_detection.data.snapshots.contracts import ExistingPlayerSnapshot
from src.tasks.player_detection.training.annotation.checkpoints import verify_export
from src.tasks.player_detection.training.annotation.contracts import (
    ModelEvaluation,
    PlayerLearningRecipe,
    PlayerSelectionPolicy,
)
from src.tasks.player_detection.training.annotation.evaluation import (
    evaluation_config,
    rejection_reasons,
    verify_model_evaluation,
)
from src.tasks.player_detection.training.annotation.inputs import resolve_inputs
from tests.support.tasks.player_detection.chat_root import SyntheticRoot
from tests.support.tasks.player_detection.learning import (
    Environment,
    tiny_evaluation,
    use_cpu_dino,
)


def test_learning_retains_native_shards_fixed_supports_and_training_settings(
    synthetic_root: SyntheticRoot,
) -> None:
    env = Environment(synthetic_root)
    inputs = resolve_inputs(env.layout, env.intent)
    snapshot = read_document(
        env.layout, env.intent.train.manifest, ExistingPlayerSnapshot
    )
    retained = set(env.learning.storage.inputs(env.intent_ref))
    assert {
        snapshot.index,
        snapshot.metadata,
        *(item.shard for item in snapshot.clips),
        inputs.train.manifest,
        inputs.fixed.validation.manifest,
        inputs.recipe.training,
    } <= retained
    assert inputs.fixed.test is not None
    assert inputs.fixed.test.manifest in retained


def test_missing_registered_support_is_not_regenerated(
    synthetic_root: SyntheticRoot,
) -> None:
    env = Environment(synthetic_root)
    inputs = resolve_inputs(env.layout, env.intent)
    path = env.layout.artifact_path(inputs.train.manifest)
    path.unlink()
    with pytest.raises(ValueError, match="lost its pinned"):
        resolve_inputs(env.layout, env.intent)
    assert not path.exists()


def test_stage_requires_supervisor_digest_task_and_owned_result_path(
    synthetic_root: SyntheticRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = Environment(synthetic_root)
    started = env.tick()
    assert started.launch is not None
    launch = read_document(env.layout, started.launch, StageLaunch)
    assert launch.input is not None
    result = Path(launch.command[-1])
    monkeypatch.delenv("TENNIS_ANNOTATION_STAGE_OPERATION", raising=False)
    with pytest.raises(ValueError, match="supervisor-owned"):
        stage_request(env.layout, launch.input, result, task="player")
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", launch.input.digest)
    with pytest.raises(ValueError, match="different task"):
        stage_request(env.layout, launch.input, result, task="ball")
    with pytest.raises(ValueError, match="outside its stage"):
        stage_request(
            env.layout, launch.input, result.parent / "other.json", task="player"
        )


def test_export_tensor_swap_is_rejected_even_with_valid_parent_metadata(
    synthetic_root: SyntheticRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = Environment(synthetic_root)
    use_cpu_dino(monkeypatch)
    started = env.tick()
    assert started.launch is not None
    receipt = read_document(
        env.layout, env.executor.finish(started.launch), TrainingReceipt
    )
    source = env.layout.artifact_path(receipt.candidate.inference_weights)
    value = torch.load(source, weights_only=False, map_location="cpu")
    value["model"]["weight"] += 100.0
    forged_path = source.parent / "forged.pth"
    torch.save(value, forged_path)
    forged = file_reference(env.layout, forged_path, role="checkpoint")
    with pytest.raises(ValueError, match="tensor state differs"):
        verify_export(env.layout, receipt.candidate.training_weights, forged)


@pytest.mark.parametrize(
    "mutation", ["order", "missing", "nan", "confidence", "geometry", "metrics"]
)
def test_validation_recomputes_metrics_and_rejects_changed_support(
    synthetic_root: SyntheticRoot,
    mutation: str,
) -> None:
    env = Environment(synthetic_root)
    inputs = resolve_inputs(env.layout, env.intent)
    output = env.layout.path("models", "validation-fixture")
    config = evaluation_config(env.layout, inputs, env.parent, env.parent, output)
    tiny_evaluation(config)
    prediction_path = output / "candidate/predictions.npz"
    metrics_path = output / "candidate/metrics.json"
    if mutation == "metrics":
        value = json.loads(metrics_path.read_bytes())
        value["diagnostics"]["negative_false_positives"] += 1
        metrics_path.write_text(json.dumps(value))
    else:
        with np.load(prediction_path) as saved:
            arrays = {key: saved[key] for key in saved.files}
        if mutation == "order":
            arrays["frame_keys"] = arrays["frame_keys"][::-1]
        elif mutation == "missing":
            arrays = {key: value[:-1] for key, value in arrays.items()}
        elif mutation == "nan":
            arrays["boxes_xyxy"][0, 0, 0] = np.nan
        elif mutation == "confidence":
            arrays["scores"][0, 0] = 1.1
        else:
            arrays["boxes_xyxy"][0, 0, 0] = 1000.0
        np.savez_compressed(prediction_path, **arrays)
    evidence = ModelEvaluation(
        model=env.parent,
        predictions=annotation_reference(env.layout, prediction_path),
        metrics=annotation_reference(env.layout, metrics_path),
    )
    with pytest.raises(ValueError, match="support|invalid native|metrics differ"):
        verify_model_evaluation(env.layout, inputs, evidence)


def test_negative_support_and_fp_are_independent_selection_requirements() -> None:
    policy = PlayerSelectionPolicy(
        minimum_validation_frames=2,
        minimum_target_instances=1,
        minimum_map=0.5,
        minimum_map_improvement=0.1,
        maximum_precision_regression=0.2,
        maximum_recall_regression=0.2,
        maximum_negative_fp_per_image=0.5,
        minimum_negative_frames=2,
    )
    before = {"map": 0.5, "precision": 1.0, "recall": 1.0}
    after = {"map": 0.75, "precision": 0.9, "recall": 0.9}
    support = {
        "images": 3.0,
        "target_instances": 1.0,
        "negative_images": 2.0,
        "negative_false_positives": 1.0,
    }
    assert rejection_reasons(before, after, support, policy) == ()
    assert "confirmed-absence FP rate exceeds" in " ".join(
        rejection_reasons(
            before, after, {**support, "negative_false_positives": 2.0}, policy
        )
    )
    assert "unmeasured" in " ".join(
        rejection_reasons(before, after, {**support, "negative_images": 0.0}, policy)
    )
    with pytest.raises(ValueError, match="positive minimum support"):
        PlayerSelectionPolicy.model_validate(
            {**policy.model_dump(), "minimum_negative_frames": 0}
        )


def test_recipe_rejects_partial_batches_and_unspecified_or_incompatible_inputs(
    synthetic_root: SyntheticRoot,
) -> None:
    env = Environment(synthetic_root)
    raw = env.recipe.model_dump(mode="json")
    with pytest.raises(ValueError, match="full batch"):
        PlayerLearningRecipe.model_validate_json(json.dumps({**raw, "batch_size": 10}))
    with pytest.raises(ValueError, match="complete eligible"):
        PlayerLearningRecipe.model_validate_json(
            json.dumps({**raw, "selection": {**raw["selection"], "frame_stride": 2}})
        )
    del raw["train_samples_per_epoch"]
    with pytest.raises(ValueError, match="Field required"):
        PlayerLearningRecipe.model_validate_json(json.dumps(raw))
