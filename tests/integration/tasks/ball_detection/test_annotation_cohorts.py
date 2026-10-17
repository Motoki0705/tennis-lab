from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.training.annotation.contracts import (
    BalancedSelectionPolicy,
    BallEvaluationMetrics,
    BallTrainingProvenance,
)
from src.tasks.ball_detection.training.annotation.inputs import resolve_inputs
from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.learning.contracts import (
    EvaluationReceipt,
    TrainingReceipt,
)
from tests.support.tasks.ball_detection.learning_pipeline import LearningEnvironment


def test_published_mixed_splits_replay_train_validate_and_final_test(tmp_path: Path) -> None:
    env = LearningEnvironment(tmp_path, cohorts=True, selection_policy=BalancedSelectionPolicy(minimum_frames_per_fps=1))
    inputs = resolve_inputs(env.layout, env.intent)
    assert inputs.fixed.recipe == env.review.plan.tasks[0].learning.evaluation_recipe
    assert inputs.fixed.original == env.review.plan.tasks[0].learning.validation
    assert env.intent.fixed_evaluation != inputs.fixed.original
    assert inputs.test_holdout is not None
    assert inputs.fixed.test is not None
    assert inputs.prepared.added_train_clips == 1
    assert len(inputs.fixed.validation.clips) == len(inputs.fixed.test.clips) == 2
    original_ref = inputs.fixed.original
    original_bytes = env.layout.artifact_path(original_ref).read_bytes()
    assert env.learning.register(env.completion, env.adapter).intent == env.intent_ref
    assert env.layout.artifact_path(original_ref).read_bytes() == original_bytes
    training = read_document(env.layout, env.executor.finish(env.tick().launch), TrainingReceipt)
    provenance = read_document(env.layout, training.provenance, BallTrainingProvenance)
    run = env.layout.artifact_path(provenance.latest_checkpoint).parent
    checkpoint = torch.load(env.layout.artifact_path(provenance.latest_checkpoint), weights_only=True, map_location="cpu")
    assert checkpoint["global_step"] == 10
    assert checkpoint["recipe"]["selection_scope"] == "balanced_existing_new"
    assert checkpoint["recipe"]["test_usage"] == "none"
    sampling = json.loads((run / "sampling-epoch-001.json").read_text())
    assert {r["group"]: r["draws"] for r in sampling["groups"]} == {"untrained": 3, "replay": 2}
    evaluation = read_document(env.layout, env.executor.finish(env.tick().launch), EvaluationReceipt)
    assert evaluation.test == inputs.fixed.test
    metrics = read_document(env.layout, evaluation.metrics, BallEvaluationMetrics)
    assert metrics.test_manifest == inputs.test_holdout.manifest
    raw = json.loads(env.layout.artifact_path(metrics.comparison).read_bytes())
    assert raw["split"] == "test" and raw["test_usage"] == "final_comparison"
    assert {"existing", "new"} <= raw["support"].keys()
    assert raw["baseline"]["evaluation"]["scopes"]["existing"]["membership_sha256"] == raw["candidate"]["evaluation"]["scopes"]["existing"]["membership_sha256"]
    outcome = env.tick()
    assert outcome.state in {"adopted", "rejected"}
    assert len(env.executor.submitted) == 2
    assert env.tick() == outcome
    # A forged final receipt cannot substitute the validation cohort for test.
    forged = evaluation.model_copy(update={"test": inputs.fixed.validation})
    with pytest.raises(ValueError, match="fixed validation changed"):
        env.adapter.validate_evaluation(env.layout, env.intent, forged)


@pytest.mark.parametrize("entrypoint", ["resolve_inputs", "register"])
@pytest.mark.parametrize("split", ["validation", "test"])
def test_lost_resolved_holdout_is_rejected_without_recreation(
    tmp_path: Path, entrypoint: str, split: str
) -> None:
    env = LearningEnvironment(tmp_path, cohorts=True)
    inputs = resolve_inputs(env.layout, env.intent)
    assert inputs.test_holdout is not None
    holdout = inputs.holdout if split == "validation" else inputs.test_holdout
    path = env.layout.artifact_path(holdout.manifest)
    path.unlink()
    with pytest.raises(ValueError, match="missing or size changed"):
        if entrypoint == "register":
            env.learning.register(env.completion, env.adapter)
        else:
            resolve_inputs(env.layout, env.intent)
    assert not path.exists()
