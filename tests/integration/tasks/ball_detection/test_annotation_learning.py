from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.evaluation.coordinates import selection_score
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallSnapshot,
)
from src.tasks.ball_detection.training.annotation.contracts import (
    BallEvaluationMetrics,
    BallSelectionPolicy,
    BallTrainingProvenance,
)
from src.tasks.ball_detection.training.annotation.evidence import validate_resume
from src.tasks.ball_detection.training.annotation.inputs import (
    output_reference,
    resolve_inputs,
    run_root,
)
from src.tasks.ball_detection.training.annotation.stages import execute
from src.tasks.ball_detection.training.coordinates import joint
from src.tasks.base.annotation.artifacts import read_document, write_asset
from src.tasks.base.annotation.execution.contracts import StageLaunch
from src.tasks.base.annotation.learning.contracts import (
    EvaluationReceipt,
    StageInput,
    TrainingReceipt,
)
from tests.support.tasks.ball_detection.learning_pipeline import (
    LearningEnvironment,
    save,
)


@pytest.mark.parametrize("configured", [False, True])
def test_published_teachers_train_and_evaluate_through_shared_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    # Deliberately impossible policy is test data, never a proposed production threshold.
    policy = (
        BallSelectionPolicy(
            minimum_common_frames_per_fps=1,
            maximum_common_error_px=0.0,
            minimum_improvement_px=0.0,
            maximum_full_regression_px=0.0,
            source_regression_limits=(),
        )
        if configured
        else None
    )
    env = LearningEnvironment(tmp_path, selection_policy=policy)
    snapshot = read_document(env.layout, env.intent.train.manifest, BallSnapshot)
    initial_inputs = resolve_inputs(env.layout, env.intent)
    retained_inputs = set(env.learning.storage.inputs(env.intent_ref))
    assert {
        snapshot.index,
        snapshot.metadata,
        *(item.shard for item in snapshot.clips),
        initial_inputs.prepared_ref,
        initial_inputs.prepared.manifest,
        initial_inputs.holdout.manifest,
        initial_inputs.recipe.augmentation,
    } <= retained_inputs
    outcome = env.tick()
    assert outcome.state == "training_pending" and len(env.executor.submitted) == 1
    assert env.tick().launch == outcome.launch and len(env.executor.submitted) == 1
    launch = read_document(env.layout, outcome.launch, StageLaunch)
    assert launch.input is not None
    result_path = Path(launch.command[-1])
    with pytest.raises(ValueError, match="supervisor-owned"):
        execute(env.layout, launch.input, result_path)
    trained = env.executor.finish(outcome.launch)
    receipt = read_document(env.layout, trained, TrainingReceipt)
    provenance = read_document(env.layout, receipt.provenance, BallTrainingProvenance)
    latest = torch.load(
        env.layout.artifact_path(provenance.latest_checkpoint),
        map_location="cpu",
        weights_only=True,
    )
    assert latest["epoch"] == 1 and latest["global_step"] == 2
    assert latest["recipe"]["optimizer_initialization"] == "fresh"
    assert latest["recipe"]["parent"]["learning_intent"] == env.intent.sha256
    assert (
        latest["recipe"]["parent"]["parent_manifest_sha256"]
        != latest["recipe"]["manifest_sha256"]
    )
    inputs = resolve_inputs(env.layout, env.intent)
    assert inputs.prepared.old_train_clips == inputs.prepared.added_train_clips == 1
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", launch.input.digest)
    with pytest.raises(ValueError, match="outside its stage attempt"):
        execute(env.layout, launch.input, result_path.parent / "another-result.json")
    assert execute(env.layout, launch.input, result_path) == trained
    assert (
        output_reference(
            env.layout, env.layout.artifact_path(provenance.latest_checkpoint)
        )
        == provenance.latest_checkpoint
    )
    evaluated = env.tick()
    assert evaluated.state == "evaluation_pending" and len(env.executor.submitted) == 2
    env.executor.finish(evaluated.launch)
    finished = env.tick()
    assert finished.state == ("rejected" if configured else "not_configured")
    assert finished.training is not None and finished.evaluation is not None
    assert {
        provenance.latest_checkpoint,
        provenance.selected_checkpoint,
        provenance.completed,
        provenance.run_config,
        provenance.best_record,
    } <= set(env.learning.storage.completed_inputs(finished.training))
    assert env.learning.models.current("ball").model == env.parent
    assert env.tick() == finished and len(env.executor.submitted) == 2
    validate_resume(env.layout, env.intent, provenance.latest_checkpoint)
    assert env.learning.register(env.completion, env.adapter).intent == env.intent_ref


@pytest.mark.parametrize("lost_finalization", [False, True])
def test_explicit_retry_recovers_completed_run_without_extra_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost_finalization: bool
) -> None:
    env = LearningEnvironment(tmp_path)
    started = env.tick()
    launch = read_document(env.layout, started.launch, StageLaunch)
    assert launch.input is not None
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", launch.input.digest)
    trained = execute(env.layout, launch.input, Path(launch.command[-1]))
    receipt = read_document(env.layout, trained, TrainingReceipt)
    provenance = read_document(env.layout, receipt.provenance, BallTrainingProvenance)
    # Stage execution failed after the model completed. A new attempt must name
    # this exact run; no implicit latest-checkpoint lookup is allowed.
    if lost_finalization:
        env.layout.artifact_path(provenance.completed).unlink()
    env.executor.record_exit(started.launch, failed=True)
    assert env.tick().state == "failed"
    retry = env.learning.retry(
        env.intent_ref,
        env.adapter,
        "train",
        reason="recover completed CPU fixture",
        resume_checkpoint=provenance.latest_checkpoint,
    )
    assert read_document(env.layout, retry, StageInput).attempt == 2
    restarted = env.tick()
    recovered = read_document(
        env.layout, env.executor.finish(restarted.launch), TrainingReceipt
    )
    assert recovered.candidate == receipt.candidate
    assert (
        read_document(
            env.layout, recovered.provenance, BallTrainingProvenance
        ).latest_checkpoint
        == provenance.latest_checkpoint
    )
    run = env.layout.artifact_path(provenance.latest_checkpoint).parent
    assert len(list(run.glob("epoch-*.pt"))) == 2
    assert (run / "COMPLETED.json").exists()


def test_altered_input_or_checkpoint_is_not_resumed(tmp_path: Path) -> None:
    env = LearningEnvironment(tmp_path)
    started = env.tick()
    trained = env.executor.finish(started.launch)
    receipt = read_document(env.layout, trained, TrainingReceipt)
    provenance = read_document(env.layout, receipt.provenance, BallTrainingProvenance)
    path = env.layout.artifact_path(provenance.latest_checkpoint)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    saved["recipe"]["parent"]["training_snapshot"] = "f" * 64
    torch.save(saved, path)
    forged = output_reference(env.layout, path)
    with pytest.raises(ValueError, match="snapshot or recipe differs"):
        validate_resume(env.layout, env.intent, forged)
    with pytest.raises(ValueError, match="missing or size changed|hash changed"):
        env.adapter.validate_training(env.layout, env.intent, receipt)
    inputs = resolve_inputs(env.layout, env.intent)
    assert inputs.prepared.manifest is not None
    env.layout.artifact_path(inputs.prepared.manifest).unlink()
    with pytest.raises(ValueError, match="missing or size changed"):
        resolve_inputs(env.layout, env.intent)
    assert not env.layout.artifact_path(inputs.prepared.manifest).exists()


@pytest.mark.parametrize("cohorts", [False, True])
def test_interrupted_epoch_resumes_only_remaining_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cohorts: bool
) -> None:
    env = LearningEnvironment(tmp_path, cohorts=cohorts)
    started = env.tick()
    original = joint.train_epoch

    def interrupted(*args, **kwargs):
        if kwargs["epoch"] == 1:
            raise RuntimeError("synthetic interruption before second epoch")
        return original(*args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(joint, "train_epoch", interrupted)
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            env.executor.finish(started.launch)
    first = output_reference(
        env.layout, run_root(env.layout, env.intent) / "attempt-001/epoch-000.pt"
    )
    interrupted_checkpoint = torch.load(
        env.layout.artifact_path(first), weights_only=True, map_location="cpu"
    )
    scope = "balanced_existing_new" if cohorts else "common"
    assert interrupted_checkpoint["recipe"]["selection_scope"] == scope
    assert interrupted_checkpoint["best_score"] == selection_score(
        interrupted_checkpoint["validation"], scope
    )
    if cohorts:
        assert interrupted_checkpoint["best_score"] != selection_score(
            interrupted_checkpoint["validation"], "common"
        )
    env.executor.record_exit(started.launch, failed=True)
    assert env.tick().state == "failed"
    env.learning.retry(
        env.intent_ref,
        env.adapter,
        "train",
        reason="resume interrupted CPU fixture",
        resume_checkpoint=first,
    )
    recovered = read_document(
        env.layout, env.executor.finish(env.tick().launch), TrainingReceipt
    )
    provenance = read_document(env.layout, recovered.provenance, BallTrainingProvenance)
    latest = torch.load(
        env.layout.artifact_path(provenance.latest_checkpoint),
        weights_only=True,
        map_location="cpu",
    )
    assert latest["global_step"] == env.recipe.total_updates and latest["epoch"] == 1
    assert output_reference(env.layout, env.layout.artifact_path(first)) == first
    assert (
        max(int(state["step"]) for state in latest["optimizer"]["state"].values())
        == env.recipe.total_updates
    )
    selected = torch.load(
        env.layout.artifact_path(provenance.selected_checkpoint),
        weights_only=True,
        map_location="cpu",
    )
    best = json.loads(env.layout.artifact_path(provenance.best_record).read_bytes())
    assert best["selection_error_px"] == latest["best_score"] == selection_score(
        selected["validation"], scope
    )


def test_model_support_and_fixed_validation_evidence_cannot_be_substituted(
    tmp_path: Path,
) -> None:
    env = LearningEnvironment(tmp_path)
    env.executor.finish(env.tick().launch)
    evaluation = read_document(
        env.layout, env.executor.finish(env.tick().launch), EvaluationReceipt
    )
    metrics = read_document(env.layout, evaluation.metrics, BallEvaluationMetrics)
    raw = json.loads(env.layout.artifact_path(metrics.comparison).read_bytes())
    for fault, message in (
        ("state", "model identities"),
        ("precision", "model identities"),
        ("source", "finite nonnegative"),
        ("validation", "fixed validation contract"),
    ):
        changed = copy.deepcopy(raw)
        if fault == "state":
            changed["candidate"]["initial_state_sha256"] = "f" * 64
        elif fault == "precision":
            changed["candidate"]["evaluation"]["precision"] = "bf16"
        elif fault == "source":
            changed["candidate"]["evaluation"]["scopes"]["full"]["by_source"][
                "tracknet"
            ]["by_frame_step"]["1"]["mean_error_px"] = float("nan")
        else:
            changed["validation_manifest_sha256"] = "f" * 64
        altered = write_asset(
            env.layout,
            env.layout.path("models", "forged-" + fault + ".json"),
            json.dumps(changed).encode(),
        )
        altered_metrics = save(
            env.layout, metrics.model_copy(update={"comparison": altered})
        )
        with pytest.raises(ValueError, match=message):
            env.adapter.validate_evaluation(
                env.layout,
                env.intent,
                evaluation.model_copy(update={"metrics": altered_metrics}),
            )
    with pytest.raises(ValueError, match="different completed candidate"):
        env.adapter.validate_evaluation(
            env.layout,
            env.intent,
            evaluation.model_copy(update={"candidate": env.parent}),
        )
    fixed_test = resolve_inputs(env.layout, env.intent).fixed.test
    assert fixed_test is not None
    with pytest.raises(ValueError, match="fixed validation changed"):
        env.adapter.validate_evaluation(
            env.layout,
            env.intent,
            evaluation.model_copy(update={"validation": fixed_test}),
        )
