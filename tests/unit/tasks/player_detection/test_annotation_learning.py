from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch

from src.tasks.base.annotation.artifacts import (
    file_reference,
    read_document,
    verify_reference,
)
from src.tasks.base.annotation.learning.contracts import TrainingReceipt
from src.tasks.player_detection.training.annotation.checkpoints import (
    checkpoint_directory,
    freeze_resume_checkpoint,
)
from src.tasks.player_detection.training.annotation.contracts import (
    PlayerSelectionPolicy,
    PlayerTrainingProvenance,
)
from src.tasks.player_detection.training.annotation.inputs import (
    resolve_inputs,
    run_root,
)
from src.tasks.player_detection.training.annotation.runner import AnnotationPlayerRunner
from tests.support.tasks.player_detection.chat_root import SyntheticRoot
from tests.support.tasks.player_detection.learning import Environment, use_cpu_dino


def test_fixed_inputs_exclude_test_and_bind_validation_support(
    synthetic_root: SyntheticRoot,
) -> None:
    env = Environment(synthetic_root)
    inputs = resolve_inputs(env.layout, env.intent)
    config = inputs.configuration(
        env.layout, env.intent, run_root(env.layout, env.intent) / "attempt-001", None
    )
    assert (
        config.run.resume is None
        and config.run.init_weights is None
        and config.run.test_after_fit is False
    )
    assert config.model.init_checkpoint == env.parent.inference_weights.path
    data = inputs.datamodule(env.layout, config)
    data.setup("fit")
    assert data.selection_stats["val"]["frames"] == inputs.validation.selected_frames
    assert data.selection_stats["val"]["negative_frames"] > 0
    with pytest.raises(ValueError, match="test data"):
        data.test_dataloader()


def test_cpu_lightning_saves_bound_best_last_and_exact_export(
    synthetic_root: SyntheticRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = Environment(synthetic_root)
    use_cpu_dino(monkeypatch)
    pending = env.tick()
    assert pending.launch is not None
    result = env.executor.finish(pending.launch)
    receipt = read_document(env.layout, result, TrainingReceipt)
    env.adapter.validate_training(env.layout, env.intent, receipt)
    provenance = read_document(env.layout, receipt.provenance, PlayerTrainingProvenance)
    latest = torch.load(
        env.layout.artifact_path(provenance.latest_checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    assert latest["epoch"] == 1 and latest["global_step"] == 4
    assert (
        latest["annotation_learning"]["initial_weights"]
        == env.parent.inference_weights.model_dump()
    )
    assert receipt.candidate.training_weights != provenance.latest_checkpoint
    assert receipt.candidate.inference_weights.role == "checkpoint"
    assert env.learning.models.current("player").model == env.parent
    next_stage = env.tick()
    assert next_stage.state == "evaluation_pending"
    # Re-reading the completed attempt does not create another optimizer run.
    assert env.executor.finish(pending.launch) == result
    assert (
        len(
            list(
                checkpoint_directory(
                    run_root(env.layout, env.intent) / "attempt-001"
                ).glob("last.ckpt")
            )
        )
        == 1
    )


@pytest.mark.parametrize("configured", [False, True])
def test_cpu_lifecycle_evaluates_same_support_and_keeps_or_adopts_model(
    synthetic_root: SyntheticRoot,
    monkeypatch: pytest.MonkeyPatch,
    configured: bool,
) -> None:
    policy = PlayerSelectionPolicy(
        minimum_validation_frames=1,
        minimum_target_instances=1,
        minimum_map=0.0,
        minimum_map_improvement=0.0,
        maximum_precision_regression=0.0,
        maximum_recall_regression=0.0,
        maximum_negative_fp_per_image=1.0,
        minimum_negative_frames=1,
    )
    env = Environment(synthetic_root, policy=policy if configured else None)
    use_cpu_dino(monkeypatch)
    training = env.tick()
    assert training.launch is not None
    env.executor.finish(training.launch)
    evaluation = env.tick()
    assert evaluation.launch is not None
    env.executor.finish(evaluation.launch)
    outcome = env.tick()
    assert outcome.state == ("adopted" if configured else "not_configured")
    assert (env.learning.models.current("player").model == env.parent) is (
        not configured
    )


@pytest.mark.parametrize("before_export", [False, True])
def test_explicit_resume_preserves_input_checkpoint_and_does_not_add_updates(
    synthetic_root: SyntheticRoot,
    monkeypatch: pytest.MonkeyPatch,
    before_export: bool,
) -> None:
    env = Environment(synthetic_root)
    use_cpu_dino(monkeypatch)
    first = env.tick()
    assert first.launch is not None
    original = AnnotationPlayerRunner.callbacks_extra

    class Interrupted(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module) -> None:
            if trainer.current_epoch == 1:
                raise RuntimeError("synthetic interrupted run")

    def fail_export(*_args):
        raise RuntimeError("synthetic interrupted run")

    with monkeypatch.context() as interruption:
        if before_export:
            interruption.setattr(
                "src.tasks.player_detection.training.annotation.train.export_player_checkpoint",
                fail_export,
            )
        else:
            interruption.setattr(
                AnnotationPlayerRunner,
                "callbacks_extra",
                lambda *args: [*original(*args), Interrupted()],
            )
        with pytest.raises(RuntimeError, match="synthetic interrupted"):
            env.executor.finish(first.launch)
    env.executor.record_exit(first.launch, failed=True)
    assert env.tick().state == "failed"
    latest_path = (
        checkpoint_directory(run_root(env.layout, env.intent) / "attempt-001")
        / "last.ckpt"
    )
    latest = file_reference(env.layout, latest_path, role="artifact")
    with pytest.raises(ValueError, match="frozen"):
        env.adapter.validate_resume(env.layout, env.intent, latest)
    frozen = freeze_resume_checkpoint(env.layout, env.intent, latest)
    assert frozen.digest == latest.digest
    assert freeze_resume_checkpoint(env.layout, env.intent, latest) == frozen
    env.learning.retry(
        env.intent_ref,
        env.adapter,
        "train",
        reason="resume CPU fixture",
        resume_checkpoint=frozen,
    )
    second = env.tick()
    assert second.launch is not None
    with monkeypatch.context() as recovery:
        if before_export:
            recovery.setattr(
                AnnotationPlayerRunner,
                "run",
                lambda *_args: pytest.fail("completed budget must not train again"),
            )
        result = env.executor.finish(second.launch)
    receipt = read_document(env.layout, result, TrainingReceipt)
    env.adapter.validate_training(env.layout, env.intent, receipt)
    value = torch.load(latest_path, weights_only=False, map_location="cpu")
    assert value["global_step"] == 4 and value["epoch"] == 1
    assert verify_reference(env.layout, frozen).is_file()
    assert not (run_root(env.layout, env.intent) / "attempt-002").exists()
