"""Synthetic CPU acceptance of player learning/export/adoption; no DINO CUDA run."""

from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from src.submodules.models.dino.person_detector import DinoPersonDetector
from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import PublicationReceipt
from src.tasks.base.annotation.learning.contracts import (
    EvaluationReceipt,
    TrainingReceipt,
)
from src.tasks.player_detection.data.snapshots.store import load_snapshot
from src.tasks.player_detection.inference import (
    DinoPlayerDetector,
    PlayerDetectionRequest,
)
from src.tasks.player_detection.training.annotation.contracts import (
    PlayerComparison,
    PlayerEvaluationMetrics,
    PlayerSelectionPolicy,
    PlayerTrainingProvenance,
)
from src.tasks.player_detection.training.annotation.inputs import resolve_inputs
from src.tasks.player_detection.training.annotation.runner import AnnotationPlayerRunner
from tests.support.tasks.player_detection.chat_root import make_synthetic_root
from tests.support.tasks.player_detection.learning import (
    Environment,
    TinyDino,
    use_cpu_dino,
)


def test_cumulative_cpu_steps_use_adopted_weights_old_and_new_empty_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = Environment(
        make_synthetic_root(tmp_path), added_sources=("src_b",), train_samples_per_epoch=10,
    )
    use_cpu_dino(monkeypatch)
    snapshot, store = load_snapshot(env.layout, env.snapshot)
    publication = read_document(env.layout, env.publication, PublicationReceipt)
    seen: set[str] = set()
    target_sizes: list[int] = []

    class Observe(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            assert trainer.global_step == 0
            assert not trainer.optimizers[0].state
            assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == 0
            parent = torch.load(
                env.layout.artifact_path(env.parent.inference_weights), map_location="cpu", weights_only=False,
            )["model"]
            for key, value in pl_module.model.dino.state_dict().items():
                assert torch.equal(value, parent[key])

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            assert torch.isfinite(outputs["loss"])
            for row, boxes, labels in zip(batch.frames, batch.boxes_cxcywh, batch.labels, strict=True):
                seen.add(snapshot.clips[store.clip_of(row).index].clip_id)
                target_sizes.append(len(labels))
                assert boxes.shape == (len(labels), 4)
                if not len(labels):
                    assert store.frames["confirmed_empty"][row]

    original = AnnotationPlayerRunner.callbacks_extra
    monkeypatch.setattr(
        AnnotationPlayerRunner, "callbacks_extra", lambda *args: [*original(*args), Observe()],
    )
    pending = env.tick()
    assert pending.launch is not None
    receipt = read_document(env.layout, env.executor.finish(pending.launch), TrainingReceipt)
    provenance = read_document(env.layout, receipt.provenance, PlayerTrainingProvenance)
    latest = torch.load(env.layout.artifact_path(provenance.latest_checkpoint), map_location="cpu", weights_only=False)
    assert latest["global_step"] == 10
    assert latest["state_dict"]["model.dino.weight"].item() != 0.0
    assert seen.intersection(publication.added_clips)
    assert seen.intersection(publication.reused_clips)
    assert 0 in target_sizes and any(target_sizes)
    assert set(env.fixed.validation.clips).isdisjoint(seen)
    assert env.fixed.test is not None and set(env.fixed.test.clips).isdisjoint(seen)


def test_cpu_export_predictions_match_fixed_evaluation_and_deployed_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = Environment(make_synthetic_root(tmp_path))
    use_cpu_dino(monkeypatch)
    training = env.tick()
    assert training.launch is not None
    trained = read_document(env.layout, env.executor.finish(training.launch), TrainingReceipt)
    evaluation = env.tick()
    assert evaluation.launch is not None
    evaluated = read_document(env.layout, env.executor.finish(evaluation.launch), EvaluationReceipt)
    metrics = read_document(env.layout, evaluated.metrics, PlayerEvaluationMetrics)
    comparison = read_document(env.layout, metrics.comparison, PlayerComparison)
    assert evaluated.candidate.inference_weights == trained.candidate.inference_weights
    inputs = resolve_inputs(env.layout, env.intent)
    _, store = load_snapshot(env.layout, inputs.validation.snapshot)

    def load_cpu(self):
        # Substitute only the CUDA model; player metadata, preprocessing and decode run.
        model = TinyDino().eval()
        model.load_state_dict(torch.load(self.checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
        self._model = model

    monkeypatch.setattr(DinoPersonDetector, "_load_impl", load_cpu)
    detector = DinoPlayerDetector(
        env.layout.artifact_path(trained.candidate.inference_weights),
        env.layout.repository / "third_party/DINO", device="cpu",
        confidence=inputs.recipe.operating_point.score_threshold,
        short_side=inputs.recipe.input_size.short_side,
        max_long_side=inputs.recipe.input_size.max_long_side,
    )
    try:
        with np.load(env.layout.artifact_path(comparison.candidate.predictions)) as predictions:
            rows = {store.frame_key(int(row)): int(row) for row in store.split_frames("val")}
            for key, boxes, scores in zip(predictions["frame_keys"], predictions["boxes_xyxy"], predictions["scores"], strict=True):
                actual = detector.predict(PlayerDetectionRequest(store.read_bgr(rows[key])))
                keep = scores >= inputs.recipe.operating_point.score_threshold
                np.testing.assert_array_equal(actual.boxes_xyxy, boxes[keep])
                np.testing.assert_array_equal(actual.scores, scores[keep])
        assert detector.checkpoint_info is not None
        assert detector.checkpoint_info.source_checkpoint_sha256 == trained.candidate.training_weights.digest
    finally:
        detector.unload()


@pytest.mark.parametrize("mode", ["train_failure", "evaluation_failure", "rejected"])
def test_player_lifecycle_failure_or_rejection_preserves_current_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    policy = PlayerSelectionPolicy(
        minimum_validation_frames=1, minimum_target_instances=1,
        minimum_map=1.0, minimum_map_improvement=1.0,
        maximum_precision_regression=0.0, maximum_recall_regression=0.0,
        maximum_negative_fp_per_image=0.0, minimum_negative_frames=1,
    )
    env = Environment(make_synthetic_root(tmp_path), policy=policy)
    use_cpu_dino(monkeypatch)
    training = env.tick()
    assert training.launch is not None

    def fail(*_):
        raise RuntimeError("synthetic player stage failure")

    if mode == "train_failure":
        monkeypatch.setattr(AnnotationPlayerRunner, "run", fail)
        with pytest.raises(RuntimeError, match="synthetic player stage failure"):
            env.executor.finish(training.launch)
        env.executor.record_exit(training.launch, failed=True)
    else:
        env.executor.finish(training.launch)
        evaluation = env.tick()
        assert evaluation.launch is not None
        assert env.learning.models.current("player").model == env.parent
        if mode == "evaluation_failure":
            monkeypatch.setattr("src.tasks.player_detection.training.annotation.evaluation.evaluate_checkpoints", fail)
            with pytest.raises(RuntimeError, match="synthetic player stage failure"):
                env.executor.finish(evaluation.launch)
            env.executor.record_exit(evaluation.launch, failed=True)
        else:
            env.executor.finish(evaluation.launch)
    outcome = env.tick()
    assert outcome.state == ("rejected" if mode == "rejected" else "failed")
    assert outcome.reason
    assert env.tick() == outcome
    assert env.learning.models.current("player").model == env.parent
    assert len(env.executor.submitted) == (1 if mode == "train_failure" else 2)
