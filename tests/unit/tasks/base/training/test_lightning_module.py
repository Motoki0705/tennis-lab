"""Unit tests for BaseLightningModule pure helpers.

Covers ``_concat_padded`` (time-axis padding), config-driven step estimation,
optimizer/scheduler construction, and test-prediction persistence. Heavy
training (a real ``training_step`` over a model) is deferred to the integration
smoke suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torch.optim.optimizer import Optimizer

from src.tasks.base.training.lightning_module import (
    BaseLightningModule,
    _concat_padded,
)
from src.tasks.base.training.repro import QueueReproDirError

pytestmark = pytest.mark.unit


class _TinyModule(BaseLightningModule):
    """Concrete module with a single trainable parameter for optimizer tests."""

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__(config)
        self.model = nn.Linear(2, 2)


@dataclass(frozen=True)
class _FrozenRuntimeDependency:
    value: int


class _ModuleWithRuntimeDependency(BaseLightningModule):
    def __init__(
        self,
        config: dict[str, object],
        *,
        dependency: _FrozenRuntimeDependency,
    ) -> None:
        super().__init__(config)
        self.dependency = dependency


class _SchedulerResult(TypedDict):
    interval: str
    scheduler: object


class _OptimizerResult(TypedDict):
    optimizer: Optimizer
    lr_scheduler: _SchedulerResult


def _configure_optimizer(module: _TinyModule) -> _OptimizerResult:
    return cast(_OptimizerResult, module.configure_optimizers())


def _config(
    *,
    artifact_root: Path = Path("/tmp/tennis-lab-test-artifacts"),
    max_epochs: int = 1,
    steps_per_epoch: int | None = 1,
    warmup_steps: int | None = 0,
    warmup_epochs: int | None = None,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
) -> dict[str, object]:
    return {
        "paths": {
            "project_root": ".",
            "data_root": "data",
            "checkpoint_root": "checkpoints",
            "artifact_root": str(artifact_root),
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "external",
        },
        "training": {
            "trainer": {
                "max_epochs": max_epochs,
                "gradient_clip_val": None,
                "deterministic": True,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "check_val_every_n_epoch": 1,
                "accumulate_grad_batches": 1,
                "reload_dataloaders_every_n_epochs": 0,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "benchmark": False,
            },
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "warmup_epochs": warmup_epochs,
            "min_lr": 0.0,
            "steps_per_epoch": steps_per_epoch,
            "optimizer": {"betas": betas},
            "checkpoint": {
                "enabled": False,
                "filename": "model-{epoch}",
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 1,
                "save_last": False,
            },
            "early_stopping": {
                "enabled": False,
                "monitor": "val/loss",
                "mode": "min",
                "patience": 1,
                "min_delta": 0.0,
                "check_on_train_epoch_end": False,
            },
            "lr_monitor": {"enabled": False, "interval": "step"},
            "qualitative_logging": {
                "enabled": False,
                "every_n_epochs": 1,
                "num_samples": 1,
                "selection_mode": "random",
                "selected_indices": None,
            },
            "gan": {
                "enabled": False,
                "target_weight": 0.0,
                "warmup_epochs": 1,
                "generator_gradient_clip_val": None,
                "discriminator_gradient_clip_val": None,
                "transition": {"start_epoch": 0},
            },
            "compile": {
                "enabled": True,
                "backend": "inductor",
                "mode": "reduce-overhead",
                "fullgraph": False,
                "dynamic": False,
            },
            "matmul_precision": "high",
            "allow_tf32": False,
        },
    }


# ---------------------------------------------------------------------------
# compilation_targets
# ---------------------------------------------------------------------------


def test_hyperparameters_only_capture_serializable_config(tmp_path: Path) -> None:
    module = _ModuleWithRuntimeDependency(
        _config(),
        dependency=_FrozenRuntimeDependency(value=1),
    )
    logger = TensorBoardLogger(save_dir=tmp_path)

    assert set(module.hparams) == {"config"}
    logger.log_hyperparams(cast("dict[str, Any]", dict(module.hparams)))
    logger.save()


def test_compilation_targets_exposes_primary_model() -> None:
    module = _TinyModule(_config())

    assert module.compilation_targets() == {"model": module.model}


def test_compilation_targets_rejects_missing_primary_model() -> None:
    module = BaseLightningModule(_config())

    with pytest.raises(RuntimeError, match="self.model"):
        module.compilation_targets()


def test_cuda_graph_compile_marks_each_outer_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _TinyModule(_config())
    calls = 0

    def mark_step_begin() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", mark_step_begin)

    module.on_train_batch_start({}, 0)
    module.on_validation_batch_start({}, 0)
    module.on_test_batch_start({}, 0)
    module.on_predict_batch_start({}, 0)

    assert calls == 4


@pytest.mark.parametrize(
    "compile_overrides",
    [
        {"enabled": False},
        {"mode": "default"},
        {"mode": "max-autotune-no-cudagraphs"},
    ],
)
def test_non_cuda_graph_compile_does_not_mark_batch(
    compile_overrides: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    compile_config = cast(
        "dict[str, object]",
        cast("dict[str, object]", config["training"])["compile"],
    )
    compile_config.update(compile_overrides)
    module = _TinyModule(config)
    calls = 0

    def mark_step_begin() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", mark_step_begin)

    module.on_train_batch_start({}, 0)

    assert calls == 0


# ---------------------------------------------------------------------------
# _concat_padded
# ---------------------------------------------------------------------------


def test_concat_padded_single_chunk_passthrough() -> None:
    chunk = np.ones((2, 3))
    assert _concat_padded([chunk]) is chunk


def test_concat_padded_pads_time_axis() -> None:
    a = np.ones((2, 3, 4))  # T=3
    b = np.ones((1, 5, 4))  # T=5
    out = _concat_padded([a, b])
    assert out.shape == (3, 5, 4)  # batch concat, T padded to 5
    # padded region of `a` (time steps 3,4) is zero
    assert (out[0:2, 3:, :] == 0).all()
    # original content preserved
    assert (out[0:2, :3, :] == 1).all()


def test_concat_padded_low_dim_arrays_concatenated() -> None:
    a = np.array([1.0, 2.0])  # ndim 1
    b = np.array([3.0])
    out = _concat_padded([a, b])
    assert out.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# _estimate_total_steps
# ---------------------------------------------------------------------------


def test_estimate_total_steps_from_steps_per_epoch_attr() -> None:
    m = _TinyModule(_config(max_epochs=4))
    m.steps_per_epoch = 50
    assert m._estimate_total_steps() == 200


def test_estimate_total_steps_from_config_steps_per_epoch() -> None:
    m = _TinyModule(_config(max_epochs=3, steps_per_epoch=10))
    assert m._estimate_total_steps() == 30


def test_estimate_total_steps_rejects_unresolved_count() -> None:
    m = _TinyModule(_config(steps_per_epoch=None))
    with pytest.raises(RuntimeError, match="step count is unresolved"):
        m._estimate_total_steps()


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------


def test_configure_optimizers_step_interval_no_warmup() -> None:
    m = _TinyModule(_config(max_epochs=2, steps_per_epoch=5))
    cfg = _configure_optimizer(m)
    assert isinstance(cfg["optimizer"], AdamW)
    assert cfg["lr_scheduler"]["interval"] == "step"
    assert isinstance(cfg["lr_scheduler"]["scheduler"], CosineAnnealingLR)


def test_configure_optimizers_epoch_warmup_uses_sequential() -> None:
    m = _TinyModule(
        _config(
            max_epochs=10,
            steps_per_epoch=None,
            warmup_steps=None,
            warmup_epochs=3,
        )
    )
    cfg = _configure_optimizer(m)
    assert cfg["lr_scheduler"]["interval"] == "epoch"
    assert isinstance(cfg["lr_scheduler"]["scheduler"], SequentialLR)


def test_configure_optimizers_respects_lr_and_betas() -> None:
    m = _TinyModule(
        _config(
            learning_rate=5e-4,
            weight_decay=0.01,
            steps_per_epoch=1,
            betas=(0.8, 0.95),
        )
    )
    opt = _configure_optimizer(m)["optimizer"]
    group = opt.param_groups[0]
    assert group["lr"] == pytest.approx(5e-4)
    assert group["weight_decay"] == pytest.approx(0.01)
    assert group["betas"] == (0.8, 0.95)


def test_optimizer_betas_are_explicit() -> None:
    m = _TinyModule(_config())
    assert m.optimizer_betas == (0.9, 0.999)


# ---------------------------------------------------------------------------
# test_prediction_payload defaults / collection / saving
# ---------------------------------------------------------------------------


def test_default_payload_empty_and_collect_noop() -> None:
    m = _TinyModule(_config())
    assert m.test_prediction_payload(batch=None, result={}) == {}
    # collecting an empty payload must not create a buffer
    m.collect_test_predictions(batch=None, result={})
    assert not getattr(m, "_test_pred_arrays", None)


def test_save_test_predictions_preserves_legacy_artifact_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PredModule(_TinyModule):
        def test_prediction_payload(self, batch, result):
            return {"position": result["position"]}

    m = _PredModule(_config(artifact_root=tmp_path))
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    m._reset_test_prediction_buffer()

    # two batches of differing time length to exercise padding
    m.collect_test_predictions(None, {"position": torch.zeros(2, 3, 3)})
    m.collect_test_predictions(None, {"position": torch.ones(1, 5, 3)})

    npz_path = m.save_test_predictions(metrics={"mae": 0.1})
    assert npz_path is not None
    assert npz_path == tmp_path / "test_predictions" / "pred_test.npz"
    loaded = np.load(npz_path)
    assert loaded["position"].shape == (3, 5, 3)  # batch 2+1, T padded to 5
    assert loaded["scene_ids"].shape == (3,)
    metrics = (npz_path.parent / "metrics.json").read_text()
    assert "mae" in metrics


def test_save_test_predictions_isolated_between_queue_repro_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PredModule(_TinyModule):
        def test_prediction_payload(self, batch, result):
            return {"position": result["position"]}

    artifact_root = tmp_path / "artifacts"
    module = _PredModule(_config(artifact_root=artifact_root))
    repro_dirs = (tmp_path / "queue-a", tmp_path / "queue-b")

    saved_paths: list[Path] = []
    for index, repro_dir in enumerate(repro_dirs):
        monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))
        module._reset_test_prediction_buffer()
        module.collect_test_predictions(
            None,
            {"position": torch.full((1, 2, 3), float(index))},
        )
        saved = module.save_test_predictions(metrics={"run": index})
        assert saved is not None
        saved_paths.append(saved)

    assert saved_paths == [
        repro_dirs[0] / "predictions" / "pred_test.npz",
        repro_dirs[1] / "predictions" / "pred_test.npz",
    ]
    for index, path in enumerate(saved_paths):
        assert np.load(path)["position"].item(0) == float(index)
        assert (path.parent / "metrics.json").is_file()
    assert not (artifact_root / "test_predictions").exists()


def test_save_test_predictions_rejects_invalid_queue_dir_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PredModule(_TinyModule):
        def test_prediction_payload(self, batch, result):
            return {"position": result["position"]}

    artifact_root = tmp_path / "artifacts"
    module = _PredModule(_config(artifact_root=artifact_root))
    module._reset_test_prediction_buffer()
    module.collect_test_predictions(None, {"position": torch.zeros(1, 2, 3)})
    monkeypatch.setenv("TENNIS_REPRO_DIR", "relative/repro")

    with pytest.raises(QueueReproDirError, match="absolute"):
        module.save_test_predictions(metrics={"mae": 0.1})

    assert not (artifact_root / "test_predictions").exists()


def test_save_test_predictions_none_when_empty(tmp_path: Path) -> None:
    m = _TinyModule(_config(artifact_root=tmp_path))
    assert m.save_test_predictions() is None  # nothing collected
