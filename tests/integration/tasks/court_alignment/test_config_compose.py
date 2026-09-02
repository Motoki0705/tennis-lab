"""Hydra composition checks for court-alignment sigma experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.tasks.court_alignment.configuration import (
    CourtAlignmentRealHeatmapRuntimeConfig,
    CourtAlignmentRuntimeConfig,
    validate_evaluation_boundary,
    validate_real_heatmap_evaluation_boundary,
    validate_training_boundary,
)
from src.tasks.court_alignment.models.cnn import CourtAlignmentCNN
from src.tasks.court_alignment.training.losses import CourtAlignmentLoss
from src.tasks.court_alignment.training.runner import CourtAlignmentTrainingRunner
from src.utils.configuration import UnknownConfigurationKeyError

pytestmark = pytest.mark.integration

_SIGMAS = (0.75, 1.0, 1.5, 2.0)


def _config_dir() -> str:
    root = Path(__file__).resolve().parents[4]
    return str(root / "src" / "tasks" / "court_alignment" / "configs")


def _compose(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(config_dir=_config_dir(), version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides or [])


@pytest.mark.parametrize("sigma_px", _SIGMAS)
def test_sigma_ablation_changes_only_the_explicit_target_width(
    sigma_px: float,
) -> None:
    cfg = _compose("train", [f"data.sigma_px={sigma_px}"])
    CourtAlignmentTrainingRunner().prepare_config(cfg)

    assert cfg.data.sigma_px == sigma_px
    assert cfg.run.seed == 42
    assert cfg.training.trainer.max_epochs == 50
    assert cfg.training.steps_per_epoch == 256
    assert cfg.training.early_stopping.enabled is False
    assert cfg.training.checkpoint.monitor == "val/loss"
    assert cfg.training.checkpoint.save_last is True
    assert cfg.model.heatmap_prior_probability == 0.1
    assert cfg.data.seed == 42
    assert cfg.metrics.minimum_common_keypoints == 4
    assert cfg.metrics.minimum_visible_keypoints == 4
    assert cfg.metrics.minimum_visible_fraction == 0.5
    assert cfg.metrics.minimum_sim2_keypoints == 4


def test_config_constructs_model_loss_metrics_and_datamodule() -> None:
    cfg = _compose("smoke")
    runner = CourtAlignmentTrainingRunner()
    runner.prepare_config(cfg)

    assert isinstance(instantiate(cfg.model), CourtAlignmentCNN)
    assert isinstance(instantiate(cfg.loss), CourtAlignmentLoss)
    assert instantiate(cfg.metrics) is not None
    assert runner.build_datamodule(cfg).__class__.__name__ == "GroundCourtDataModule"


def test_smoke_config_is_cpu_only_and_keeps_test_bundle_enabled() -> None:
    cfg = _compose("smoke")

    assert cfg.run.gpus == 0
    assert cfg.run.test_after_fit is True
    assert cfg.training.trainer.max_epochs == 1
    assert cfg.training.compile.enabled is False
    assert cfg.training.checkpoint.enabled is True
    assert cfg.data.num_workers == 0


def test_hydra_boundaries_validate_resolved_train_and_evaluate_contracts() -> None:
    train_cfg = _compose("train")
    evaluate_cfg = _compose("evaluate")

    validate_training_boundary(train_cfg)
    validate_evaluation_boundary(evaluate_cfg)
    assert (
        CourtAlignmentRuntimeConfig.from_config(train_cfg).evaluation_checkpoint is None
    )


def test_real_heatmap_config_composes_explicit_ablation_boundaries(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    checkpoint_root = tmp_path / "checkpoint-root"
    output_root = tmp_path / "output-root"
    cfg = _compose(
        "evaluate_real_heatmap",
        [
            f"paths.data_root={data_root}",
            f"paths.checkpoint_root={checkpoint_root}",
            f"paths.output_root={output_root}",
            "real_evaluation.archive_path=b00/heatmaps.npz",
            "real_evaluation.manifest_path=b00/manifest.json",
            "real_evaluation.alignment_path=b00/alignment.json",
            "real_evaluation.checkpoint_path=sigma-200/best.ckpt",
            "real_evaluation.output_dir=b00-sigma-200-max",
            "real_evaluation.preprocess.method=max",
            "real_evaluation.preprocess.content_fraction=0.8372",
            "decoder.threshold=0.3",
        ],
    )

    validate_real_heatmap_evaluation_boundary(cfg)
    request = CourtAlignmentRealHeatmapRuntimeConfig.from_config(cfg).require_request()

    assert request.archive_path == (data_root / "b00" / "heatmaps.npz").resolve()
    assert (
        request.checkpoint_path
        == (checkpoint_root / "sigma-200" / "best.ckpt").resolve()
    )
    assert request.output_dir == (output_root / "b00-sigma-200-max").resolve()
    assert request.preprocess.method == "max"
    assert request.preprocess.content_fraction == pytest.approx(0.8372)
    assert request.decoder.threshold == pytest.approx(0.3)


def test_real_heatmap_config_rejects_unknown_preprocess() -> None:
    cfg = _compose(
        "evaluate_real_heatmap",
        ["real_evaluation.preprocess.method=not-registered"],
    )

    with pytest.raises(ValueError, match="Unknown real-heatmap preprocess"):
        validate_real_heatmap_evaluation_boundary(cfg)


def test_real_heatmap_config_rejects_invalid_content_fraction() -> None:
    cfg = _compose(
        "evaluate_real_heatmap",
        ["real_evaluation.preprocess.content_fraction=0.0"],
    )

    with pytest.raises(ValueError, match="content_fraction"):
        validate_real_heatmap_evaluation_boundary(cfg)


def test_hydra_boundary_rejects_unknown_model_key() -> None:
    cfg = _compose("train", ["+model.unexpected=true"])
    with pytest.raises(UnknownConfigurationKeyError, match=r"model\.unexpected"):
        validate_training_boundary(cfg)


def test_runner_rejects_disabled_checkpoint_selection() -> None:
    cfg = _compose("smoke", ["training.checkpoint.enabled=false"])

    with pytest.raises(ValueError, match="requires checkpointing"):
        CourtAlignmentTrainingRunner().prepare_config(cfg)


def test_runner_allows_disabled_checkpointing_when_post_fit_test_is_disabled() -> None:
    cfg = _compose(
        "smoke",
        [
            "run.test_after_fit=false",
            "training.checkpoint.enabled=false",
        ],
    )

    CourtAlignmentTrainingRunner().prepare_config(cfg)

    runtime = CourtAlignmentRuntimeConfig.from_config(cfg)
    assert runtime.runtime.run.test_after_fit is False
    assert runtime.runtime.training.checkpoint.enabled is False


def test_relative_evaluation_checkpoint_resolves_under_checkpoint_role(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "configured-checkpoints"
    cfg = _compose("evaluate")
    cfg.paths.checkpoint_root = str(checkpoint_root)
    cfg.evaluation.checkpoint_path = "sigma-1/best.ckpt"

    runtime = CourtAlignmentRuntimeConfig.from_config(cfg, evaluation=True)

    assert (
        runtime.evaluation_checkpoint
        == (checkpoint_root / "sigma-1" / "best.ckpt").resolve()
    )


def test_evaluate_tests_the_typed_explicit_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "configured-checkpoints"
    checkpoint_path = checkpoint_root / "sigma-1" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    cfg = _compose("evaluate")
    cfg.paths.checkpoint_root = str(checkpoint_root)
    cfg.paths.output_root = str(tmp_path / "outputs")
    cfg.evaluation.checkpoint_path = "sigma-1/best.ckpt"
    runner = CourtAlignmentTrainingRunner()
    tested_checkpoints: list[str] = []

    class _Trainer:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def test(self, *args: object, **kwargs: object) -> None:
            del args
            tested_checkpoints.append(cast(str, kwargs["ckpt_path"]))

    monkeypatch.setattr(runner, "seed_everything", lambda _: None)
    monkeypatch.setattr(runner, "apply_runtime_settings", lambda _: None)
    monkeypatch.setattr(runner, "build_datamodule", lambda _: cast(Any, object()))
    monkeypatch.setattr(
        runner,
        "build_lightning_module",
        lambda *_, **__: cast(Any, object()),
    )
    monkeypatch.setattr(runner, "select_devices", lambda _: ("cpu", 1))
    monkeypatch.setattr(
        "src.tasks.court_alignment.training.runner.pl.Trainer",
        _Trainer,
    )
    monkeypatch.setattr(
        "src.tasks.court_alignment.training.runner.resolve_queue_repro_dir",
        lambda: None,
    )

    runner.evaluate(cfg)

    assert tested_checkpoints == [str(checkpoint_path.resolve())]


def test_evaluate_rejects_a_missing_typed_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "configured-checkpoints"
    cfg = _compose("evaluate")
    cfg.paths.checkpoint_root = str(checkpoint_root)
    cfg.evaluation.checkpoint_path = "missing.ckpt"
    runner = CourtAlignmentTrainingRunner()
    monkeypatch.setattr(runner, "seed_everything", lambda _: None)
    monkeypatch.setattr(runner, "apply_runtime_settings", lambda _: None)

    with pytest.raises(FileNotFoundError, match=str(checkpoint_root / "missing.ckpt")):
        runner.evaluate(cfg)


def test_sigma_runs_share_samples_and_change_only_dense_targets() -> None:
    samples: list[dict[str, object]] = []
    for sigma_px in _SIGMAS:
        cfg = _compose("smoke", [f"data.sigma_px={sigma_px}"])
        datamodule = CourtAlignmentTrainingRunner().build_datamodule(cfg)
        datamodule.setup("test")
        dataset = datamodule.test_dataloader().dataset
        samples.append(dataset[0])

    reference = samples[0]
    for sample in samples[1:]:
        assert torch.equal(sample["image"], reference["image"])
        assert torch.equal(sample["keypoints"], reference["keypoints"])
        assert torch.equal(sample["visibility"], reference["visibility"])
        assert sample["sample_id"] == reference["sample_id"]
        assert not torch.equal(sample["target_heatmaps"], reference["target_heatmaps"])
