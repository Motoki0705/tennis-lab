"""Hydra composition checks for the three DINO input-mode ablations."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.tasks.court_alignment.configuration import validate_training_boundary
from src.tasks.court_alignment.data.dataset import _sample_instances
from src.tasks.court_alignment.data.splits import GroundCourtSplit, stable_sample_seed
from src.tasks.court_alignment.geometry.court import (
    COURT_LENGTH_M,
    DOUBLES_WIDTH_M,
    court_doubles_footprint_for_instance,
)
from src.tasks.court_alignment.training.detr_losses import CourtDetrCriterion
from src.tasks.court_alignment.training.detr_metrics import CourtDetrMetrics
from src.tasks.court_alignment.training.runner import CourtAlignmentTrainingRunner

pytestmark = pytest.mark.integration


def _compose(overrides: list[str] | None = None) -> DictConfig:
    root = Path(__file__).resolve().parents[4]
    config_dir = root / "src" / "tasks" / "court_alignment" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="train_dino", overrides=overrides or [])


@pytest.mark.parametrize("input_mode", ["repeat_rgb", "learnable_1x1", "red_only"])
def test_dino_input_mode_ablation_composes_with_fixed_pretrained_resize(
    input_mode: str,
) -> None:
    cfg = _compose([f"model.input_mode={input_mode}"])

    validate_training_boundary(cfg)

    assert cfg.model.input_mode == input_mode
    assert cfg.model.short_side == 800
    assert cfg.model.max_long_side == 1333
    assert cfg.data.image_size == 800
    assert cfg.data.batch_size == 1
    assert cfg.data.max_scale_px_per_metre == pytest.approx(18.0)
    assert cfg.data.court_margin_px == pytest.approx(236.0)
    assert cfg.data.footprint_overlap_tolerance_px == pytest.approx(1.0)
    assert cfg.data.max_sampling_attempts == 512
    assert cfg.training.compile.enabled is False
    assert cfg.run.output_dir == f"court_alignment/dino_{input_mode}"
    assert isinstance(instantiate(cfg.loss), CourtDetrCriterion)
    assert isinstance(instantiate(cfg.metrics), CourtDetrMetrics)


def test_input_modes_change_no_experimental_axis_besides_channel_mapping() -> None:
    composed = {
        mode: _compose([f"model.input_mode={mode}"])
        for mode in ("repeat_rgb", "learnable_1x1", "red_only")
    }
    reference = composed["repeat_rgb"]
    reference_model = OmegaConf.to_container(reference.model, resolve=True)
    assert isinstance(reference_model, dict)
    reference_model.pop("input_mode")
    for mode, config in composed.items():
        model = OmegaConf.to_container(config.model, resolve=True)
        assert isinstance(model, dict)
        model.pop("input_mode")
        assert model == reference_model
        for section in ("data", "loss", "decoder", "metrics", "training"):
            assert OmegaConf.to_container(
                config[section], resolve=True
            ) == OmegaConf.to_container(reference[section], resolve=True), mode


def test_dino_margin_covers_worst_case_rotated_court_footprint() -> None:
    config = _compose()
    half_diagonal_m = math.hypot(COURT_LENGTH_M / 2.0, DOUBLES_WIDTH_M / 2.0)
    maximum_corner_radius_px = (
        half_diagonal_m * config.data.max_scale_px_per_metre
    )

    assert config.data.court_margin_px >= maximum_corner_radius_px
    assert config.data.court_margin_px < (config.data.image_size - 1) / 2.0


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_every_configured_sample_has_only_fully_visible_court_footprints(
    split: GroundCourtSplit,
) -> None:
    config = instantiate(_compose().data).dataset_config
    height, width = config.image_size
    for index in range(config.split.size(split)):
        rng = np.random.default_rng(
            stable_sample_seed(config.split.seed, split, index)
        )
        instances = _sample_instances(config, rng)
        for instance in instances:
            footprint = court_doubles_footprint_for_instance(instance)
            assert bool(
                (
                    (footprint[:, 0] >= 0.0)
                    & (footprint[:, 0] <= width - 1)
                    & (footprint[:, 1] >= 0.0)
                    & (footprint[:, 1] <= height - 1)
                ).all()
            ), (split, index, instance)


def test_runner_selects_dino_lightning_boundary_without_loading_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose()
    runner = CourtAlignmentTrainingRunner()
    datamodule = runner.build_datamodule(cfg)
    marker = pl.LightningModule()
    monkeypatch.setattr(
        "src.tasks.court_alignment.training.runner.DinoCourtAlignmentLightningModule",
        lambda config: marker,
    )

    module = runner.build_lightning_module(cfg, datamodule, steps_per_epoch=17)

    assert module is marker
    assert module.steps_per_epoch == 17


def test_dino_boundary_rejects_non_pretrained_resolution() -> None:
    cfg = _compose(["data.image_size=256"])

    with pytest.raises(ValueError, match="800x800"):
        validate_training_boundary(cfg)
