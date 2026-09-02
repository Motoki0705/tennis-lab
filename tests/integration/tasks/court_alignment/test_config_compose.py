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
from src.tasks.court_alignment.data.augmentation import (
    GroundCourtAugmentationConfig,
    build_augmentation,
)
from src.tasks.court_alignment.data.datamodule import GroundCourtDataModule
from src.tasks.court_alignment.data.dataset import GroundCourtDataset
from src.tasks.court_alignment.models.cnn import CourtAlignmentCNN
from src.tasks.court_alignment.training.lightning_module import (
    CourtAlignmentLightningModule,
)
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


def _augmentations(
    datamodule: GroundCourtDataModule,
) -> tuple[GroundCourtAugmentationConfig, ...]:
    return cast(
        tuple[GroundCourtAugmentationConfig, ...],
        datamodule.dataset_config.augmentations,
    )


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


@pytest.mark.parametrize(
    ("config_name", "augmentation_names"),
    [
        ("b00_scale_v1", ["identity"]),
        (
            "b00_appearance_v1",
            [
                "random_line_morphology",
                "random_heatmap_blur",
                "random_probability_noise",
            ],
        ),
        ("b00_structure_v1", ["random_line_dropout", "random_ghost_lines"]),
        (
            "b00_v1",
            [
                "random_line_morphology",
                "random_heatmap_blur",
                "random_line_dropout",
                "random_ghost_lines",
                "random_probability_noise",
            ],
        ),
        (
            "b00_weak_structure_v2",
            [
                "random_line_morphology",
                "random_heatmap_blur",
                "random_line_dropout",
                "random_ghost_lines",
                "random_probability_noise",
            ],
        ),
    ],
)
def test_b00_pilot_profiles_compose_as_explicit_data_overrides(
    config_name: str,
    augmentation_names: list[str],
) -> None:
    cfg = _compose("train", [f"data={config_name}"])
    runner = CourtAlignmentTrainingRunner()
    runner.prepare_config(cfg)
    datamodule = cast(GroundCourtDataModule, runner.build_datamodule(cfg))

    assert cfg.data.sigma_px == pytest.approx(2.0)
    assert cfg.data.min_scale_px_per_metre == pytest.approx(3.0)
    assert cfg.data.max_scale_px_per_metre == pytest.approx(9.0)
    assert [item.name for item in _augmentations(datamodule)] == augmentation_names


def test_b00_weak_structure_v2_keeps_appearance_v1_and_exact_weak_noise() -> None:
    weak_cfg = _compose("train", ["data=b00_weak_structure_v2"])
    appearance_cfg = _compose("train", ["data=b00_appearance_v1"])
    runner = CourtAlignmentTrainingRunner()
    weak = _augmentations(
        cast(GroundCourtDataModule, runner.build_datamodule(weak_cfg))
    )
    appearance = _augmentations(
        cast(GroundCourtDataModule, runner.build_datamodule(appearance_cfg))
    )

    for weak_index, appearance_index in ((0, 0), (1, 1), (4, 2)):
        assert weak[weak_index].name == appearance[appearance_index].name
        assert dict(weak[weak_index].params) == dict(
            appearance[appearance_index].params
        )
    assert dict(weak[2].params) == {
        "probability": 0.08,
        "gap_count_range": [1, 2],
        "gap_length_px_range": [2.0, 8.0],
        "gap_width_px_range": [1.0, 3.0],
    }
    assert dict(weak[3].params) == {
        "probability": 0.18,
        "copy_count_range": [1, 1],
        "offset_px_range": [3, 10],
        "amplitude_range": [0.35, 0.70],
        "long_line_count_range": [0, 1],
        "long_line_length_px_range": [48.0, 180.0],
        "long_line_width_px_range": [1.0, 3.0],
        "long_line_amplitude_range": [0.35, 0.70],
    }


def test_b00_combined_profile_keeps_validation_and_test_clean() -> None:
    cfg = _compose(
        "train",
        [
            "data=b00_v1",
            "data.train_samples=1",
            "data.val_samples=1",
            "data.test_samples=1",
            "data.num_workers=0",
        ],
    )
    datamodule = cast(
        GroundCourtDataModule,
        CourtAlignmentTrainingRunner().build_datamodule(cfg),
    )
    datamodule.setup(None)
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    identity = build_augmentation("identity")

    for split, actual_dataset in (
        ("train", datamodule.train_dataset),
        ("val", datamodule.val_dataset),
        ("test", datamodule.test_dataset),
    ):
        clean_dataset = GroundCourtDataset(
            datamodule.dataset_config,
            split=cast(Any, split),
            augmentation=identity,
        )
        actual = actual_dataset[0]
        clean = clean_dataset[0]
        for key in ("keypoints", "visibility", "centers", "instance_ids"):
            torch.testing.assert_close(
                cast(torch.Tensor, actual[key]),
                cast(torch.Tensor, clean[key]),
            )
        if split == "train":
            assert not torch.equal(
                cast(torch.Tensor, actual["image"]),
                cast(torch.Tensor, clean["image"]),
            )
        else:
            torch.testing.assert_close(
                cast(torch.Tensor, actual["image"]),
                cast(torch.Tensor, clean["image"]),
            )


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


def test_b00_v1_decoder_profile_composes_explicit_success_settings() -> None:
    cfg = _compose("evaluate_real_heatmap", ["decoder=b00_v1"])

    validate_real_heatmap_evaluation_boundary(cfg)

    assert cfg.decoder.threshold == pytest.approx(0.25)
    assert cfg.decoder.nms_kernel == 3
    assert cfg.decoder.max_peaks == 8
    assert cfg.decoder.subpixel_refine is True
    assert cfg.decoder.cluster_distance_px == pytest.approx(8.0)
    assert cfg.decoder.max_instances == 2


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
    test_calls: list[dict[str, object]] = []
    datamodule = object()
    module = object()

    class _Trainer:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def test(self, *args: object, **kwargs: object) -> None:
            del args
            test_calls.append(kwargs)

    monkeypatch.setattr(runner, "seed_everything", lambda _: None)
    monkeypatch.setattr(runner, "apply_runtime_settings", lambda _: None)
    monkeypatch.setattr(runner, "build_datamodule", lambda _: cast(Any, datamodule))
    monkeypatch.setattr(
        runner,
        "build_lightning_module",
        lambda *_, **__: cast(Any, module),
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

    assert len(test_calls) == 1
    assert test_calls[0]["datamodule"] is datamodule
    assert test_calls[0]["ckpt_path"] == str(checkpoint_path.resolve())
    assert test_calls[0]["weights_only"] is False


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
        datamodule = cast(
            GroundCourtDataModule,
            CourtAlignmentTrainingRunner().build_datamodule(cfg),
        )
        datamodule.setup("test")
        dataset = datamodule.test_dataloader().dataset
        samples.append(dataset[0])

    reference = samples[0]
    for sample in samples[1:]:
        assert torch.equal(
            cast(torch.Tensor, sample["image"]),
            cast(torch.Tensor, reference["image"]),
        )
        assert torch.equal(
            cast(torch.Tensor, sample["keypoints"]),
            cast(torch.Tensor, reference["keypoints"]),
        )
        assert torch.equal(
            cast(torch.Tensor, sample["visibility"]),
            cast(torch.Tensor, reference["visibility"]),
        )
        assert sample["sample_id"] == reference["sample_id"]
        assert not torch.equal(
            cast(torch.Tensor, sample["target_heatmaps"]),
            cast(torch.Tensor, reference["target_heatmaps"]),
        )


def test_runner_strict_warm_starts_historical_model_only_payload(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    checkpoint_path = checkpoint_root / "historical-sigma2.ckpt"
    cfg = _compose(
        "smoke",
        [
            f"paths.checkpoint_root={checkpoint_root}",
            "run.init_weights=historical-sigma2.ckpt",
            "model.base_channels=4",
            "model.group_norm_groups=4",
        ],
    )
    runner = CourtAlignmentTrainingRunner()
    datamodule = runner.build_datamodule(cfg)
    source = cast(
        CourtAlignmentLightningModule,
        runner.build_lightning_module(cfg, datamodule),
    )
    target = cast(
        CourtAlignmentLightningModule,
        runner.build_lightning_module(cfg, datamodule),
    )
    with torch.no_grad():
        for parameter in source.model.parameters():
            parameter.fill_(0.125)
        for parameter in target.model.parameters():
            parameter.zero_()
    torch.save(
        {
            "epoch": 47,
            "global_step": 12_288,
            "optimizer_states": [{"state": "must-be-ignored"}],
            "lr_schedulers": [{"state": "must-be-ignored"}],
            "state_dict": {
                f"model.{key}": value.clone()
                for key, value in source.model.state_dict().items()
            },
        },
        checkpoint_path,
    )

    runner.maybe_load_init_weights(runner.validate_runtime_config(cfg), target)

    for key, expected in source.model.state_dict().items():
        torch.testing.assert_close(target.model.state_dict()[key], expected)
