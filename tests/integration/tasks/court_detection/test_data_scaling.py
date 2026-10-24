"""CPU train/resume/evaluation with packed stores and isolated real val/test."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.synthetic_data_generation.dataset.court.sample_store import (
    LEGACY_SAMPLE_FILES,
    finish_court_store,
    read_court_manifest,
    read_court_rgb,
)
from src.tasks.court_detection.ablation.evaluation import (
    DENSE_METRICS,
    evaluate_checkpoint,
)
from src.tasks.court_detection.configuration import (
    CourtTrainingConfig,
    SyntheticCourtStoresSourceConfig,
)
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.scaling.comparison import compare_evaluations
from src.tasks.court_detection.scaling.provenance import data_condition
from src.tasks.court_detection.scaling.real_split import make_registry, real_source
from src.tasks.court_detection.scaling.training import (
    build_step_trainer,
    step_checkpoints,
)
from src.tasks.court_detection.training.runner import (
    CourtDetectionTrainingRunner,
    resolve_training_config,
)
from src.utils.artifact_store import ArtifactStore
from src.utils.data.image_record_store import ImageRecordWriter, encode_rgb_jpeg
from tests.integration.tasks.court_detection.test_composable_pipeline import (
    _configure_tiny_model,
    _image_points,
    _write_synthetic_court_singleton,
)
from tests.unit.tasks.court_detection.data.inputs.fixtures import write_tennis_records

pytestmark = pytest.mark.integration
_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


def _config(tmp_path: Path) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_data_scaling",
            overrides=[
                "data/source=synthetic_court_stores",
                "data.source.store_roots=[versions/condition/part]",
                "data.source.data_version=fixture-v1",
                "data.source.generation_seed=17",
                "mixed.sources.tennis_court_detector.registry=real/registry.json",
                "training.warmup_steps=1",
                "training.step_schedule.max_steps=6",
                "training.step_schedule.validation_every_n_steps=2",
                "training.step_schedule.checkpoint_every_n_steps=3",
                "run.output_dir=court_detection/train/data-scaling-fixture/s42",
            ],
        )
    config.paths.project_root = str(tmp_path)
    config.paths.data_root = "data"
    config.paths.output_root = "outputs"
    config.paths.artifact_root = "artifacts"
    config.paths.checkpoint_root = "checkpoints"
    config.paths.external_asset_root = "external"
    return config


def _inputs(tmp_path: Path) -> DictConfig:
    _write_synthetic_court_singleton(
        tmp_path / "legacy", schema="v3", calibrated_pose=True
    )
    original = tmp_path / "legacy/V3/datasets/court"
    manifest = read_court_manifest(original)
    manifest["metadata_fields"] = ["fixture"]
    root = tmp_path / "data/versions/condition/part"
    with ImageRecordWriter(root / "samples") as writer:
        for sample in manifest["samples"]:
            writer.append(
                encode_rgb_jpeg(read_court_rgb(original, sample)),
                {
                    key: value
                    for key, value in sample.items()
                    if key not in LEGACY_SAMPLE_FILES
                },
            )
        finish_court_store(root, writer, manifest)
    real_root = tmp_path / "data/court_detection/tennis_court_detector-v1"
    write_tennis_records(
        real_root,
        [
            {"id": sample, "split": split, "kps": _image_points()}
            for sample, split in [
                ("vid_a_1", "train"),
                ("vid_a_2", "val"),
                ("vid_b_1", "val"),
                ("vid_c_1", "train"),
                ("vid_d_1", "train"),
                ("QszoUKyCOHo_600", "train"),
            ]
        ],
    )
    registry = make_registry(real_source(real_root), version="fixture-real-v1", seed=42)
    destination = tmp_path / "artifacts/real/registry.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(registry))
    return _config(tmp_path)


class _PublishedCheckpoints(ArtifactStore):
    def __init__(self, local_root: Path) -> None:
        super().__init__(local_root)
        self.published: list[Path] = []
        self.removed: list[Path] = []
        self.files: dict[Path, bytes] = {}

    @property
    def enabled(self) -> bool:
        return True

    def publish_file(self, path: Path) -> None:
        self.files[self.relative_path(path)] = path.read_bytes()
        self.published.append(path)

    def fetch_file(self, path: Path) -> None:
        raise AssertionError(f"Unexpected checkpoint fetch: {path}")

    def remove_file(self, path: Path) -> None:
        del self.files[self.relative_path(path)]
        self.removed.append(path)

    def sync_tree(self) -> None:
        return None


class _ValidationLosses(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.validation_steps: list[int] = []

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.weight.square()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        # Include a worsening validation so finite top-k is checked by rank.
        losses = (4.0, 2.0, 3.0, 1.0)
        self.validation_steps.append(self.global_step)
        self.log("val/loss", losses[self.global_step // 2 - 1], batch_size=1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


@pytest.mark.parametrize(
    ("save_top_k", "published_steps", "removed_steps"),
    [
        (-1, (2, 4, 6, 8), ()),
        (0, (), ()),
        (1, (2, 4, 8), (2, 4)),
        (2, (2, 4, 6, 8), (2, 6)),
    ],
)
def test_cpu_published_checkpoint_retention(
    tmp_path: Path,
    save_top_k: int,
    published_steps: tuple[int, ...],
    removed_steps: tuple[int, ...],
) -> None:
    torch.set_num_threads(1)
    config = _config(tmp_path)
    config.training.step_schedule.max_steps = 8
    config.training.step_schedule.checkpoint_every_n_steps = 4
    config.training.checkpoint.save_top_k = save_top_k
    config.training.trainer.precision = "32-true"
    config.training.trainer.enable_progress_bar = False
    config.training.trainer.enable_model_summary = False
    logger = TensorBoardLogger(save_dir=str(tmp_path), name="logs", version=0)
    store = _PublishedCheckpoints(tmp_path)
    callbacks = step_checkpoints(config, logger)
    trainer = build_step_trainer(
        config,
        callbacks,
        logger,
        accelerator="cpu",
        devices=1,
        artifact_store=store,
    )
    module = _ValidationLosses()

    trainer.fit(
        module,
        train_dataloaders=DataLoader(torch.zeros(3, 1), batch_size=1),
        val_dataloaders=DataLoader(torch.zeros(1, 1), batch_size=1),
    )

    assert trainer.global_step == 8 and module.validation_steps == [2, 4, 6, 8]
    periodic = {"update-00000004.ckpt", "update-00000008.ckpt"}
    published = {f"best-val-{step:08d}.ckpt" for step in published_steps}
    removed = [f"best-val-{step:08d}.ckpt" for step in removed_steps]
    assert len(store.published) == len(periodic | published)
    assert {path.name for path in store.published} == periodic | published
    assert [path.name for path in store.removed] == removed
    retained = periodic | (published - set(removed))
    directory = Path(logger.log_dir) / "checkpoints"
    assert {path.name for path in directory.glob("*.ckpt")} == retained
    assert {path.name for path in store.files} == retained
    assert all(
        (tmp_path / path).read_bytes() == data for path, data in store.files.items()
    )
    if save_top_k != 0:
        assert Path(callbacks[1].best_model_path).name == "best-val-00000008.ckpt"
        assert callbacks[1].best_model_score == 1.0


def test_fixed_model_and_legacy_configs_remain_separate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _, mixed = resolve_training_config(config)
    runtime = CourtTrainingConfig.from_config(config)
    assert mixed.train_batch_counts == {
        "synthetic_court": 8,
        "tennis_court_detector": 0,
    }
    assert runtime.model.encoder.backbone_name == "dinov3_vitb16"
    assert runtime.model.encoder.lora is not None
    assert (
        runtime.model.encoder.train_mode == "frozen"
        and not runtime.model.encoder.lora.enabled
    )
    transformer = runtime.model.transformer_encoder
    assert (
        transformer.dim,
        transformer.depth,
        transformer.num_heads,
        transformer.ffn_dim,
    ) == (768, 8, 12, 2048)
    assert runtime.model.decoder.channels == 512 and runtime.loss.pose.enabled
    assert runtime.data.augmentation.train_scales == (512,)
    for name in (
        "train",
        "train_dinov3_size_ablation_s",
        "train_dinov3_size_ablation_splus",
        "train_dinov3_size_ablation_b",
        "train_dinov3_size_ablation_l",
    ):
        with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
            old = compose(config_name=name)
        _, previous = resolve_training_config(old)
        assert previous.train_batch_counts == {
            "synthetic_court": 4,
            "tennis_court_detector": 4,
        }
        assert "step_schedule" not in old.training
        assert old.model.transformer_encoder.dim == 1024
        assert old.training.trainer.max_epochs == 20 and old.training.warmup_epochs == 3


@pytest.mark.parametrize(
    "override",
    [
        "data.source.store_roots=[]",
        "data.source.store_roots=[versions/condition/part,versions/condition/part]",
        "data.source.schema=v2",
        "training.warmup_epochs=1",
        "training.warmup_steps=6",
        "training.trainer.max_epochs=2",
        "training.trainer.accumulate_grad_batches=2",
        "training.step_schedule.max_steps=7",
        "training.step_schedule.validation_every_n_steps=0",
        "training.early_stopping.enabled=true",
        "run.test_after_fit=true",
        "mixed.train_batch_counts.tennis_court_detector=1",
    ],
)
def test_ambiguous_schedule_or_real_training_is_rejected(
    tmp_path: Path, override: str
) -> None:
    config = _config(tmp_path)
    from omegaconf import OmegaConf

    config = OmegaConf.merge(config, OmegaConf.from_dotlist([override]))
    with pytest.raises(ValueError):
        resolve_training_config(config)


def test_explicit_stores_train_all_and_refuse_missing_and_corrupt_versions(
    tmp_path: Path,
) -> None:
    config = _inputs(tmp_path)
    source_config = CourtTrainingConfig.from_config(config).data.source
    source = build_court_input(source_config)
    assert source.available_splits == ("train",) and len(source.records("train")) == 3
    assert {
        source.load(record).pose_authority is not None
        for record in source.records("train")
    } == {True}
    with pytest.raises(ValueError, match="no accepted"):
        source.records("test")
    assert isinstance(source_config, SyntheticCourtStoresSourceConfig)
    root = source_config.store_roots[0]
    # Corruption in an unsampled image must fail before model/worker startup.
    shard = next((root / "samples/shards").glob("*.bin"))
    value = shard.read_bytes()
    shard.write_bytes(value[:-1] + bytes([value[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        build_court_input(source_config)
    config.data.source.store_roots = ["versions/missing"]
    with pytest.raises(FileNotFoundError):
        build_court_input(CourtTrainingConfig.from_config(config).data.source)


def test_cpu_updates_checkpoints_and_separate_fixed_evaluations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    config = _inputs(tmp_path)
    _configure_tiny_model(config, monkeypatch)
    config.data.batch_size = 1
    config.mixed.train_batch_counts.synthetic_court = 1
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.data.augmentation.train_scales = [32]
    config.data.augmentation.val_short_side = 32
    config.training.trainer.precision = "32-true"
    config.training.trainer.enable_progress_bar = False
    config.training.trainer.enable_model_summary = False
    config.training.compile.enabled = False
    config.training.lr_monitor.enabled = False
    dm = CourtDetectionDataModule(config)
    dm.setup("fit")
    assert dm._train_source_lengths == {"synthetic_court": 3}
    assert dm.val_dataset is not None
    assert (
        len(dm.val_dataset)
        + len(dm.source_eval_dataloader("tennis_court_detector", "test").dataset)
        == 5
    )
    runner = CourtDetectionTrainingRunner()
    output = tmp_path / "outputs/court_detection/train/data-scaling-fixture/s42"
    output.mkdir(parents=True)
    module = runner.build_lightning_module(config, dm, steps_per_epoch=999)
    assert module._estimate_total_steps() == 6
    logger = TensorBoardLogger(save_dir=str(output), name="logs")
    callbacks = step_checkpoints(config, logger)

    class ValidationSteps(pl.Callback):
        def __init__(self) -> None:
            self.steps: list[int] = []

        def on_validation_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule
        ) -> None:
            self.steps.append(trainer.global_step)

    validation = ValidationSteps()
    trainer = build_step_trainer(
        config,
        [*callbacks, validation],
        logger,
        accelerator="cpu",
        devices=1,
        artifact_store=None,
    )
    trainer.fit(module, datamodule=dm)
    assert trainer.global_step == 6 and validation.steps == [2, 4, 6]
    checkpoint = Path(logger.log_dir) / "checkpoints/update-00000006.ckpt"
    assert (
        checkpoint.is_file() and (checkpoint.parent / "update-00000003.ckpt").is_file()
    )
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["hyper_parameters"]["data_condition"] == data_condition(dm)
    assert payload["lr_schedulers"][0]["last_epoch"] == 6
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(
        config.training.min_lr
    )
    resumed_module = runner.build_lightning_module(config, dm, steps_per_epoch=2)
    resumed_logger = TensorBoardLogger(save_dir=str(output), name="resume")
    resumed_trainer = build_step_trainer(
        config,
        step_checkpoints(config, resumed_logger),
        resumed_logger,
        accelerator="cpu",
        devices=1,
        artifact_store=None,
    )
    resume_path = str(checkpoint.parent / "update-00000003.ckpt")
    with runner.resume_checkpoint_load_env(resume_path):
        resumed_trainer.fit(resumed_module, datamodule=dm, ckpt_path=resume_path)
    assert resumed_trainer.global_step == 6
    assert resumed_trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(
        config.training.min_lr
    )
    invalid_config = config.copy()
    invalid_config.training.warmup_steps = 0
    invalid_module = runner.build_lightning_module(
        invalid_config, dm, steps_per_epoch=2
    )
    with pytest.raises(ValueError, match="changed training"):
        invalid_module.on_load_checkpoint(payload)
    relocated = tmp_path / "relocated-artifacts"
    shutil.copytree(tmp_path / "artifacts", relocated)
    for split in ("val", "test"):
        result = evaluate_checkpoint(
            checkpoint,
            tmp_path / f"eval-{split}",
            device="cpu",
            expected_step=6,
            real_split=split,
            series="A1",
            path_overrides={"artifact_root": str(relocated)},
        )
        measured = result["splits"][f"tennis_court_detector-{split}"]
        assert set(measured["metrics"]) == set(DENSE_METRICS)
        assert (
            not measured["pose_supervised"]
            and result["data_condition"]["accepted_images"] == 3
        )
    with pytest.raises(ValueError, match="expected-step"):
        evaluate_checkpoint(
            checkpoint,
            tmp_path / "wrong-step",
            device="cpu",
            expected_step=5,
            real_split="test",
            series="A1",
        )
    with pytest.raises(ValueError, match="explicit"):
        evaluate_checkpoint(
            checkpoint,
            tmp_path / "implicit-split",
            device="cpu",
            expected_step=6,
            series="A1",
        )
    report = compare_evaluations(
        [tmp_path / "eval-val/evaluation.json", tmp_path / "eval-test/evaluation.json"],
        tmp_path / "comparison",
    )
    assert "fixture-v1" in report.read_text()
    assert (report.parent / "scaling.png").stat().st_size > 0
    config.data.source.data_version = "changed"
    module.config = config
    payload["hyper_parameters"]["config"] = config
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="condition"):
        evaluate_checkpoint(
            checkpoint,
            tmp_path / "drift",
            device="cpu",
            expected_step=6,
            real_split="val",
            series="A1",
        )
