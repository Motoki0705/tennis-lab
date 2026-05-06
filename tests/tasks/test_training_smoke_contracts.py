from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning_fabric.utilities.exceptions import MisconfigurationException
from omegaconf import DictConfig

from src.tasks.blcs.training.runner import BLCSTrainingRunner
from src.tasks.court_detection.training.runner import CourtDetectionTrainingRunner
from src.tasks.event_detection.training.runner import EventDetectionTrainingRunner
from src.tasks.plcs.training.runner import PLCSTrainingRunner
from src.tasks.trajectory_completion.training.runner import (
    TrajectoryCompletionTrainingRunner,
)

RunnerFactory = Any  # Callable[[DictConfig], runner_instance]


@dataclass(frozen=True)
class SmokeCase:
    name: str
    runner: type[Any]
    config_dir: Path
    config_name: str
    overrides: tuple[str, ...]
    supports_test_phase: bool
    expect_qualitative: bool
    expect_gan_training: bool = False
    runner_factory: RunnerFactory | None = None


@dataclass(frozen=True)
class SmokeVariant:
    name: str
    overrides: tuple[str, ...] = ()
    expect_gan_training: bool = False


@dataclass(frozen=True)
class SmokeTaskSpec:
    name: str
    runner: type[Any]
    config_dir: Path
    config_name: str
    default_overrides: tuple[str, ...]
    variants: tuple[SmokeVariant, ...]
    supports_test_phase: bool
    expect_qualitative: bool
    runner_factory: RunnerFactory | None = None


REPO_ROOT = Path(__file__).resolve().parents[2]


def _blcs_chunked_runner_factory(config: DictConfig) -> BLCSTrainingRunner:
    """Build a BLCSTrainingRunner with generator_config for chunked training."""
    from src.tasks.blcs.scripts.generate_dataset import build_generator_config

    generator_config = build_generator_config(config)
    return BLCSTrainingRunner(generator_config=generator_config)


COMMON_OVERRIDES = (
    "run.gpus=0",
    "data.batch_size=1",
    "data.num_workers=0",
    "data.pin_memory=false",
    "training.trainer.max_epochs=1",
    "training.trainer.limit_train_batches=1",
    "training.trainer.limit_val_batches=1",
    "training.trainer.num_sanity_val_steps=0",
    "training.trainer.precision=32-true",
    "training.trainer.enable_progress_bar=false",
    "training.trainer.enable_model_summary=false",
    "training.trainer.log_every_n_steps=1",
    "training.checkpoint.save_top_k=1",
    "training.checkpoint.save_last=true",
    "training.qualitative_logging.enabled=true",
    "training.qualitative_logging.every_n_epochs=1",
    "training.qualitative_logging.num_samples=1",
    "training.qualitative_logging.selection_mode=fixed_indices",
    "training.qualitative_logging.selected_indices=[0]",
)


SMOKE_TASK_SPECS = (
    SmokeTaskSpec(
        name="blcs",
        runner=BLCSTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/blcs/configs",
        config_name="train",
        default_overrides=(
            "data.seq_len_range=[16,16]",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
        ),
        variants=(
            SmokeVariant(name="single"),
            SmokeVariant(
                name="single_gan",
                overrides=(
                    "training.trainer.max_epochs=2",
                    "training.gan.transition.patience=0",
                    "training.gan.warmup_epochs=1",
                    "training.gan.discriminator.hidden_dim=32",
                    "training.gan.discriminator.num_layers=2",
                    "training.gan.discriminator.num_heads=4",
                    "training.gan.discriminator.ffn_dim=64",
                    "training.gan.discriminator.max_seq_len=64",
                ),
                expect_gan_training=True,
            ),
            SmokeVariant(
                name="multiview",
                overrides=(
                    "model=multiview",
                    "data=multiview",
                ),
            ),
            SmokeVariant(
                name="multiview_num_court_kp_12",
                overrides=(
                    "model=multiview",
                    "data=multiview",
                    "data.num_court_kp=12",
                ),
            ),
            SmokeVariant(
                name="multiview_axial",
                overrides=(
                    "model=multiview_axial",
                    "data=multiview",
                ),
            ),
            SmokeVariant(
                name="multiview_axial_num_court_kp_12",
                overrides=(
                    "model=multiview_axial",
                    "data=multiview",
                    "data.num_court_kp=12",
                ),
            ),
        ),
        supports_test_phase=True,
        expect_qualitative=True,
    ),
    SmokeTaskSpec(
        name="plcs",
        runner=PLCSTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/plcs/configs",
        config_name="train",
        default_overrides=(
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
        ),
        variants=(
            SmokeVariant(name="frame"),
            SmokeVariant(
                name="sequence",
                overrides=(
                    "data=sequence",
                    "loss=sequence",
                ),
            ),
            SmokeVariant(
                name="multiview",
                overrides=(
                    "model=multiview",
                    "data=multiview",
                    "loss=multiview_sequence",
                    "data.seq_len_range=[16,16]",
                    "model.max_seq_len=64",
                ),
            ),
            SmokeVariant(
                name="multiview_num_court_kp_12",
                overrides=(
                    "model=multiview",
                    "data=multiview",
                    "loss=multiview_sequence",
                    "data.seq_len_range=[16,16]",
                    "data.num_court_kp=12",
                    "model.max_seq_len=64",
                ),
            ),
            SmokeVariant(
                name="multiview_axial",
                overrides=(
                    "model=multiview_axial",
                    "data=multiview",
                    "loss=multiview_sequence",
                    "data.seq_len_range=[16,16]",
                    "model.max_seq_len=64",
                ),
            ),
            SmokeVariant(
                name="multiview_axial_num_court_kp_12",
                overrides=(
                    "model=multiview_axial",
                    "data=multiview",
                    "loss=multiview_sequence",
                    "data.seq_len_range=[16,16]",
                    "data.num_court_kp=12",
                    "model.max_seq_len=64",
                ),
            ),
        ),
        supports_test_phase=True,
        expect_qualitative=True,
    ),
    SmokeTaskSpec(
        name="plcs_chunked",
        runner=PLCSTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/plcs/configs",
        config_name="train_chunked",
        default_overrides=(
            "run.device=cpu",
            "data.seq_len_range=[16,16]",
            "data.chunk.scenes_per_chunk=3",
            "data.chunk.epochs_per_chunk=1",
            "data.chunk.prefetch_chunks=0",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
        ),
        variants=(
            SmokeVariant(name="multiview"),
            SmokeVariant(
                name="multiview_gan",
                overrides=(
                    "training.gan.enabled=true",
                    "training.trainer.max_epochs=2",
                    "training.gan.transition.patience=0",
                    "training.gan.warmup_epochs=1",
                    "training.gan.discriminator.hidden_dim=32",
                    "training.gan.discriminator.num_layers=2",
                    "training.gan.discriminator.num_heads=4",
                    "training.gan.discriminator.ffn_dim=64",
                    "training.gan.discriminator.max_seq_len=64",
                ),
                expect_gan_training=True,
            ),
        ),
        supports_test_phase=True,
        expect_qualitative=True,
    ),
    SmokeTaskSpec(
        name="plcs_chunked",
        runner=PLCSTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/plcs/configs",
        config_name="train_chunked",
        default_overrides=(
            "data.seq_len_range=[16,16]",
            "data.chunk.scenes_per_chunk=2",
            "data.chunk.epochs_per_chunk=1",
            "data.chunk.prefetch_chunks=0",
            "data.chunk.generation_workers=2",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
        ),
        variants=(SmokeVariant(name="multiview"),),
        supports_test_phase=True,
        expect_qualitative=True,
    ),
    SmokeTaskSpec(
        name="event_detection_uv",
        runner=EventDetectionTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/event_detection/configs",
        config_name="train_uv",
        default_overrides=(
            "data.seq_len_range=[16,16]",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
        ),
        variants=(
            SmokeVariant(name="uv_transformer"),
            SmokeVariant(
                name="uv_transformer_nocourt",
                overrides=("model=uv_transformer_nocourt",),
            ),
        ),
        supports_test_phase=False,
        expect_qualitative=True,
    ),
    SmokeTaskSpec(
        name="event_detection_uv_chunked",
        runner=EventDetectionTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/event_detection/configs",
        config_name="train_uv_chunked",
        default_overrides=(
            "data.seq_len_range=[16,16]",
            "data.chunk.scenes_per_chunk=3",
            "data.chunk.epochs_per_chunk=1",
            "data.chunk.prefetch_chunks=0",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
        ),
        variants=(
            SmokeVariant(name="uv_transformer"),
            SmokeVariant(
                name="uv_transformer_gan",
                overrides=(
                    "training.gan.enabled=true",
                    "training.trainer.max_epochs=2",
                    "training.gan.transition.patience=0",
                    "training.gan.warmup_epochs=1",
                    "training.gan.discriminator.hidden_dim=32",
                    "training.gan.discriminator.num_layers=2",
                    "training.gan.discriminator.num_heads=4",
                    "training.gan.discriminator.ffn_dim=64",
                    "training.gan.discriminator.max_seq_len=64",
                ),
                expect_gan_training=True,
            ),
        ),
        supports_test_phase=False,
        expect_qualitative=True,
    ),
    SmokeTaskSpec(
        name="court_detection",
        runner=CourtDetectionTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/court_detection/configs",
        config_name="train",
        default_overrides=(),
        variants=(
            SmokeVariant(
                name="seg",
                overrides=(
                    "data=court_seg",
                    "model=court_seg",
                    "loss=seg",
                ),
            ),
            SmokeVariant(
                name="kp",
                overrides=(
                    "data=court_kp",
                    "model=court_kp",
                    "loss=kp",
                ),
            ),
            SmokeVariant(
                name="line",
                overrides=(
                    "data=court_line",
                    "model=court_line",
                    "loss=line",
                ),
            ),
        ),
        supports_test_phase=False,
        expect_qualitative=True,
    ),
    SmokeTaskSpec(
        name="event_detection_3d",
        runner=EventDetectionTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/event_detection/configs",
        config_name="train_3d",
        default_overrides=(
            "data.seq_len_range=[16,16]",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
        ),
        variants=(SmokeVariant(name="traj3d_transformer"),),
        supports_test_phase=False,
        expect_qualitative=True,
    ),
    SmokeTaskSpec(
        name="event_detection_3d_chunked",
        runner=EventDetectionTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/event_detection/configs",
        config_name="train_3d_chunked",
        default_overrides=(
            "data.seq_len_range=[16,16]",
            "data.chunk.scenes_per_chunk=3",
            "data.chunk.epochs_per_chunk=1",
            "data.chunk.prefetch_chunks=0",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
        ),
        variants=(
            SmokeVariant(name="traj3d_transformer"),
            SmokeVariant(
                name="traj3d_transformer_gan",
                overrides=(
                    "training.gan.enabled=true",
                    "training.trainer.max_epochs=2",
                    "training.gan.transition.patience=0",
                    "training.gan.warmup_epochs=1",
                    "training.gan.discriminator.hidden_dim=32",
                    "training.gan.discriminator.num_layers=2",
                    "training.gan.discriminator.num_heads=4",
                    "training.gan.discriminator.ffn_dim=64",
                    "training.gan.discriminator.max_seq_len=64",
                ),
                expect_gan_training=True,
            ),
        ),
        supports_test_phase=False,
        expect_qualitative=True,
    ),
    SmokeTaskSpec(
        name="blcs_chunked",
        runner=BLCSTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/blcs/configs",
        config_name="train_chunked",
        default_overrides=(
            "data.seq_len_range=[16,16]",
            "data.chunk.scenes_per_chunk=3",
            "data.chunk.epochs_per_chunk=1",
            "data.chunk.prefetch_chunks=0",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
        ),
        variants=(
            SmokeVariant(name="multiview"),
            SmokeVariant(
                name="multiview_gan",
                overrides=(
                    "training.trainer.max_epochs=2",
                    "training.gan.transition.patience=0",
                    "training.gan.warmup_epochs=1",
                    "training.gan.discriminator.hidden_dim=32",
                    "training.gan.discriminator.num_layers=2",
                    "training.gan.discriminator.num_heads=4",
                    "training.gan.discriminator.ffn_dim=64",
                    "training.gan.discriminator.max_seq_len=64",
                ),
                expect_gan_training=True,
            ),
        ),
        supports_test_phase=True,
        expect_qualitative=True,
        runner_factory=_blcs_chunked_runner_factory,
    ),
    SmokeTaskSpec(
        name="trajectory_completion",
        runner=TrajectoryCompletionTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/trajectory_completion/configs",
        config_name="train",
        default_overrides=(
            "data.seq_len_range=[16,16]",
            "model.hidden_dim=32",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
        ),
        variants=(
            SmokeVariant(
                name="uv_transformer",
                overrides=(
                    "model.num_ball_layers=2",
                    "model.num_query_layers=1",
                ),
            ),
            SmokeVariant(
                name="uv_transformer_nocourt",
                overrides=(
                    "model=uv_transformer_nocourt",
                    "model.num_layers=2",
                ),
            ),
        ),
        supports_test_phase=True,
        expect_qualitative=True,
    ),
    SmokeTaskSpec(
        name="trajectory_completion_chunked",
        runner=TrajectoryCompletionTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/trajectory_completion/configs",
        config_name="train_chunked",
        default_overrides=(
            "data.seq_len_range=[16,16]",
            "data.chunk.scenes_per_chunk=3",
            "data.chunk.epochs_per_chunk=1",
            "data.chunk.prefetch_chunks=0",
            "model.hidden_dim=32",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
        ),
        variants=(
            SmokeVariant(
                name="uv_transformer",
                overrides=(
                    "model.num_ball_layers=2",
                    "model.num_query_layers=1",
                ),
            ),
            SmokeVariant(
                name="uv_transformer_gan",
                overrides=(
                    "model.num_ball_layers=2",
                    "model.num_query_layers=1",
                    "training.gan.enabled=true",
                    "training.trainer.max_epochs=2",
                    "training.gan.transition.patience=0",
                    "training.gan.warmup_epochs=1",
                    "training.gan.discriminator.hidden_dim=32",
                    "training.gan.discriminator.num_layers=2",
                    "training.gan.discriminator.num_heads=4",
                    "training.gan.discriminator.ffn_dim=64",
                    "training.gan.discriminator.max_seq_len=64",
                ),
                expect_gan_training=True,
            ),
        ),
        supports_test_phase=True,
        expect_qualitative=True,
    ),
)


def _build_smoke_cases(task_specs: tuple[SmokeTaskSpec, ...]) -> tuple[SmokeCase, ...]:
    cases: list[SmokeCase] = []
    for task_spec in task_specs:
        for variant in task_spec.variants:
            variant_name = f"{task_spec.name}_{variant.name}"
            cases.append(
                SmokeCase(
                    name=variant_name,
                    runner=task_spec.runner,
                    config_dir=task_spec.config_dir,
                    config_name=task_spec.config_name,
                    overrides=(
                        *COMMON_OVERRIDES,
                        *task_spec.default_overrides,
                        *variant.overrides,
                    ),
                    supports_test_phase=task_spec.supports_test_phase,
                    expect_qualitative=task_spec.expect_qualitative,
                    expect_gan_training=variant.expect_gan_training,
                    runner_factory=task_spec.runner_factory,
                )
            )
    return tuple(cases)


SMOKE_CASES = _build_smoke_cases(SMOKE_TASK_SPECS)


def _compose_config(case: SmokeCase, output_dir: Path) -> DictConfig:
    GlobalHydra.instance().clear()
    overrides = [_normalize_override(f"run.output_dir={output_dir.as_posix()}")]
    overrides.extend(_normalize_override(override) for override in case.overrides)
    with initialize_config_dir(version_base="1.3", config_dir=str(case.config_dir)):
        return compose(config_name=case.config_name, overrides=overrides)


def _normalize_override(override: str) -> str:
    if override.startswith(("+", "~")):
        return override
    key, separator, value = override.partition("=")
    if not separator:
        return override
    if "." not in key:
        return override
    return f"++{key}={value}"


def _assert_common_artifacts(output_dir: Path, *, expect_qualitative: bool) -> None:
    assert (output_dir / "config.yaml").is_file()

    log_dir = output_dir / "logs" / "version_0"
    assert log_dir.is_dir()

    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert event_files

    checkpoint_dir = log_dir / "checkpoints"
    assert checkpoint_dir.is_dir()
    assert (checkpoint_dir / "last.ckpt").is_file()

    saved_checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    assert saved_checkpoints

    qualitative_dir = log_dir / "qualitative" / "epoch_0000"
    if expect_qualitative:
        qualitative_images = list(qualitative_dir.glob("*.png"))
        assert qualitative_images
    else:
        assert not qualitative_dir.exists()


def _assert_gan_training(case: SmokeCase, callbacks: list[Any], lightning_module: Any) -> None:
    if not case.expect_gan_training:
        return

    from src.tasks.base.training.gan_transition_callback import GANTransitionCallback

    gan_callbacks = [cb for cb in callbacks if isinstance(cb, GANTransitionCallback)]
    assert gan_callbacks

    gan_callback = gan_callbacks[0]
    assert gan_callback.has_switched_to_gan
    assert lightning_module.gan_phase_active
    assert lightning_module.supervised_only_step_count > 0
    assert lightning_module.hybrid_gan_step_count > 0
    assert lightning_module.current_gan_weight > 0


@pytest.mark.integration
@pytest.mark.local_data
@pytest.mark.slow
@pytest.mark.parametrize("case", SMOKE_CASES, ids=[case.name for case in SMOKE_CASES])
def test_scene_training_smoke_contracts(case: SmokeCase, tmp_path: Path) -> None:
    output_dir = tmp_path / case.name
    config = _compose_config(case, output_dir)

    if case.runner_factory is not None:
        runner = case.runner_factory(config)
    else:
        runner = case.runner()
    runner.seed_everything(config)
    runner.apply_runtime_settings(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    runner.save_config(config, output_dir)

    datamodule = runner.build_datamodule(config)
    lightning_module = runner.build_lightning_module(config, datamodule)
    logger = runner.build_logger(config, output_dir)
    callbacks = runner.build_callbacks(config, datamodule, logger)
    trainer = runner.build_trainer(config, callbacks, logger)

    trainer.fit(lightning_module, datamodule=datamodule)
    _assert_gan_training(case, callbacks, lightning_module)

    if case.supports_test_phase:
        trainer.test(lightning_module, datamodule=datamodule)
    else:
        with pytest.raises(
            (AttributeError, MisconfigurationException, NotImplementedError, RuntimeError)
        ):
            trainer.test(lightning_module, datamodule=datamodule)

    _assert_common_artifacts(output_dir, expect_qualitative=case.expect_qualitative)
