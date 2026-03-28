"""Train a PLCS model with Hydra-managed configuration.

Example commands:
    `uv run python -m src.tasks.plcs.scripts.train`
    `uv run python -m src.tasks.plcs.scripts.train run.gpus=0 training.max_epochs=1`
    `uv run python -m src.tasks.plcs.scripts.train run.dry_run=true`

Config entry point: `src/tasks/plcs/configs/train.yaml`
"""

# mypy: disable-error-code=misc

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from collections.abc import Callable
from typing import TypeVar, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.tasks.plcs.training.runner import PLCSTrainingRunner

F = TypeVar("F", bound=Callable[..., object])
hydra.main = cast(Callable[..., Callable[[F], F]], hydra.main)


@dataclass
class ModelIOConfig:
    input_profile: str
    strict_input: bool


@dataclass
class MoEConfigSchema:
    moe_inter_dim: int
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    n_expert_groups: int
    n_limited_groups: int
    score_func: str
    route_scale: float


@dataclass
class ModelConfig:
    name: str
    io: ModelIOConfig
    hidden_dim: int
    num_layers: int | None
    num_heads: int
    ffn_dim: int | None
    dropout: float
    invisible_init_std: float
    num_register_tokens: int | None
    use_kp_id_embedding: bool | None
    use_rope: bool | None
    rope_dim: int | None
    rope_theta: float
    yarn: dict[str, int | float] | None
    use_moe: bool
    moe_config: MoEConfigSchema | None
    max_views: int | None
    max_seq_len: int
    num_player_layers: int | None
    num_query_layers: int | None
    query_init_std: float | None
    architecture: str | None


@dataclass
class AugmentationConfig:
    keypoint_noise_std: float
    visibility_drop_prob: float


@dataclass
class DataConfig:
    scene_dir: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    mode: str
    camera_mode: str
    seq_len_range: list[int]
    num_views_range: list[int]
    scene_sampler: bool
    scenes_per_batch: int
    chunk_max_scenes: int
    adapter_camera_index: int
    augmentation: AugmentationConfig


@dataclass
class TemporalLossTermConfig:
    weight: float
    order: int
    robust: bool


@dataclass
class TemporalLossConfig:
    position_gt: TemporalLossTermConfig
    position_inertia: TemporalLossTermConfig
    rotation_gt: TemporalLossTermConfig
    rotation_inertia: TemporalLossTermConfig


@dataclass
class LossConfig:
    position_weight: float
    rotation_weight: float
    temporal: TemporalLossConfig


@dataclass
class MetricsConfig:
    position_threshold_m: float
    angle_threshold_deg: float
    velocity_threshold_m: float


@dataclass
class TrainerConfig:
    max_epochs: int
    gradient_clip_val: float | None
    deterministic: bool
    precision: str | None
    log_every_n_steps: int
    check_val_every_n_epoch: int
    benchmark: bool | None


@dataclass
class CheckpointConfig:
    enabled: bool
    filename: str
    monitor: str
    mode: str
    save_top_k: int
    save_last: bool


@dataclass
class EarlyStoppingConfig:
    enabled: bool
    monitor: str
    mode: str
    patience: int
    min_delta: float | None
    check_on_train_epoch_end: bool


@dataclass
class LRMonitorConfig:
    enabled: bool
    interval: str


@dataclass
class OptimizerConfig:
    betas: list[float] | None


@dataclass
class TrainingConfig:
    trainer: TrainerConfig
    learning_rate: float
    weight_decay: float
    warmup_steps: int | None
    warmup_epochs: int | None
    scheduler: str
    min_lr: float
    matmul_precision: str
    allow_tf32: bool
    optimizer: OptimizerConfig
    checkpoint: CheckpointConfig
    early_stopping: EarlyStoppingConfig
    lr_monitor: LRMonitorConfig
    steps_per_epoch: int | None
    num_samples_per_epoch: int | None
    max_epochs: int


@dataclass
class RunConfig:
    output_dir: str
    seed: int | None
    gpus: int
    resume: str | None
    fast_dev_run: bool
    dry_run: bool


@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig
    metrics: MetricsConfig
    run: RunConfig


if TYPE_CHECKING:
    StructuredTrainConfig = TrainConfig
    RuntimeTrainConfig = TrainConfig
else:
    StructuredTrainConfig = DictConfig
    RuntimeTrainConfig = DictConfig


def _validate_structured_config(config: DictConfig) -> DictConfig:
    schema = OmegaConf.structured(TrainConfig)
    return cast(DictConfig, OmegaConf.merge(schema, config))


def run_training(config: RuntimeTrainConfig) -> None:
    """Execute PLCS training with the provided configuration."""
    runner = PLCSTrainingRunner()
    runner.run(cast(DictConfig, config))


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: StructuredTrainConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for PLCS training."""
    validated = _validate_structured_config(cast(DictConfig, config))
    run_training(validated)


if __name__ == "__main__":
    main()
