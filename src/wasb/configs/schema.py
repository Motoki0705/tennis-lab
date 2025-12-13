"""Structured configuration schemas for WASB training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class ResumeConfig:
    enabled: bool = False
    ckpt_path: str | None = None
    auto_last: bool = True


@dataclass
class TrainingConfig:
    max_epochs: int = 20
    freeze_backbone_epochs: int = 10
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    bce_weight: float = 1.0
    mse_weight: float = 1.0
    precision: str = "bf16-mixed"
    resume: ResumeConfig = field(default_factory=ResumeConfig)


@dataclass
class DataConfig:
    root_dir: str = "data/tennis"
    train_matches: list[str] = field(
        default_factory=lambda: [
            "game1",
            "game2",
            "game3",
            "game4",
            "game5",
            "game6",
            "game7",
            "game11",
            "game12",
            "game13",
            "game15",
            "game16",
        ]
    )
    val_matches: list[str] = field(
        default_factory=lambda: ["game8", "game9", "game14"]
    )
    test_matches: list[str] = field(default_factory=lambda: ["game10"])
    frames_in: int = 1
    frames_out: int = 1
    step: int = 1
    visibility_mode: str = "all_visible"
    image_ext: str = ".jpg"
    csv_filename: str = "Label.csv"
    batch_size: int = 4
    num_workers: int = 4
    resize_hw: list[int] | None = field(default_factory=lambda: [288, 512])
    heatmap_hw: list[int] | None = field(default_factory=lambda: [288, 512])
    heatmap_sigma: float | None = 2.0
    pin_memory: bool = True


@dataclass
class MetricsConfig:
    target_rmse_px: float = 5.0
    target_detection_rate: float = 0.9
    accuracy_thresh_px: float = 10.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class RunConfig:
    output_dir: str = "outputs/wasb"
    seed: int = 42
    gpus: int = 1
    fast_dev_run: bool = False
    dry_run: bool = False


@dataclass
class WasbTrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    run: RunConfig = field(default_factory=RunConfig)
    model: dict[str, Any] = field(default_factory=dict)


def register_configs() -> None:
    """Register structured configs for Hydra validation and tab completion."""

    cs = ConfigStore.instance()
    cs.store(name="wasb_train_config", node=WasbTrainConfig)
    cs.store(group="training", name="default", node=TrainingConfig())
    cs.store(group="data", name="default", node=DataConfig())
    cs.store(group="logging", name="default", node=LoggingConfig())
    cs.store(group="metrics", name="default", node=MetricsConfig())
    cs.store(group="run", name="default", node=RunConfig())

