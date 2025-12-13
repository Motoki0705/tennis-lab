"""Hydra configuration package for PLCS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    """Model hyperparameters."""

    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    use_court_context: bool = True
    max_seq_len: Optional[int] = 120
    architecture: str = "frame"


@dataclass
class DataConfig:
    """Data loading and augmentation settings."""

    scene_dir: str = "data/plcs/scenes"
    batch_size: int = 64
    num_workers: int = 4
    val_split: float = 0.1
    test_split: float = 0.1
    camera_mode: str = "random"
    keypoint_noise_std: float = 0.01
    visibility_drop_prob: float = 0.05
    mode: str = "frame"
    seq_len: int = 16
    seq_stride: int = 16


@dataclass
class TrainingConfig:
    """Training and runtime configuration."""

    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    position_loss_weight: float = 1.0
    rotation_loss_weight: float = 1.0
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    seed: int = 42
    gpus: int = 1
    fast_dev_run: bool = False
    resume: Optional[str] = None
    output_dir: str = "outputs/plcs"


@dataclass
class CameraConfig:
    """Camera sampling configuration for simulation."""

    z_min: float = 3.0
    z_max: float = 5.0
    r_in: float = 1.0
    r_out: float = 2.0
    hfov_deg: float = 60.0
    image_size: tuple[int, int] = (1280, 720)


@dataclass
class MetricsConfig:
    """Evaluation thresholds."""

    position_threshold_m: float = 0.5
    angle_threshold_deg: float = 15.0
    velocity_threshold_m: float = 1.0


@dataclass
class PLCSConfig:
    """Base configuration for frame-based PLCS training."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


@dataclass
class PLCSSequenceConfig(PLCSConfig):
    """Configuration for sequence PLCS training."""

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            num_layers=8,
            max_seq_len=120,
            architecture="sequence",
        )
    )
    data: DataConfig = field(
        default_factory=lambda: DataConfig(
            camera_mode="all",
            mode="sequence",
            seq_len=32,
            seq_stride=8,
        )
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(output_dir="outputs/plcs_sequence")
    )


@dataclass
class SimulationSettings:
    """Simulation batch parameters."""

    num_scenes: int = 3000
    num_cameras: int = 5
    output_dir: str = "data/plcs"
    human_visibility_threshold: float = 0.8
    court_visibility_threshold: int = 15


@dataclass
class MotionSource:
    """Motion source weighting and paths."""

    paths: List[str]
    weight: float


@dataclass
class SimulationConfig:
    """Simulation and data generation configuration."""

    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    camera: CameraConfig = field(default_factory=CameraConfig)
    motion_sources: Dict[str, MotionSource] = field(
        default_factory=lambda: {
            "running": MotionSource(
                paths=[
                    "data/ACCAD/Female1Running_c3d",
                    "data/ACCAD/Male1Running_c3d",
                    "data/ACCAD/Male2Running_c3d",
                    "data/ACCAD/s008",
                    "data/ACCAD/s009",
                ],
                weight=0.5,
            ),
            "walking": MotionSource(
                paths=[
                    "data/ACCAD/Female1Walking_c3d",
                    "data/ACCAD/Male1Walking_c3d",
                    "data/ACCAD/Male2Walking_c3d",
                    "data/ACCAD/s007",
                    "data/ACCAD/s011",
                ],
                weight=0.4,
            ),
            "general": MotionSource(
                paths=[
                    "data/ACCAD/Female1General_c3d",
                    "data/ACCAD/Female1Gestures_c3d",
                    "data/ACCAD/Male1General_c3d",
                    "data/ACCAD/Male2General_c3d",
                    "data/ACCAD/s001",
                ],
                weight=0.1,
            ),
        }
    )
    smplh_model_path: str = "data/smplx/smplh"
    category: Optional[str] = None
    seed: int = 42
    device: str = "auto"


@dataclass
class VisualizationConfig:
    """Configuration for visualization and prediction CLI."""

    mode: str = "visualize"
    scene_path: str = "data/plcs/scenes/scene_000000.npz"
    frame: int = 0
    view: str = "multi"
    camera: int = 0
    animation_view: str = "2d_topdown"
    fps: Optional[float] = None
    save: Optional[str] = None
    info: bool = False
    checkpoint: Optional[str] = None
    device: str = "auto"


def register_configs() -> None:
    """Register PLCS configs with the Hydra ConfigStore."""

    cs = ConfigStore.instance()

    if not cs.exists("plcs"):
        cs.store(name="plcs", node=PLCSConfig())

    if not cs.exists("plcs_sequence"):
        cs.store(name="plcs_sequence", node=PLCSSequenceConfig())

    if not cs.exists("plcs_simulation"):
        cs.store(name="plcs_simulation", node=SimulationConfig())

    if not cs.exists("plcs_visualization"):
        cs.store(name="plcs_visualization", node=VisualizationConfig())


__all__ = [
    "CameraConfig",
    "DataConfig",
    "ModelConfig",
    "MotionSource",
    "PLCSConfig",
    "PLCSSequenceConfig",
    "SimulationConfig",
    "SimulationSettings",
    "TrainingConfig",
    "VisualizationConfig",
    "register_configs",
]
