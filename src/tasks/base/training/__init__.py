"""Base training components."""

from src.tasks.base.training.batch_transfer import (
    move_batch_to_device_preserving_frozen_metadata,
)
from src.tasks.base.training.chunk_rotation_callback import ChunkRotationCallback
from src.tasks.base.training.compilation import CompilationTargetError, compile_modules
from src.tasks.base.training.gan_loss import LSGANLoss
from src.tasks.base.training.gan_training import (
    ManualGANSupportMixin,
    ManualGANTrainingStrategy,
)
from src.tasks.base.training.gan_transition_callback import GANTransitionCallback
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.metric_logging import (
    MetricContractError,
    MetricLoggingContract,
    StageMetricContract,
    WeightedMetricAccumulator,
    evaluation_only_metric_logging_contract,
    uniform_metric_logging_contract,
)
from src.tasks.base.training.qualitative_callback import QualitativeLoggingCallback
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig

__all__ = [
    "BaseLightningModule",
    "BaseTrainingRunner",
    "ChunkRotationCallback",
    "CompilationTargetError",
    "GANTransitionCallback",
    "LSGANLoss",
    "ManualGANSupportMixin",
    "ManualGANTrainingStrategy",
    "MetricContractError",
    "MetricLoggingContract",
    "QualitativeLoggingCallback",
    "StageMetricContract",
    "TrackingMetricConfig",
    "WeightedMetricAccumulator",
    "compile_modules",
    "evaluation_only_metric_logging_contract",
    "move_batch_to_device_preserving_frozen_metadata",
    "uniform_metric_logging_contract",
]
