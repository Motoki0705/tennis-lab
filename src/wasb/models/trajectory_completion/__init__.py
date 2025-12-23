"""Pure `torch.nn.Module` architectures for trajectory completion.

This package intentionally contains only model architectures (no inference-time
wrappers such as `complete()` or checkpoint IO). Those live under
`src/wasb/inference/trajectory/trajectory_completion.py`.
"""

from .bilstm import TrajectoryBiLSTM
from .refiner import TrajectoryDeltaTransformer
from .transformer import TrajectoryTransformer

__all__ = [
    "TrajectoryBiLSTM",
    "TrajectoryDeltaTransformer",
    "TrajectoryTransformer",
]
