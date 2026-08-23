"""Canonical typed model I/O API for SLCS."""

from src.tasks.slcs.model_io.adapter import SLCSModelIOAdapter, SLCSModelIOSpec
from src.tasks.slcs.model_io.checkpoints import (
    load_slcs_checkpoint_mapping,
    prepare_slcs_checkpoint_config,
)
from src.tasks.slcs.model_io.contracts import (
    SLCSClipPrediction,
    SLCSDecodedOutput,
    SLCSPhysicalOutput,
    SLCSRawOutput,
    SLCSTrainingTargets,
)
from src.tasks.slcs.model_io.frame_tokens import (
    BoundSLCSFrameTokenEncoder,
    FrameTokenBackbone,
    FrameTokenCall,
    SLCSFrameTokenIOAdapter,
)

__all__ = [
    "SLCSClipPrediction",
    "BoundSLCSFrameTokenEncoder",
    "FrameTokenBackbone",
    "FrameTokenCall",
    "SLCSDecodedOutput",
    "SLCSModelIOAdapter",
    "SLCSModelIOSpec",
    "SLCSFrameTokenIOAdapter",
    "SLCSPhysicalOutput",
    "SLCSRawOutput",
    "SLCSTrainingTargets",
    "load_slcs_checkpoint_mapping",
    "prepare_slcs_checkpoint_config",
]
