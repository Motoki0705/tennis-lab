"""Composition root for the sole SLCS model and its I/O adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.slcs.configuration import (
    SLCSDataRuntimeConfig,
    SLCSModelConfig,
    SLCSPrecomputeConfig,
)
from src.tasks.slcs.model_io.adapter import SLCSModelIOAdapter, SLCSModelIOSpec
from src.tasks.slcs.model_io.contracts import SLCSDecodedOutput, SLCSRawOutput
from src.tasks.slcs.model_io.frame_tokens import (
    BoundSLCSFrameTokenEncoder,
    SLCSFrameTokenIOAdapter,
)
from src.tasks.slcs.models.slcs_model import SLCSFusionModel
from src.utils.device import resolve_device
from src.utils.models.loading.dinov3 import load_dinov3_backbone

SLCSBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object], SLCSRawOutput, SLCSDecodedOutput
]


def create_slcs_model_io(
    model_config: SLCSModelConfig,
    data_config: SLCSDataRuntimeConfig,
) -> tuple[SLCSFusionModel, SLCSModelIOAdapter, SLCSBoundModelIO]:
    """Construct and bind the canonical model-adapter pair exactly once."""
    model = SLCSFusionModel.from_config(model_config, data_config)
    dino = data_config.pipeline.dino_spec
    adapter = SLCSModelIOAdapter(
        SLCSModelIOSpec(
            num_players=data_config.pipeline.num_players,
            num_court_kp=data_config.pipeline.num_court_kp,
            max_seq_len=data_config.pipeline.window_size,
            dino_num_tokens=dino.num_tokens,
            dino_encoded_num_tokens=model.dino_encoder.num_tokens,
            dino_embed_dim=dino.embed_dim,
            log_b_min=model_config.log_b_min,
            log_b_max=model_config.log_b_max,
        ),
        data_config.pipeline.court_coordinate_normalization,
    )
    adapter.validate_model(model)
    binding = bind_model_io(model, adapter)
    return model, adapter, binding


def create_slcs_frame_token_encoder(
    config: SLCSPrecomputeConfig,
) -> BoundSLCSFrameTokenEncoder:
    """Load and bind the configured DINO backbone before precompute starts."""
    spec = config.data.pipeline.dino_spec
    device = resolve_device(config.device)
    model = load_dinov3_backbone(
        repository_path=config.repository_path,
        checkpoint_path=config.checkpoint_path,
        backbone_name=spec.backbone,
        strict=config.strict,
    )
    model.to(device)
    model.eval()
    adapter = SLCSFrameTokenIOAdapter(spec, device)
    backbone = adapter.validate_model(model)
    return BoundSLCSFrameTokenEncoder(model=backbone, adapter=adapter)


__all__ = [
    "SLCSBoundModelIO",
    "create_slcs_frame_token_encoder",
    "create_slcs_model_io",
]
