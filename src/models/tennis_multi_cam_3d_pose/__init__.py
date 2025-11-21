"""Tennis-specific multi-view pose models."""

from __future__ import annotations

from .config import TennisDetrConfig
from .config_v2 import TennisDetrV2Config
from .factory import (
    create_default_config,
    create_tennis_model,
    get_available_model_versions,
    validate_config_for_version,
)
from .model import TennisDETR
from .model_v2 import TennisDETR_v2
from .model_v2_5 import TennisDETR_v2_5

__all__ = [
    "TennisDetrConfig",
    "TennisDetrV2Config",
    "TennisDETR",
    "TennisDETR_v2",
    "TennisDETR_v2_5",
    "create_tennis_model",
    "get_available_model_versions",
    "validate_config_for_version",
    "create_default_config",
]
