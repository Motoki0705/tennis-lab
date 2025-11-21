"""Tennis-specific multi-view pose models."""

from __future__ import annotations

from .config import TennisDetrConfig
from .config_v2 import TennisDetrV2Config
from .config_v3 import TennisDetrV3Config
from .factory import (
    create_default_config,
    create_tennis_model,
    get_available_model_versions,
    validate_config_for_version,
)
from .model import TennisDETR
from .model_v2 import TennisDETR_v2
from .model_v2_5 import TennisDETR_v2_5
from .model_v3 import TennisDETR_v3

__all__ = [
    "TennisDetrConfig",
    "TennisDetrV2Config",
    "TennisDetrV3Config",
    "TennisDETR",
    "TennisDETR_v2",
    "TennisDETR_v2_5",
    "TennisDETR_v3",
    "create_tennis_model",
    "get_available_model_versions",
    "validate_config_for_version",
    "create_default_config",
]
