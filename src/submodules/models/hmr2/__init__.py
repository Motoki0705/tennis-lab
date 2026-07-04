"""Image-feature model (HMR2.0a)."""

from src.submodules.models.hmr2.feature_extractor import (
    DEFAULT_HMR2_CHECKPOINT,
    Hmr2FeatureExtractor,
    ImageFeatureRequest,
    ImageFeatureResult,
)

__all__ = [
    "DEFAULT_HMR2_CHECKPOINT",
    "Hmr2FeatureExtractor",
    "ImageFeatureRequest",
    "ImageFeatureResult",
]
