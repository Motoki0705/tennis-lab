"""Image-feature model (HMR2.0a)."""

from src.submodules.models.hmr2.feature_extractor import (
    Hmr2FeatureExtractor,
    ImageFeatureRequest,
    ImageFeatureResult,
)

__all__ = [
    "Hmr2FeatureExtractor",
    "ImageFeatureRequest",
    "ImageFeatureResult",
]
