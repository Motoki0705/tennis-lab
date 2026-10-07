"""MDD-only visual encoder and pose-conditioned coordinate detector."""

from .config import MDDPoseConfig
from .model import MDDPoseDetector

__all__ = ["MDDPoseConfig", "MDDPoseDetector"]
