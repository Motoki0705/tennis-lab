"""Local BLCS inference web UI.

The package serves a browser client that selects a checkpoint and a generated
scene, runs GPU trajectory inference, and compares the prediction against the
ground-truth ball trajectory on a 3D tennis court.
"""

from .service import InferenceService
from .web import create_app

__all__ = ["InferenceService", "create_app"]
