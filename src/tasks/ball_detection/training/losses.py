"""Loss functions for supervised ball detection.

The focal heatmap loss previously implemented here as
``BallDetectionFocalLoss`` now lives in the shared foundation as
:class:`src.tasks.base.training.losses.FocalBCEWithLogitsLoss`. Use that class
directly with ``validate_shape=True`` to preserve the strict shape check.
"""

from __future__ import annotations

__all__: list[str] = []
