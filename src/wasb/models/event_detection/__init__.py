"""Event detection models for WASB.

These models take a ball (x, y) trajectory sequence and predict a per-frame
event class (0: none, 1: shot, 2: bounce).
"""

from .transformer import TrajectoryEventTransformer

__all__ = ["TrajectoryEventTransformer"]

