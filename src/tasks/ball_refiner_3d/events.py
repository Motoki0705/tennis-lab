"""Ordered channels shared by the targets, model and prediction artifacts."""

from src.utils.physics.ball.record import EVENT_BITS

EVENT_CLASSES = ("shot", "bounce", "net")
EVENT_CLASS_BITS = tuple(EVENT_BITS[kind] for kind in EVENT_CLASSES)
