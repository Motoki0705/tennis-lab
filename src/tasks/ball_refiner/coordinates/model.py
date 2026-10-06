"""Public coordinate generator API; implementations live in models/."""

from src.tasks.ball_refiner.coordinates.models.contracts import validate_input
from src.tasks.ball_refiner.coordinates.models.generators import CoordinateRefiner

__all__ = ["CoordinateRefiner", "validate_input"]
