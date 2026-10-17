"""Coordinate learning with explicit scratch, encoder or full-model initialization."""
from src.utils.configuration.cli import dispatch_cli


def main() -> None:
    dispatch_cli("initialize", {
        "scratch": "src.tasks.ball_detection.training.coordinates.scratch",
        "encoder": "src.tasks.ball_detection.training.coordinates.finetune",
        "full_model": "src.tasks.ball_detection.training.coordinates.full_model",
    }, description="Coordinate regression; choose the model initialization explicitly.")
