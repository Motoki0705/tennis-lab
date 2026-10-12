"""Coordinate learning with explicit scratch or pretrained-encoder initialization."""
from src.utils.configuration.cli import dispatch_cli


def main() -> None:
    dispatch_cli("initialize", {
        "scratch": "src.tasks.ball_detection.training.coordinates.scratch",
        "encoder": "src.tasks.ball_detection.training.coordinates.finetune",
    }, description="Coordinate regression; choose the model initialization explicitly.")
