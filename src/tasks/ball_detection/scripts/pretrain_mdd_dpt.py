"""Train only the CNN+DPT heatmap stage. Submit CUDA runs through training queue."""
from src.tasks.ball_detection.training.heatmap_pretraining.runner import (
    PATH_BOUNDARY,
    main,
)

__all__ = ["PATH_BOUNDARY", "main"]

if __name__ == "__main__":
    main()
