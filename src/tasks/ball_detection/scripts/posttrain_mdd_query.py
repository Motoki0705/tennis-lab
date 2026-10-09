"""Train the selected pretrained CNN's query decoder with video augmentation."""
from src.tasks.ball_detection.training.posttraining.runner import PATH_BOUNDARY, main

__all__ = ["PATH_BOUNDARY", "main"]

if __name__ == "__main__":
    main()
