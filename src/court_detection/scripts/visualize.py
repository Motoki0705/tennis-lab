"""Visualize court keypoint detection results.

Example:
    # Visualize predictions from a checkpoint
    uv run python -m src.court_detection.scripts.visualize \
        visualization.checkpoint=outputs/court_detection/checkpoints/last.ckpt \
        visualization.input_path=data/test_images/

Config entry point: `src/court_detection/configs/visualize.yaml`
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from src.court_detection.inference.predictor import CourtKeypointPredictor
from src.court_detection.inference.visualization import visualize_keypoints

LOGGER = logging.getLogger(__name__)


def visualize_predictions(
    checkpoint_path: Path,
    input_path: Path,
    output_dir: Path,
    num_samples: int = 10,
    vis_config: dict | None = None,
) -> None:
    """Visualize model predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictor
    predictor = CourtKeypointPredictor.load_from_checkpoint(
        checkpoint_path,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )

    # Find input files
    if input_path.is_dir():
        # Check for images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(input_path.glob(ext))
        image_files = sorted(image_files)[:num_samples]

        # Process images
        for image_file in image_files:
            image = np.array(Image.open(image_file).convert("RGB"))
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Run prediction
            result = predictor.predict(image)

            # Convert tensors to numpy for visualization
            keypoints = result["keypoints"].numpy()
            visibility = result["visibility"].numpy()

            # Visualize
            vis = visualize_keypoints(
                image_bgr,
                keypoints,
                visibility,
                config=vis_config,
            )

            # Save
            output_path = output_dir / f"{image_file.stem}_pred.png"
            cv2.imwrite(str(output_path), vis)
            LOGGER.info("Saved: %s", output_path)
    else:
        # Single file
        image = np.array(Image.open(input_path).convert("RGB"))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        result = predictor.predict(image)

        # Convert tensors to numpy for visualization
        keypoints = result["keypoints"].numpy()
        visibility = result["visibility"].numpy()

        vis = visualize_keypoints(
            image_bgr,
            keypoints,
            visibility,
            config=vis_config,
        )

        output_path = output_dir / f"{input_path.stem}_pred.png"
        cv2.imwrite(str(output_path), vis)
        LOGGER.info("Saved: %s", output_path)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="visualize",
)
def main(cfg: DictConfig) -> None:
    """Visualize court keypoint detection results."""
    LOGGER.info("Starting visualization")

    input_path = cfg.visualization.get("input_path")
    output_dir = Path(cfg.visualization.get("output_dir", "outputs/court_detection/visualize"))
    num_samples = cfg.visualization.get("num_samples", 10)
    checkpoint_path = cfg.visualization.get("checkpoint")

    vis_config = OmegaConf.to_container(cfg.get("visualization", {}))

    if input_path is None:
        raise ValueError("visualization.input_path is required")

    input_path = Path(input_path)

    if checkpoint_path is None:
        raise ValueError("visualization.checkpoint is required")
    visualize_predictions(
        checkpoint_path=Path(checkpoint_path),
        input_path=input_path,
        output_dir=output_dir,
        num_samples=num_samples,
        vis_config=vis_config,
    )

    LOGGER.info("Visualization complete")


if __name__ == "__main__":
    main()
