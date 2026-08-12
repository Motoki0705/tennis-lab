"""Visualize one exact canonical generated-dataset view as an annotated MP4.

Usage:
    python -m src.synthetic_data_generation.scripts.visualize_dataset visualization.domain=court visualization.dataset_root=scenes/scene-000/datasets/court visualization.trajectory_id=orbit-000

Notes:
    - Hydra loads configuration from `src/synthetic_data_generation/configs/visualize_dataset.yaml`.
    - Court requires `trajectory_id`; BLCS and PLCS require `logical_scene_id` and `camera_id`.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.synthetic_data_generation.visualization import visualize_dataset
from src.synthetic_data_generation.visualization.configuration import (
    build_visualization_request,
)
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="visualize_dataset",
    version_base="1.3",
    validation_boundary="synthetic.dataset_visualization",
)
def main(config: DictConfig) -> int:  # pragma: no cover - Hydra CLI boundary
    """Render and report one deterministic visualization publication."""
    request = build_visualization_request(config)
    result = visualize_dataset(request)
    print(f"video={result.video_path}")
    print(f"metadata={result.metadata_path}")
    print(f"frames={result.frame_count}")
    print(f"resolution={result.width}x{result.height}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
