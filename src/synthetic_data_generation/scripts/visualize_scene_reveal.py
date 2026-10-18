"""Render the staged reveal animation of one canonical scene.

Usage:
    python -m src.synthetic_data_generation.scripts.visualize_scene_reveal
    python -m src.synthetic_data_generation.scripts.visualize_scene_reveal reveal.mode=stills 'reveal.still_times_seconds=[2,20,30]'

Notes:
    - Hydra loads configuration from `src/synthetic_data_generation/configs/visualize_scene_reveal.yaml`.
    - The viewpoint is a court-frame look-at camera; tune it with stills mode first.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.synthetic_data_generation.visualization.reveal import render_scene_reveal
from src.synthetic_data_generation.visualization.reveal.configuration import (
    build_scene_reveal_request,
)
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="visualize_scene_reveal",
    version_base="1.3",
    validation_boundary="synthetic.scene_reveal",
)
def main(config: DictConfig) -> int:  # pragma: no cover - Hydra CLI boundary
    """Render and report one staged reveal publication."""
    result = render_scene_reveal(build_scene_reveal_request(config))
    if result.video_path is not None:
        print(f"video={result.video_path}")
    for path in result.still_paths:
        print(f"still={path}")
    print(f"metadata={result.metadata_path}")
    print(f"frames={result.frame_count}")
    print(f"nht_states={result.nht_state_count}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
