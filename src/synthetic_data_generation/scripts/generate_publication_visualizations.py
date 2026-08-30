"""Generate one validated PNG/GIF publication bundle for a canonical synthetic scene.

Usage:
    python -m src.synthetic_data_generation.scripts.generate_publication_visualizations publication.scene_id=B00 publication.court.trajectory_id=<id> publication.court.frame_indices='[0,<last>]' publication.blcs.logical_scene_id=<id> publication.blcs.camera_id=<id> publication.blcs.frame_indices='[0,<last>]' publication.blcs.camera_ids='[<ids>]' publication.plcs.logical_scene_id=<id> publication.plcs.camera_id=<id> publication.plcs.frame_indices='[0,<last>]' publication.plcs.camera_ids='[<ids>]' publication.captured.camera_ids='[<ids>]'

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/generate_publication_visualizations.yaml`.
    - Every dataset selection includes both timeline endpoints and every camera inventory is explicit.
    - The complete bundle is validated in private staging before one atomic directory publication.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.synthetic_data_generation.visualization.publication import (
    generate_publication_bundle,
)
from src.synthetic_data_generation.visualization.publication.configuration import (
    build_publication_request,
)
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="generate_publication_visualizations",
    version_base="1.3",
    validation_boundary="synthetic.publication_visualization",
)
def main(config: DictConfig) -> int:  # pragma: no cover - Hydra CLI boundary
    """Generate, validate, and report one complete publication bundle."""
    result = generate_publication_bundle(build_publication_request(config))
    print(f"bundle={result.bundle_path}")
    print(f"manifest={result.manifest_path}")
    print(f"artifacts={len(result.manifest.artifacts)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
