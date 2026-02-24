"""Inspect discovered video layout for pseudo-label generation."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.tasks.ball_detection.data.io.layout import discover_video_layouts


@hydra.main(config_path="../configs", config_name="generate_pseudo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    layouts = discover_video_layouts(
        str(cfg.data.video_root_dir),
        extensions=tuple(str(ext) for ext in cfg.data.video_extensions),
    )
    print(
        {
            "total_videos": len(layouts),
            "games": [x.game_name for x in layouts],
        }
    )


if __name__ == "__main__":
    main()
