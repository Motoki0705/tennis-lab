"""Inspect discovered clip layout and annotation availability."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.ball_detection.data.io.layout import discover_clip_layouts


@hydra.main(config_path="../configs", config_name="generate_pseudo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    layouts = discover_clip_layouts(str(cfg.data.root_dir))
    n_with_csv = sum(1 for x in layouts if x.label_csv.exists())
    print(
        {
            "total_clips": len(layouts),
            "with_label_csv": n_with_csv,
            "without_label_csv": len(layouts) - n_with_csv,
        }
    )


if __name__ == "__main__":
    main()
