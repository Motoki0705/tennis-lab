"""Orchestrates a full DINOv3 SSL data collection run from Hydra config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from src.tasks.dino_ssl.generate_dataset.collectors import (
    CollectedImage,
    collect_from_source,
    deduplicate_images,
    reset_image_dir,
    write_synthetic_images,
)
from src.tasks.dino_ssl.generate_dataset.manifest import write_manifest


class DinoSSLCollectionRunner:
    """Run all configured sources and emit a manifest-backed image folder."""

    def run(self, config: Any) -> Path:
        collector_cfg = config.collector
        output_dir = Path(to_absolute_path(str(collector_cfg.output_dir)))
        image_dir = output_dir / "images"

        if bool(collector_cfg.get("overwrite", True)):
            reset_image_dir(image_dir)

        collected: list[CollectedImage] = []
        for source in collector_cfg.sources:
            source_dict = OmegaConf.to_container(source, resolve=True)
            assert isinstance(source_dict, dict)
            collected.extend(collect_from_source(source_dict, image_dir))

        min_images = int(collector_cfg.get("min_images", 0))
        synthetic_cfg = collector_cfg.get("synthetic_fallback", {}) or {}
        if len(collected) < min_images and bool(synthetic_cfg.get("enabled", False)):
            missing = min_images - len(collected)
            print(
                f"[collect] only {len(collected)} real image(s); "
                f"adding {missing} synthetic fallback image(s)."
            )
            collected.extend(
                write_synthetic_images(
                    out_dir=image_dir,
                    count=missing,
                    size=int(synthetic_cfg.get("size", 256)),
                    seed=int(config.run.get("seed", 0)),
                )
            )

        if bool(collector_cfg.get("dedup", True)):
            collected = deduplicate_images(collected)

        if not collected:
            raise RuntimeError(
                "DINOv3 SSL collection produced no images. Check the configured "
                "sources or enable a synthetic fallback."
            )

        manifest = write_manifest(
            root=output_dir,
            images=collected,
            extra={"name": str(collector_cfg.get("name", output_dir.name))},
        )
        print(
            f"[collect] wrote {manifest.num_images} image(s) and manifest to "
            f"{output_dir}"
        )
        return output_dir


__all__ = ["DinoSSLCollectionRunner"]
