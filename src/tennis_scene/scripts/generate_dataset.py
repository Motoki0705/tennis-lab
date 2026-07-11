"""Generate incremental ``tennis_scene`` pseudo annotations for exported clips.

Usage:
    python -m src.tennis_scene.scripts.generate_dataset dataset_dir=data/tennis_scene_real
    python -m src.tennis_scene.scripts.generate_dataset dataset_dir=data/tennis_scene_real clip_ids='[match1/clip_000]' overwrite=true

Notes:
    - The dataset must have been created by ``export_clips`` and contain
      ``dataset.json`` plus ``clips/<recording_id>/<clip_name>/clip.json``.
    - The existing ``pipeline.yaml`` is loaded as the single source of model
      configuration; ``pipeline_overrides`` applies explicit Hydra dot-list values.
    - A clip is complete only after ``annotation.json`` is published. Any clip
      failure is recorded and produces a non-zero process exit status.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.hydra import hydra_main

LOGGER = logging.getLogger(__name__)


def _load_pipeline_config(cfg: DictConfig, dataset_dir: Path) -> DictConfig:
    config_path = Path(to_absolute_path(str(cfg.pipeline_config_path)))
    if not config_path.exists():
        raise FileNotFoundError(f"pipeline config not found: {config_path}")
    pipeline_cfg = OmegaConf.load(config_path)
    pipeline_cfg = OmegaConf.merge(
        pipeline_cfg,
        OmegaConf.create({"output_dir": str(dataset_dir / ".pipeline_work")}),
    )
    raw_overrides = cfg.get("pipeline_overrides", [])
    if not isinstance(raw_overrides, (list, tuple, ListConfig)):
        raise ValueError("pipeline_overrides must be a list of Hydra dot-list values")
    overrides = [str(value) for value in raw_overrides]
    if overrides:
        pipeline_cfg = OmegaConf.merge(pipeline_cfg, OmegaConf.from_dotlist(overrides))
    if not isinstance(pipeline_cfg, DictConfig):
        raise TypeError("pipeline config must resolve to a mapping")
    return pipeline_cfg


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="generate_dataset",
)
def main(cfg: DictConfig) -> int:
    """Run pseudo annotation generation for selected or pending clips."""
    from src.tennis_scene.generate_dataset import generate_pseudo_annotations
    from src.tennis_scene.io import SceneResult
    from src.tennis_scene.pipeline import TennisSceneOrchestrator

    dataset_dir = Path(to_absolute_path(str(cfg.dataset_dir))).resolve()
    raw_clip_ids = cfg.get("clip_ids")
    if raw_clip_ids is None:
        clip_ids = None
    elif isinstance(raw_clip_ids, (list, tuple, ListConfig)):
        clip_ids = [str(value) for value in raw_clip_ids]
    else:
        LOGGER.error("clip_ids must be null or a list")
        return 1

    try:
        pipeline_cfg = _load_pipeline_config(cfg, dataset_dir)
        pipeline_config_yaml = OmegaConf.to_yaml(pipeline_cfg, resolve=True)
        orchestrator = TennisSceneOrchestrator.from_config(pipeline_cfg)

        def run_clip(
            video_paths: Sequence[Path], camera_ids: Sequence[str]
        ) -> SceneResult:
            return orchestrator.run(
                video_paths=video_paths,
                camera_ids=camera_ids,
                max_frames=None,
                frame_index=int(pipeline_cfg.court_kp.get("frame_index", 0)),
            )

        outcomes = generate_pseudo_annotations(
            dataset_dir,
            run_clip,
            pipeline_config_yaml=pipeline_config_yaml,
            clip_ids=clip_ids,
            overwrite=bool(cfg.get("overwrite", False)),
            continue_on_error=bool(cfg.get("continue_on_error", True)),
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as error:
        LOGGER.error(str(error))
        return 1

    for outcome in outcomes:
        if outcome.status == "failed":
            LOGGER.error(f"{outcome.clip_id}: {outcome.error}")
        else:
            LOGGER.info(f"{outcome.clip_id}: {outcome.status}")
    generated = sum(outcome.status == "generated" for outcome in outcomes)
    skipped = sum(outcome.status == "skipped" for outcome in outcomes)
    failed = sum(outcome.status == "failed" for outcome in outcomes)
    LOGGER.info(f"Summary: generated={generated}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    main()
