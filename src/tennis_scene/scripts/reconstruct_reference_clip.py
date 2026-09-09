"""Reconstruct a reference-camera clip and render an evaluation video.

Run ``python -m src.tennis_scene.scripts.reconstruct_reference_clip stage=observe``
then ``stage=infer`` and ``stage=render``. Settings live in reference_clip.yaml.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.utils.hydra import hydra_main


@hydra_main(version_base="1.3", config_path="../configs", config_name="reference_clip")
def main(cfg: DictConfig) -> None:
    from src.tennis_scene.reference_pipeline.observations import (
        import_ball,
        observe_people,
    )

    clip, output = Path(cfg.clip_dir).resolve(), Path(cfg.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / f"{cfg.stage}.config.yaml")
    if cfg.stage == "observe":
        import_ball(clip, output)
        observe_people(cfg, clip, output)
    elif cfg.stage == "infer":
        from src.tennis_scene.reference_pipeline.reconstruction import reconstruct

        reconstruct(cfg, clip, output)
    elif cfg.stage == "render":
        from src.tennis_scene.reference_pipeline.rendering import render

        render(cfg, clip, output)
    else:
        raise ValueError(f"Unknown stage: {cfg.stage}")


if __name__ == "__main__":
    main()
