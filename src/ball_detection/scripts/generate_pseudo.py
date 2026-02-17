"""Generate pseudo labels from unlabeled data."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.ball_detection.pseudo.orchestrator import PseudoLabelOrchestrator, PseudoOrchestratorConfig


@hydra.main(config_path="../configs", config_name="generate_pseudo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pseudo_cfg = PseudoOrchestratorConfig(
        labeled_root_dir=str(cfg.data.root_dir),
        pseudo_output_dir=str(cfg.pseudo.output_dir),
        ensemble_checkpoints=tuple(str(p) for p in cfg.ensemble.checkpoints),
        ensemble_weights=tuple(float(w) for w in cfg.ensemble.weights) if cfg.ensemble.weights is not None else None,
        trajectory_checkpoint=cfg.pseudo.trajectory_checkpoint,
        event_checkpoint=cfg.pseudo.event_checkpoint,
        confidence_threshold=float(cfg.pseudo.confidence_threshold),
        min_clip_length=int(cfg.pseudo.min_clip_length),
        max_gap=int(cfg.pseudo.max_gap),
        image_h=int(cfg.data.image_h),
        image_w=int(cfg.data.image_w),
        allow_overwrite=bool(cfg.pseudo.allow_overwrite),
        device=str(cfg.run.device),
    )
    orchestrator = PseudoLabelOrchestrator(pseudo_cfg)
    result = orchestrator.run()
    print(result)


if __name__ == "__main__":
    main()
