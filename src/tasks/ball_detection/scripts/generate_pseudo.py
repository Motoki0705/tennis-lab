"""Generate pseudo labels from unlabeled data."""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from src.tasks.ball_detection.pseudo.orchestrator import PseudoLabelOrchestrator, PseudoOrchestratorConfig


@hydra.main(config_path="../configs", config_name="generate_pseudo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = str(cfg.run.device)
    if device != "cuda":
        raise ValueError("generate_pseudo requires run.device=cuda (CPU inference is disabled).")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. CPU inference is disabled for generate_pseudo.")

    raw_max_frames = cfg.data.max_frames
    raw_model_config_paths = cfg.ensemble.get("model_config_paths", None)
    model_config_paths = None
    if raw_model_config_paths is not None:
        model_config_paths = tuple(
            str(path) if path is not None and str(path).strip() != "" else None
            for path in raw_model_config_paths
        )

    pseudo_cfg = PseudoOrchestratorConfig(
        video_root_dir=str(cfg.data.video_root_dir),
        video_extensions=tuple(str(ext) for ext in cfg.data.video_extensions),
        pseudo_output_dir=str(cfg.pseudo.output_dir),
        ensemble_checkpoints=tuple(str(p) for p in cfg.ensemble.checkpoints),
        ensemble_weights=tuple(float(w) for w in cfg.ensemble.weights) if cfg.ensemble.weights is not None else None,
        ensemble_model_config_paths=model_config_paths,
        visibility_threshold=float(cfg.ensemble.get("visibility_threshold", 0.5)),
        trajectory_checkpoint=cfg.pseudo.trajectory_checkpoint,
        event_checkpoint=cfg.pseudo.event_checkpoint,
        confidence_threshold=float(cfg.pseudo.confidence_threshold),
        min_clip_length=int(cfg.pseudo.min_clip_length),
        max_gap=int(cfg.pseudo.max_gap),
        image_h=int(cfg.data.image_h),
        image_w=int(cfg.data.image_w),
        batch_size=max(1, int(cfg.data.batch_size)),
        max_frames=int(raw_max_frames) if raw_max_frames is not None else None,
        jpeg_quality=int(cfg.pseudo.jpeg_quality),
        allow_overwrite=bool(cfg.pseudo.allow_overwrite),
        device=device,
    )
    orchestrator = PseudoLabelOrchestrator(pseudo_cfg)
    result = orchestrator.run()
    print(result)


if __name__ == "__main__":
    main()
