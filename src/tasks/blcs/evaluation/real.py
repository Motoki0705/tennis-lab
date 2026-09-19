"""Compare a BLCS checkpoint on the fixed, recording-disjoint Meiji test split."""

from __future__ import annotations

import json

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.tasks.blcs.model_io.training import compose_blcs_training
from src.utils.checksum import dual_sha256 as sha256
from src.utils.configuration import PathRole
from src.utils.io import save_json_atomic

from .configuration import RealEvaluationConfig


def evaluate(args: RealEvaluationConfig) -> None:
    cfg = args.training
    output, checkpoint_path = args.output, args.checkpoint
    if output.exists():
        raise FileExistsError(f"Choose a new evaluation run: {output}")
    pl.seed_everything(int(cfg.run.seed), workers=True)
    composition = compose_blcs_training(cfg, generator_config=None)
    module = composition.lightning_module
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module.on_load_checkpoint(checkpoint)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    trainer = pl.Trainer(
        accelerator=args.device,
        devices=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(output),
    )
    report = trainer.test(module, datamodule=composition.datamodule)[0]
    save_json_atomic(
        {
            "metrics": report,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "dataset_provenance_sha256": sha256(
                args.resolver.resolve(
                    PathRole.DATA, cfg.data.scene_dir, "provenance.json"
                )
            ),
            "interpretation": "held-out recording; geometric pseudo-3D, not measured ground truth",
        },
        output / "evaluation.json",
    )
    print(json.dumps(report))
