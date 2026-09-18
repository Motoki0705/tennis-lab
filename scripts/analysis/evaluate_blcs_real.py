"""Compare a BLCS checkpoint on the fixed, recording-disjoint Meiji test split."""

from __future__ import annotations

import argparse
import json

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.blcs.model_io.training import compose_blcs_training
from src.tasks.blcs.training.runner import BLCSTrainingRunner
from src.tennis_scene.reference_pipeline.observations import sha256
from src.utils.configuration import PathRole
from src.utils.io import save_json_atomic
from src.utils.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint-root relative")
    parser.add_argument("--output", required=True, help="output-root relative run")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tasks/blcs/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="train_meiji_real_rgb",
            overrides=[f"run.output_dir={args.output}", "data.num_workers=0"],
        )
    runtime = BLCSTrainingRunner().validate_runtime_config(cfg)
    output = runtime.resolver.resolve(PathRole.OUTPUT, args.output)
    checkpoint_path = runtime.resolver.resolve(PathRole.CHECKPOINT, args.checkpoint)
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
                runtime.resolver.resolve(
                    PathRole.DATA, cfg.data.scene_dir, "provenance.json"
                )
            ),
            "interpretation": "held-out recording; geometric pseudo-3D, not measured ground truth",
        },
        output / "evaluation.json",
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
