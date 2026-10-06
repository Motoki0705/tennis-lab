"""Evaluate the validation-selected BLCS checkpoint on its held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.blcs.configuration import validate_training_boundary
from src.tasks.blcs.model_io.checkpoints import load_checkpoint_runtime
from src.tasks.blcs.model_io.training import compose_blcs_training
from src.utils.checksum import dual_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    checkpoint_fragment = Path(args.checkpoint)
    output_fragment = Path(args.output_dir)
    for fragment in (checkpoint_fragment, output_fragment):
        if fragment.is_absolute() or ".." in fragment.parts:
            raise ValueError("Checkpoint and output must be root-relative paths.")
    checkpoint = root / "ckpt" / checkpoint_fragment
    output = root / "outputs" / output_fragment
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    runtime = load_checkpoint_runtime(
        checkpoint, runtime_court_keypoints="physical_v1"
    )
    config = OmegaConf.create(OmegaConf.to_container(runtime.config, resolve=True))
    config.paths.project_root = str(root)
    config.paths.data_root = str(root / "data")
    config.paths.checkpoint_root = str(root / "ckpt")
    config.paths.artifact_root = str(root / "outputs")
    config.paths.output_root = str(root / "outputs")
    config.run.output_dir = args.output_dir
    config.run.resume = None
    config.run.init_weights = None
    TrainingRuntimeConfig.from_config(config, repository_root=root)
    validate_training_boundary(config)
    pl.seed_everything(int(config.run.seed))
    composition = compose_blcs_training(config, generator_config=None)
    composition.datamodule.setup("test")
    scene_count = len(composition.datamodule.test_dataset)
    output.mkdir(parents=True)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        deterministic=True,
        enable_progress_bar=False,
        logger=TensorBoardLogger(str(output), name="logs"),
    )
    metrics = trainer.test(
        composition.lightning_module,
        datamodule=composition.datamodule,
        ckpt_path=str(checkpoint),
        weights_only=False,
    )[0]
    receipt = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": dual_sha256(checkpoint),
        "selection": "minimum val/position_error_m; test was not used for selection",
        "split": "test",
        "scene_count": scene_count,
        "sequence_length": 128,
        "window": "deterministic centre crop from the training dataset reader",
        "camera_range": [3, 6],
        "camera_selection": "sample-local RNG with run seed 42",
        "augmentation": False,
        "metrics": metrics,
    }
    (output / "evaluation.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
