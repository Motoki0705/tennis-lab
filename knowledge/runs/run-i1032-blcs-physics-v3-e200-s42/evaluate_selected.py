"""Evaluate the preselected best-validation checkpoint on the frozen test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
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
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    selection = json.loads(args.selection.read_text())
    checkpoint = Path(selection["source"]).resolve(strict=True)
    output_fragment = Path(args.output_dir)
    if output_fragment.is_absolute() or ".." in output_fragment.parts:
        raise ValueError("Output must be relative to the outputs root")
    output = root / "outputs" / output_fragment
    if output.exists():
        raise FileExistsError(output)
    digest = dual_sha256(checkpoint)
    if digest != selection["sha256"] or selection["selection_used_test"]:
        raise ValueError("Checkpoint does not match the pre-test selection receipt")
    runtime = load_checkpoint_runtime(checkpoint, runtime_court_keypoints="physical_v1")
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
    pl.seed_everything(int(config.run.seed), workers=True)
    torch.set_float32_matmul_precision(str(config.training.matmul_precision))
    torch.backends.cuda.matmul.allow_tf32 = bool(config.training.allow_tf32)
    composition = compose_blcs_training(config, generator_config=None)
    composition.datamodule.setup("test")
    scene_count = len(composition.datamodule.test_dataset)
    output.mkdir(parents=True)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=str(config.training.trainer.precision),
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
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "selection": selection,
        "split": "test",
        "scene_count": scene_count,
        "sequence_length": list(config.data.seq_len_range),
        "window": "deterministic centre crop from the training dataset reader",
        "camera_range": list(config.data.num_views_range),
        "camera_selection": "sample-local RNG with run seed 42",
        "augmentation": False,
        "precision": str(config.training.trainer.precision),
        "matmul_precision": str(config.training.matmul_precision),
        "allow_tf32": bool(config.training.allow_tf32),
        "metrics": metrics,
    }
    (output / "evaluation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
