"""Fixed 4,000-update progressive-loss experiment, always launched by the queue."""

from __future__ import annotations

import hashlib
import json
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.tasks.blcs.training.runner import BLCSTrainingRunner
from src.utils.paths import PROJECT_ROOT

ARMS = (
    "position",
    "reprojection",
    "velocity",
    "rollout_position",
    "rollout_velocity",
    "parameters",
)


def experiment_config(
    *,
    data_root: Path,
    output: str,
    arm: str,
    seed: int,
    smoke: bool = False,
    chunked: bool = False,
    gan: bool = False,
) -> DictConfig:
    if arm not in ARMS:
        raise ValueError(f"Unknown experiment arm: {arm}")
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tasks/blcs/configs"),
        version_base="1.3",
    ):
        config = compose(
            config_name="train_chunked" if chunked else "train",
            overrides=["training=gan_small"] if gan else [],
        )
    config.paths.data_root = str(data_root)
    config.data.scene_dir = "blcs/single_object-physics-v3-cc863cacd-20261007"
    config.data.seq_len_range = [128, 128]
    config.data.num_views_range = [3, 6]
    config.data.batch_size = 2
    config.data.num_workers = 0 if smoke else 2
    config.model.predict_velocity = True
    config.model.physics_head = arm == "parameters"
    config.run.seed = seed
    config.run.output_dir = output
    config.run.test_after_fit = False
    config.training.weight_decay = 0.01
    config.training.compile.enabled = False
    config.training.trainer.accumulate_grad_batches = 1 if smoke else 8
    config.training.trainer.max_epochs = 200
    config.training.trainer.check_val_every_n_epoch = 1 if smoke else 5
    config.training.trainer.enable_progress_bar = False
    config.training.qualitative_logging.enabled = False
    config.training.checkpoint.monitor = "val/trajectory_rmse_m"
    config.training.checkpoint.save_top_k = 1
    config.training.checkpoint.save_last = True
    config.training.checkpoint.filename = "blcs-{epoch:03d}-{step:05d}"
    config.training.early_stopping.enabled = False
    index = ARMS.index(arm)
    config.loss.reprojection_weight = 0.1 if index >= 1 else 0.0
    config.loss.velocity_weight = 0.1 if index >= 2 else 0.0
    config.loss.rollout_position_weight = 1.0 if index >= 3 else 0.0
    config.loss.rollout_velocity_weight = 0.1 if index >= 4 else 0.0
    if arm == "parameters":
        # B is a separate architecture package. The first five arms isolate A's
        # loss additions; this arm does not identify individual B auxiliary effects.
        config.loss.parameters_weight = 0.01
        config.loss.event_weight = 0.01
        config.loss.reconstruction_weight = 0.1
        config.loss.parameter_consistency_weight = 0.1
    if chunked and smoke:
        config.data.chunk.scenes_per_chunk = 4
        config.data.chunk.epochs_per_chunk = 1
        config.data.chunk.prefetch_chunks = 1
        config.data.chunk.generation_workers = 2
        config.data.chunk.chunks_dir = "blcs/i1044/" + output.split("/")[-1] + "-chunks"
    if smoke:
        config.training.warmup_steps = 0
        config.loss.physics_start_step = 0
        config.loss.physics_warmup_steps = 0
        if gan:
            config.training.gan.transition.start_epoch = 0
            config.training.gan.warmup_epochs = 1
    return config


class FixedUpdateRunner(BLCSTrainingRunner):
    def __init__(
        self, *, steps: int, smoke: bool, generator_config: Any = None
    ) -> None:
        super().__init__(generator_config=generator_config)
        self.steps, self.smoke = steps, smoke
        self.trainer: Any = None

    def build_trainer(
        self,
        config: Any,
        callbacks: list[Any],
        logger: Any,
        *,
        artifact_store: Any = None,
    ) -> pl.Trainer:
        trainer = super().build_trainer(
            config, callbacks, logger, artifact_store=artifact_store
        )
        trainer.fit_loop.epoch_loop.max_steps = self.steps
        if self.smoke:
            trainer.limit_val_batches = 2
            trainer.num_sanity_val_steps = 1
        self.trainer = trainer
        return trainer


def run_experiment(args: Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("Optimizer update budget must be positive")
    if args.gan and not args.smoke:
        raise ValueError("The fixed comparison protocol disables GAN")
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("GPU experiments must run through the shared training queue")
    torch.set_num_threads(2)
    config = experiment_config(
        data_root=args.data_root,
        output=args.output,
        arm=args.arm,
        seed=args.seed,
        smoke=args.smoke,
        chunked=args.chunked,
        gan=args.gan,
    )
    from src.tasks.blcs.model_io.training import optional_standard_generator_config

    runner = FixedUpdateRunner(
        steps=args.steps,
        smoke=args.smoke,
        generator_config=optional_standard_generator_config(config),
    )
    output = Path(config.paths.output_root) / args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    OmegaConf.save(config, output / "experiment-config.yaml", resolve=True)
    began = time.monotonic()
    runner.run(config)
    trainer = runner.trainer
    if trainer.global_step != args.steps:
        raise RuntimeError(f"Budget mismatch: {trainer.global_step} != {args.steps}")
    # Includes the partial final epoch in checkpoint selection without using test.
    final_metrics = trainer.validate(
        trainer.lightning_module, datamodule=trainer.datamodule, verbose=False
    )[0]
    checkpoint = trainer.checkpoint_callback
    final_score = float(final_metrics["val/trajectory_rmse_m"])
    if not torch.isfinite(torch.tensor(final_score)):
        raise RuntimeError("Final validation trajectory RMSE is undefined/nonfinite")
    if checkpoint.best_model_score is None or final_score < float(
        checkpoint.best_model_score
    ):
        selected = output / "checkpoints" / "final-selected.ckpt"
        trainer.save_checkpoint(selected)
        selected_score = final_score
    else:
        selected = Path(checkpoint.best_model_path)
        selected_score = float(checkpoint.best_model_score)
    with selected.open("rb") as source:
        selected_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    receipt = {
        "protocol": "issue1044-v1",
        "arm": args.arm,
        "seed": args.seed,
        "optimizer_updates": trainer.global_step,
        "elapsed_seconds": time.monotonic() - began,
        "checkpoint": str(selected.resolve()),
        "checkpoint_sha256": selected_sha256,
        "validation_trajectory_rmse_m": selected_score,
        "selection_used_test": False,
        "run_id": os.environ["TENNIS_RUN_ID"],
        "smoke": args.smoke,
        "chunked": args.chunked,
        "gan": args.gan,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    OmegaConf.save(config, output / "experiment-config.yaml", resolve=True)
    print(json.dumps(receipt), flush=True)
