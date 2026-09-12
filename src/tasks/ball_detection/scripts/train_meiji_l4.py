"""Prepare Meiji videos, measure the L4 batch limit, then start a fresh training run."""

from __future__ import annotations

import json

from omegaconf import DictConfig

from src.tasks.ball_detection.configuration import BallRuntimePaths, validate_training
from src.tasks.ball_detection.data.multiview_datamodule import MultiviewBallDataModule
from src.tasks.ball_detection.training.l4_calibration import calibrate, require_l4
from src.tasks.ball_detection.training.runner import BallDetectionTrainingRunner
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_meiji_3cam",
    validation_boundary="ball.train",
)
def main(config: DictConfig) -> None:
    """One fixed-GPU recipe; probes never contribute weights to the real training."""
    require_l4()
    validate_training(config)
    if config.run.resume is not None or config.run.dry_run or config.run.fast_dev_run:
        raise ValueError("The L4 calibration launcher requires a fresh full run.")
    if (
        config.model.name != "conv_next_unet"
        or config.data.source != "multiview"
        or config.training.gan.enabled
    ):
        raise ValueError(
            "This recipe requires ConvNeXtUNet, multiview data and supervised training."
        )
    if config.training.trainer.accumulate_grad_batches != 4 or config.run.gpus != 1:
        raise ValueError(
            "This L4 recipe requires one GPU and accumulate_grad_batches=4."
        )
    output = BallRuntimePaths.from_config(config).output(str(config.run.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    dm = MultiviewBallDataModule(config)
    summary = dm.summary()
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    dm.prepare_data()
    (output / "preprocessing.json").write_text(
        json.dumps(dm.preparation_report, indent=2), encoding="utf-8"
    )
    config.data.batch_size = calibrate(config, output / "calibration")
    print(f"[L4 calibration] selected batch_size={config.data.batch_size}", flush=True)
    BallDetectionTrainingRunner().run(config)


if __name__ == "__main__":
    main()
