"""Run the fixed Court ViT suite, one GPU process at a time."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from .checkpoints import validate_checkpoint

HERE = Path(__file__).parent
VARIANTS = {
    "b": ("vitb16", "73cec8be", 768, [2, 5, 8, 11]),
    "s": ("vits16", "08c60483", 384, [2, 5, 8, 11]),
    "splus": ("vits16plus", "4057cbaa", 384, [2, 5, 8, 11]),
    "l": ("vitl16", "8aa4cbdd", 1024, [5, 11, 17, 23]),
}


def variant_config(size: str, overrides: list[str]) -> DictConfig:
    config = OmegaConf.merge(
        OmegaConf.load(HERE / "baseline.yaml"), OmegaConf.from_dotlist(overrides)
    )
    assert isinstance(config, DictConfig)
    name, suffix, _width, indices = VARIANTS[size]
    config.model.encoder.backbone_name = f"dinov3_{name}"
    config.model.encoder.checkpoint_path = (
        f"dinov3/checkpoints/dinov3_{name}_pretrain_lvd1689m-{suffix}.pth"
    )
    config.model.encoder.out_indices = indices
    if size != "b":
        config.model.feature_adapter = {"output_channels": 768}
    config.run.output_dir = f"{config.run.output_dir}/{size}"
    if config.run.artifact_store.mode != "rclone":
        raise ValueError("This suite requires workflow-managed rclone persistence")
    config.run.artifact_store.remote_root = (
        f"{config.run.artifact_store.remote_root}/{size}"
    )
    # Existing runs are immutable inputs. Fresh outputs belong to this new workflow.
    resume = Path(f"ckpt/court_vit_ablation/{size}.ckpt")
    config.run.resume = f"court_vit_ablation/{size}.ckpt" if resume.is_file() else None
    if size == "b" and not resume.is_file():
        raise FileNotFoundError("The B baseline full-state checkpoint is mandatory")
    return config


def train_one(
    size: str, overrides: list[str], *, smoke: bool, continuation: bool
) -> None:
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint

    from src.tasks.court_detection.training.runner_mixed import (
        MixedCourtDetectionTrainingRunner,
    )

    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This experiment requires an NVIDIA L4 GPU")
    config = variant_config(size, overrides)
    candidates = list(Path(f"ckpt/court_vit_ablation/{size}").glob("**/*.ckpt"))
    if config.run.resume:
        candidates.append(Path(config.paths.checkpoint_root) / config.run.resume)
    ranked = []
    for path in candidates:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        validate_checkpoint(
            checkpoint, config, initial=size == "b" and not continuation
        )
        ranked.append((int(checkpoint["global_step"]), int(checkpoint["epoch"]), path))
        del checkpoint
    if ranked:
        step, epoch, selected = max(ranked)
        config.run.resume = selected.relative_to(
            config.paths.checkpoint_root
        ).as_posix()
        print(
            f"VALIDATED RESUME {size}: epoch={epoch} global_step={step}; full optimizer/scheduler state present",
            flush=True,
        )
    if smoke:
        # Exercise the maximum resolution, actual mixed data and backward pass;
        # do not consume/overwrite a production run's optimizer or checkpoint.
        config.run.resume = None
        config.run.fast_dev_run = True
        config.run.test_after_fit = False
        config.data.augmentation.train_scales = [512]
        config.data.augmentation.val_short_side = 512
        config.run.output_dir += "/smoke"
        config.run.artifact_store.remote_root += "/smoke"

    class DurableRunner(MixedCourtDetectionTrainingRunner):
        def callbacks_extra(self, cfg: Any, datamodule: Any, logger: Any) -> list[Any]:
            callbacks = super().callbacks_extra(cfg, datamodule, logger)
            if not smoke:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=Path(logger.log_dir) / "checkpoints",
                        filename="recovery",
                        monitor=None,
                        save_top_k=1,
                        every_n_train_steps=100,
                        save_on_train_epoch_end=False,
                        enable_version_counter=False,
                    )
                )
            return cast(list[Any], callbacks)

    print(
        f"START size={size} smoke={smoke} gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    runner = DurableRunner()
    if continuation and ranked and not smoke:
        from src.tasks.court_detection.training.runner_mixed import (
            resolve_mixed_training_config,
        )

        standard, _ = resolve_mixed_training_config(config)
        runtime = runner.validate_runtime_config(standard)
        output = runner.prepare_output_dir(runtime)
        carried = output / "resume" / "last.ckpt"
        carried.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected, carried)
        runner.build_artifact_store(runtime, output).publish_file(carried)
        print(
            f"CARRIED RESUME {size}: step={step}; retained even if training is already complete",
            flush=True,
        )
    runner.run(config)
    print(
        f"FINISHED size={size} smoke={smoke} peak_allocated_bytes={torch.cuda.max_memory_allocated()}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=VARIANTS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--continuation", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.size:
        train_one(
            args.size, overrides, smoke=args.smoke, continuation=args.continuation
        )
        return
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    for size in VARIANTS:
        command = [
            sys.executable,
            "-m",
            "scripts.colab.train.court_vit_ablation.run",
            "--size",
            size,
        ]
        if args.smoke:
            command.append("--smoke")
        if args.continuation:
            command.append("--continuation")
        subprocess.run([*command, *overrides], check=True, env=environment)
    print(json.dumps({"suite": "court_vit_ablation", "state": "completed"}), flush=True)


if __name__ == "__main__":
    main()
