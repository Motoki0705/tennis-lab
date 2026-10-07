"""Validate real checkpoint qualitative output through the shared callback.

Run as a module from the checkout root, through the shared training queue.
The command performs validation only; checkpoint/model weights must not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader, Subset

import src.utils.hydra  # noqa: F401 -- register the public configuration resolvers
from src.tasks.base.training.qualitative_callback import QualitativeLoggingCallback
from src.utils.paths import PROJECT_ROOT


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _configuration(task: str, assets: Path, saved: Mapping[str, Any]) -> DictConfig:
    with initialize_config_dir(
        version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tasks" / task / "configs")
    ):
        config = compose(config_name="train")
    # Preserve checkpoint-owned architecture, preprocessing and scoring contracts.
    sections = ("model", "data", "loss", "render_style") if task == "court_detection" else (
        "model", "data", "evaluation"
    )
    for key in sections:
        config[key] = OmegaConf.create(OmegaConf.to_container(saved[key], resolve=True))
    config.paths.project_root = str(PROJECT_ROOT)
    for key, value in {
        "data_root": "data", "checkpoint_root": "ckpt", "external_asset_root": "third_party",
        "output_root": "outputs", "artifact_root": "outputs", "cache_root": ".cache",
    }.items():
        config.paths[key] = str(assets / value)
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.training.compile.enabled = False
    config.training.qualitative_logging.enabled = True
    config.training.qualitative_logging.every_n_epochs = 1
    config.training.qualitative_logging.num_samples = 2
    config.training.qualitative_logging.selection_mode = "fixed_indices"
    config.training.qualitative_logging.selected_indices = [0, 1]
    if task == "court_detection":
        # Explicit migrations of historical runtime settings, not target/model schemas.
        with open_dict(config.data.processing):
            if "derived_target_root" in config.data.processing:
                del config.data.processing.derived_target_root
        config.data.augmentation.train_scales = [int(config.data.augmentation.val_short_side)]
        previous = str(config.model.encoder.checkpoint_path)
        if previous.startswith("dinov3/checkpoints/"):
            config.model.encoder.checkpoint_path = "dinov3/" + Path(previous).name
    return config


def _build(
    task: str, config: DictConfig, checkpoint: Mapping[str, Any], indices: list[int],
) -> tuple[pl.LightningModule, DataLoader[Any], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    if task == "court_detection":
        from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
        from src.tasks.court_detection.training.lightning_module import (
            CourtDetectionLightningModule,
        )
        from src.tasks.court_detection.training.runner import resolve_training_config

        standard, mixed = resolve_training_config(config)
        data = CourtDetectionDataModule(standard, mixed_config=mixed)
        data.setup("validate")
        original_loader = data.val_dataloader()
        module: pl.LightningModule = CourtDetectionLightningModule(
            config, target_bundle_state=checkpoint["hyper_parameters"]["target_bundle_state"],
        )
        for index in indices:
            sample = original_loader.dataset[index]
            selected.append({
                "dataset_index": index, "sample_id": sample["sample_id"],
                "source_kind": sample["metadata"]["source_kind"],
            })
    else:
        from src.tasks.player_detection.configuration import PlayerTrainingConfig
        from src.tasks.player_detection.data.datamodule import PlayerDetectionDataModule
        from src.tasks.player_detection.data.detection_dataset import (
            PlayerDetectionDataset,
        )
        from src.tasks.player_detection.training.lightning_module import (
            PlayerDetectionLightningModule,
        )

        runtime = PlayerTrainingConfig.from_config(config)
        player_data = PlayerDetectionDataModule(runtime.data)
        original_loader = player_data.val_dataloader()
        dataset = cast(PlayerDetectionDataset, original_loader.dataset)
        module = PlayerDetectionLightningModule(config, runtime)
        for index in indices:
            row = int(dataset.selection.frames[index])
            clip = dataset.store.clip_of(row)
            selected.append({
                "dataset_index": index, "store_row": row, "clip_index": clip.index,
                "clip_id": clip.clip_id, "source_id": clip.source_id, "split": clip.split,
                "video_sha256": clip.video_sha256, "frame_count": clip.frame_count,
                "nominal_fps": clip.nominal_fps, "width": clip.width, "height": clip.height,
            })
    result = module.load_state_dict(checkpoint["state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    loader = DataLoader(
        Subset(original_loader.dataset, indices), batch_size=1, shuffle=False,
        num_workers=0, collate_fn=original_loader.collate_fn,
    )
    return module, loader, selected


def _artifacts(log_dir: Path, task: str) -> list[dict[str, Any]]:
    files = sorted((log_dir / "qualitative/epoch_0000").glob("*"))
    expected_count = 8 if task == "court_detection" else 2
    assert len(files) == expected_count, files
    result: list[dict[str, Any]] = []
    for path in files:
        with Image.open(path) as media:
            dimensions = media.size
            frames = int(getattr(media, "n_frames", 1))
            for index in range(frames):
                media.seek(index)
                media.load()
                assert media.size == dimensions
            if task == "court_detection":
                assert media.format == "PNG" and frames == 1
            else:
                assert media.format == "GIF" and frames > 1
            result.append({
                "path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size,
                "width": dimensions[0], "height": dimensions[1], "frames": frames,
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("court_detection", "player_detection"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-indices", type=int, nargs=2, required=True)
    parser.add_argument("--player-max-frames", type=int, default=60)
    parser.add_argument("--player-display-width", type=int, default=640)
    args = parser.parse_args()
    output: Path = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict[str, Any] = {
        "task": args.task, "checkpoint": str(args.checkpoint.resolve()),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "status": "running", "checkpoint_sha256": _sha256(args.checkpoint),
        "sample_indices": args.sample_indices,
        "mode": "standalone validation; interval=1; no optimizer or fit",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("This real-checkpoint verification requires CUDA and the shared queue.")
        pl.seed_everything(42, workers=True)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
        saved = checkpoint["hyper_parameters"]["config"]
        saved = saved if isinstance(saved, DictConfig) else OmegaConf.create(saved)
        config = _configuration(args.task, args.asset_root.resolve(), saved)
        config.run.output_dir = str(output.relative_to(args.asset_root.resolve() / "outputs"))
        if args.task == "player_detection":
            config.qualitative.max_frames = args.player_max_frames
            config.qualitative.display_width = args.player_display_width
        OmegaConf.save(config, output / "config.yaml", resolve=True)
        torch.set_float32_matmul_precision(str(config.training.matmul_precision))
        torch.backends.cuda.matmul.allow_tf32 = bool(config.training.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.training.allow_tf32)
        torch.cuda.reset_peak_memory_stats()
        report.update({
            "checkpoint_epoch": checkpoint["epoch"], "checkpoint_global_step": checkpoint["global_step"],
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "lightning": pl.__version__, "state_tensor_count": len(checkpoint["state_dict"]),
        })
        module, loader, selected = _build(args.task, config, checkpoint, args.sample_indices)
        report["selected_samples"] = selected
        report["strict_load"] = True
        callback = QualitativeLoggingCallback(
            enabled=True, every_n_epochs=1, num_samples=2,
            selection_mode="fixed_indices", selected_indices=[0, 1],
        )
        logger = TensorBoardLogger(str(output), name="logs", version=0)
        trainer = pl.Trainer(
            accelerator="gpu", devices=1, precision="32-true", logger=logger,
            callbacks=[callback], enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
        )
        metrics = trainer.validate(module, dataloaders=loader, verbose=False)
        logger.experiment.flush()
        report["artifacts"] = _artifacts(Path(logger.log_dir), args.task)
        events = EventAccumulator(logger.log_dir, size_guidance={"images": 0}).Reload()
        image_tags = events.Tags()["images"]
        assert len(image_tags) == len(report["artifacts"]), image_tags
        report["tensorboard_image_tags"] = image_tags
        assert callback.state_dict() == {"last_logged_epoch": 1}
        report["callback_state"] = callback.state_dict()
        after = module.state_dict()
        assert set(after) == set(checkpoint["state_dict"])
        assert all(torch.equal(value.detach().cpu(), checkpoint["state_dict"][key]) for key, value in after.items())
        assert _sha256(args.checkpoint) == report["checkpoint_sha256"]
        report["weights_unchanged"] = True
        report["validation_metrics_subset_only"] = {
            key: float(value) if math.isfinite(float(value)) else None
            for key, value in metrics[0].items()
        }
        report["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if os.environ.get("TENNIS_REPRO_DIR"):
            repro = Path(os.environ["TENNIS_REPRO_DIR"])
            (repro / "output_dir.txt").write_text(str(output) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
