"""Real-data Re-ID fit/validation/test/reload diagnostic (CUDA via training queue)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.compilation import compile_modules
from src.tasks.plcs.data.association_datamodule import PLCSAssociationDataModule
from src.tasks.plcs.training.association_lightning_module import (
    PLCSAssociationLightningModule,
)
from src.utils.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    torch.set_num_threads(4)
    overrides = [f"paths.data_root={args.data_root}", f"data.scene_dir={args.scene_dir}",
        f"paths.output_root={args.output.parent}", f"run.output_dir={args.output.name}",
        "data.num_workers=0", "data.batch_size=2", "training.trainer.max_epochs=1",
        "training.trainer.accumulate_grad_batches=1", "training.warmup_steps=0",
        f"training.compile.enabled={'true' if args.device == 'cuda' else 'false'}"]
    if args.small:
        overrides += ["model.hidden_dim=32", "model.ffn_dim=64", "model.num_heads=4",
            "model.rope_dim=8", "model.num_stages=2", "data.seq_len_range=[32,32]"]
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "src/tasks/plcs/configs"), version_base="1.3"):
        cfg = compose(config_name="train_reid", overrides=overrides)
    data = PLCSAssociationDataModule(cfg)
    module = PLCSAssociationLightningModule(cfg)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnostic requires a CUDA device")
        torch.cuda.reset_peak_memory_stats()
        compile_modules(module.compilation_targets(), module.training_config.compile)
    args.output.mkdir(parents=True, exist_ok=True)
    callback = ModelCheckpoint(dirpath=args.output / "checkpoints", monitor="val/loss", save_top_k=1, filename="reid-smoke")
    trainer = pl.Trainer(accelerator="gpu" if args.device == "cuda" else "cpu", devices=1,
        max_epochs=1, limit_train_batches=4, limit_val_batches=2, limit_test_batches=2,
        num_sanity_val_steps=0, precision="bf16-mixed" if args.device == "cuda" else "32-true",
        callbacks=[callback], logger=TensorBoardLogger(str(args.output), name="logs"),
        enable_progress_bar=False, log_every_n_steps=1, gradient_clip_val=1.)
    start = time.monotonic()
    trainer.fit(module, datamodule=data)
    test = trainer.test(module, datamodule=data, ckpt_path=callback.best_model_path, weights_only=False)
    restored = PLCSAssociationLightningModule.load_from_checkpoint(callback.best_model_path, map_location="cpu", weights_only=False).eval()
    batch = next(iter(data.test_dataloader()))
    with torch.no_grad():
        output = restored.model_io.run(batch)
    if not all(torch.isfinite(value).all() for value in output.values()):
        raise RuntimeError("Reloaded model emitted non-finite output")
    report = {"device": args.device, "seconds": time.monotonic() - start, "checkpoint": callback.best_model_path,
        "test": test, "matching_threshold": float(restored.matching_threshold),
        "parameters": sum(p.numel() for p in module.model.parameters()),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else 0,
        "shapes": {k: list(v.shape) for k, v in output.items()}}
    (args.output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
