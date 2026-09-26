"""Evaluate an explicit headless export and compare its retained embeddings."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import open_dict

from src.tasks.plcs.data.association_datamodule import PLCSAssociationDataModule
from src.tasks.plcs.training.association_lightning_module import (
    PLCSAssociationLightningModule,
)
from src.utils.checksum import dual_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    stored = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = deepcopy(stored["hyper_parameters"]["config"])
    with open_dict(config):
        config.paths.output_root = str(args.output.parent)
        config.paths.artifact_root = str(args.output.parent)
        config.run.output_dir = args.output.name
        config.data.num_workers = 0
    pl.seed_everything(int(config.run.seed), workers=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision(str(config.training.matmul_precision))
    torch.backends.cuda.matmul.allow_tf32 = bool(config.training.allow_tf32)
    module = PLCSAssociationLightningModule.load_from_checkpoint(args.checkpoint, config=config, map_location="cpu", weights_only=False)
    data = PLCSAssociationDataModule(config)
    trainer = pl.Trainer(accelerator="gpu" if args.device == "cuda" else "cpu", devices=1,
        precision="bf16-mixed" if args.device == "cuda" else "32-true",
        deterministic=True, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False)
    metrics = trainer.test(module, datamodule=data)
    with np.load(args.reference_predictions, allow_pickle=False) as reference, np.load(args.output / "predictions/pred_test.npz", allow_pickle=False) as actual:
        for name in ("sample_index", "track_person_id", "track_valid", "side_target", "reference_view_index", "track_observation_count"):
            np.testing.assert_array_equal(actual[name], reference[name])
        assert "is_player_logit" not in actual
        z, old = actual["track_embedding"], reference["track_embedding"]
        valid = actual["track_valid"]
        difference = float(np.max(np.abs(z - old)))
        similarity = np.sum(z[valid].astype(np.float64) * old[valid].astype(np.float64), axis=-1)
        if not np.isfinite(z).all() or float(similarity.min()) < .9999:
            raise AssertionError("Exported embeddings diverge from the saved source predictions")
        report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": dual_sha256(args.checkpoint),
            "source_checkpoint_sha256": stored["reid_export_provenance"]["source_sha256"],
            "device": args.device, "retrained": False, "scene_count": len(z),
            "test": metrics, "embeddings_bitwise_equal": bool(np.array_equal(z, old)),
            "max_embedding_difference": difference, "minimum_embedding_cosine": float(similarity.min())}
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
