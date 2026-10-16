"""Record per-point diagnostics without changing the original Court metric."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.court_detection.ablation.evaluation import sample_identity
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.inference.checkpoint import file_sha256
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.tasks.court_detection.training.metrics import CourtPoseGeometryMetrics


class RecordedGeometry(CourtPoseGeometryMetrics):
    def __init__(self, *, min_depth_m: float) -> None:
        super().__init__(min_depth_m=min_depth_m)
        self.sample_ids: list[str] = []
        self.rows: list[dict[str, Any]] = []

    def _record_pose_distances(
        self,
        pose_distance: torch.Tensor,
        depth: torch.Tensor,
        point_visible: torch.Tensor,
    ) -> None:
        super()._record_pose_distances(pose_distance, depth, point_visible)
        if len(self.sample_ids) != pose_distance.shape[0]:
            raise ValueError("Sample IDs do not match the supervised metric batch")
        errors = pose_distance.detach().float().cpu().tolist()
        depths = depth.detach().float().cpu().tolist()
        visibility = point_visible.detach().cpu().tolist()
        for sample_id, error, z, visible in zip(
            self.sample_ids, errors, depths, visibility, strict=True
        ):
            self.rows.append(
                {
                    "sample_id": sample_id,
                    "error_px": error,
                    "depth_m": z,
                    "visible": visible,
                }
            )


class RecordBatchIds(pl.Callback):
    def __init__(self, recorder: RecordedGeometry) -> None:
        self.recorder = recorder

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch_idx, dataloader_idx
        self.recorder.sample_ids = list(batch["sample_id"])


def describe(points: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([p["error_px"] for p in points], dtype=np.float64)
    if not len(values):
        return {
            "count": 0,
            "mean_px": None,
            "median_px": None,
            "p95_px": None,
            "max_px": None,
        }
    return {
        "count": len(values),
        "mean_px": float(values.mean()),
        "median_px": float(np.median(values)),
        "p95_px": float(np.quantile(values, 0.95)),
        "max_px": float(values.max()),
        "error_sum_px": float(values.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This diagnostic requires the existing Colab CUDA environment"
        )
    original = json.loads(args.evaluation.read_text())
    checkpoint = Path(original["checkpoint"])
    if file_sha256(checkpoint) != original["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the completed evaluation")
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    hyper = payload["hyper_parameters"]
    saved = hyper["config"]
    cfg = OmegaConf.create(
        OmegaConf.to_container(saved, resolve=True)
        if isinstance(saved, DictConfig)
        else saved
    )
    pl.seed_everything(int(cfg.run.seed), workers=True)
    dm = CourtDetectionDataModule(cfg)
    module = CourtDetectionLightningModule(
        cfg, target_bundle_state=hyper["target_bundle_state"]
    )
    if dm.target_bundle_spec != module.target_bundle:
        raise ValueError("Target schema changed")
    module.load_state_dict(payload["state_dict"], strict=True)
    del payload
    recorder = RecordedGeometry(
        min_depth_m=module._pose_geometry_metrics["val"].min_depth_m
    )
    module._pose_geometry_metrics["val"] = recorder
    loader = dm.source_eval_dataloader("synthetic_court", "test")
    identity = sample_identity(loader.dataset)
    expected = original["splits"]["synthetic_court-test"]
    if identity["sample_ids"] != expected["sample_ids"]:
        raise ValueError("Evaluation sample IDs/order changed")
    args.output.mkdir(parents=True, exist_ok=False)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        precision=original["precision"],
        deterministic="warn",
        callbacks=[RecordBatchIds(recorder)],
    )
    measured = trainer.validate(module, dataloaders=loader, verbose=False)[0]
    if [row["sample_id"] for row in recorder.rows] != expected["sample_ids"]:
        raise ValueError(
            "Captured metric rows do not match the complete evaluation order"
        )
    points = []
    for row in recorder.rows:
        for index, (error, depth, visible) in enumerate(
            zip(row["error_px"], row["depth_m"], row["visible"], strict=True)
        ):
            if visible:
                points.append(
                    {
                        "sample_id": row["sample_id"],
                        "semantic_point_index": index,
                        "error_px": error,
                        "depth_m": depth,
                        "invalid_depth": depth <= recorder.min_depth_m,
                    }
                )
    invalid = [p for p in points if p["invalid_depth"]]
    valid = [p for p in points if not p["invalid_depth"]]
    pooled = describe(points)
    observed_mean = float(measured["val/pose_reprojection_mean_distance_px"])
    original_mean = expected["metrics"]["pose_reprojection_mean_distance_px"]
    if len(points) != int(expected["metrics"]["visible_point_count"]):
        raise ValueError("Visible point count changed")
    if len(invalid) != round(expected["metrics"]["invalid_depth_rate"] * len(points)):
        raise ValueError("Invalid depth count changed")
    if not all(
        math.isclose(pooled["mean_px"], value, rel_tol=1e-4, abs_tol=1e-3)
        for value in (observed_mean, original_mean)
    ):
        raise ValueError(
            f"Diagnostic aggregate does not reproduce the original: {pooled['mean_px']}, {observed_mean}, {original_mean}"
        )
    result = {
        "schema_version": 1,
        "scope": "Supplementary per-point diagnosis; the original all-visible-point metric is unchanged.",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint_sha256": original["checkpoint_sha256"],
        "ordered_sample_ids_sha256": identity["ordered_sample_ids_sha256"],
        "min_depth_m": recorder.min_depth_m,
        "original_reported_mean_px": original_mean,
        "rerun_reported_mean_px": observed_mean,
        "all_visible": pooled,
        "depth_valid_only": describe(valid),
        "depth_invalid_only": describe(invalid),
        "invalid_error_fraction": sum(p["error_px"] for p in invalid)
        / pooled["error_sum_px"]
        if pooled["error_sum_px"]
        else None,
        "invalid_points": invalid,
        "largest_errors": sorted(points, key=lambda p: p["error_px"], reverse=True)[
            :20
        ],
    }
    (args.output / "points.json").write_text(
        json.dumps(recorder.rows, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "diagnostic.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
