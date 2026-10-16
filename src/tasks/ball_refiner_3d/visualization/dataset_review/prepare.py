"""Explicit CPU generation of hash-bound test predictions for the read-only UI."""

from __future__ import annotations

import json

import torch

from src.tasks.ball_refiner_3d.data.preprocessing import prepare
from src.tasks.ball_refiner_3d.evaluation.evaluator import evaluate
from src.tasks.ball_refiner_3d.model_io.checkpoint import load_checkpoint
from src.tasks.ball_refiner_3d.visualization.dataset_review.artifacts import (
    write_bundle,
)
from src.tasks.ball_refiner_3d.visualization.dataset_review.contracts import (
    Augmentation,
)
from src.tasks.ball_refiner_3d.visualization.dataset_review.service import ReviewService


def prepare_saved_predictions(service: ReviewService) -> None:
    """Recompute; legacy test arrays never acquire a retrospective model hash."""
    service.catalog(refresh=True)
    entries = [
        entry
        for entry in service.checkpoints.entries.values()
        if entry.info["compatible"] and entry.predictions is not None
    ]
    for index, entry in enumerate(entries, 1):
        if entry.info["saved_available"]:
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(entries),
                        "checkpoint": entry.info["id"],
                        "status": "already_verified",
                    }
                ),
                flush=True,
            )
            continue
        profile = entry.info["evaluation_profile"]
        digest = entry.info["sha256"]
        service.checkpoints.get(entry.info["id"], entry.info["dimensions"], digest)
        model, _ = load_checkpoint(entry.path, torch.device("cpu"))
        corrupted = prepare(
            service.dataset.split("test"),
            entry.info["dimensions"],
            Augmentation.model_validate(profile["augmentation"]).config(),
            profile["augmentation_seed"],
            event_sigma_frames=entry.info["event_sigma_frames"],
        )
        _, predictions = evaluate(
            model,
            corrupted,
            torch.device("cpu"),
            seed=profile["flow_seed"],
            batch_size=32,
            physics=False,
        )
        assert entry.predictions is not None
        write_bundle(
            entry.predictions,
            predictions,
            checkpoint=entry.path,
            checkpoint_hash=digest,
            manifest_hash=service.dataset.manifest_hash,
            profile=profile,
        )
        del model, corrupted, predictions
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(entries),
                    "checkpoint": entry.info["id"],
                    "status": "created",
                }
            ),
            flush=True,
        )
    service.catalog(refresh=True)
