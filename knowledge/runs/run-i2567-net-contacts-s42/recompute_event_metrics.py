"""Recompute the three-class metrics from the frozen synthetic test predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.tasks.ball_refiner_3d.data.dataset import SharedDataset
from src.tasks.ball_refiner_3d.evaluation.events import event_report
from src.tasks.ball_refiner_3d.events import EVENT_CLASSES
from src.utils.paths import PROJECT_ROOT

DATASET_SHA256 = "b54876c074c22c055d65033a9387d2e167263fbf7e4971e2e5fe8b0009653ca0"
PREDICTION_SHA256 = "670029b88db868f3685d45bde24e59591c134c6798b076b19b05c16735a8ef8c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    if hashlib.sha256(args.predictions.read_bytes()).hexdigest() != PREDICTION_SHA256:
        raise ValueError("Unexpected synthetic prediction SHA-256")
    dataset = SharedDataset(args.dataset)
    if dataset.manifest_hash != DATASET_SHA256:
        raise ValueError("Unexpected synthetic dataset manifest SHA-256")
    test = dataset.split("test")
    with np.load(args.predictions, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["event_classes"], EVENT_CLASSES)
        probabilities = saved["event_probability"]
        rally_ids, frame_ids = saved["rally_id"], saved["frame_id"]
    np.testing.assert_array_equal(np.unique(rally_ids), [rally.index for rally in test])
    if probabilities.shape != (len(rally_ids), len(EVENT_CLASSES)):
        raise ValueError("Invalid prediction shape")
    predictions = []
    for rally in test:
        select = rally_ids == rally.index
        np.testing.assert_array_equal(frame_ids[select], np.arange(len(rally.events)))
        predictions.append(probabilities[select])
    events = event_report(predictions, [rally.events for rally in test], dataset.fps)
    net_by_outcome = {}
    for outcome in ("crossed", "dropped", None):
        indices = [
            i
            for i, rally in enumerate(test)
            if dataset.manifest["records"][rally.index]["net_contact"] == outcome
        ]
        net = event_report(
            [predictions[i] for i in indices],
            [test[i].events for i in indices],
            dataset.fps,
        )["by_kind"]["net"]
        net_by_outcome[outcome or "no_contact"] = {
            key: net[key] for key in ("predicted", "true", "matched")
        }
    report = {
        "evaluation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "matching": "maximize match count, then minimize total absolute frame error",
        "dataset_manifest_sha256": DATASET_SHA256,
        "prediction_sha256": PREDICTION_SHA256,
        "rallies": len(test),
        "frames": len(probabilities),
        "events": events,
        "net_by_outcome": net_by_outcome,
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
