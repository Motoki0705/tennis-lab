"""Add framing accumulators to the unchanged, ordered scaling evaluator."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from framing import GROUPS, RULE

import src.tasks.court_detection.ablation.evaluation as evaluation
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.model_io.contracts import CourtPoseTrainingResult
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.tasks.court_detection.training.metrics import CourtDetectionMetrics

NAMES = {
    "kp": {
        "mean_distance_px": "kp_mean_distance_px",
        "median_distance_px": "kp_median_distance_px",
    },
    "seg": {"miou": "seg_miou"},
    "line": {"dice": "line_dice"},
    "semantic_line": {"miou": "semantic_line_miou"},
}


class FramingAccumulator:
    def __init__(self, assignments: Path, split: str) -> None:
        rows = [json.loads(line) for line in assignments.read_text().splitlines()]
        selected = [
            row for row in rows if row["source"] == "real" and row["split"] == split
        ]
        self.groups = {row["sample_id"]: row["framing"] for row in selected}
        if len(self.groups) != len(selected) or not self.groups:
            raise ValueError("Expected unique real framing assignments")
        self.meters: dict[str, dict[str, CourtDetectionMetrics]] = {}
        self.counts: Counter[str] = Counter()
        self.seen: list[str] = []
        self.batch_seconds: list[float] = []

    def update(self, batch: Mapping[str, Any], result: CourtPoseTrainingResult) -> None:
        torch.cuda.synchronize()
        started = time.perf_counter()
        logits = result.output.dense_logits
        if not self.meters:
            self.meters = {
                group: {
                    kind: CourtDetectionMetrics(kind, value.shape[1])
                    for kind, value in logits.items()
                }
                for group in (*GROUPS, "all")
            }
        for index, sample_id in enumerate(batch["sample_id"]):
            group = self.groups[sample_id]
            self.seen.append(sample_id)
            self.counts[group] += 1
            self.counts["all"] += 1
            for kind, value in logits.items():
                raw = batch["targets"][kind]
                target = (
                    {key: item[index : index + 1] for key, item in raw.items()}
                    if isinstance(raw, Mapping)
                    else raw[index : index + 1]
                )
                for bucket in (group, "all"):
                    self.meters[bucket][kind].update(
                        value[index : index + 1],
                        target,
                        image_size=batch["image_size"][index : index + 1],
                    )
        torch.cuda.synchronize()
        self.batch_seconds.append(time.perf_counter() - started)

    def metrics(self, group: str) -> dict[str, float] | None:
        if not self.counts[group]:
            return None
        return {
            display: meter.compute()[name]
            for kind, meter in self.meters[group].items()
            for name, display in NAMES[kind].items()
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name():
        raise RuntimeError("Run the pilot evaluator on Colab L4")
    accumulator = FramingAccumulator(args.assignments, args.split)
    original = evaluation.CourtDetectionLightningModule
    original_data = evaluation.CourtDetectionDataModule
    original_validate = evaluation.pl.Trainer.validate
    timings: dict[str, float] = {}

    class TimedDataModule(CourtDetectionDataModule):
        def __init__(self, *arguments: Any, **kwargs: Any) -> None:
            started = time.perf_counter()
            super().__init__(*arguments, **kwargs)
            timings["data_preflight_seconds"] = time.perf_counter() - started

    def timed_validate(trainer: Any, *arguments: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_validate(trainer, *arguments, **kwargs)
        finally:
            torch.cuda.synchronize()
            timings["full_validation_seconds"] = time.perf_counter() - started

    class InstrumentedModule(CourtDetectionLightningModule):
        def _shared_step(self, batch: Mapping[str, object], stage: str) -> Any:
            result = super()._shared_step(batch, stage)
            if stage != "val" or not isinstance(result, CourtPoseTrainingResult):
                raise ValueError(
                    "Expected the fixed pose-training model's validation path"
                )
            accumulator.update(cast(Mapping[str, Any], batch), result)
            return result

    # Only the evaluator-local class reference changes. The model, state dict,
    # targets, batch composition, and standard accumulators remain authoritative.
    evaluation.CourtDetectionLightningModule = InstrumentedModule
    evaluation.CourtDetectionDataModule = TimedDataModule
    evaluation.pl.Trainer.validate = timed_validate
    started = time.perf_counter()
    try:
        result = evaluation.evaluate_checkpoint(
            args.checkpoint,
            args.output,
            expected_step=args.expected_step,
            real_split=args.split,
            series="pilot",
        )
    finally:
        evaluation.CourtDetectionLightningModule = original
        evaluation.CourtDetectionDataModule = original_data
        evaluation.pl.Trainer.validate = original_validate
    elapsed = time.perf_counter() - started
    standard = result["splits"][f"tennis_court_detector-{args.split}"]
    assert accumulator.seen == standard["sample_ids"]
    assert (
        len(accumulator.seen) == len(set(accumulator.seen)) == len(accumulator.groups)
    )
    all_metrics = accumulator.metrics("all")
    assert all_metrics is not None
    for name, value in all_metrics.items():
        assert math.isclose(
            value, standard["metrics"][name], rel_tol=1e-5, abs_tol=1e-5
        ), (name, value, standard["metrics"][name])
    summary = {
        "schema": "court_pilot_framing_evaluation_v1",
        "rule": RULE,
        "split": args.split,
        "expected_step": args.expected_step,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "batch_size": standard["batch_size"],
        "ordered_sample_ids_sha256": standard["ordered_sample_ids_sha256"],
        "groups": {
            group: {
                "images": accumulator.counts[group],
                "metrics": accumulator.metrics(group),
            }
            for group in (*GROUPS, "all")
        },
        "standard_metrics_agree": True,
        "wall_seconds": elapsed,
        "timings": timings,
        "additional_group_accumulation_seconds": sum(accumulator.batch_seconds),
        "interpretation": "Short training is a timing probe. Framing trends are descriptive, not a scaling result.",
    }
    (args.output / "framing.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
