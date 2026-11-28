"""Audit the paired full training logs and terminal weights on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from run import STEM_WEIGHT, budget

from data import (
    CONDITIONS,
    LABEL_KINDS,
    PACK_SIZE,
    RUN,
    SEED,
    FrozenInputs,
    require_sha,
)
from src.tasks.ball_detection.checkpoints.temporal import json_bytes, save_json
from src.tasks.ball_detection.training.temporal.packs import pack_index


def verify(training: Path, inputs: FrozenInputs) -> dict[str, Any]:
    finish = json.loads((training / "completed.json").read_bytes())
    plan_file = training / "plan.json"
    require_sha(plan_file, finish["plan_sha256"])
    require_sha(RUN / "plan.json", finish["plan_sha256"])
    plan = json.loads(plan_file.read_bytes())
    updates = plan["updates"]
    if plan["inputs"] != inputs.audit() or set(finish["conditions"]) != set(CONDITIONS):
        raise ValueError(
            "Completion does not cover the three conditions on the frozen inputs"
        )

    draws, shared, stages = set(), set(), {}
    for name, frames in CONDITIONS.items():
        directory = training / name
        stage = finish["conditions"][name]
        recipe = json.loads((directory / "recipe.json").read_bytes())
        config = recipe["model"]
        channels = 3 if frames == 1 else 5
        if (
            json.loads((directory / "completed.json").read_bytes()) != stage
            or recipe["plan_sha256"] != finish["plan_sha256"]
            or recipe["budget"] != plan["training_budget"][name]
            or recipe["code"] != plan["code"]
            or recipe["runtime"] != plan["runtime"]
            or recipe["initialization"] != stage["train"]["initialization"]
            or (config["frames"], config["input_channels"]) != (frames, channels)
            or f"{recipe['width']}x{recipe['height']}" != plan["selected"]
        ):
            raise ValueError(f"Stage recipe disagrees with the frozen plan: {name}")

        checkpoint = directory / "final.pt"
        require_sha(checkpoint, stage["weights_sha256"])
        model = torch.load(checkpoint, weights_only=True, map_location="cpu")
        weights = model["weights"]
        if (
            model["recipe"] != recipe
            or model["updates"] != updates
            or weights[STEM_WEIGHT].shape[1] != channels
            or sum(t.numel() for t in weights.values())
            != recipe["initialization"]["parameters"]
            or not all(bool(torch.isfinite(t).all()) for t in weights.values())
        ):
            raise ValueError(f"Invalid final checkpoint: {name}")
        shared.add(recipe["initialization"]["shared_sha256"])

        log = directory / "train.jsonl"
        require_sha(log, stage["train"]["train_log_sha256"])
        rows = [json.loads(line) for line in log.read_bytes().splitlines()]
        expected_budget = budget(updates, frames)
        if len(rows) != updates or stage["train"]["updates"] != updates:
            raise ValueError(f"Training update count differs from the plan: {name}")
        counts: Counter[str] = Counter()
        for step, row in enumerate(rows, 1):
            index = pack_index(SEED, step - 1, len(inputs.train))
            block = inputs.train[index]
            codes = inputs.labels[block.clip_index].codes[block.start : block.stop]
            expected = {
                kind: int((codes == i).sum()) for i, kind in enumerate(LABEL_KINDS)
            }
            if (
                row["update"] != step
                or row["pack"] != [index, block.clip_index, block.start]
                or row["input_frames"] != PACK_SIZE
                or row["teacher_counts"] != expected
                or row["lr"] != expected_budget.lr(step - 1)
                or not all(math.isfinite(row[k]) for k in ("loss", "grad_norm"))
            ):
                raise ValueError(
                    f"Training log disagrees with the frozen draw or schedule: {name}, {step}"
                )
            counts.update(row["teacher_counts"])
        digest = hashlib.sha256(
            json_bytes([[r["pack"], r["teacher_counts"]] for r in rows])
        ).hexdigest()
        if (
            digest != stage["train"]["pack_draws_sha256"]
            or sum(counts.values()) != updates * PACK_SIZE
        ):
            raise ValueError(f"Training draw digest or frame count differs: {name}")
        draws.add(digest)
        stages[name] = dict(
            updates=updates,
            input_frames=sum(counts.values()),
            teacher_counts=counts,
            lr_first=rows[0]["lr"],
            lr_warmup_end=rows[expected_budget.warmup_updates - 1]["lr"],
            lr_final=rows[-1]["lr"],
            final_weights_sha256=stage["weights_sha256"],
            train_log_sha256=stage["train"]["train_log_sha256"],
        )
    if len(draws) != 1 or shared != {
        plan["estimates"][plan["selected"]]["shared_initialization_sha256"]
    }:
        raise ValueError(
            "The conditions did not share the planned initialization and pack draws"
        )
    return dict(
        execution_commit=finish["commit"],
        plan_sha256=finish["plan_sha256"],
        queue_job=finish["queue_job"],
        updates=updates,
        seed=plan["seed"],
        pack_draws_sha256=draws.pop(),
        shared_initialization_sha256=shared.pop(),
        stages=stages,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    save_json(
        args.output, verify(args.training, FrozenInputs(args.dataset_root.resolve()))
    )


if __name__ == "__main__":
    main()
