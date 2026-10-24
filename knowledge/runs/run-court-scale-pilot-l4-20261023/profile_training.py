"""Run the existing train CLI with timing around artifact and checkpoint I/O."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path
from typing import Any

from src.tasks.base.training.artifact_store import PublishingCheckpointIO
from src.tasks.court_detection.training.runner import CourtDetectionTrainingRunner
from src.utils.artifact_store import RcloneArtifactStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    if not overrides or args.receipt.exists():
        raise ValueError("Supply the normal train CLI arguments and a new receipt")
    calls: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    preflight: list[dict[str, Any]] = []
    original_call = RcloneArtifactStore._run
    original_save = PublishingCheckpointIO.save_checkpoint
    original_data = CourtDetectionTrainingRunner.build_datamodule

    def traced_data(self: Any, *arguments: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_data(self, *arguments, **kwargs)
        preflight.append(
            {
                "wall_seconds": time.perf_counter() - started,
                "training_images": len(
                    result._train_pipelines["synthetic_court"].input_layer.records(
                        "train"
                    )
                ),
            }
        )
        return result

    def traced_call(self: Any, *arguments: str, **kwargs: Any) -> None:
        started = time.perf_counter()
        entry = {
            "operation": arguments[0],
            "started_unix": time.time(),
            "arguments": list(arguments[1:]),
            "success": False,
        }
        try:
            original_call(self, *arguments, **kwargs)
            entry["success"] = True
        finally:
            entry["wall_seconds"] = time.perf_counter() - started
            calls.append(entry)

    def traced_save(
        self: Any, checkpoint: Any, path: Any, *positional: Any, **kwargs: Any
    ) -> None:
        started = time.perf_counter()
        entry = {
            "path": str(path),
            "global_step": checkpoint["global_step"],
            "started_unix": time.time(),
            "success": False,
        }
        try:
            original_save(self, checkpoint, path, *positional, **kwargs)
            entry["success"] = True
        finally:
            entry["wall_seconds"] = time.perf_counter() - started
            entry["bytes"] = Path(path).stat().st_size if Path(path).is_file() else None
            checkpoints.append(entry)

    RcloneArtifactStore._run = traced_call
    PublishingCheckpointIO.save_checkpoint = traced_save
    CourtDetectionTrainingRunner.build_datamodule = traced_data
    old_argv = sys.argv
    started = time.perf_counter()
    try:
        sys.argv = ["train", *overrides]
        runpy.run_module("src.tasks.court_detection.scripts.train", run_name="__main__")
    finally:
        sys.argv = old_argv
        RcloneArtifactStore._run = original_call
        PublishingCheckpointIO.save_checkpoint = original_save
        CourtDetectionTrainingRunner.build_datamodule = original_data
        receipt = {
            "schema": "court_pilot_training_io_profile_v1",
            "wall_seconds": time.perf_counter() - started,
            "rclone_calls": calls,
            "checkpoint_saves": checkpoints,
            "data_preflight": preflight,
            "model_training_entrypoint": "src.tasks.court_detection.scripts.train",
            "instrumentation": "Timers only; original methods receive identical arguments. No model, data, RNG or schedule changes.",
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.receipt.with_suffix(".partial")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(args.receipt)


if __name__ == "__main__":
    main()
