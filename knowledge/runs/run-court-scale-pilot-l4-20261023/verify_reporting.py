"""CPU checks for framing boundaries, aggregation, and command receipts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import profile_training
import torch
from evaluate_framing import FramingAccumulator
from framing import group_for_count, in_frame_count
from numpy.typing import NDArray

from src.tasks.court_detection.training.metrics import CourtDetectionMetrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert [group_for_count(i) for i in (0, 11, 12, 13, 14)] == [
        "partial",
        "partial",
        "near_full",
        "near_full",
        "full",
    ]
    points = np.ones((14, 2))
    assert in_frame_count(points, 10, 10) == 14
    points[0] = [10, 0]
    front: NDArray[np.bool_] = np.ones(14, dtype=bool)
    front[1] = False
    assert in_frame_count(points, 10, 10) == 13
    assert in_frame_count(points, 10, 10, front) == 12
    for bad in (-1, 15):
        try:
            group_for_count(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid KP counts must fail")
    assignments = args.output / "assignments.jsonl"
    ids = ["full_1", "near_1", "partial_1"]
    assignments.write_text(
        "".join(
            json.dumps(
                {"sample_id": key, "source": "real", "split": "val", "framing": group}
            )
            + "\n"
            for key, group in zip(ids, ("full", "near_full", "partial"), strict=True)
        )
    )
    accumulator = FramingAccumulator(assignments, "val")
    generator = torch.Generator().manual_seed(42)
    channels = {"kp": 14, "seg": 7, "line": 1, "semantic_line": 12}
    logits = {
        kind: torch.randn((3, count, 4, 4), generator=generator)
        for kind, count in channels.items()
    }
    targets = {
        "kp": {
            "points_xy": torch.rand((3, 14, 1, 2), generator=generator),
            "point_visible": torch.ones((3, 14, 1), dtype=torch.bool),
        },
        "line": torch.rand((3, 1, 4, 4), generator=generator),
        "seg": torch.softmax(torch.randn((3, 7, 4, 4), generator=generator), dim=1),
        "semantic_line": torch.softmax(
            torch.randn((3, 12, 4, 4), generator=generator), dim=1
        ),
    }
    size = torch.full((3, 2), 4, dtype=torch.long)
    with patch("torch.cuda.synchronize"):
        accumulator.update(
            {"sample_id": ids, "targets": targets, "image_size": size},
            SimpleNamespace(output=SimpleNamespace(dense_logits=logits)),
        )
    assert accumulator.counts == {"all": 3, "full": 1, "near_full": 1, "partial": 1}
    assert set(accumulator.metrics("all") or {}) == {
        "kp_mean_distance_px",
        "kp_median_distance_px",
        "seg_miou",
        "semantic_line_miou",
        "line_dice",
    }
    for kind, count in channels.items():
        standard = CourtDetectionMetrics(kind, count)
        standard.update(logits[kind], targets[kind], image_size=size)
        assert standard.compute() == accumulator.meters["all"][kind].compute()
    timer = Path(__file__).with_name("timed_command.py")
    for code in (0, 7):
        receipt = args.output / f"command-{code}.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(timer),
                "--receipt",
                str(receipt),
                "--",
                sys.executable,
                "-c",
                f"raise SystemExit({code})",
            ],
            check=False,
        )
        record = json.loads(receipt.read_text())
        assert completed.returncode == record["returncode"] == code
        assert record["wall_seconds"] > 0 and not record["sampled_gpu"]
    checkpoint = args.output / "fake-checkpoint.txt"
    storage_options = {"test": True}

    def fake_save(owner: Any, payload: Any, path: Any, passed_options: Any) -> None:
        assert payload["global_step"] == 32 and passed_options is storage_options
        Path(path).write_text("CPU fixture")

    def fake_train(module: str, *, run_name: str) -> None:
        assert (
            module == "src.tasks.court_detection.scripts.train"
            and run_name == "__main__"
        )
        assert sys.argv[1:] == ["--config-name", "train_data_scaling", "run.seed=42"]
        profile_training.CourtDetectionTrainingRunner.build_datamodule(None, "config")
        profile_training.RcloneArtifactStore._run(None, "copy", "local", "remote")
        profile_training.PublishingCheckpointIO.save_checkpoint(
            None, {"global_step": 32}, checkpoint, storage_options
        )

    trace = args.output / "fake-training-trace.json"
    fake_data = SimpleNamespace(
        _train_pipelines={
            "synthetic_court": SimpleNamespace(
                input_layer=SimpleNamespace(records=lambda split: tuple(range(7)))
            )
        }
    )
    with (
        patch.object(profile_training.RcloneArtifactStore, "_run", return_value=None),
        patch.object(
            profile_training.PublishingCheckpointIO, "save_checkpoint", fake_save
        ),
        patch.object(
            profile_training.CourtDetectionTrainingRunner,
            "build_datamodule",
            return_value=fake_data,
        ),
        patch.object(profile_training.runpy, "run_module", fake_train),
        patch.object(
            sys,
            "argv",
            [
                "profile_training",
                "--receipt",
                str(trace),
                "--",
                "--config-name",
                "train_data_scaling",
                "run.seed=42",
            ],
        ),
    ):
        profile_training.main()
    evidence = json.loads(trace.read_text())
    assert evidence["rclone_calls"][0]["success"]
    assert evidence["checkpoint_saves"][0]["success"]
    assert evidence["checkpoint_saves"][0]["bytes"] == len("CPU fixture")
    assert evidence["data_preflight"][0]["training_images"] == 7
    print(
        "Framing boundaries, whole-batch metric agreement, and success/failure receipts passed (CPU only)."
    )


if __name__ == "__main__":
    main()
