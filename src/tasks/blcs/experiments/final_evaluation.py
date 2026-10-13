"""Evaluate all ten frozen runs, historical weights and fixed diagnostic controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluation import evaluate, file_sha256, recompute, save_json
from .issue1044 import ARMS
from .real_observations import evaluate_real
from .repeat_selected import select_additional_arm


def trained_runs(root: Path) -> list[dict[str, Any]]:
    selection = select_additional_arm(root)
    conditions = [(arm, 42) for arm in ARMS] + [
        (arm, seed)
        for seed in (43, 44)
        for arm in ("reprojection", selection["selected_arm"])
    ]
    result = []
    for arm, seed in conditions:
        name = f"{arm}-s{seed}-u4000"
        receipt = json.loads((root / name / "selection.json").read_text())
        if (
            receipt["optimizer_updates"] != 4000
            or receipt["seed"] != seed
            or receipt["arm"] != arm
            or receipt["smoke"]
            or receipt["selection_used_test"]
        ):
            raise ValueError(f"Invalid completed run: {name}")
        if file_sha256(Path(receipt["checkpoint"])) != receipt["checkpoint_sha256"]:
            raise ValueError(f"Selected checkpoint changed: {name}")
        result.append({"name": name, "receipt": receipt})
    if len(result) != 10:
        raise AssertionError("Protocol requires ten completed training runs")
    return result


def validate_saved_metrics(directory: Path) -> None:
    saved = json.loads((directory / "evaluation.json").read_text())
    restored = recompute(directory / "pred_test.npz")
    if restored["schema"] != saved["schema"]:
        raise ValueError("Saved metric schema changed")
    for key, value in saved["metrics"].items():
        actual = restored["metrics"][key]
        if value is None:
            if actual is not None:
                raise ValueError(f"Undefined saved metric changed: {key}")
        elif actual is None or not np.isclose(value, actual, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Saved metric mismatch: {key}: {value} != {actual}")
    save_json(directory / "recomputed.json", restored)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--real-store", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("Final evaluation must run through the shared queue")
    torch.set_num_threads(2)
    root = Path("outputs/blcs/i1044")
    runs = trained_runs(root)
    save_json(root / "completed-training.json", runs)
    conditions = [
        {
            "name": row["name"],
            "checkpoint": Path(row["receipt"]["checkpoint"]),
            "baseline": None,
            "historical": False,
        }
        for row in runs
    ] + [
        {
            "name": "historical-u9000",
            "checkpoint": args.historical_checkpoint,
            "baseline": None,
            "historical": True,
        }
    ]
    conditions += [
        {"name": name, "checkpoint": None, "baseline": name, "historical": False}
        for name in ("gt", "geometry", "jitter", "time_shift", "collision_crossing")
    ]
    for condition in conditions:
        name = condition["name"]
        output = root / "evaluations" / str(name)
        print(f"Final synthetic evaluation: {name}", flush=True)
        checkpoint = condition["checkpoint"]
        if not (output / "evaluation.json").exists():
            evaluate(
                data_root=args.data_root,
                output=output,
                split="test",
                checkpoint=checkpoint,
                baseline=condition["baseline"],
                historical=condition["historical"],
            )
        else:
            saved = json.loads((output / "evaluation.json").read_text())
            expected = file_sha256(checkpoint) if checkpoint else None
            if (
                saved["checkpoint_sha256"] != expected
                or saved["baseline"] != condition["baseline"]
                or saved["split"] != "test"
            ):
                raise ValueError(f"Cannot reuse mismatched evaluation: {name}")
        validate_saved_metrics(output)
        if checkpoint is not None or name == "geometry":
            real_output = root / "real" / str(name)
            print(f"Final real observation diagnostics: {name}", flush=True)
            if not (real_output / "evaluation.json").exists():
                evaluate_real(
                    source=args.real_store,
                    data_root=args.data_root,
                    output=real_output,
                    checkpoint=checkpoint,
                    historical=condition["historical"],
                )
            else:
                saved_real = json.loads((real_output / "evaluation.json").read_text())
                expected = file_sha256(checkpoint) if checkpoint else None
                if saved_real["checkpoint_sha256"] != expected or saved_real[
                    "source_index_sha256"
                ] != file_sha256(args.real_store / "scene.json"):
                    raise ValueError(
                        f"Cannot reuse mismatched real observations: {name}"
                    )
        torch.cuda.empty_cache()
    from .report import create_report

    create_report(root)
    save_json(
        root / "evaluation-complete.json",
        {
            "conditions": [row["name"] for row in conditions],
            "training_runs": 10,
            "optimizer_updates_per_run": 4000,
            "used_test_for_selection": False,
            "run_id": os.environ["TENNIS_RUN_ID"],
        },
    )


if __name__ == "__main__":
    main()
