"""Evaluate all ten frozen runs, historical weights and fixed diagnostic controls."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluation import clean_json, evaluate, file_sha256, recompute, save_json
from .issue1044 import ARMS
from .real_observations import evaluate_real, recompute_real
from .repeat_selected import select_additional_arm


def verify_dataset_snapshot(data_root: Path, output: Path) -> None:
    """Reject a changed snapshot before evaluating the frozen checkpoints."""
    manifest = Path(
        "knowledge/runs/run-i1032-blcs-physics-v3-e200-s42/dataset_files.jsonl.gz"
    )
    raw = gzip.decompress(manifest.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != "36c1c978867016173a32e0f75bae06165bb557bfd981a32db27bec2a291e9544":
        raise ValueError("The fixed dataset manifest changed")
    root = data_root / "blcs/single_object-physics-v3-cc863cacd-20261007"
    began = time.monotonic()
    records = [json.loads(line) for line in raw.splitlines()]
    expected = {row["path"] for row in records}
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    mismatches = [
        {"path": name, "reason": "missing"} for name in sorted(expected - actual)
    ] + [{"path": name, "reason": "unexpected"} for name in sorted(actual - expected)]
    for row in records:
        path = root / row["path"]
        if not path.is_file():
            continue
        if path.stat().st_size != row["size"] or file_sha256(path) != row["sha256"]:
            mismatches.append({"path": row["path"], "reason": "content changed"})
    save_json(
        output,
        {
            "manifest": str(manifest),
            "manifest_decompressed_sha256": digest,
            "root": str(root),
            "file_count": len(records),
            "total_bytes": sum(row["size"] for row in records),
            "mismatches": mismatches,
            "elapsed_s": time.monotonic() - began,
        },
    )
    if mismatches:
        raise ValueError(f"Dataset snapshot changed: see {output}")


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


def assert_recomputed(expected: Any, actual: Any, key: str = "metrics") -> None:
    """Require the same support/undefined structure and numerical values."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"Saved metric keys changed: {key}")
        for name in expected:
            assert_recomputed(expected[name], actual[name], f"{key}/{name}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Saved support changed: {key}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            assert_recomputed(left, right, f"{key}/{index}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if actual is None or not np.isclose(expected, actual, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Saved metric mismatch: {key}: {expected} != {actual}")
    elif expected != actual:
        raise ValueError(f"Saved value changed: {key}: {expected} != {actual}")


def validate_saved_metrics(directory: Path) -> None:
    saved = json.loads((directory / "evaluation.json").read_text())
    if saved["split"] not in {"val", "test"}:
        raise ValueError("Saved evaluation has an unknown split")
    restored = recompute(
        directory / f"pred_{saved['split']}.npz", include_diagnostics=True
    )
    assert_recomputed(saved["schema"], restored["schema"])
    assert_recomputed(saved["metrics"], restored["metrics"])
    for mode in ("oracle", "predicted"):
        key = f"integrated_{mode}"
        if key in saved:
            assert_recomputed(saved[key], restored[key], key)
    originals = {row["scene_id"]: row for row in saved["clips"]}
    for row in restored.get("diagnostics", []):
        original = originals[row["scene_id"]]
        for key in row:
            assert_recomputed(original[key], row[key], f"{row['scene_id']}/{key}")
    save_json(directory / "recomputed.json", restored)


def validate_real_metrics(directory: Path) -> None:
    saved = json.loads((directory / "evaluation.json").read_text())
    restored = clean_json(recompute_real(directory))
    for key, value in restored.items():
        assert_recomputed(saved[key], value, key)
    save_json(directory / "recomputed.json", restored)


def run_final_evaluation(args: Namespace) -> None:
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("Final evaluation must run through the shared queue")
    torch.set_num_threads(2)
    root = args.run_root
    runs = trained_runs(root)
    verify_dataset_snapshot(args.data_root, root / "final-dataset-verification.json")
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
            validate_real_metrics(real_output)
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
