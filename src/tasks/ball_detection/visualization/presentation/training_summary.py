"""Read loss/LR columns extracted unchanged from the recorded training logs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.utils.checksum import dual_sha256

_BASE_COLUMNS = ("epoch", "global_step", "train_loss", "learning_rate")
TRAINING_COLUMNS = {
    "pretraining": _BASE_COLUMNS,
    "posttraining": (*_BASE_COLUMNS, "encoder_frozen", "encoder_learning_rate"),
}


def training_source(bundle: Path) -> dict[str, Any]:
    """Use the deletion record as the authority for the original bytes/hash."""
    manifest = json.loads((bundle / "artifact-manifest.json").read_text())
    matches = [
        (index, record)
        for index, record in enumerate(manifest["files"])
        if record["path"] == "train-summary.jsonl"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one train-summary.jsonl source record: {bundle}")
    index, record = matches[0]
    run = f"knowledge/runs/{manifest['run_id']}"
    return {
        "source_path": f"{run}/train-summary.jsonl",
        "source_manifest": f"{run}/artifact-manifest.json#/files/{index}",
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }


def training_rows(
    columns: dict[str, Any], *, stage: str, count: int, steps: Mapping[str, int]
) -> tuple[dict[str, Any], ...]:
    """Reject missing samples/columns and changed update indices explicitly."""
    if (
        type(count) is not int
        or count <= 0
        or set(columns) != set(TRAINING_COLUMNS[stage])
        or any(
            not isinstance(values, list) or len(values) != count
            for values in columns.values()
        )
    ):
        raise ValueError(f"Training column/count mismatch: {stage}")
    if (
        set(steps) != {"first", "last", "stride"}
        or any(type(value) is not int or value <= 0 for value in steps.values())
        or columns["global_step"][-1] != steps["last"]
        or columns["global_step"]
        != list(range(steps["first"], steps["last"] + 1, steps["stride"]))
    ):
        raise ValueError(f"Training step mismatch: {stage}")
    for key, values in columns.items():
        for value in values:
            if key == "encoder_frozen":
                valid = type(value) is bool
            elif key in ("epoch", "global_step"):
                valid = type(value) is int and value >= 0
            else:
                valid = (
                    type(value) in (int, float) and math.isfinite(value) and value >= 0
                )
            if not valid:
                raise ValueError(f"Invalid training scalar: {stage}.{key}")
    return tuple(
        dict(zip(columns, values, strict=True))
        for values in zip(*columns.values(), strict=True)
    )


def read_training_summary(
    path: Path, bundles: Mapping[str, Path]
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Verify the compact asset, its derivation, and both original source hashes."""
    payload = json.loads(path.read_text())
    receipt = json.loads(path.with_name("derivation.json").read_text())
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "kind", "runs"}
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["kind"] != "ball_learning_scalars"
        or not isinstance(receipt, dict)
        or set(receipt)
        != {"schema_version", "kind", "policy", "sources", "output", "sha256"}
        or type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["kind"] != "measured_learning_scalar_summary"
        or set(payload["runs"]) != set(TRAINING_COLUMNS)
        or set(receipt["sources"]) != set(TRAINING_COLUMNS)
        or receipt["output"] != path.name
    ):
        raise ValueError("Invalid ball learning scalar summary schema")
    if dual_sha256(path) != receipt["sha256"]:
        raise ValueError("Training summary SHA-256 differs from derivation")
    result = {}
    for stage, bundle in bundles.items():
        run, source = payload["runs"][stage], receipt["sources"][stage]
        expected = training_source(bundle)
        if (
            set(run) != {"source_sha256", "columns"}
            or set(source) != {*expected, "columns", "count", "steps"}
            or any(source[key] != value for key, value in expected.items())
            or run["source_sha256"] != expected["sha256"]
            or source["columns"] != list(TRAINING_COLUMNS[stage])
        ):
            raise ValueError(f"Training summary source/columns mismatch: {stage}")
        result[stage] = training_rows(
            run["columns"], stage=stage, count=source["count"], steps=source["steps"]
        )
    return result
