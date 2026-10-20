"""Extract only the presentation's loss/LR columns, without transforming values.

The caller supplies the original logs; the retained artifact manifests fix their
SHA-256 and byte counts. No raw logs are restored to knowledge or copied here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from src.tasks.ball_detection.visualization.presentation.training_summary import (
    TRAINING_COLUMNS,
    training_rows,
    training_source,
)
from src.utils.checksum import dual_sha256


def export_scalars(
    raw_files: Mapping[str, Path], bundles: Mapping[str, Path], output_dir: Path
) -> None:
    """Stop on changed originals, missing samples, or changed step sequences."""
    runs, sources = {}, {}
    for stage, columns in TRAINING_COLUMNS.items():
        raw = raw_files[stage]
        source = training_source(bundles[stage])
        if (
            dual_sha256(raw) != source["sha256"]
            or raw.stat().st_size != source["size_bytes"]
        ):
            raise ValueError(
                f"Raw training log differs from the recorded source: {stage}"
            )
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        if any(
            not isinstance(row, dict) or not set(columns) <= row.keys() for row in rows
        ):
            raise ValueError(f"Missing training columns: {stage}")
        selected = {key: [row[key] for row in rows] for key in columns}
        # These are the agreed runs: 1,200 summaries at completed updates 50..60k.
        steps = {"first": 50, "last": 60000, "stride": 50}
        training_rows(selected, stage=stage, count=1200, steps=steps)
        if dual_sha256(raw) != source["sha256"]:
            raise ValueError(f"Raw training log changed while reading: {stage}")
        runs[stage] = {"source_sha256": source["sha256"], "columns": selected}
        sources[stage] = {
            **source,
            "columns": list(columns),
            "count": len(rows),
            "steps": steps,
        }
    payload = {"schema_version": 1, "kind": "ball_learning_scalars", "runs": runs}
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
    if len(encoded.encode()) > 256 * 1024:
        raise ValueError("Scalar summary exceeds 256 KiB")
    receipt = {
        "schema_version": 1,
        "kind": "measured_learning_scalar_summary",
        "policy": "Preserve selected original columns and every step/value; no smoothing, resampling, normalization or scheduler reconstruction.",
        "sources": sources,
        "output": "scalars.json",
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }
    if any(
        (output_dir / name).exists() for name in ("scalars.json", "derivation.json")
    ):
        raise FileExistsError(f"Scalar summary already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scalars.json").open("x") as stream:
        stream.write(encoded)
    with (output_dir / "derivation.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in TRAINING_COLUMNS:
        parser.add_argument(f"--{stage}-raw", type=Path, required=True)
        parser.add_argument(f"--{stage}-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = vars(parser.parse_args())
    export_scalars(
        {stage: args[f"{stage}_raw"] for stage in TRAINING_COLUMNS},
        {stage: args[f"{stage}_bundle"] for stage in TRAINING_COLUMNS},
        args["output_dir"],
    )


if __name__ == "__main__":
    main()
