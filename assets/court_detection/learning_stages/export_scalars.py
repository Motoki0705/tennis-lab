"""Export the presentation's measured scalars without retaining raw events.

The raw event is verified against tensorboard-sources.json. Tag, step, value
and wall_time are preserved, including the original epoch-start LR samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tags-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads((args.bundle / "tensorboard-sources.json").read_text())
    (source,) = [item for item in sources["sources"] if item["role"] == "training"]
    with args.raw_file.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source["sha256"] or args.raw_file.stat().st_size != source["bytes"]:
        raise ValueError("Raw training event differs from the recorded source")
    events = EventAccumulator(
        str(args.raw_file), size_guidance={"scalars": 0}, purge_orphaned_data=False
    ).Reload()
    tags = list(OmegaConf.load(args.tags_config).tags.values())
    scalars = {}
    for tag in tags:
        rows = events.Scalars(tag)
        scalars[tag] = {
            "steps": [row.step for row in rows],
            "values": [row.value for row in rows],
            "wall_times": [row.wall_time for row in rows],
        }
    payload = {
        "schema_version": 1,
        "kind": "court_learning_scalars",
        "source_sha256": digest,
        "scalars": scalars,
    }
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
    if len(encoded.encode()) > 256 * 1024:
        raise ValueError("Scalar summary exceeds 256 KiB")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "scalars.json"
    with output.open("x") as stream:
        stream.write(encoded)
    receipt = {
        "schema_version": 1,
        "kind": "measured_learning_scalar_summary",
        "source_manifest": "tensorboard-sources.json",
        "source_sha256": digest,
        "policy": "Preserve selected original tag, step, value and wall_time; no smoothing or scheduler reconstruction.",
        "counts": {tag: len(values["steps"]) for tag, values in scalars.items()},
        "output": output.name,
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }
    (args.output_dir / "derivation.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
