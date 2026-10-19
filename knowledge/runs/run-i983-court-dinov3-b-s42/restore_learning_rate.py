"""Restore epoch LR scalars omitted by the original train/val-only curation.

Use the hash-verified raw training event recorded in tensorboard-sources.json.
The original curated events and their derivation receipt remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.record_writer import RecordWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
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
    tag = "lr-AdamW"
    values = events.Scalars(tag)
    if len(values) != 20 or [value.step for value in values] != [
        1658 * epoch for epoch in range(20)
    ]:
        raise ValueError("Expected 20 epoch-start learning-rate measurements")
    output = args.bundle / "tensorboard/version_0/events.out.tfevents.learning_rate"
    with output.open("xb") as stream:
        writer = RecordWriter(stream)
        writer.write(Event(file_version="brain.Event:2").SerializeToString())
        for value in values:
            writer.write(
                Event(
                    wall_time=value.wall_time,
                    step=value.step,
                    summary=Summary(
                        value=[Summary.Value(tag=tag, simple_value=value.value)]
                    ),
                ).SerializeToString()
            )
        writer.flush()
    with output.open("rb") as stream:
        output_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {
        "schema_version": 1,
        "kind": "restored_learning_rate_scalars",
        "source_manifest": "tensorboard-sources.json",
        "source": source,
        "tag": tag,
        "count": len(values),
        "policy": "Retain original tag, step, value and wall_time; no scheduler reconstruction.",
        "output": output.relative_to(args.bundle).as_posix(),
        "sha256": output_digest,
    }
    (args.bundle / "tensorboard-lr-derivation.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
