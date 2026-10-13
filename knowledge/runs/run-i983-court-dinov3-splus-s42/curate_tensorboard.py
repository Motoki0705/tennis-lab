"""Derive scalar evidence from the hash-verified sources and their explicit filters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((args.bundle / "tensorboard-sources.json").read_text())
    rows: list[tuple[int, str, float, float]] = []
    terminal: dict[str, Any] = {}
    counts: list[dict[str, Any]] = []
    for source in spec["sources"]:
        path = args.raw_dir / source["name"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != source["sha256"] or path.stat().st_size != source["bytes"]:
            raise ValueError(f"Raw evidence differs from its recorded version: {path}")
        accumulator = EventAccumulator(
            str(path),
            size_guidance={"scalars": 0, "images": 1},
            purge_orphaned_data=False,
        ).Reload()
        included = discarded = 0
        for tag in accumulator.Tags()["scalars"]:
            for event in accumulator.Scalars(tag):
                if source["role"] == "terminal_test":
                    if tag.startswith("test/"):
                        if tag in terminal:
                            raise ValueError(f"Repeated terminal test scalar: {tag}")
                        terminal[tag] = {
                            "step": event.step,
                            "wall_time": event.wall_time,
                            "value": event.value,
                        }
                    continue
                if not tag.startswith(("train/", "val/")):
                    continue
                before = source.get("keep_step_before")
                if before is not None and event.step >= before:
                    discarded += 1
                    continue
                rows.append((event.step, tag, event.wall_time, event.value))
                included += 1
        counts.append(
            {"source": source["name"], "included": included, "discarded": discarded}
        )
    keys = [(step, tag) for step, tag, _, _ in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "Duplicate tag/step remains after the source filters"
        )
    output = args.bundle / "tensorboard" / "version_0"
    output.mkdir(parents=True, exist_ok=False)
    writer = EventFileWriter(str(output))
    try:
        for step, tag, wall_time, value in sorted(rows):
            writer.add_event(
                Event(
                    wall_time=wall_time,
                    step=step,
                    summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]),
                )
            )
    finally:
        writer.close()
    check = EventAccumulator(str(output), size_guidance={"scalars": 0}).Reload()
    actual = sum(len(check.Scalars(tag)) for tag in check.Tags()["scalars"])
    if actual != len(rows):
        raise ValueError("Derived event did not preserve the selected scalar count")
    report = {
        "schema_version": 1,
        "kind": "derived_training_curves",
        "source_manifest": "tensorboard-sources.json",
        "policy": spec["policy"],
        "sources": counts,
        "selected_scalars": len(rows),
        "duplicate_tag_steps": 0,
        "terminal_test_scalars": len(terminal),
    }
    (args.bundle / "tensorboard-derivation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.bundle / "terminal-test-metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "Automatic test of the final epoch model; not the best-checkpoint comparison evaluation.",
                "metrics": terminal,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
