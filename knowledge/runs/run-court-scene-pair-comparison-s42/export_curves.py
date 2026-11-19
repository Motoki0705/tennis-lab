"""Export accepted TensorBoard history, then render it with kg_curves.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

TAGS = (
    "train/loss",
    "val/loss",
    "val/kp_mean_dist",
    "val/seg_miou",
    "val/line_dice",
    "val/semantic_line_miou",
)
REPO = Path(__file__).resolve().parents[3]
SKILL = REPO / ".agents/skills/knowledge-control/scripts"


def select_points(
    before: list[list[float]], after: list[list[float]], boundary: int | None
) -> tuple[list[list[float]], list[list[float]]]:
    if boundary is None:
        if after:
            raise ValueError("Unexpected resumed attempt")
        kept, discarded = before, []
    else:
        if any(point[0] <= boundary for point in after):
            raise ValueError("Resumed history overlaps the retained saved boundary")
        kept = [point for point in before if point[0] <= boundary] + after
        discarded = [point for point in before if point[0] > boundary]
    points: dict[int, float] = {}
    for step, value in kept:
        if not math.isfinite(value) or int(step) != step or not 0 < step <= 33160:
            raise ValueError("Nonfinite fit value or invalid completed update")
        if int(step) in points:
            raise ValueError("Duplicate fit step")
        points[int(step)] = value
    return [[step, value] for step, value in sorted(points.items())], discarded


def export(events_root: Path, sources_path: Path, canonical_root: Path) -> None:
    import yaml
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    sources = json.loads(sources_path.read_text())
    event_manifest = json.loads(sources_path.with_name("curve-inputs.json").read_text())
    actual_paths = {
        p.relative_to(events_root).as_posix()
        for p in events_root.rglob("events.out.tfevents.*")
    }
    if actual_paths != set(event_manifest["files"]):
        raise ValueError("Event inventory differs from verified Drive inputs")
    for relative, expected in event_manifest["files"].items():
        path = events_root / relative
        if (
            path.stat().st_size != expected["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]
            or expected["drive_download_verified"] is not True
        ):
            raise ValueError(f"Verified Drive event content changed: {relative}")
    source_receipts: dict[str, Any] = {
        "schema": "court_scene_pair_curve_sources_v1",
        "conditions": {},
    }
    for condition in ("P02", "P03", "P12", "P13", "P23"):
        restarts = [
            r
            for r in sources["restarts"]
            if r["condition"] == condition and not r["probe"] and r["resume_update"] > 0
        ]
        if len(restarts) > 1:
            raise ValueError("Unexpected multiple resumed attempts")
        boundary = restarts[0]["resume_update"] if restarts else None
        paths = sorted(
            (events_root / f"{condition}-s42/logs").glob(
                "version_*/events.out.tfevents.*"
            )
        )
        if len(paths) != (2 if boundary is not None else 1):
            raise ValueError(f"Unexpected event inventory for {condition}")
        attempts = []
        receipts = []
        for path in paths:
            accumulator = EventAccumulator(
                str(path), size_guidance={"scalars": 0}, purge_orphaned_data=False
            ).Reload()
            attempts.append(
                {
                    tag: [
                        [point.step + 1, float(point.value)]
                        for point in accumulator.Scalars(tag)
                    ]
                    for tag in TAGS
                }
            )
            receipts.append(
                {
                    "drive_path": "outputs/court_detection/train/scene-pair-5500-s384-colab/"
                    + path.relative_to(events_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        series, discarded = {}, {}
        for tag in TAGS:
            points, tail = select_points(
                attempts[0][tag],
                attempts[1][tag] if boundary is not None else [],
                boundary,
            )
            series[tag] = points
            discarded[tag] = tail
        if [int(p[0]) for p in series["val/loss"]] != list(range(3316, 33161, 3316)):
            raise ValueError("Expected all ten scheduled accepted validation updates")
        node_id = f"run-court-scene-pair-{condition.lower()}-s42"
        bundle = REPO / "knowledge/runs" / node_id
        payload = {
            "schema": "court_scene_pair_accepted_fit_curves_v1",
            "condition": condition,
            "completed_update": "TensorBoard step + 1",
            "validation_domain": "fit-time aggregate synthetic500 + real2211, distinct from final per-domain evaluations",
            "resume_update": boundary,
            "source_restarts": [r["record"] for r in restarts],
            "sources": receipts,
            "series": series,
            "discarded_old_tail": discarded,
        }
        (bundle / "curves.json").write_text(json.dumps(payload, indent=2) + "\n")
        canonical = canonical_root / condition
        if canonical.exists():
            raise ValueError(
                "Use a new canonical event directory, never merge stale histories"
            )
        writer = EventFileWriter(str(canonical))
        for tag, values in series.items():
            for step, value in values:
                writer.add_event(
                    Event(
                        wall_time=0,
                        step=int(step),
                        summary=Summary(
                            value=[Summary.Value(tag=tag, simple_value=value)]
                        ),
                    )
                )
        writer.close()
        node = next((REPO / "knowledge/nodes/court_detection").glob(f"*-{node_id}.md"))
        _, frontmatter, body = node.read_text().split("---", 2)
        meta = yaml.safe_load(frontmatter)
        meta["artifacts"]["tb_logdir"] = canonical.relative_to(REPO).as_posix()
        node.write_text(
            "---\n"
            + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
            + "---"
            + body
        )
        subprocess.run(
            [sys.executable, str(SKILL / "kg_curves.py"), node_id], check=True
        )
        source_receipts["conditions"][condition] = {
            "resume_update": boundary,
            "sources": receipts,
            "discarded_train_loss_points": len(discarded["train/loss"]),
            "accepted_validation_updates": [int(p[0]) for p in series["val/loss"]],
        }
    (sources_path.parent / "curve-sources.json").write_text(
        json.dumps(source_receipts, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument(
        "--sources", type=Path, default=Path(__file__).with_name("sources.json")
    )
    parser.add_argument("--canonical-root", type=Path, required=True)
    args = parser.parse_args()
    export(args.events, args.sources, args.canonical_root.resolve())
