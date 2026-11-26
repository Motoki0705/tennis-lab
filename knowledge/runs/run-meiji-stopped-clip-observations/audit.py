"""Check the visual review against the extracted DINO rows and count the reported cases."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

BUNDLE = Path(__file__).resolve().parent
CAMERAS = ("cam0", "cam1", "cam2", "cam3")
ROLES = ("T", "W", "C", "D")
VISIBILITY = ("outside", "clipped", "occluded", "missed")
EVENTS = (
    "off_target_box",
    "merged_box",
    "split_box",
    "detection_gap",
    "id_switch",
    "fragment",
    "selection_gap",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_review(extract: Path) -> dict[str, Any]:
    clips = read_json(extract / "index.json")["clips"]
    with (BUNDLE / "anchor-labels.csv").open(newline="") as handle:
        labels = list(csv.DictReader(handle))
    reviewed = read_json(BUNDLE / "events.json")
    edits = read_json(BUNDLE / "replay-edits.json")
    clip_names = {clip["clip"] for clip in clips}
    if (
        len(clips) != 14
        or clip_names != {r["clip"] for r in labels}
        or not (set(edits) | {r["clip"] for r in reviewed}) <= clip_names
    ):
        raise ValueError("Visual review must cover the same 14 stopped clips")

    result = {}
    for clip in clips:
        name = clip["clip"]
        target = extract / name
        observations = read_json(target / "observations.json")
        frames = clip["frames"]
        if (
            len(set(clip["anchors"])) != 3
            or not set(clip["anchors"]) <= set(frames)
            or set(map(int, observations)) != set(frames)
        ):
            raise ValueError(f"{name}: image and observation frame indices disagree")
        row_lookup = {}
        for frame in frames:
            for file in (
                f"f{frame:04d}_raw.jpg",
                f"f{frame:04d}_overlay.jpg",
                *(f"native/f{frame:04d}_{camera}.jpg" for camera in CAMERAS),
            ):
                if (target / file).stat().st_size <= 0:
                    raise ValueError(f"{name}: missing extract {file}")
            if tuple(observations[str(frame)]) != CAMERAS:
                raise ValueError(f"{name}/{frame}: camera axis mismatch")
            for camera in CAMERAS:
                row_lookup[frame, camera] = {
                    row["row"]: row for row in observations[str(frame)][camera]
                }

        counts: Counter[str] = Counter()
        focus_by_role: Counter[str] = Counter()
        seen = set()
        for label in (r for r in labels if r["clip"] == name):
            frame, camera = int(label["frame"]), label["camera"]
            key = frame, camera
            if key in seen or frame not in clip["anchors"]:
                raise ValueError(
                    f"{name}/{camera}/{frame}: duplicate or unexpected visual label"
                )
            seen.add(key)
            rows = row_lookup[key]
            assigned = set()
            for role in ROLES:
                text = label[role]
                if text in VISIBILITY:
                    counts[text] += 1
                    continue
                row_id = int(text)
                if row_id in assigned:
                    raise ValueError(
                        f"{name}/{camera}/{frame}: a player row was used for two roles"
                    )
                assigned.add(row_id)
                row = rows[row_id]
                counts["detected"] += 1
                counts["tracked"] += row["track"] is not None
                counts["selected"] += row["selected"]
                counts["carrier"] += row["group"] is not None
                for focus in clip["focus"]:
                    if (
                        focus["camera"] == camera
                        and focus["start"] <= frame < focus["end"]
                        and row[focus["kind"]] == focus["identity"]
                    ):
                        focus_by_role[role] += 1
            if label["chair"]:
                chair = int(label["chair"])
                if chair in assigned:
                    raise ValueError(
                        f"{name}/{camera}/{frame}: umpire chair row also labelled as a player"
                    )
                counts["chair_box"] += 1
                counts["chair_box_selected"] += rows[chair]["selected"]
            counts["unlabelled_selected_rows"] += sum(
                row["selected"] for i, row in rows.items() if i not in assigned
            )
        if seen != {(frame, camera) for frame in clip["anchors"] for camera in CAMERAS}:
            raise ValueError(f"{name}: anchor labels are incomplete")

        event_counts: Counter[str] = Counter()
        checked_rows: dict[str, set[int]] = {camera: set() for camera in CAMERAS}
        for event in (row for row in reviewed if row["clip"] == name):
            kind = event["kind"]
            if kind not in EVENTS or not set(event["frames"]) <= set(frames):
                raise ValueError(f"{name}: visual event has an unknown type or frame")
            evidence = {
                r
                for frame in event["frames"]
                for r in row_lookup[frame, event["camera"]]
            }
            ids = set(event.get("rows", ()))
            if not ids <= evidence:
                raise ValueError(
                    f"{name}: visual event uses DINO rows outside its reviewed frames"
                )
            if kind in {"off_target_box", "merged_box", "split_box"}:
                checked_rows[event["camera"]].update(ids)
            units = (
                len(event["frames"])
                if kind in {"off_target_box", "merged_box", "split_box"}
                else 1
            )
            if kind == "id_switch":
                units = len(event.get("tracks", [None]))
            event_counts[kind] += units
        for camera, rows in (
            edits.get(name, {}).get("remove_detection_rows", {}).items()
        ):
            if len(rows) != len(set(rows)) or not set(rows) <= checked_rows[camera]:
                raise ValueError(
                    f"{name}/{camera}: a CPU intervention lacks a reviewed DINO row"
                )
        result[name] = {
            "frames": frames,
            "anchors": clip["anchors"],
            "reason": clip["decision"]["reason"],
            "anchor_counts": dict(counts),
            "events": {key: event_counts[key] for key in EVENTS},
            "focus_carrier_matches_by_role": dict(focus_by_role),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = count_review(args.extract)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Validated {len(result)} clips, {sum(len(row['frames']) for row in result.values())} synchronized extracts and 168 anchor camera frames"
    )


if __name__ == "__main__":
    main()
