"""Build the chapter's 2D tracking ablation table from the two saved reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def table() -> list[dict[str, str | int]]:
    runs = Path(__file__).resolve().parents[1]
    r10 = "run-i964-tracker-hybrids-r10-20260930"
    r11 = "run-i964-default-merge-r11-20260930"
    merge = rows(runs / r11 / "merge_counts.csv")
    detections = sum(int(row["input_boxes"]) for row in merge)
    dropped = sum(int(row["dropped_boxes"]) for row in merge)
    edited = sum(int(row["changed_frames"]) for row in merge)
    result: list[dict[str, str | int]] = []
    for run in (r10, r11):
        comparison = runs / run / "comparison.csv"
        association = runs / run / "association.csv"
        candidates = {
            row["variant"]: row
            for row in rows(association)
            if row["encoder"] == "clipreid_vitb16_market1501"
        }
        for row in rows(comparison):
            if row["camera"] != "all" or row["near_far"] != "all":
                continue
            method = row["variant"]
            pair = candidates[method]
            merge_on = "merge_on" in method
            result.append(
                {
                    "run": run,
                    "variant": method,
                    **{
                        k: row[k]
                        for k in (
                            "idf1",
                            "idtp",
                            "idfp",
                            "idfn",
                            "id_switches",
                            "fragments",
                            "player_units_kept",
                            "player_units",
                            "nonplayer_units_kept",
                            "nonplayer_units",
                            "group_idf1",
                            "group_idtp",
                            "group_idfp",
                            "group_idfn",
                            "group_id_switches",
                            "group_fragments",
                        )
                    },
                    **{
                        k: pair[k]
                        for k in (
                            "encoder",
                            "pair_f1",
                            "tp",
                            "fp",
                            "fn",
                            "decided_clips",
                            "total_clips",
                            "matched_label_boxes",
                            "label_boxes",
                        )
                    },
                    "input_detection_boxes": detections if run == r11 else "",
                    "dropped_detection_boxes": (dropped if merge_on else 0)
                    if run == r11
                    else "",
                    "changed_camera_frames": (edited if merge_on else 0)
                    if run == r11
                    else "",
                    "tracking_source": f"../{run}/comparison.csv",
                    "pair_source": f"../{run}/association.csv",
                    "merge_source": f"../{run}/merge_counts.csv" if run == r11 else "",
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = table()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
