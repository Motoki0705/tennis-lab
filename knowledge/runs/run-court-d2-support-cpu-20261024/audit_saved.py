"""Reclassify the saved pilot, including geometry/QA rejections, without rendering."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from support import (
    GROUPS,
    PREVIOUS,
    count_points,
    framing,
    partitions,
    stats,
    target_points,
    write,
)

from src.synthetic_data_generation.dataset.court.sample_store import (
    open_court_store,
    read_court_manifest,
)
from src.synthetic_data_generation.dataset.court.scale.publication import (
    iter_plan_groups,
    load_dataset_version,
    read_json,
)
from src.utils.data.image_record_store import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    version = load_dataset_version(args.version, verify_images=True)
    assert (
        file_sha256(args.version)
        == "5a879b3f371d10792613d9f19beb5b2bba96c4154f609bbc4c10a07995ff452e"
    )
    expected = json.loads((args.diagnosis / "baseline-geometry-rows.json").read_text())
    geometry = {(row["scene"], row["sample_id"]): row for row in expected}
    rows = []
    statuses: dict[str, list[dict[str, Any]]] = {
        "qa_accepted": [],
        "qa_rejected": [],
        "geometry_rejected": [],
    }
    quality_reasons: Counter[str] = Counter()
    missing_projection = 0
    camera_matches = 0
    for scene in version["scenes"]:
        groups = iter_plan_groups(
            args.version.parent, read_json(args.version.parent / scene["plan_path"])
        )
        planned = {
            record["sample_id"]: record
            for group in groups
            for record in group["samples"]
        }
        for cache in scene["render_caches"]:
            completion = read_json(args.version.parent / cache["path"])
            ledger = read_json(args.version.parent / completion["ledger_path"])
            for item in ledger["records"]:
                row = dict(geometry[scene["scene_id"], item["sample_id"]])
                record = item["record"]
                assert record["camera"] == planned[item["sample_id"]]["camera"]
                camera_matches += 1
                if record["projection"] is not None:
                    w, h = record["projection"]["resolution"]
                    assert row["in_frame_kp"] == count_points(
                        target_points(record), w, h
                    )
                else:
                    assert item["state"] == "geometry_rejected"
                    missing_projection += 1
                state = {
                    "accepted": "qa_accepted",
                    "qa_rejected": "qa_rejected",
                    "geometry_rejected": "geometry_rejected",
                }[item["state"]]
                row["state"] = state
                row["qa"] = item["qa"]
                statuses[state].append(row)
                if item["state"] == "qa_rejected":
                    quality_reasons.update(item["reasons"])
        for entry in scene["stores"]:
            root = args.version.parent / entry["path"]
            store = open_court_store(root)
            manifest = read_court_manifest(root, store=store)
            for sample in manifest["samples"]:
                points = target_points(sample)
                w, h = sample["projection"]["resolution"]
                count = count_points(points, w, h)
                row = geometry[scene["scene_id"], sample["sample_id"]]
                assert row["in_frame_kp"] == count
                rows.append({**row, "framing": framing(count), "qa": sample["qa"]})
    assert len(rows) == 1102
    assert len({(r["scene"], r["sample_id"]) for r in rows}) == len(rows)
    observed = {
        scene: {
            g: sum(r["framing"] == g and r["scene"] == scene for r in rows)
            for g in GROUPS
        }
        for scene in ("B00", "B01", "B02", "B03")
    }
    assert observed == {
        "B00": {"full": 0, "near_full": 1, "partial": 275},
        "B01": {"full": 6, "near_full": 8, "partial": 331},
        "B02": {"full": 0, "near_full": 0, "partial": 243},
        "B03": {"full": 2, "near_full": 1, "partial": 235},
    }
    previous_synthetic = json.loads(
        (PREVIOUS / "evidence/framing-summary.json").read_text()
    )["synthetic"]
    for scene, prior in previous_synthetic.items():
        histogram = Counter(r["in_frame_kp"] for r in rows if r["scene"] == scene)
        assert {str(k): v for k, v in histogram.items()} == prior[
            "in_frame_kp_histogram"
        ]
        assert observed[scene] == prior["groups"]
    write(args.output / "accepted-rows.json", rows)
    qa_summary = {}
    for state in ("qa_accepted", "qa_rejected"):
        qa_summary[state] = {
            field: stats(
                [r["qa"][field] for r in statuses[state] if r["qa"][field] is not None]
            )
            for field in ("target_visible_kp", "roi_hole_fraction", "roi_alpha_lt_0_9")
        }
    write(
        args.output / "saved-pilot.json",
        {
            "schema": "court_d2_saved_pilot_audit_v1",
            "rule": "in_frame_kp14_v1",
            "images_verified": len(rows),
            "dataset_file_sha256": file_sha256(args.version),
            "camera_records_matched": camera_matches,
            "null_semantic_projection_geometric_rejections": missing_projection,
            "by_scene": partitions(rows, "scene"),
            "by_framing": partitions(rows, "framing"),
            "states": {
                name: partitions(selected, "scene")
                for name, selected in statuses.items()
            },
            "qa_statistics": qa_summary,
            "qa_rejection_reasons": dict(quality_reasons),
            "previous_counts_and_histograms_match": True,
        },
    )
    # Stored JPEGs cannot re-establish alpha/depth gates; their immutable ledgers do.
    assert all(
        r["qa"]["target_visible_kp"] >= 4 and r["qa"]["roi_hole_fraction"] <= 0.01
        for r in rows
    )
    assert all(np.isfinite(r["qa"]["roi_alpha_lt_0_9"]) for r in rows)
    print(
        json.dumps(
            {
                "accepted": len(rows),
                "framing": observed,
                "states": {s: len(v) for s, v in statuses.items()},
                "reasons": dict(quality_reasons),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
