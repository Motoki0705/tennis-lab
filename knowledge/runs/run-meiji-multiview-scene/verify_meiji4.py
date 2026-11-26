"""Verify the 69 adopted-input replays and compare the two completion tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median

import numpy as np

from src.tennis_scene.multiview_scene.inputs import INPUT_NODE, RECIPE
from src.tennis_scene.multiview_scene.report import report_clip
from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.tennis_scene.pipeline.storage.scene_index import (
    annotation_scene_path,
    indexed_scene_path,
)
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic

BALL = ("ball_uv", "ball_vis", "ball_3d", "ball_3d_valid", "ball_rejection_code")
BODY_STAGE = (
    "body_view_selection",
    "hmr2_features",
    "gvhmr",
    "body_placement",
    "scene_assembly",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    indexed = {row["clip_id"]: row for row in rows}
    expected = {f"video_000/clip_{idx:03d}" for idx in range(69)}
    if len(rows) != 69 or set(indexed) != expected:
        raise ValueError(f"Wrong clip cohort: {path}")
    return indexed


def profile(path: Path) -> dict:
    # The recorder writes these top-level fields before the resource samples
    # and event table, which are several GiB for the archived run.
    with path.open(encoding="utf-8") as file:
        prefix = file.read(2 * 1024 * 1024)
    info = {}
    for field in ("status", "started_at", "total_wall_seconds", "resources"):
        match = re.search(rf'"{field}"\s*:\s*', prefix)
        if match is None:
            raise ValueError(f"Missing recorder field {field}: {path}")
        info[field] = json.JSONDecoder().raw_decode(prefix[match.end() :])[0]
    return info


def verify_clip(
    legacy: StoreReader, new: StoreReader, expected: dict[str, str]
) -> dict:
    old_archive = annotation_scene_path(
        legacy.root, read_json(legacy.root / "annotation.json")
    )
    archive = indexed_scene_path(new.index_path)
    meta = read_json(archive.with_suffix(".metadata.json"))
    old_meta = read_json(old_archive.with_suffix(".metadata.json"))
    pins = new.descriptor(INPUT_NODE)["identity"]["pins"]
    old_deps = legacy.descriptor("scene_assembly")["dependencies"]
    if (
        legacy.source_sha256 != new.source_sha256
        or pins["legacy_index_sha256"] != dual_sha256(legacy.index_path)
        or pins["legacy_scene"] != legacy.references["scene_assembly"]
        or meta["status"] != "ok"
        or meta["player_reconstruction_method"] != RECIPE
        or meta["reused_ball"] != pins["ball"]
        or pins["ball"] != old_deps["ball"]
        or meta["reused_alignment"] != pins["alignment"]
        or pins["alignment"] != old_deps["alignment"]
        or meta["court_reference"] != old_meta["court_reference"]
    ):
        raise ValueError(
            f"Input lineage or adopted camera gauge changed: {new.source['clip_id']}"
        )
    placement = new.descriptor("body_placement")
    directory = (new.root / new.references["body_placement"]["path"]).parent
    mask = unpack_value(
        placement["payload"]["players"]["smpl_valid"], directory, placement["arrays"]
    )
    with (
        np.load(archive, allow_pickle=False) as current,
        np.load(old_archive, allow_pickle=False) as old,
    ):
        for field in BALL:
            if current[field].dtype != old[field].dtype or not np.array_equal(
                current[field], old[field]
            ):
                raise ValueError(f"Ball array differs: {new.source['clip_id']} {field}")
        np.testing.assert_array_equal(current["player_smpl_valid"], mask)
        if int(mask.all(axis=0).sum()) != int(expected["all_placed_frames"]):
            raise ValueError(
                f"Published placement coverage changed: {new.source['clip_id']}"
            )
    actual = report_clip(new, players_per_side=2, new=True)
    if not actual["complete"]:
        raise ValueError(
            f"New scene is incomplete: {new.source['clip_id']} {actual['reason']}"
        )
    verify_saved_row(actual, expected)
    return {
        "scene": new.references["scene_assembly"],
        "scene_npz_sha256": new.document["exports"]["scene"]["files"]["scene"][
            "sha256"
        ],
        "gvhmr": new.references["gvhmr"]["artifact_id"],
        "placement": new.references["body_placement"]["artifact_id"],
        "legacy_scene": legacy.references["scene_assembly"]["artifact_id"],
        "ball": pins["ball"],
        "alignment": pins["alignment"],
    }


def verify_saved_row(actual: dict, expected: dict[str, str]) -> None:
    for field, value in actual.items():
        serialized = "" if value is None else str(value)
        if serialized != expected[field]:
            raise ValueError(
                f"Completion row {actual['clip_id']} {field} differs: {serialized}"
            )


def totals(rows: dict[str, dict[str, str]], paired: list[str]) -> dict:
    result = {
        field: sum(int(row[field]) for row in rows.values())
        for field in (
            "source_frames",
            "joint_player_frames",
            "all_gvhmr_frames",
            "all_placed_frames",
            "ball_3d_frames",
            "unused_view_points",
        )
    }
    result["clip_complete"] = sum(row["complete"] == "True" for row in rows.values())
    result["noncomplete"] = {
        k: row["reason"] for k, row in rows.items() if row["complete"] != "True"
    }
    result["joint_coverage"] = result["joint_player_frames"] / (
        4 * result["source_frames"]
    )
    for key in ("all_placed_frames", "all_gvhmr_frames", "ball_3d_frames"):
        result[key.replace("_frames", "_ratio")] = result[key] / result["source_frames"]
    result["paired_unused_view_points"] = sum(
        int(rows[clip]["unused_view_points"]) for clip in paired
    )
    result["paired_clip_median_of_unused_medians_px"] = median(
        float(rows[clip]["unused_view_median_px"]) for clip in paired
    )
    result["paired_clip_median_of_unused_p95_px"] = median(
        float(rows[clip]["unused_view_p95_px"]) for clip in paired
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i2363-run", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene, baseline = args.scenes, args.i2363_run
    tables = {
        route: read_table(scene / f"{route}-completion.csv")
        for route in ("legacy", "multiview")
    }
    replay = read_json(scene / "run.json")
    pilot = read_json(scene / "pilot-verification.json")["run"]["clips"]
    (progress_file,) = (baseline / "all/progress").glob("*/progress.json")
    progress = read_json(progress_file)
    (full_recording,) = scene.glob("recording-*meiji-multiview-scene-all.json")
    if replay["errors"] or set(replay["clips"]) != set(tables["multiview"]):
        raise ValueError("The batch did not produce the expected 69 records")
    manifest = {}
    for key, expected in sorted(tables["multiview"].items()):
        video, clip = key.split("/")
        old_path = (
            baseline
            / "inputs/dataset/videos"
            / video
            / "clips"
            / clip
            / "annotations/tennis_scene/scene.json"
        )
        legacy = StoreReader(old_path)
        new = StoreReader(scene / key / "scene.json")
        manifest[key] = verify_clip(legacy, new, expected)
        original = progress["clips"][key]
        reason = (
            str(original.get("reason") or original.get("error") or "")
            if original["status"] == "failed"
            else ""
        )
        old_row = report_clip(legacy, players_per_side=2, new=False, reason=reason)
        verify_saved_row(old_row, tables["legacy"][key])
        print(key, "verified", flush=True)
    both = [
        key
        for key in manifest
        if all(int(table[key]["unused_view_points"]) > 0 for table in tables.values())
    ]
    old_nodes = progress["nodes"]
    old_completed = [
        key for key in manifest if progress["clips"][key]["status"] == "generated"
    ]
    comparison = {route: totals(table, both) for route, table in tables.items()}
    comparison["legacy"]["initial_player_stops"] = sum(
        progress["clips"][key]["status"] == "failed" for key in manifest
    )
    comparison["multiview"]["initial_player_stops"] = len(replay["errors"])
    old_times = {
        stage: sum(
            old_nodes[key][stage]["seconds"]
            for key in old_completed
            if old_nodes[key][stage]["status"] == "executed"
        )
        for stage in BODY_STAGE
    }
    new_times = {
        stage: sum(
            (pilot[key] if key in pilot else replay["clips"][key])["nodes"][stage][
                "seconds"
            ]
            for key in old_completed
        )
        for stage in BODY_STAGE
    }
    for route, times in (("legacy", old_times), ("multiview", new_times)):
        comparison[route]["executed_common_node_seconds_on_legacy_completed_clips"] = (
            sum(times.values())
        )
        comparison[route]["common_node_seconds_by_stage"] = times
    comparison["legacy"]["profile"] = profile(progress_file.with_name("recording.json"))
    comparison["multiview"]["profile"] = profile(full_recording)
    comparison["multiview"]["pilot_wall_seconds"] = read_json(
        scene / "pilot-verification.json"
    )["total_wall_seconds"]
    comparison["common_timing_clips"] = len(old_completed)
    comparison["common_unused_view_clips"] = both
    for name, doc in (("meiji4.json", comparison), ("verified-scenes.json", manifest)):
        write_json_atomic(args.output / name, doc)


if __name__ == "__main__":
    main()
