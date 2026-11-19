"""Summarize C1 traces and compare saved benchmark artifacts on CPU."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from benchmark import CLIPS, RUN, SPEC, attempts, pinned_inputs, read, sha, write

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.generate_dataset.manifest import load_dataset_manifest
from src.tennis_scene.pipeline.dataset_publication import (
    ANNOTATION_RELATIVE_DIR,
    validate_dataset_result,
)
from src.tennis_scene.pipeline.storage.codec import verified_array_path
from src.tennis_scene.pipeline.storage.scene_index import (
    annotation_scene_path,
    read_component_descriptor,
)

TOLERANCE = read(RUN / "tolerances.json")["units"]
SCENE_UNITS = {
    "ball_uv": "normalized_image", "court_kp": "normalized_image", "human_kp_2d": "normalized_image",
    "ball_3d": "m", "player_position": "m", "gvhmr_aligned_player_position": "m", "player_kp_3d": "m",
    "player_canonical_pose": "m", "smpl_vertices_local": "m", "gvhmr_aligned_smpl_vertices_local": "m",
    "player_yaw": "rad", "gvhmr_aligned_player_yaw": "rad", "smpl_body_pose": "rad",
    "smpl_global_orient": "rad", "gvhmr_aligned_smpl_global_orient": "rad", "fps": "fps",
}


def as_seconds(event):
    return (event["end_ns"] - event["start_ns"]) * 1e-9


def batch_metrics(events):
    groups = defaultdict(Counter)
    for event in events:
        attrs = event["attributes"]
        if event["stage"] == "inference" and "batch_shape" in attrs and event["status"] == "ok":
            node = attrs.get("node", "unscoped").split("/")[0]
            groups[(attrs["model"], node)][attrs["batch_items"]] += 1
    rows = []
    for (model, stage), counts in sorted(groups.items()):
        cap = SPEC["native_batch_caps"].get(model)
        batches = sum(counts.values())
        items = sum(size * count for size, count in counts.items())
        rows.append({"model": model, "stage": stage, "batches": batches, "items": items, "items_histogram": dict(counts),
                     "configured_cap": cap, "fill_fraction": None if cap is None else items / (batches * cap)})
    return rows


def clip_evidence(dataset, record):
    store = (dataset / record.path / ANNOTATION_RELATIVE_DIR).resolve()
    index = read(store / "scene.json") if (store / "scene.json").is_file() else None
    report = {"clip_id": record.clip_id, "scene": None, "court_side": None, "camera_stages": {}}
    if index is not None:
        for camera in SPEC["camera_ids"]:
            observed = {}
            for stage in ("person_detection", "ball_detection", "ball_points"):
                node = f"{stage}/{camera}"
                if node not in index["artifacts"]:
                    continue
                payload = component_payload(store, index, node)
                if stage == "person_detection":
                    observed[stage] = int(np.count_nonzero(np.diff(payload["frame_offsets"])))
                else:
                    observed[stage] = int(np.count_nonzero(payload["observed"]))
            report["camera_stages"][camera] = observed
        if "court_side" in index["artifacts"]:
            side = component_payload(store, index, "court_side")
            turns = dict(zip(side["camera_ids"], side["view_half_turns"], strict=True))
            actual = [turns.get(camera) for camera in SPEC["camera_ids"]]
            report["court_side"] = {
                "view_half_turns": actual, "margin": side.get("margin"),
                "matches_ball_reference": actual == SPEC["expected_view_half_turns"],
            }
    marker_path = store / "annotation.json"
    if marker_path.is_file():
        archive = annotation_scene_path(store, read(marker_path))
        scene = load_scene_result(archive)
        validate_dataset_result(scene, record)
        report["scene"] = {"archive": str(archive), "reload": "validated", "players": int(scene.player_position.shape[0]),
                           "num_frames": scene.num_frames, "ball_valid_frames": int(np.count_nonzero(scene.ball_3d_valid))}
    failure = store.parent / "tennis_scene.failure.json"
    if failure.is_file():
        report["failure"] = read(failure)["error"]
    return report


def summarize(root, destination):
    terminal_path = root / "measurement" / "terminal.json"
    terminal = read(terminal_path) if terminal_path.is_file() else {}
    measurements = []
    for directory in attempts(root):
        path = directory / "measurement.json"
        if not path.is_file():
            continue
        measurement = read(path)
        recording_path = directory / measurement["recording"]
        trace = read(recording_path)
        resources = trace["resources"]
        if trace["status"] == "complete" and not resources.get("cuda"):
            raise ValueError(f"Completed recorder has no CUDA allocator peaks: {recording_path}")
        events = trace["events"]
        dataset = directory / "dataset"
        records = load_dataset_manifest(dataset).clips
        clips = [clip_evidence(dataset, records[key]) for key in CLIPS]
        finished = [records[clip["clip_id"]] for clip in clips if clip["scene"] is not None]
        wall = measurement["wall_seconds"] + measurement["worker_startup_seconds"]
        measurements.append({
            **measurement, "end_to_end_wall_seconds": wall, "recorder_wall_seconds": trace["total_wall_seconds"],
            "completed_clips_per_hour": len(finished) * 3600 / wall,
            "completed_camera_frames_per_second": sum(r.num_frames * r.num_cameras for r in finished) / wall,
            "recorder_status": trace["status"], "clips": clips, "all_generated": (
                len(finished) == len(CLIPS) and all(o["status"] == "generated" for o in measurement["outcomes"].values())
            ),
            "stage_wall_seconds": trace["stage_wall_seconds"], "models": trace["models"], "batches": batch_metrics(events),
            "resources": resources,
            "intermediate_array_save_seconds": sum(as_seconds(e) for e in events
                                                   if e["attributes"].get("operation") == "artifact_codec"),
            "scene_export_seconds": sum(as_seconds(e) for e in events
                                        if e["attributes"].get("operation") == "export_scene"),
            "dataset_marker_publication_seconds": sum(as_seconds(e) for e in events
                                                      if e["attributes"].get("operation") == "publish_annotation"),
            "recorder_sha256": sha(recording_path),
        })
    regimes = {}
    for regime in SPEC["regimes"]:
        values = [m["end_to_end_wall_seconds"] for m in measurements
                  if m["regime"] == regime and m["all_generated"] and m["recorder_status"] == "complete"]
        if values:
            regimes[regime] = {"n": len(values), "median_seconds": statistics.median(values),
                               "min_seconds": min(values), "max_seconds": max(values),
                               "sample_stdev_seconds": statistics.stdev(values) if len(values) > 1 else None}
    write(destination, {"terminal": terminal, "measurements": measurements, "complete_run_wall_by_regime": regimes})
    return bool(measurements)


def unit_for(node, path):
    field = path[-1]
    if field in SCENE_UNITS:
        return SCENE_UNITS[field]
    if field.endswith("_px") or field in {"boxes_xyxy", "boxes_xys", "intrinsic"}:
        return "px"
    if field.endswith("_m") or field in {"xyz", "position", "positions", "vertices_local", "transl", "translation"}:
        return "m"
    if field.endswith("_rad") or field in {"yaw", "body_pose", "global_orient"}:
        return "rad"
    if node == "ball_points" and field in {"smoothed", "coordinates"}:
        return "px"
    if node in {"court_detection", "court_calibration"} and field == "keypoints":
        return "normalized_image"
    if node in {"person_tracking", "player_selection"} and path == ("reconstruction", "boxes"):
        return "px"
    return "unitless"


def compare_array(left, right, unit):
    dtype = str(left.dtype)
    result = {"shape": list(left.shape), "dtype": dtype, "unit": unit}
    if left.shape != right.shape or left.dtype != right.dtype:
        return {**result, "shape_candidate": list(right.shape), "dtype_candidate": str(right.dtype),
                "exact": False, "close": False}
    floats = left.dtype.kind == "f"
    atol, rtol = TOLERANCE[unit][dtype] if floats else (0, 0)
    changed = violations = offset = 0
    max_abs = 0.0
    first_violations = []
    for a, b in np.nditer([left, right], flags=["external_loop", "buffered", "zerosize_ok"], buffersize=262144, order="C"):
        equal = a == b
        if floats:
            equal |= np.isnan(a) & np.isnan(b)
            finite = np.isfinite(a) & np.isfinite(b)
            delta = np.abs(b[finite].astype(np.float64) - a[finite].astype(np.float64))
            max_abs = max(max_abs, float(delta.max(initial=0)))
            outside = ~equal
            outside[finite] = delta > (atol + rtol * np.abs(a[finite].astype(np.float64)))
        else:
            outside = ~equal
        changed += int(np.count_nonzero(~equal))
        violations += int(np.count_nonzero(outside))
        if len(first_violations) < 8:
            first_violations.extend(int(i) + offset for i in np.flatnonzero(outside)[:8 - len(first_violations)])
        offset += a.size
    return {**result, "exact": changed == 0, "close": violations == 0, "changed": changed, "violations": violations,
            "max_abs_finite": max_abs if floats else None, "first_violation_flat_indexes": first_violations}


def compare_value(left, right, node, dataset_roots, path=()):
    if type(left) is not type(right):
        return {"exact": False, "close": False, "types": [type(left).__name__, type(right).__name__]}
    if isinstance(left, np.ndarray):
        pose_with_confidence = ((node == "person_pose_features" and path == ("keypoints",))
                                or (node in {"person_tracking", "player_selection"} and path == ("evidence", "poses")))
        if pose_with_confidence and left.shape == right.shape and left.ndim and left.shape[-1] == 3:
            return combine({
                "uv": compare_array(left[..., :2], right[..., :2], "px"),
                "confidence": compare_array(left[..., 2], right[..., 2], "unitless"),
            })
        if path in {("court_vis",), ("human_kp_vis",)}:
            return combine({"confidence": compare_array(left, right, "unitless"),
                            "present": compare_array(left > 0, right > 0, "unitless")})
        return compare_array(left, right, unit_for(node, path))
    if isinstance(left, dict):
        return combine({key: compare_value(left[key], right[key], node, dataset_roots, (*path, key))
                        for key in sorted(left.keys() & right.keys())},
                       missing=sorted(left.keys() - right.keys()), extra=sorted(right.keys() - left.keys()))
    if isinstance(left, list):
        return combine({str(index): compare_value(a, b, node, dataset_roots, (*path, str(index)))
                        for index, (a, b) in enumerate(zip(left, right, strict=False))}, lengths=[len(left), len(right)])
    if isinstance(left, str):
        left = left.removeprefix(str(dataset_roots[0]) + "/")
        right = right.removeprefix(str(dataset_roots[1]) + "/")
    if left == right:
        return {"exact": True, "close": True}
    close = False
    if isinstance(left, float):
        atol, rtol = TOLERANCE[unit_for(node, path)]["float64"]
        close = math.isfinite(left) and math.isfinite(right) and abs(right - left) <= atol + rtol * abs(left)
    return {"exact": False, "close": close, "reference": left, "candidate": right}


def combine(children, *, missing=(), extra=(), lengths=None):
    keys_match = not missing and not extra and (lengths is None or lengths[0] == lengths[1])
    result = {"exact": keys_match and all(c["exact"] for c in children.values()),
              "close": keys_match and all(c["close"] for c in children.values()),
              "changes": {key: child for key, child in children.items() if not child["exact"]}}
    if not keys_match:
        result.update(missing=missing, extra=extra, lengths=lengths)
    return result


def component_payload(store, index, node):
    ref = index["artifacts"][node]
    descriptor = read_component_descriptor(store, ref, node=node, source_sha256=index["source_sha256"])
    directory = (store / ref["path"]).parent
    arrays = descriptor["arrays"]

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"array"}:
                name = value["array"]
                path = verified_array_path(directory, name, arrays)
                array = np.load(path, allow_pickle=False, mmap_mode="r")
                if list(array.shape) != arrays[name]["shape"] or str(array.dtype) != arrays[name]["dtype"]:
                    raise ValueError(f"Artifact schema mismatch: {path}")
                return array
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    return decode(descriptor["payload"])


def compare_clip(reference, candidate, clip_id):
    roots = [attempt / "dataset" for attempt in (reference, candidate)]
    records = [load_dataset_manifest(root).clips[clip_id] for root in roots]
    stores = [(root / record.path / ANNOTATION_RELATIVE_DIR).resolve() for root, record in zip(roots, records, strict=True)]
    indexes = [read(store / "scene.json") if (store / "scene.json").is_file() else {"artifacts": {}} for store in stores]
    measurements = [read(attempt / "measurement.json") for attempt in (reference, candidate)]
    outcomes = [{key: data["outcomes"][clip_id].get(key) for key in ("status", "reason")} for data in measurements]
    evidence = [clip_evidence(root, record) for root, record in zip(roots, records, strict=True)]
    node_sets = [index["artifacts"].keys() for index in indexes]
    common = sorted(node_sets[0] & node_sets[1])
    checked = {}
    upstream = {}
    for node in common:
        refs = [index["artifacts"][node] for index in indexes]
        schemas = [(ref["schema"], ref["version"]) for ref in refs]
        versions_match = schemas[0] == schemas[1]
        value = compare_value(*(component_payload(store, index, node) for store, index in zip(stores, indexes, strict=True)),
                              node.split("/")[0], roots)
        checked[node] = {**value, "schemas": schemas, "exact": versions_match and value["exact"],
                         "close": versions_match and value["close"]}
        descriptor = read(stores[0] / refs[0]["path"])
        dependencies = {ref["artifact_id"] for ref in descriptor["dependencies"].values()}
        upstream[node] = {key for key, ref in indexes[0]["artifacts"].items() if ref["artifact_id"] in dependencies}

    scenes = [row["scene"] for row in evidence]
    archived = all(scene is not None for scene in scenes)
    archive_diff = {"exact": False, "close": False}
    if archived:
        with np.load(scenes[0]["archive"], allow_pickle=False) as a, np.load(scenes[1]["archive"], allow_pickle=False) as b:
            archive_diff = combine({key: compare_array(a[key], b[key], unit_for("scene_assembly", (key,)))
                                    for key in sorted(set(a.files) & set(b.files))},
                                   missing=sorted(set(a.files) - set(b.files)), extra=sorted(set(b.files) - set(a.files)))
    changed = {node for node, value in checked.items() if not value["exact"]}
    absent = [sorted(node_sets[0] - node_sets[1]), sorted(node_sets[1] - node_sets[0])]
    status_matches = outcomes[0] == outcomes[1]
    sides_match = all(row["court_side"] and row["court_side"]["matches_ball_reference"] for row in evidence)
    return {
        "clip_id": clip_id, "outcomes": outcomes, "outcomes_match": status_matches, "evidence": evidence,
        "missing_candidate_nodes": absent[0], "extra_candidate_nodes": absent[1], "compared_nodes": len(common),
        "first_changed_nodes": sorted(n for n in changed if not upstream[n] & changed),
        "changed_nodes": {node: checked[node] for node in sorted(changed)}, "archive_arrays": archive_diff,
        "quality_agrees": (bool(common) and status_matches and not any(absent)
                           and all(v["close"] for v in checked.values()) and (not archived or archive_diff["close"])
                           and (all(scene is None for scene in scenes) or archived)),
        "both_scenes_validated": archived, "player_side_matches_reference": sides_match,
    }


def compare(reference, candidate, destination):
    data = pinned_inputs()
    reports = [compare_clip(reference, candidate, clip) for clip in CLIPS]
    pins = [read(attempt.parents[1] / "preflight.json") for attempt in (reference, candidate)]
    measurements = [read(attempt / "measurement.json") for attempt in (reference, candidate)]
    same_runtime = all(measurements[0][name] == measurements[1][name] for name in ("torch", "cuda", "threads"))
    fields = ("settings", "checkpoints", "models", "video_court_cameras", "video_court_drift_cameras")
    same_publication_pins = all(pins[0]["publication_identity"].get(k) == pins[1]["publication_identity"].get(k)
                                for k in fields)
    same_input_pins = all(pins[0][name] == pins[1][name] for name in ("input_sha256", "tolerances_sha256"))
    accepted = same_runtime and same_publication_pins and same_input_pins and all(
        r["quality_agrees"] and r["player_side_matches_reference"] for r in reports
    ) and any(r["both_scenes_validated"] for r in reports)
    write(destination, {"clips": reports, "quality_acceptance": accepted, "same_input_pins": same_input_pins,
                        "same_publication_pins": same_publication_pins, "same_runtime": same_runtime,
                        "same_commit": pins[0]["commit"] == pins[1]["commit"],
                        "selected_camera_frames": sum(r["num_cameras"] * r["num_frames"] for r in data["selected_records"])})
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--root", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True, help="A serial attempts/<repeat> directory")
    comparison.add_argument("--candidate", type=Path, required=True, help="A candidate attempts/<repeat> directory")
    for command in (summary, comparison):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        return 0 if summarize(args.root.resolve(), args.output) else 1
    return 0 if compare(args.reference.resolve(), args.candidate.resolve(), args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
