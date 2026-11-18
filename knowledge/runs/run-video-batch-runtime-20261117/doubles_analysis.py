"""CPU-only summaries and a one-clip comparison for the doubles qualification."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
C3 = REPO / "knowledge/runs/run-i1751-runtime-gpu-a5"
sys.path.insert(0, str(C3))
from health_guard import write  # noqa: E402

RUNTIME_MODELS = ("ball", "dino", "vitpose", "appearance", "hmr2", "gvhmr", "body_geometry")
PER_CLIP = {"court_calibration"}


def read(path):
    return json.loads(path.read_text())


def lifetimes(events):
    loaded = Counter(e["model"] for e in events if e["action"] == "load" and e["status"] == "complete")
    unloaded = Counter(e["model"] for e in events if e["action"] == "unload" and e["status"] == "complete")
    active, maximum = set(), 0
    for event in events:
        if event["status"] != "complete":
            continue
        if event["action"] == "load":
            active.add(event["instance"])
            maximum = max(maximum, len(active))
        else:
            active.discard(event["instance"])
    return {"loads": dict(loaded), "unloads": dict(unloaded), "unclosed": sorted(active),
            "maximum_owned": maximum, "failed_events": [e for e in events if e["status"] != "complete"],
            "each_target_loaded_once": all(loaded[name] == unloaded[name] == 1 for name in RUNTIME_MODELS)
                                       and not active and maximum == 1
                                       and all(e["status"] == "complete" for e in events)}


def identities(diagnostics):
    selection = diagnostics["selection"]
    by_id = {item["identity"]: item for item in diagnostics["identities"]}
    return {side: {"count": len(selection[side]["players"]),
                   "identities": [by_id[value] for value in selection[side]["players"]],
                   "runner_up": selection[side]["runner_up"]} for side in ("-1", "1")}


def clip_summary(dataset, clip_id):
    from src.tennis_scene.archive import load_scene_result
    from src.tennis_scene.pipeline.storage.scene_index import annotation_scene_path
    from src.tennis_scene.schema import SCENE_MASK_FIELDS

    video, clip = clip_id.split("/")
    store = dataset / "videos" / video / "clips" / clip / "annotations/tennis_scene"
    marker = store / "annotation.json"
    result = {"clip_id": clip_id, "marker_exists": marker.exists()}
    if not marker.exists():
        return result
    scene_path = annotation_scene_path(store, read(marker))
    scene = load_scene_result(scene_path)
    index = read(store / "scene.json")
    association = read(store / index["artifacts"]["player_association"]["path"])["payload"]["diagnostics"]["association"]
    result.update(scene_status=scene.metadata["status"], num_frames=scene.num_frames,
                  scene_players=int(scene.player_position.shape[0]), sides=identities(association),
                  validity_statistics=scene.metadata["validity_statistics"], scene_path=str(scene_path),
                  body_exports_present=scene.smpl_vertices_local is not None and bool(scene.player_smpl_valid.any()),
                  mask_counts={name: {"valid": int(getattr(scene, name).sum()), "total": int(getattr(scene, name).size)}
                               for name in SCENE_MASK_FIELDS if getattr(scene, name) is not None})
    return result


def stage_estimates(nodes, events, sample_frames, sample_clips, full_frames, full_clips):
    totals, lifecycle = defaultdict(float), defaultdict(float)
    for clip_nodes in nodes.values():
        for name, record in clip_nodes.items():
            if record["seconds"] is not None:
                totals[name.split("/")[0]] += record["seconds"]
    for event in events:
        name = event["attributes"].get("node", "").split("/")[0]
        if event["stage"] in ("model_load", "model_unload") and name in totals:
            lifecycle[name] += (event["end_ns"] - event["start_ns"]) / 1e9
    estimates = {}
    for name, seconds in totals.items():
        fixed = lifecycle[name]
        ratio = full_clips / sample_clips if name in PER_CLIP else full_frames / sample_frames
        estimates[name] = {"measured_node_seconds": seconds, "included_lifecycle_seconds": fixed,
                           "scaling": "clip_count" if name in PER_CLIP else "source_frames",
                           "estimated_69clip_seconds": fixed + max(0, seconds - fixed) * ratio}
    return estimates


def overhead_estimates(recording, estimates, sample_clips, full_clips):
    # Outermost attributed export/publication intervals are disjoint from nodes.
    events = recording["events"]
    spans = []
    for name in ("scene_export", "clip_publication"):
        rows = [e for e in events if e["attributes"].get("node") == name]
        outer = [e for e in rows if not any(
            p["start_ns"] <= e["start_ns"] and p["end_ns"] >= e["end_ns"] and p["depth"] < e["depth"]
            for p in rows)]
        seconds = sum((e["end_ns"] - e["start_ns"]) / 1e9 for e in outer)
        spans.append({"stage": name, "measured_seconds": seconds,
                      "estimated_69clip_seconds": seconds * full_clips / sample_clips})
    node_seconds = sum(row["measured_node_seconds"] for row in estimates.values())
    remaining = recording["total_wall_seconds"] - node_seconds - sum(row["measured_seconds"] for row in spans)
    if remaining < -0.1:
        raise ValueError("Overlapping duration attribution")
    # Includes shared load/identity and plan/progress IO; show a range because
    # some of this remaining work is video-fixed and some scales by clips.
    remaining = max(0, remaining)
    return {"stages": spans, "remaining_seconds": remaining,
            "estimated_69clip_remaining_seconds_range": [remaining, remaining * full_clips / sample_clips]}


def compare_reference(root, clip_id, terminal):
    from assess_parent_acceptance import assess
    from compare_completed import compare

    from src.tennis_scene.pipeline.storage.scene_index import annotation_scene_path

    analysis = root / "comparison"
    video, clip = clip_id.split("/")
    for name, phase in (("main", "single"), ("runtime", "batch")):
        directory = analysis / name
        directory.mkdir(parents=True, exist_ok=True)
        store = root / "datasets" / phase / "videos" / video / "clips" / clip / "annotations/tennis_scene"
        for target, source in ((directory / "store", store),
                               (directory / "scene.npz", annotation_scene_path(store, read(store / "annotation.json")))):
            if not target.exists():
                target.symlink_to(source)
        metadata = directory / "scene.metadata.json"
        if not metadata.exists():
            metadata.symlink_to((directory / "scene.npz").resolve().with_suffix(".metadata.json"))
        write(directory / "measurement.json", read(root / phase / "entry.json"))
        write(analysis / "health" / f"{name}.json", terminal["outcomes"][phase])
    comparison = compare(analysis)
    comparison["entry_bindings"] = {"main": "single CLI on the same runtime", "runtime": "reference clip in the three-clip batch CLI"}
    write(analysis / "comparison-completed.json", comparison)
    write(analysis / "comparison.json", comparison)
    life = lifetimes(read(root / "batch/entry.json")["runtime_events"])
    write(analysis / "acceptance-summary.json", {"runtime_unclosed_instances": life["unclosed"],
                                               "runtime_failed_events": life["failed_events"]})
    write(analysis / "terminal.json", {"status": "complete" if comparison["numerical_acceptance"] else "failed", "comparison_returncode": 0 if comparison["numerical_acceptance"] else 2,
                                      "outcomes": {"main": terminal["outcomes"]["single"], "runtime": terminal["outcomes"]["batch"]}})
    manifest = read(root / "input" / "videos" / video / "clips" / clip / "clip.json")
    write(analysis / "inputs.json", {"settings": {"ball_detection": {"precision": "bf16"}},
                                    "source": {"videos": [{"camera_id": c} for c in manifest["camera_ids"]]}})
    policy = read(C3 / "parent-acceptance-policy.json")
    reference = assess(analysis, policy)
    reference["scope"] = "Reference only: the policy was explicitly adopted for C3; strict C4 result is retained"
    write(analysis / "c3-policy-reference.json", reference)
    return {"strict": comparison["numerical_acceptance"], "all_arrays_exact": comparison["all_arrays_exact"],
            "all_discrete_exact": reference["checks"]["all_discrete_observed_masks_exact"],
            "c3_policy_reference": reference["parent_acceptance"],
            "ball_detection_error_px": reference["ball_detection_error_px"],
            "ball_final_position_error_m": reference["ball_final_position_error_m"]}


def report(root):
    from doubles_acceptance import CLIPS

    terminal = read(root / "terminal.json")
    result = {"clips": [], "all_three_completed": False, "doubles_identity_counts_ok": False,
              "body_exports_present": False, "comparison_strict": False,
              "runtime_lifetimes_ok": False,
              "cuda_anomaly": (root / "anomaly.json").exists()}
    result["clips"] = [clip_summary(root / "datasets/batch", clip) for clip in CLIPS]
    result["all_three_completed"] = all(clip["marker_exists"] for clip in result["clips"])
    result["doubles_identity_counts_ok"] = result["all_three_completed"] and all(
        clip["scene_players"] == 4 and all(side["count"] == 2 for side in clip["sides"].values()) for clip in result["clips"])
    result["body_exports_present"] = result["all_three_completed"] and all(c["body_exports_present"] for c in result["clips"])
    entry_path = root / "batch/entry.json"
    if entry_path.exists():
        entry = read(entry_path)
        result["runtime_lifetimes"] = lifetimes(entry["runtime_events"])
        result["runtime_lifetimes_ok"] = result["runtime_lifetimes"]["each_target_loaded_once"]
        if entry.get("recording"):
            recording = read(Path(entry["recording"]))
            progress = read(Path(entry["progress"]))
            result.update(resources=recording["resources"], model_recording=recording["models"],
                          total_recorded_wall_seconds=recording["total_wall_seconds"], entry_wall_seconds=entry["entry_wall_seconds"],
                          nodes=progress["nodes"], outcomes=progress["clips"], stage_wall_seconds=recording["stage_wall_seconds"])
            dataset = read(root / "source-dataset.json")
            records = [v for v in dataset["clips"] if v["video_id"] == "video_000"]
            full_frames = sum(v["num_frames"] for v in records)
            result["full_video"] = {"clips": len(records), "frames_per_camera": full_frames}
            estimates = stage_estimates(progress["nodes"], recording["events"], 883, 3, full_frames, len(records))
            result["stage_estimates"] = estimates
            result["estimated_69clip_node_hours"] = sum(row["estimated_69clip_seconds"] for row in estimates.values()) / 3600
            result["overhead_estimates"] = overhead_estimates(recording, estimates, 3, len(records))
            overhead = result["overhead_estimates"]
            base = sum(row["estimated_69clip_seconds"] for row in estimates.values()) + sum(row["estimated_69clip_seconds"] for row in overhead["stages"])
            result["estimated_69clip_wall_hours_range"] = [(base + value) / 3600 for value in overhead["estimated_69clip_remaining_seconds_range"]]
            result["estimate_complete_sample"] = result["all_three_completed"]
            result["estimate_limits"] = "Lifecycle is fixed once; calibration/export/publication scale by clips, other nodes by source frames. Remaining recorded overhead has fixed-to-clip-linear bounds. Includes first compile and clip/person/disk differences. Entry import/preflight outside the recorder is additional. No speedup ratio against the one-clip reference."
    reference = terminal.get("reference_clip")
    if reference and not result["cuda_anomaly"] and terminal["outcomes"].get("single", {}).get("status") == "complete":
        result["comparison"] = compare_reference(root, reference, terminal)
        result["comparison_strict"] = result["comparison"]["strict"]
    write(root / "summary.json", result)
    return {key: value for key, value in result.items() if key in (
        "all_three_completed", "doubles_identity_counts_ok", "body_exports_present", "runtime_lifetimes_ok", "comparison_strict", "comparison", "cuda_anomaly")}
