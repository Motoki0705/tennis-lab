"""Recompute the paper's timing and output-parity tables from the saved run."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import yaml
from parity import compare_conditions, load_scene
from run import BUNDLE, CASES, CLIPS, CONDITION, MODELS, read_json

from src.utils.checksum import dual_sha256

MANAGED = ("dino", "vitpose", "appearance", "hmr2", "gvhmr", "body_geometry")
MODEL_STAGES = {
    "dino": ("person_detection", "detector", "checkpoint"),
    "vitpose": ("person_pose_features", "pose", "checkpoint"),
    "clip": ("person_appearance_features", "encoder", "checkpoint"),
    "hmr2": ("hmr2_features", "features", "checkpoint"),
    "gvhmr": ("gvhmr", "mesh", "gvhmr"),
}
FORWARD_LABELS = {
    "dino": "DINO",
    "vitpose": "ViTPose",
    "clip": "CLIP-ReID",
    "hmr2": "HMR2",
    "gvhmr": "GVHMR-network",
}
MIB = 1024**2
GIB = 1024**3


def recorder_reader() -> Any:
    script = BUNDLE.parent / "run-i2363-trt-player-video/analyze.py"
    spec = importlib.util.spec_from_file_location("prior_video_analysis", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the recorder reader from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.recording_header


def appearance_wall_seconds(path: Path, offset: int) -> float:
    """Union the CLIP-ReID processing and runtime lifecycle labels."""
    prefix = b'{"id":'
    labels = (b'"model":"appearance"', b'"model":"clip"', b'"model":"CLIP-ReID"')
    intervals = []

    def collect(fragment: bytes) -> None:
        if any(label in fragment for label in labels):
            event = json.loads(prefix + fragment)
            if event["attributes"].get("model") in {"appearance", "clip", "CLIP-ReID"}:
                intervals.append((event["start_ns"], event["end_ns"]))

    with path.open("rb") as stream:
        stream.seek(offset)
        if stream.read(len(prefix)) != prefix:
            raise ValueError(f"Expected recorder events at {path}")
        tail = b""
        while block := stream.read(4 * MIB):
            chunks = (tail + block).split(b"}," + prefix)
            tail = chunks.pop()
            for fragment in chunks:
                collect(fragment + b"}")
        collect(
            tail[:-3]
        )  # remove the closing event-array bracket, object brace and newline
    total = end = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(end, start))
        end = max(end, stop)
    return total / 1e9


def verify_progress(
    progress: dict[str, Any],
    process: dict[str, Any],
    case: str,
    measurement: dict[str, Any],
) -> None:
    clips = process["clips"]
    if (progress["dataset_id"], progress["video_id"], progress["status"]) != (
        CONDITION["dataset_id"],
        CONDITION["video_id"],
        "complete",
    ):
        raise ValueError("Incomplete progress or wrong input scope")
    if (
        progress["passes"] != [{"index": 0, "clips": clips}]
        or progress["batching"]["enabled"] != (case in CASES[2:])
        or progress["batching"]["store_cache_bytes"] != 0
        or progress["overwrite"]
        or process["exit_code"] != 0
    ):
        raise ValueError(f"Wrong execution conditions for {process['directory']}")
    outcomes = {clip: outcome["status"] for clip, outcome in progress["clips"].items()}
    if outcomes != process["statuses"] or outcomes != dict.fromkeys(clips, "generated"):
        raise ValueError("Incomplete publication of the fixed clips")
    if not all(
        progress["nodes"][clip]
        and all(
            node["status"] == "executed" for node in progress["nodes"][clip].values()
        )
        for clip in clips
    ):
        raise ValueError("A component was not executed from an empty store")
    if [target["record"]["clip_id"] for target in progress["targets"]] != clips:
        raise ValueError("The measured input order differs from the campaign")
    for target in progress["targets"]:
        record = target["record"]
        pinned = CONDITION["clips"][record["clip_id"]]
        if (
            record["num_frames"],
            record["num_cameras"],
            target["manifest_sha256"],
            target["media_sha256"],
        ) != (
            pinned["frames"],
            4,
            pinned["manifest_sha256"],
            pinned["media_sha256"],
        ):
            raise ValueError(f"Measured inputs differ for {record['clip_id']}")
    court = progress["video_court"]
    if (court["artifact_digest"], court["drift_digest"]) != (
        CONDITION["court_digest"],
        CONDITION["court_drift_digest"],
    ):
        raise ValueError("Measured court or drift differs from the pinned artifact")
    pipeline = yaml.safe_load(progress["pipeline_config_yaml"])
    if (
        pipeline["court_side"]["mode"] != "clip"
        or pipeline["court_side"]["evidence"] != "player"
        or pipeline["player_association"]["players_per_side"] != 2
        or pipeline["ball_detection"]["enabled"]
        or pipeline["ball_reconstruction"]["enabled"]
        or not pipeline["people_models"]["runtime"]["vitpose"]["flip_test"]
        or any(
            pipeline["people_models"]["runtime"][key]["batch_size"]
            != CONDITION["batches"][name]
            for name, key in MODELS.items()
        )
        or any(
            progress["batching"]["sizes"][name] != CONDITION["batches"][name]
            for name in MODELS
            if name != "gvhmr"
        )
    ):
        raise ValueError("Measured pipeline settings differ from the campaign")
    models = progress["publication_identity"]["models"]
    engines = measurement["engine_preparation"]["engines"]["models"]
    shim = dual_sha256(BUNDLE / "pipeline_cli.py")
    for name, (stage, alias, checkpoint) in MODEL_STAGES.items():
        spec = models[stage][alias]
        assets = spec["assets"]
        if (
            spec["dtype"] != "float16"
            or assets[checkpoint]["sha256"]
            != engines[name]["export"]["sources"]["checkpoint"]
        ):
            raise ValueError(f"Wrong inference dtype or checkpoint for {name}")
        if case == CASES[3]:
            engine = engines[name]
            expected = {
                "engine_0/engine.json": engine["engine_manifest_sha256"],
                "engine_0/export.json": engine["engine"]["export_sha256"],
                "engine_0/engine.plan": engine["engine"]["plan_sha256"],
            }
            if spec["settings"]["backend"] != "tensorrt" or any(
                assets[file]["sha256"] != digest for file, digest in expected.items()
            ):
                raise ValueError(
                    f"Prepared TensorRT assets differ from the scene's {name} assets"
                )
        elif any(key.startswith("engine_") for key in assets):
            raise ValueError(f"PyTorch condition bound a TensorRT engine for {name}")
        elif name in {"clip", "hmr2", "gvhmr"} and (
            spec["settings"]["compute_precision"],
            spec["settings"]["experiment_sha256"],
        ) != ("cuda_autocast_fp16", shim):
            raise ValueError(f"Wrong PyTorch FP16 adapter for {name}")
        elif (
            name == "dino"
            and assets["extension"]["sha256"] != measurement["dino_extension_sha256"]
        ):
            raise ValueError("DINO CUDA extension differs from the recorded binary")


def forward_batches(progress: dict[str, Any]) -> dict[str, dict[str, int | float]]:
    results = {}
    for model, label in FORWARD_LABELS.items():
        rows = progress["batch_shapes"][label]
        count = items = 0
        divisor = 2 if model == "vitpose" else 1
        batch_limit = CONDITION["batches"][model]
        for shape, row in rows.items():
            rows_in_tensor = json.loads(shape)[0]
            batch, remainder = divmod(rows_in_tensor, divisor)
            if (
                row["failed"]
                or row["items"] != batch
                or remainder
                or not 1 <= batch <= batch_limit
            ):
                raise ValueError(f"Invalid {model} forward batch: {shape}")
            count += row["succeeded"]
            items += batch * row["succeeded"]
        results[model] = batch_counts(count, items, count * batch_limit)
    return results


def batch_counts(batches: int, items: int, capacity: int) -> dict[str, int | float]:
    if not 0 < batches <= items <= capacity:
        raise ValueError("Inconsistent inference batch counts")
    return {
        "batches": batches,
        "items": items,
        "total_capacity": capacity,
        "fill_fraction": items / capacity,
    }


def aggregate_batches(passes: list[dict[str, Any]], key: str) -> dict[str, Any]:
    aggregate = {}
    fields = ("batches", "items", "total_capacity")
    for name in sorted({model for process in passes for model in process[key]}):
        counts = {
            field: sum(
                process[key][name][field] for process in passes if name in process[key]
            )
            for field in fields
        }
        aggregate[name] = batch_counts(
            counts["batches"], counts["items"], counts["total_capacity"]
        )
        if key == "batch_dispatches":
            aggregate[name]["cross_clip_batches"] = sum(
                row[key][name]["cross_clip_batches"]
                for row in passes
                if name in row[key]
            )
    return aggregate


def timing_row(
    run: Path,
    process: dict[str, Any],
    progress: dict[str, Any],
    header: Any,
    case: str,
    measurement: dict[str, Any],
) -> dict[str, Any]:
    trace = run / process["recording"]
    record, samples, offset = header(trace)
    wall, stages = record["total_wall_seconds"], record["stage_wall_seconds"]
    timing = record["timing"]
    if (
        timing["gpu"] != "host_wall"
        or timing["sample_interval_seconds"] != CONDITION["sampling_interval_seconds"]
        or not math.isclose(sum(stages.values()), wall, rel_tol=1e-6)
        or not 0 < wall <= process["wall_seconds"]
        or record["metadata"]["batching"] != progress["batching"]
    ):
        raise ValueError(f"Unexpected recorder timing contract at {trace}")
    lifecycle = {}
    engines = measurement["engine_preparation"]["engines"]["models"]
    for row in record["models"]:
        name = row["model"]
        if row["failed_count"] or name in lifecycle:
            raise ValueError(f"Unexpected model lifecycle event for {name} at {trace}")
        if name in MANAGED and name != "body_geometry":
            model = "clip" if name == "appearance" else name
            if row["dtype"] != "float16":
                raise ValueError(f"Recorded dtype is not FP16 for {name} at {trace}")
            if (
                case == CASES[3]
                and row["weights"]["engine_0/engine.plan"]["sha256"]
                != engines[model]["engine"]["plan_sha256"]
            ):
                raise ValueError(
                    f"The loaded TensorRT plan does not match preparation for {name}"
                )
        lifecycle[name] = {
            key: row[key]
            for key in (
                "dtype",
                "load_count",
                "unload_count",
                "load_seconds",
                "unload_seconds",
            )
        }
    if any(
        lifecycle[name]["load_count"] != 1 or lifecycle[name]["unload_count"] != 1
        for name in MANAGED
    ):
        raise ValueError(
            f"A managed model was not loaded and released once in {process['directory']}"
        )
    dispatches = record["batch_dispatches"]
    expected_dispatches = set(MODEL_STAGES) - {"gvhmr"} if case in CASES[2:] else set()
    if set(dispatches) != expected_dispatches or any(
        row["failed"] for row in dispatches.values()
    ):
        raise ValueError(f"Unexpected adapter dispatch summary at {trace}")
    by_model = {
        name: row["wall_seconds"] for name, row in record["timings"]["by_model"].items()
    }
    by_model["clip_reid_union"] = appearance_wall_seconds(trace, offset)
    return {
        "directory": process["directory"],
        "clips": process["clips"],
        "cli_exit_code": process["exit_code"],
        "cli_wall_seconds": process["wall_seconds"],
        "recorder_wall_seconds": wall,
        "progress": process["progress"],
        "progress_sha256": dual_sha256(run / process["progress"]),
        "recording": process["recording"],
        "recording_sha256": dual_sha256(trace),
        "sample_stats": samples,
        "resources": record["resources"],
        "video_pass": record["timings"]["by_stage"]["video_pass"],
        "stage_wall_seconds": stages,
        "model_wall_seconds": by_model,
        "lifecycle": lifecycle,
        "batch_dispatches": dispatches,
        "forward_batches": forward_batches(progress),
    }


def aggregate_timing(passes: list[dict[str, Any]]) -> dict[str, Any]:
    cli = sum(row["cli_wall_seconds"] for row in passes)
    recorder = sum(row["recorder_wall_seconds"] for row in passes)
    gpu_time = sum(row["video_pass"]["gpu_observed_wall_seconds"] for row in passes)
    peak_vram = max(
        row["resources"]["nvml"]["cuda:0"]["used_peak_bytes"] for row in passes
    )
    peak_rss = max(
        row["resources"]["ram_process_lifetime_peak_bytes"] for row in passes
    )
    lifecycle = {}
    for name in sorted({model for row in passes for model in row["lifecycle"]}):
        lifecycle[name] = {
            field: sum(
                row["lifecycle"][name][field]
                for row in passes
                if name in row["lifecycle"]
            )
            for field in (
                "load_count",
                "unload_count",
                "load_seconds",
                "unload_seconds",
            )
        }
    batches = aggregate_batches(passes, "batch_dispatches")
    forwards = aggregate_batches(passes, "forward_batches")
    return {
        "processes": passes,
        "cli_wall_seconds": cli,
        "recorder_wall_seconds": recorder,
        "outside_recorder_wall_seconds": cli - recorder,
        "gpu_observed_wall_seconds": gpu_time,
        "video_pass_wall_seconds": sum(
            row["video_pass"]["wall_seconds"] for row in passes
        ),
        **{
            key: sum(
                row["video_pass"][key] * row["video_pass"]["gpu_observed_wall_seconds"]
                for row in passes
            )
            / gpu_time
            for key in ("gpu_active_fraction", "gpu_mean_utilization_percent")
        },
        "peak_vram_mib": peak_vram / MIB,
        "peak_vram_gib": peak_vram / GIB,
        "peak_rss_mib": peak_rss / MIB,
        "peak_rss_gib": peak_rss / GIB,
        "models": lifecycle,
        "managed_model_loads": sum(lifecycle[name]["load_count"] for name in MANAGED),
        "managed_model_unloads": sum(
            lifecycle[name]["unload_count"] for name in MANAGED
        ),
        "batch_dispatches": batches,
        "forward_batches": forwards,
        **{
            key: {
                name: sum(row[key].get(name, 0) for row in passes)
                for name in sorted({name for row in passes for name in row[key]})
            }
            for key in ("stage_wall_seconds", "model_wall_seconds")
        },
    }


def reports(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    captured = read_json(BUNDLE / "run.json")
    measurement = read_json(run / "outcome.json")
    if (
        captured["commit"] != measurement["head"]
        or captured["resource"] != "all"
        or captured["run_id"] != measurement["queue_job"]
        or set(measurement["cases"]) != set(CASES)
        or dual_sha256(BUNDLE / "condition.json") != measurement["condition_sha256"]
    ):
        raise ValueError("Queue recording and frozen run identity disagree")
    header = recorder_reader()
    timings: dict[str, Any] = {}
    scenes = {}
    code = set()
    directories = set()
    previous = None
    for case in CASES:
        passes = measurement["cases"][case]
        selected = [[clip] for clip in CLIPS] if case == CASES[0] else [list(CLIPS)]
        if [row["clips"] for row in passes] != selected:
            raise ValueError(f"Wrong per-process clip selection for {case}")
        rows = []
        scenes[case] = {}
        for process in passes:
            if process["directory"] in directories:
                raise ValueError("Repeated component store in the comparison")
            directories.add(process["directory"])
            progress = read_json(run / process["progress"])
            verify_progress(progress, process, case, measurement)
            identity = progress["publication_identity"]
            code.add(identity["code_sha256"])
            rows.append(timing_row(run, process, progress, header, case, measurement))
            for target in progress["targets"]:
                scenes[case][target["record"]["clip_id"]] = load_scene(
                    run / process["directory"] / "dataset", target, identity
                )
        current = aggregate_timing(rows)
        current["speedup_vs_c1"] = (timings[CASES[0]] if previous else current)[
            "cli_wall_seconds"
        ] / current["cli_wall_seconds"]
        current["change_vs_previous"] = (
            None
            if previous is None
            else {
                "saved_seconds": previous["cli_wall_seconds"]
                - current["cli_wall_seconds"],
                "speedup": previous["cli_wall_seconds"] / current["cli_wall_seconds"],
            }
        )
        timings[case] = previous = current
    if len(code) != 1:
        raise ValueError("The pipeline code differs between the measured processes")
    metrics = {
        "measurement_head": measurement["head"],
        "source_sha256": code.pop(),
        "condition_sha256": measurement["condition_sha256"],
        "dino_extension_sha256": measurement["dino_extension_sha256"],
        "queue_job": measurement["queue_job"] + ".job",
        "engine_preparation": measurement["engine_preparation"],
        "conditions": timings,
    }
    return metrics, compare_conditions(scenes)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in CASES:
        timing = metrics["conditions"][case]
        previous = timing["change_vs_previous"] or {}
        row = {
            "condition": case,
            **{
                key: timing[key]
                for key in (
                    "cli_wall_seconds",
                    "recorder_wall_seconds",
                    "speedup_vs_c1",
                    "peak_vram_mib",
                    "peak_vram_gib",
                    "peak_rss_mib",
                    "peak_rss_gib",
                    "gpu_active_fraction",
                    "gpu_mean_utilization_percent",
                    "managed_model_loads",
                    "managed_model_unloads",
                )
            },
            "saved_seconds_vs_previous": previous.get("saved_seconds"),
            "speedup_vs_previous": previous.get("speedup"),
        }
        if case == CASES[0]:
            for process in timing["processes"]:
                row[f"{process['clips'][0].split('/')[-1]}_cli_wall_seconds"] = process[
                    "cli_wall_seconds"
                ]
        for category in ("stage_wall_seconds", "model_wall_seconds"):
            for name, seconds in timing[category].items():
                row[f"{category}_{name}"] = seconds
        for name, lifecycle in timing["models"].items():
            for field in ("load_count", "unload_count"):
                row[f"model_{name}_{field}"] = lifecycle[field]
        for name in FORWARD_LABELS:
            row[f"{name}_forward_fill_fraction"] = timing["forward_batches"][name][
                "fill_fraction"
            ]
            dispatch = timing["batch_dispatches"].get(name)
            row[f"{name}_adapter_fill_fraction"] = (
                None if dispatch is None else dispatch["fill_fraction"]
            )
            row[f"{name}_cross_clip_batches"] = (
                0 if dispatch is None else dispatch["cross_clip_batches"]
            )
        rows.append(row)
    return rows


def parity_rows(parity: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in CASES:
        condition = parity["conditions"][case]
        for result in (*condition["clips"], condition["totals"]):
            row = {
                "condition": case,
                "clip_id": result.get("clip_id", "total"),
                "scene_status": result.get("scene_status"),
                "generated_clips": result.get("generated_clips", 1),
                **{
                    key: result[key]
                    for key in (
                        "source_frames",
                        "player_frames",
                        "valid_joint_frames",
                        "valid_joint_points",
                        "valid_smpl_frames",
                        "matched_player_ids",
                        "same_numeric_id",
                    )
                },
                **result["joint_position_l2"],
            }
            for name in ("joint_point_mask", "joint_frame_mask", "smpl_mask"):
                row.update(
                    {f"{name}_{key}": value for key, value in result[name].items()}
                )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    metrics, parity = reports(args.raw_run)
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, document in (("metrics", metrics), ("parity", parity)):
        (args.destination / f"{name}.json").write_text(
            json.dumps(
                document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
    write_csv(args.destination / "comparison.csv", comparison_rows(metrics))
    write_csv(args.destination / "parity.csv", parity_rows(parity))


if __name__ == "__main__":
    main()
