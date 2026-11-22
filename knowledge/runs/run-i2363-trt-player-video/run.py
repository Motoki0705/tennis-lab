"""Build, qualify and evaluate the five TensorRT FP16 player models on the pinned video."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from src.tennis_scene.generate_dataset.manifest import (
    DatasetClipRecord,
    load_dataset_manifest,
)
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import EXPORT_MANIFEST, EngineBundle

CONFIG = json.loads(Path(__file__).with_name("condition.json").read_text())
OUTPUTS = Path("outputs").resolve(strict=True)
COURT_DIR = "i2363/court"
DRIFT_DIR = "i2363/court-drift"
TRT = "src.tennis_scene.scripts.tensorrt"
QUALIFY = "src.tennis_scene.scripts.qualify_gvhmr_tensorrt"
BATCH = "src.tennis_scene.scripts.run_video_batch"
PHASE2_TAG = "phase2-b2-v1"


def run_module(module: str, args: list[str], log: Path) -> dict[str, Any]:
    command = [sys.executable, "-B", "-m", module, *args]
    print(f"{log.stem}: {command}", flush=True)
    started = perf_counter()
    with log.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    row = {"command": command, "wall_seconds": perf_counter() - started, "exit_code": result.returncode, "log": str(log)}
    write_json_atomic(log.with_suffix(".command.json"), row)
    if result.returncode != 0:
        print(f"{log.stem}: exit {result.returncode} ({log})", flush=True)
    return row


def checked_run(module: str, args: list[str], log: Path) -> None:
    if run_module(module, args, log)["exit_code"] != 0:
        raise RuntimeError(f"{module} failed. See {log}")


def media_selection(source: Path) -> list[DatasetClipRecord]:
    dataset = load_dataset_manifest(source)
    clips = [record for _, record in sorted(dataset.clips.items()) if record.video_id == CONFIG["video_id"]]
    if dataset.dataset_id != CONFIG["dataset_id"] or len(clips) != CONFIG["clip_count"]:
        raise ValueError(f"Unexpected dataset/video selection at {source}")
    return clips


def copy_dataset(source: Path, data_root: Path, records: list[DatasetClipRecord]) -> Path:
    dataset = data_root / "dataset"
    dataset.mkdir(parents=True)
    shutil.copy2(source / "dataset.json", dataset / "dataset.json")
    for record in records:
        relative = Path(record.path)
        target = dataset / relative
        target.mkdir(parents=True)
        original = source / relative
        shutil.copy2(original / "clip.json", target / "clip.json")
        shutil.copytree(original / "media", target / "media")
    return dataset


def copy_court_store(source: Path, target: Path, digest: str) -> None:
    # Copy only the content-addressed manifest and its camera documents.
    root = source / "objects"
    manifest = root / f"{digest}.json"
    digests = [digest, *json.loads(manifest.read_text())["cameras"].values()]
    destination = target / "objects"
    destination.mkdir(parents=True, exist_ok=True)
    for value in digests:
        name = f"{value}.json"
        if not (destination / name).exists():
            shutil.copy2(root / name, destination / name)


def engine_overrides(engines: dict[str, str]) -> list[str]:
    settings = [
        "device=cuda:0", "court_kp.mode=video_shared", "court_kp.video_shared.action=load",
        f"court_kp.video_shared.directory={COURT_DIR}",
        f"court_kp.video_shared.dataset_id={CONFIG['dataset_id']}",
        f"court_kp.video_shared.video_id={CONFIG['video_id']}",
        f"court_kp.video_shared.artifact_digest={CONFIG['court_digest']}",
        f"court_kp.video_shared.drift_directory={DRIFT_DIR}",
        f"court_kp.video_shared.drift_digest={CONFIG['court_drift_digest']}",
        "court_kp.video_shared.unobservable_policy=warn",
        "court_side.mode=clip", "court_side.evidence=player", "player_association.players_per_side=2",
        "ball_detection.enabled=false", "ball_reconstruction.enabled=false",
    ]
    for model, options in CONFIG["models"].items():
        key = f"people_models.runtime.{options['pipeline_key']}"
        settings.extend([f"{key}.backend=tensorrt", f"{key}.precision=fp16",
                         f"{key}.batch_size={options['batch_size']}", f"{key}.engines=[{engines[model]}]"])
    return settings


def build_engines(args: argparse.Namespace, run_dir: Path, frames: int) -> dict[str, str]:
    engines = {}
    manifests = {}
    for model, options in CONFIG["models"].items():
        relative = f"tensorrt/{model}/{options['shape']}/{args.tag}"
        root = OUTPUTS / relative
        engines[model] = relative
        log = run_dir / f"{model}-build.log"
        batch = str(options["batch_size"])
        build = ["build", str(root), "--opt-batch", batch, "--max-batch", batch]
        if model == "gvhmr":
            checked_run(TRT, ["export", model, str(root), "--keep-mask-reference"], run_dir / "gvhmr-export.log")
            checked_run(TRT, [*build, "--frames", "2", "177", str(frames)], log)
        else:
            previous = args.phase2 / model / options["shape"] / PHASE2_TAG
            bundle = EngineBundle.read(previous)
            root.mkdir(parents=True)
            files = bundle.files if model == "dino" else (
                previous / EXPORT_MANIFEST, *(previous / name for name in bundle.export.onnx_files),
            )
            for path in files:
                shutil.copy2(path, root / path.name)
            if model != "dino":
                checked_run(TRT, build, log)
        manifests[model] = EngineBundle.read(root).engine.model_dump(mode="json")
        write_json_atomic(run_dir / "engines.json", {"directories": engines, "manifests": manifests})
    return engines


def reconstruction_arguments(
    target: Path, dataset: Path, engines: dict[str, str], *, synchronize: bool, clip: str | None = None,
) -> list[str]:
    progress = (target / "progress").relative_to(OUTPUTS)
    args = [
        f"paths.data_root={dataset.parent}", f"dataset_directory={dataset.name}", f"paths.output_root={OUTPUTS}",
        f"video_id={CONFIG['video_id']}", f"progress_directory={progress}", "batching.enabled=true",
        "record_cuda=true", f"record_cuda_synchronize={str(synchronize).lower()}",
        f"pipeline_overrides={json.dumps(engine_overrides(engines))}",
    ]
    if clip is not None:
        args.append(f"clip_ids=[{clip}]")
    return args


def reconstruct(
    target: Path, dataset: Path, engines: dict[str, str], *, synchronize: bool, clip: str | None = None,
) -> dict[str, Any]:
    args = reconstruction_arguments(target, dataset, engines, synchronize=synchronize, clip=clip)
    target.mkdir()
    log = target / "pipeline.log"
    result = run_module(BATCH, args, log)
    receipts = list(target.glob("progress/*/progress.json"))
    if len(receipts) != 1:
        raise RuntimeError(f"Video batch exit {result['exit_code']} produced {len(receipts)} progress receipts. See {log}")
    progress = receipts[0]
    document = json.loads(progress.read_text())
    result["progress"] = str(progress)
    result["recording"] = str(progress.with_name("recording.json"))
    result["statuses"] = {clip_id: outcome["status"] for clip_id, outcome in document["clips"].items()}
    write_json_atomic(target / "outcome.json", result)
    return result


def summarize_representative(passes: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for name, outcome in passes.items():
        recording = json.loads(Path(outcome["recording"]).read_text())
        if recording["status"] != "complete":
            raise ValueError(f"Incomplete inference recording for {name}")
        shapes = recording["batch_shapes"]
        if missing := {"DINO", "ViTPose", "CLIP-ReID", "HMR2", "GVHMR-network"} - shapes.keys():
            raise ValueError(f"Representative {name} lacks model forwards: {sorted(missing)}")
        resources = recording["resources"]
        gpu = resources["nvml"]["cuda:0"]
        rss_bytes = resources["ram_process_lifetime_peak_bytes"]
        if gpu["used_peak_bytes"] >= min(gpu["total_bytes"], 16 * 1024**3) or rss_bytes >= 40 * 1024**3:
            raise ValueError(f"Representative {name} reached the VRAM or WSL memory limit")
        summary[name] = {
            "cli_wall_seconds": outcome["wall_seconds"], "recorded_wall_seconds": recording["total_wall_seconds"],
            "cuda_sync_seconds": recording["stage_wall_seconds"].get("cuda_sync", 0.),
            "gpu_timing": recording["timing"]["gpu"],
            "sample_cost_median_ms": median(sample["sample_duration_ns"] for sample in recording["resource_samples"]) / 1e6,
            "resources": resources, "batch_shapes": shapes,
        }
    summary["relative_cli_wall_change"] = summary["sync"]["cli_wall_seconds"] / summary["host"]["cli_wall_seconds"] - 1
    return summary


def representative(args: argparse.Namespace, run_dir: Path, clips: list[DatasetClipRecord]) -> int:
    copy_court_store(args.court / "court-video-shared-acceptance-20261117/store",
                     OUTPUTS / COURT_DIR, CONFIG["court_digest"])
    copy_court_store(args.court / "court-video-drift-lines-20261117/drift-store",
                     OUTPUTS / DRIFT_DIR, CONFIG["court_drift_digest"])
    frames = max(len(sampled_frame_indices(record.num_frames, record.fps, max_frames=2048)) for record in clips)
    if CONFIG["models"]["gvhmr"]["shape"] != f"l2-177-{frames}":
        raise ValueError(f"Registered clip lengths changed: largest GVHMR sequence is {frames} frames")
    if args.prepared is None:
        engines = build_engines(args, run_dir, frames)
    else:
        previous = verify_prepared_engines(args.prepared)
        engines = previous["directories"]
        write_json_atomic(run_dir / "engines.json", previous)
    gvhmr = str(OUTPUTS / engines["gvhmr"])
    shapes = [part for batch in (1, 2, 3, 4) for length in (119, 120, 121, 145, 177)
              for part in ("--shape", str(batch), str(length))]
    checked_run(QUALIFY, [gvhmr, str(run_dir / "gvhmr-qualification.json"), *shapes], run_dir / "gvhmr-qualification.log")
    clip = CONFIG["representative"]
    selected = [record for record in clips if record.clip_id == clip]
    dataset = copy_dataset(args.dataset, run_dir / "inputs", selected)
    clip_dir = dataset / selected[0].path
    passes = {}
    for synchronize in (False, True):
        name = "sync" if synchronize else "host"
        case = run_dir / name
        outcome = reconstruct(case, dataset, engines, synchronize=synchronize, clip=clip)
        passes[name] = outcome
        if outcome["exit_code"] != 0 or outcome["statuses"] != {clip: "generated"}:
            raise RuntimeError(f"Representative {name} pass failed: {outcome}")
        archive = case / "annotations"
        shutil.move(clip_dir / "annotations", archive)
        outcome["retained_annotations"] = str(archive)
        write_json_atomic(case / "outcome.json", outcome)
    checked_run(QUALIFY, [
        gvhmr, str(run_dir / "gvhmr-representative-shapes.json"), "--ort-only",
        *(arg for result in passes.values() for arg in ("--progress", result["progress"])),
    ], run_dir / "gvhmr-representative-shapes.log")
    write_json_atomic(run_dir / "representative.json", passes)
    write_json_atomic(run_dir / "representative-summary.json", summarize_representative(passes))
    return 0


def verify_prepared_engines(prepared: Path) -> dict[str, Any]:
    previous: dict[str, Any] = json.loads((prepared / "engines.json").read_text())
    engines = previous["directories"]
    if engines.keys() != CONFIG["models"].keys() or engines.keys() != previous["manifests"].keys():
        raise ValueError(f"Prepared run does not declare the five configured engines: {prepared}")
    for model, directory in engines.items():
        if EngineBundle.read(OUTPUTS / directory).engine.model_dump(mode="json") != previous["manifests"][model]:
            raise ValueError(f"Prepared engine identity changed: {model}")
    return previous


def all_clips(args: argparse.Namespace, run_dir: Path, clips: list[DatasetClipRecord]) -> int:
    engines = verify_prepared_engines(args.prepared)["directories"]
    dataset = copy_dataset(args.dataset, run_dir / "inputs", clips)
    outcome = reconstruct(run_dir / "all", dataset, engines, synchronize=False)
    checked_run(QUALIFY, [
        str(OUTPUTS / engines["gvhmr"]), str(run_dir / "gvhmr-all-video-shapes.json"),
        "--ort-only", "--progress", outcome["progress"],
    ], run_dir / "gvhmr-all-video-shapes.log")
    return outcome["exit_code"] or int(set(outcome["statuses"].values()) != {"generated"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--dataset", type=Path, required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("representative")
    engine_source = prepare.add_mutually_exclusive_group(required=True)
    engine_source.add_argument("--phase2", type=Path)
    engine_source.add_argument("--prepared", type=Path)
    prepare.add_argument("--court", type=Path, required=True)
    full = modes.add_parser("all")
    full.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("Run qualification and inference through the shared training queue")
    if not args.tag.isascii() or not args.tag.replace("-", "").isalnum():
        parser.error("Use a run tag containing letters, numbers or hyphens")
    run_dir = OUTPUTS / "i2363" / args.tag
    run_dir.mkdir(parents=True, exist_ok=False)
    (Path(os.environ["TENNIS_REPRO_DIR"]) / "output_dir.txt").write_text(f"{run_dir}\n")
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    clips = media_selection(args.dataset)
    write_json_atomic(run_dir / "input.json", {
        "dataset_sha256": dual_sha256(args.dataset / "dataset.json"),
        "clips": [record.to_dict() for record in clips], "condition": CONFIG,
    })
    if args.mode == "representative":
        return representative(args, run_dir, clips)
    return all_clips(args, run_dir, clips)


if __name__ == "__main__":
    raise SystemExit(main())
