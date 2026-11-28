"""Run the fixed #2696 E2E comparison in one exclusive queue reservation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from src.tennis_scene.generate_dataset.manifest import (
    DatasetClipRecord,
    load_dataset_manifest,
)
from src.tennis_scene.pipeline.clip_inputs import resolve_dataset_clip
from src.tennis_scene.pipeline.video_court.diagnostics import (
    DriftReference,
    load_video_drift,
)
from src.tennis_scene.pipeline.video_court.store import VideoCourtStore
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic

BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[2]
CONDITION = json.loads((BUNDLE / "condition.json").read_text())
CLIPS = tuple(CONDITION["clips"])
MODELS = {
    "dino": "dino_detector",
    "vitpose": "vitpose",
    "clip": "clip_reid",
    "hmr2": "hmr2",
    "gvhmr": "gvhmr",
}
CASES = ("C1_serial_pt16", "C2_shared_pt16", "C3_batched_pt16", "C4_batched_trt16")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def command(arguments: list[str], log: Path) -> dict[str, Any]:
    with log.open("x") as stream:
        started = perf_counter()
        result = subprocess.run(
            arguments, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
        wall = perf_counter() - started
    receipt = {
        "command": arguments,
        "wall_seconds": wall,
        "exit_code": result.returncode,
    }
    write_json_atomic(log.with_suffix(".command.json"), receipt)
    print(f"{log}: exit={result.returncode}, {wall:.3f} s", flush=True)
    return receipt


def require_head(expected: str) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "third_party",
            str(BUNDLE),
        ],
        text=True,
    )
    if head != expected or changed:
        raise ValueError(
            f"Expected clean experiment source at {expected}, got {head}\n{changed}"
        )


def verify_inputs(
    dataset: Path, court: Path, drift: Path
) -> tuple[DatasetClipRecord, ...]:
    index = load_dataset_manifest(dataset)
    if (
        index.dataset_id != CONDITION["dataset_id"]
        or dual_sha256(dataset / "dataset.json") != CONDITION["dataset_manifest_sha256"]
    ):
        raise ValueError(f"Dataset differs from the reference: {dataset}")
    records = tuple(index.clips[clip] for clip in CLIPS)
    for record in records:
        inputs = resolve_dataset_clip(dataset, record)
        pinned = CONDITION["clips"][record.clip_id]
        if (
            (record.num_frames, record.num_cameras) != (pinned["frames"], 4)
            or inputs.manifest_sha256 != pinned["manifest_sha256"]
            or dict(inputs.media_sha256) != pinned["media_sha256"]
        ):
            raise ValueError(
                f"Clip or media differs from the reference: {record.clip_id}"
            )
    load_video_drift(
        VideoCourtStore(court).load(CONDITION["court_digest"]),
        DriftReference(drift, CONDITION["court_drift_digest"]),
    )
    return records


def copy_court(source: Path, destination: Path, digest: str) -> None:
    objects = destination / "objects"
    objects.mkdir(parents=True)
    root = source / "objects"
    manifest = root / f"{digest}.json"
    for key in (digest, *read_json(manifest)["cameras"].values()):
        shutil.copy2(root / f"{key}.json", objects / f"{key}.json")


def copy_dataset(
    source: Path,
    dataset: Path,
    records: tuple[DatasetClipRecord, ...],
    clips: tuple[str, ...],
) -> None:
    dataset.mkdir(parents=True)
    shutil.copy2(source / "dataset.json", dataset / "dataset.json")
    for record in records:
        if record.clip_id in clips:
            original = source / record.path
            target = dataset / record.path
            target.mkdir(parents=True)
            shutil.copy2(original / "clip.json", target / "clip.json")
            shutil.copytree(original / "media", target / "media")


def pipeline_overrides(
    output_root: Path, run: Path, engines: dict[str, str] | None
) -> list[str]:
    relative = run.relative_to(output_root)
    overrides = [
        "recipe=standard",
        "device=cuda:0",
        "people_models.detector=dino",
        "people_models.pose=vitpose-h",
        "person_tracking.method=strongsort_pp_pose",
        "person_tracking.encoder=clipreid_vitb16_market1501",
        "court_kp.mode=video_shared",
        "court_kp.video_shared.action=load",
        f"court_kp.video_shared.directory={relative}/court",
        f"court_kp.video_shared.dataset_id={CONDITION['dataset_id']}",
        f"court_kp.video_shared.video_id={CONDITION['video_id']}",
        f"court_kp.video_shared.artifact_digest={CONDITION['court_digest']}",
        f"court_kp.video_shared.drift_directory={relative}/court-drift",
        f"court_kp.video_shared.drift_digest={CONDITION['court_drift_digest']}",
        "court_kp.video_shared.unobservable_policy=warn",
        "court_side.mode=clip",
        "court_side.evidence=player",
        "player_association.players_per_side=2",
        "ball_detection.enabled=false",
        "ball_reconstruction.enabled=false",
        "event_detection.method=rule",
    ]
    for model, key in MODELS.items():
        prefix = f"people_models.runtime.{key}"
        precision = (
            "fp16" if engines is not None or model in {"dino", "vitpose"} else "fp32"
        )
        engine_list = [] if engines is None else [engines[model]]
        overrides += [
            f"{prefix}.backend={'pytorch' if engines is None else 'tensorrt'}",
            f"{prefix}.precision={precision}",
            f"{prefix}.batch_size={CONDITION['batches'][model]}",
            f"{prefix}.engines={json.dumps(engine_list)}",
        ]
    return overrides


def cli_arguments(
    case: str,
    output_root: Path,
    run: Path,
    pass_dir: Path,
    clips: tuple[str, ...],
    engines: dict[str, str],
) -> list[str]:
    trt = case == CASES[3]
    overrides = pipeline_overrides(output_root, run, engines if trt else None)
    return [
        sys.executable,
        "-B",
        str(BUNDLE / "pipeline_cli.py"),
        "trt" if trt else "pt",
        f"paths.data_root={pass_dir}",
        f"paths.artifact_root={output_root}",
        f"paths.output_root={output_root}",
        "dataset_directory=dataset",
        f"video_id={CONDITION['video_id']}",
        f"clip_ids={json.dumps(clips)}",
        f"progress_directory={(pass_dir / 'progress').relative_to(output_root)}",
        f"batching.enabled={'true' if case in CASES[2:] else 'false'}",
        *(
            f"batching.sizes.{model}={CONDITION['batches'][model]}"
            for model in MODELS
            if model != "gvhmr"
        ),
        "batching.store_cache_bytes=0",
        "record_cuda=true",
        "record_cuda_synchronize=false",
        f"record_sample_interval={CONDITION['sampling_interval_seconds']}",
        f"pipeline_overrides={json.dumps(overrides)}",
    ]


def measure(
    output_root: Path,
    run: Path,
    case: str,
    pass_dir: Path,
    clips: tuple[str, ...],
    engines: dict[str, str],
) -> dict[str, Any]:
    if any((pass_dir / "dataset").glob("videos/*/clips/*/annotations")):
        raise ValueError(
            f"Component stores must be empty before measurement: {pass_dir}"
        )
    receipt = command(
        cli_arguments(case, output_root, run, pass_dir, clips, engines),
        pass_dir / "pipeline.log",
    )
    paths = list((pass_dir / "progress").glob("*/progress.json"))
    if len(paths) > 1:
        raise ValueError(f"Ambiguous progress receipts at {pass_dir}")
    progress = paths[0] if paths else None
    recording = progress.with_name("recording.json") if progress else None
    document = read_json(progress) if progress else {}
    statuses = {
        clip: outcome["status"] for clip, outcome in document.get("clips", {}).items()
    }
    return {
        **receipt,
        "clips": clips,
        "statuses": statuses,
        "directory": str(pass_dir.relative_to(run)),
        "progress": str(progress.relative_to(run)) if progress else None,
        "recording": str(recording.relative_to(run))
        if recording and recording.is_file()
        else None,
    }


def run_experiment(args: argparse.Namespace) -> int:
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_REPRO_DIR"):
        raise RuntimeError("Run GPU inference through the shared training queue")
    require_head(args.expect_head)
    output_root = Path("outputs").resolve(strict=True)
    run = output_root / "i2696" / args.tag
    run.mkdir(parents=True)
    queue_job = os.environ["TENNIS_RUN_ID"]
    (Path(os.environ["TENNIS_REPRO_DIR"]) / "output_dir.txt").write_text(f"{run}\n")
    threads = str(CONDITION["cpu_threads"])
    os.environ.update(
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS=threads,
        TENNIS_LAB_DINO_EXTENSION=str(args.dino_extension.resolve(strict=True)),
        PYTHONHASHSEED=str(CONDITION["seed"]),
    )
    court, drift = run / "court", run / "court-drift"
    copy_court(args.prior / "i2363/court", court, CONDITION["court_digest"])
    copy_court(args.prior / "i2363/court-drift", drift, CONDITION["court_drift_digest"])
    records = verify_inputs(args.dataset, court, drift)
    jobs = {case: [(run / case / "video", CLIPS)] for case in CASES}
    jobs[CASES[0]] = [(run / CASES[0] / clip.split("/")[-1], (clip,)) for clip in CLIPS]
    for passes in jobs.values():
        for pass_dir, clips in passes:
            copy_dataset(args.dataset, pass_dir / "dataset", records, clips)
    preparation = command(
        [
            sys.executable,
            "-B",
            str(BUNDLE / "prepare_engines.py"),
            str(args.prior),
            str(output_root),
            str(run),
        ],
        run / "prepare-engines.log",
    )
    if preparation["exit_code"] != 0:
        return int(preparation["exit_code"])
    preparation["engines"] = read_json(run / "engines.json")
    engines = {
        model: item["directory"]
        for model, item in preparation["engines"]["models"].items()
    }
    result: dict[str, Any] = {
        "head": args.expect_head,
        "queue_job": queue_job,
        "condition_sha256": dual_sha256(BUNDLE / "condition.json"),
        "dino_extension_sha256": dual_sha256(args.dino_extension),
        "engine_preparation": preparation,
        "cases": {},
    }
    outcome = run / "outcome.json"
    write_json_atomic(outcome, result)
    failed = False
    for case, passes in jobs.items():
        result["cases"][case] = []
        for pass_dir, clips in passes:
            require_head(args.expect_head)
            receipt = measure(output_root, run, case, pass_dir, clips, engines)
            result["cases"][case].append(receipt)
            write_json_atomic(outcome, result)
            failed |= (
                receipt["exit_code"] != 0
                or receipt["statuses"] != dict.fromkeys(clips, "generated")
                or not receipt["recording"]
            )
    require_head(args.expect_head)
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--expect-head")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--prior", type=Path, required=True, help="#2363 worktree output root"
    )
    parser.add_argument("--dino-extension", type=Path, required=True)
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate pinned inputs and DINO binary on CPU",
    )
    args = parser.parse_args()
    if Path.cwd() != REPO:
        parser.error("Run from the repository root")
    if not re.fullmatch(r"[a-zA-Z0-9-]+", args.tag):
        parser.error("Tag must contain only ASCII letters, digits or hyphens")
    args.dataset = args.dataset.resolve(strict=True)
    args.prior = args.prior.resolve(strict=True)
    args.dino_extension = args.dino_extension.resolve(strict=True)
    if args.check_inputs:
        from src.submodules.models import validate_dino_extension

        os.environ["TENNIS_LAB_DINO_EXTENSION"] = str(args.dino_extension)
        print(validate_dino_extension())
        verify_inputs(
            args.dataset, args.prior / "i2363/court", args.prior / "i2363/court-drift"
        )
        print(
            "Verified pinned dataset, 3 clip manifests, 12 media files, court and drift"
        )
        return 0
    if not args.expect_head or not re.fullmatch(r"[0-9a-f]{40}", args.expect_head):
        parser.error("--expect-head requires the complete measurement commit")
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
