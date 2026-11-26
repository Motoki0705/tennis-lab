"""Run the pinned 3D Ball Refiner at its synthetic-test peak setting."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch
from protocol import (
    BENCHMARK_ID,
    CLASSES,
    PINS,
    read_json,
    save_json,
    sha256,
    verify_hash,
)

SEED = 42
FPS_TOLERANCE = 0.06
THREADS = 4


def require_checkout() -> Path:
    root = Path.cwd()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if head != PINS["refiner_commit"] or status:
        raise ValueError("Run from the clean checkout of the pinned Refiner commit")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "scoring", "tuning"), default="all")
    args = parser.parse_args()
    start = perf_counter()
    refiner_root = require_checkout()
    sys.path.insert(0, str(refiner_root))

    from src.tasks.ball_refiner_3d.inference.scene import predict_scene_events

    from src.tasks.ball_refiner_3d.inference.predictor import RefinerPredictor
    from src.tasks.ball_refiner_3d.inference.segmentation import DEFAULT_PICKING

    manifest = read_json(args.benchmark / "manifest.json", PINS["manifest_sha256"])
    selection = read_json(args.selection, PINS["selection_sha256"])
    roles = {row["clip_id"]: row["role"] for row in selection["clips"]}
    if (
        manifest["benchmark_id"] != BENCHMARK_ID
        or selection["manifest_sha256"] != PINS["manifest_sha256"]
    ):
        raise ValueError("Benchmark and selection disagree")
    verify_hash(args.checkpoint, PINS["checkpoint_sha256"])

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    predictor = RefinerPredictor.from_checkpoint(
        args.checkpoint.resolve(), device="cpu"
    )
    config = {
        "benchmark_id": BENCHMARK_ID,
        "manifest_sha256": PINS["manifest_sha256"],
        "selection_sha256": PINS["selection_sha256"],
        "method": "refiner3",
        "method_version": PINS["checkpoint_sha256"],
        "code_commit": PINS["refiner_commit"],
        "checkpoint_fps": predictor.metadata["fps"],
        "fps_tolerance": FPS_TOLERANCE,
        "seed": SEED,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "cpu_threads": THREADS,
        "peak_threshold": DEFAULT_PICKING.threshold,
        "peak_min_separation_frames": DEFAULT_PICKING.min_separation,
    }
    config_path = args.output / "refiner-config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Refiner output already belongs to a different configuration")
    save_json(config_path, config)

    progress = {}
    ok = failed = 0
    for clip in manifest["clips"]:
        key = clip["clip_id"]
        relative_path = Path("predictions/refiner3") / f"{key}.json"
        output_path = args.output / relative_path
        scene_path = args.benchmark / clip["scene"]["path"]
        expected = {
            "schema": "ball_event_predictions.v1",
            "clip_id": key,
            "fps": clip["fps"],
            "frames": clip["frames"],
            "method": "refiner3",
            "method_version": PINS["checkpoint_sha256"],
            "inputs": {"scene_sha256": clip["scene"]["sha256"]},
        }
        if output_path.exists():
            document = read_json(output_path)
        elif args.scope == "all" or roles.get(key) == args.scope:
            try:
                verify_hash(scene_path, clip["scene"]["sha256"])
                _, document = predict_scene_events(
                    predictor,
                    scene_path,
                    clip_id=key,
                    checkpoint_sha256=PINS["checkpoint_sha256"],
                    seed=SEED,
                    fps_tolerance=FPS_TOLERANCE,
                    picking=DEFAULT_PICKING,
                )
            except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
                actual = sha256(scene_path) if scene_path.is_file() else None
                document = {
                    **expected,
                    "inputs": {"scene_sha256": actual},
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                    "events": [],
                }
            save_json(output_path, document)
            print(
                f"{key} {document['status']} {len(document['events'])} events",
                flush=True,
            )
        else:
            progress[key] = {"status": "pending"}
            continue

        fields = (
            expected
            if document["status"] == "ok"
            else {k: v for k, v in expected.items() if k != "inputs"}
        )
        if any(document.get(k) != v for k, v in fields.items()):
            raise ValueError(f"Prediction provenance or frame grid mismatch for {key}")
        status = document["status"]
        if status not in {"ok", "failed"}:
            raise ValueError(f"Unexpected prediction status for {key}: {status}")
        ok += status == "ok"
        failed += status == "failed"
        counts = Counter(event["kind"] for event in document["events"])
        progress[key] = {
            "status": status,
            "reason": document["reason"],
            "predictions": relative_path.as_posix(),
            "sha256": sha256(output_path),
            "events": {kind: counts[kind] for kind in CLASSES},
            "valid_ball_frames": clip["ball"]["valid_3d_frames"],
        }

    require_checkout()
    pending = len(progress) - ok - failed
    save_json(
        args.output / "refiner-progress.json",
        {
            "schema": "ball_event_prediction_progress.v1",
            "state": "complete" if not pending else "partial",
            **config,
            "total": len(progress),
            "ok": ok,
            "failed": failed,
            "pending": pending,
            "clips": progress,
        },
    )
    print(
        f"Refiner: ok={ok} failed={failed} pending={pending} elapsed_s={perf_counter() - start:.2f}"
    )


if __name__ == "__main__":
    main()
