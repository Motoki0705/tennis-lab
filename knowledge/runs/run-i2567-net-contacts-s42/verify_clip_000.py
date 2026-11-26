"""Run the pinned three-channel checkpoint on multiview-v2 clip_000."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.tasks.ball_refiner_3d.events import EVENT_CLASSES
from src.tasks.ball_refiner_3d.inference.predictor import RefinerPredictor
from src.tasks.ball_refiner_3d.inference.scene import predict_scene_events, scene_input
from src.tennis_scene.archive import load_scene_result

BENCHMARK_SHA256 = "0d11010fa2a10d73ba4b03ba01f107c90641868fbb8dd86978503cc761d4ac44"
CHECKPOINT_SHA256 = "b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0"
CLIP_ID = "video_000/clip_000"
FPS_TOLERANCE = 0.06


def require_sha256(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.benchmark / "manifest.json"
    require_sha256(manifest_path, BENCHMARK_SHA256)
    clips = json.loads(manifest_path.read_text(encoding="utf-8"))["clips"]
    entry = next(clip for clip in clips if clip["clip_id"] == CLIP_ID)
    scene_path = args.benchmark / entry["scene"]["path"]
    require_sha256(scene_path, entry["scene"]["sha256"])
    require_sha256(args.checkpoint, CHECKPOINT_SHA256)
    torch.set_num_threads(2)

    coordinates, missing, fps = scene_input(load_scene_result(scene_path))
    if coordinates.shape != (1, entry["frames"], 3) or fps != entry["fps"]:
        raise ValueError("Scene frames or FPS differ from the benchmark manifest")
    predictor = RefinerPredictor.from_checkpoint(
        args.checkpoint.resolve(), device="cpu"
    )
    prediction, events = predict_scene_events(
        predictor,
        scene_path,
        clip_id=CLIP_ID,
        checkpoint_sha256=CHECKPOINT_SHA256,
        seed=42,
        fps_tolerance=FPS_TOLERANCE,
    )
    probability = prediction.event_probability.numpy()
    if probability.shape != (1, entry["frames"], len(EVENT_CLASSES)):
        raise ValueError(f"Unexpected event probability shape {probability.shape}")
    fitted_fps = float(predictor.metadata["fps"])
    summary = {
        "clip_id": CLIP_ID,
        "scene_sha256": entry["scene"]["sha256"],
        "manifest_sha256": BENCHMARK_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "input_shape": list(coordinates.shape),
        "probability_shape": list(probability.shape),
        "missing_frames": int(missing.sum().item()),
        "fps": fps,
        "checkpoint_fps": fitted_fps,
        "fps_difference": abs(fitted_fps - fps),
        "fps_tolerance": FPS_TOLERANCE,
        "event_classes": list(EVENT_CLASSES),
        "events": {
            kind: {
                "peaks": sum(event["kind"] == kind for event in events["events"]),
                "mean": float(values.mean()),
                "max": float(values.max()),
                "argmax_frame": int(values.argmax()),
            }
            for kind, values in zip(EVENT_CLASSES, probability[0].T, strict=True)
        },
    }

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "events.json").write_text(
        json.dumps(events, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "event_probabilities.npz",
        allow_pickle=False,
        event_probability=probability,
        event_classes=np.asarray(EVENT_CLASSES),
        input_missing=missing.numpy(),
        fps=np.asarray(fps),
        checkpoint_sha256=np.asarray(CHECKPOINT_SHA256),
        scene_sha256=np.asarray(entry["scene"]["sha256"]),
    )
    summary_text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (args.output / "summary.json").write_text(summary_text, encoding="utf-8")
    print(summary_text, end="")


if __name__ == "__main__":
    main()
