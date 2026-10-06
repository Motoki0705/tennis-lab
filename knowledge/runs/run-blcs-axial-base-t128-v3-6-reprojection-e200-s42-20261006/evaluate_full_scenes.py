"""Fixed-camera full-scene diagnostic, including short held-out BLCS scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.tasks.blcs.visualization.inference.service import InferenceService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    service = InferenceService(root / "outputs/blcs", root / "ckpt/blcs", root / "data", device="cuda")
    scenes = service.scenes(form="single_object", split="test", limit=2000)["scenes"]
    checkpoint = "checkpoints:axial-base-kp14-t128-v3-6-e200-s42-epoch189.ckpt"
    errors = []
    endpoints = []
    rows = []
    for scene in scenes:
        result = service.infer(checkpoint=checkpoint, form="single_object", scene_id=scene["id"],
            cameras=[0, 1, 2, 3], reference_camera_id=None, device="cuda", window=128)
        prediction = np.asarray(result["prediction"]["positions"]).reshape(-1, 3)
        target = np.asarray(result["gt"]["positions"]).reshape(-1, 3)
        error = np.linalg.norm(prediction - target, axis=-1)
        if not np.isfinite(error).all():
            raise RuntimeError(f"Nonfinite error: {scene['id']}")
        errors.append(error)
        endpoints.append(float(error[-1]))
        rows.append({"scene_id": scene["id"], "frames": len(error), **result["metrics"]})
    all_errors = np.concatenate(errors)
    receipt = {"split": "test", "scene_count": len(rows), "frames": len(all_errors),
        "checkpoint": checkpoint, "cameras": [0, 1, 2, 3], "window": 128,
        "protocol": "all held-out scenes, including short scenes; non-overlapping windows; frame-weighted errors",
        "precision": "float32 predictor; separate from bf16 centre-window Trainer test",
        "metrics": {"position_error_m": float(all_errors.mean()), "median_position_error_m": float(np.median(all_errors)),
            "p95_position_error_m": float(np.percentile(all_errors, 95)), "accuracy_0p3m": float(np.mean(all_errors <= .3)),
            "mean_scene_endpoint_error_m": float(np.mean(endpoints))}, "scenes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in receipt.items() if key != "scenes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
