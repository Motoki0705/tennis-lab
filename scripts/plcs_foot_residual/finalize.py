"""Evaluate the validation-selected checkpoint and render the paired artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.schema import SceneResult


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--baseline-config", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--clip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    checkpoint_dir = args.run_dir / "logs/version_0/checkpoints"
    checkpoints = list(checkpoint_dir.glob("plcs-epoch=*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one validation-best checkpoint: {checkpoints}")
    best = checkpoints[0].resolve()
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    selected = [
        v
        for k, v in checkpoint["callbacks"].items()
        if k.startswith("ModelCheckpoint")
        and v.get("monitor") == "val/position_error_m"
    ]
    if len(selected) != 1 or Path(selected[0]["best_model_path"]).resolve() != best:
        raise ValueError(
            "Selected checkpoint does not match the recorded validation best"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection.json").write_text(
        json.dumps(
            {
                "checkpoint": str(best),
                "epoch": checkpoint["epoch"],
                "validation_position_error_m": float(selected[0]["best_model_score"]),
                "selection_rule": "minimum validation position error; no test or real-clip selection",
            },
            indent=2,
        )
    )
    # Keep a model-only checkpoint for inference; retain the trainer checkpoint
    # separately for restart/optimizer state.
    checkpoint.pop("optimizer_states", None)
    checkpoint.pop("lr_schedulers", None)
    inference_path = args.output / "residual_best.ckpt"
    torch.save(checkpoint, inference_path)
    del checkpoint
    baseline_receipt = json.loads(
        (args.output / "baseline_best_clip_metrics.json").read_text()
    )
    baseline_path = Path(baseline_receipt["checkpoint"])
    if (
        hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        != baseline_receipt["checkpoint_sha256"]
    ):
        raise ValueError("Baseline checkpoint changed since its initial evaluation.")
    # Both final test predictions use the same GPU, float32 policy and loader.
    for model_path, label in [
        (baseline_path, "baseline_best"),
        (inference_path, "residual"),
    ]:
        subprocess.run(
            [
                sys.executable,
                "scripts/plcs_foot_residual/evaluate.py",
                "--checkpoint",
                str(model_path),
                "--label",
                label,
                "--baseline-config",
                str(args.baseline_config),
                "--dataset",
                str(args.dataset),
                "--clip",
                str(args.clip),
                "--output",
                str(args.output),
            ],
            check=True,
        )
    # Native scene artifact: reuse the frozen upstream observations/SMPL and
    # replace only PLCS position/yaw with the newly inferred physical outputs.
    source_scene = args.clip / "annotations/tennis_scene/scene.npz"
    with np.load(source_scene) as source:
        payload = {key: source[key] for key in source.files}
    for scalar_key, scalar_type in [
        ("num_frames", int),
        ("fps", float),
        ("width", int),
        ("height", int),
    ]:
        payload[scalar_key] = scalar_type(payload[scalar_key])
    with np.load(args.output / "residual_clip.npz") as predicted:
        payload["player_position"] = predicted["position"].astype(np.float32)
        payload["player_yaw"] = predicted["yaw"].astype(np.float32)
    metadata = json.loads((args.output / "reference_context.json").read_text())
    metadata.update(
        plcs=json.loads((args.output / "residual_clip_metrics.json").read_text()),
        upstream_source_scene=str(source_scene.resolve()),
        upstream_source_sha256=hashlib.sha256(source_scene.read_bytes()).hexdigest(),
        upstream_stages="frozen manual court, associated 2D pose and GVHMR/SMPL cache",
        video_paths=[
            str((args.clip / "media" / f"cam{i}.mp4").resolve()) for i in range(3)
        ],
    )
    native_path = args.output / "residual_scene.npz"
    save_scene_result(SceneResult(**payload, metadata=metadata), native_path)
    restored = load_scene_result(native_path)
    np.testing.assert_array_equal(restored.player_position, payload["player_position"])
    np.testing.assert_array_equal(restored.player_yaw, payload["player_yaw"])
    subprocess.run(
        [
            sys.executable,
            "scripts/plcs_foot_residual/render.py",
            "--baseline-label",
            "baseline_best",
            "--comparison",
            str(args.output),
            "--clip",
            str(args.clip),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
