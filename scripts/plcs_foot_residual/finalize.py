"""Evaluate the validation-selected checkpoint and render the paired artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


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
    subprocess.run(
        [
            sys.executable,
            "scripts/plcs_foot_residual/evaluate.py",
            "--checkpoint",
            str(inference_path),
            "--label",
            "residual",
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
    subprocess.run(
        [
            sys.executable,
            "scripts/plcs_foot_residual/render.py",
            "--comparison",
            str(args.output),
            "--clip",
            str(args.clip),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
