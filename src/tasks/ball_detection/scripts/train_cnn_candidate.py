"""Queue payload: GPU smoke must pass before a candidate's scratch pretraining."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from src.tasks.ball_detection.models.mdd_pretrain import MDDPretrainConfig
from src.tasks.ball_detection.training.heatmap_pretraining.runner import source_identity
from src.tasks.ball_detection.training.posttraining.checkpoint import (
    completed_pretraining,
)
from src.tasks.ball_detection.training.posttraining.paths import resolver
from src.utils.checksum import dual_sha256
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_detection.train_cnn_candidate", fields=(
    BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("model_config", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    BoundaryPathField("smoke_output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
))


def training_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-m", "src.tasks.ball_detection.scripts.pretrain_mdd_dpt",
        "--manifest", str(args.manifest), "--model-config", str(args.model_config), "--output", str(args.output),
        "--epochs", "10", "--windows-per-epoch", "6000", "--learning-rate", ".0002", "--warmup-updates", "500",
        "--seed", "42", "--device", "cuda", "--precision", "bf16", "--batch-size", "1", "--num-workers", "8",
        "--prefetch-factor", "4", "--cpu-threads", "2", "--pin-memory", "--jpeg-decoder", "nvjpeg",
        "--input-verification", "upfront", "--compile-mode", "default", "--selection-scope", "common",
        "--log-every", "50", "--preview-clips", "3"]
    if args.prefetch_mode == "overlap":
        command.append("--image-prefetch")
    return command


def validate_serial_prefix(args: argparse.Namespace) -> dict:
    output = args.output
    stop = json.loads((output / "DIAGNOSTIC_STOP.json").read_text())
    if (stop.get("stage") != "heatmap_pretraining_diagnostic_stop" or stop.get("complete") is not False
            or stop["epoch"] != 0 or stop["global_step"] != 6000 or stop["total_updates"] != 60000
            or stop["checkpoint"] != "epoch-000.pt" or stop["sha256"] != dual_sha256(output / "epoch-000.pt")):
        raise ValueError("Serial diagnostic did not complete exactly the first epoch of the full schedule")
    saved = torch.load(output / "epoch-000.pt", weights_only=True, map_location="cpu")
    recipe = saved["recipe"]
    declared = json.loads(json.dumps(asdict(MDDPretrainConfig.load(args.model_config))))
    if (json.loads(json.dumps(recipe["model"])) != declared
            or saved["code"]["source_sha256"] != source_identity()["source_sha256"]
            or recipe["manifest_sha256"] != dual_sha256(args.manifest)
            or recipe["runtime"]["image_prefetch"] is not False
            or recipe["runtime"]["precision"] != "bf16" or recipe["runtime"]["jpeg_decoder"] != "nvjpeg"
            or saved["global_step"] != 6000 or saved["epoch"] != 0
            or recipe["epochs"] != 10 or recipe["windows_per_epoch"] != 6000):
        raise ValueError("Serial diagnostic checkpoint has a different model/data/runtime/budget")
    if not all(bool(torch.isfinite(value).all()) for value in saved["state_dict"].values()):
        raise ValueError("Serial diagnostic produced nonfinite weights")
    rows = [json.loads(s) for s in (output / "train.jsonl").read_text().splitlines()]
    prefix = [row for row in rows if row["epoch"] == 0]
    if not prefix or prefix[-1]["global_step"] != 6000 or not all(
            math.isfinite(row["train_loss"]) and math.isfinite(row["grad_norm"])
            and all(value is not None and math.isfinite(value) and value > 0 for value in row["temporal_gradient_norms"])
            for row in prefix):
        raise ValueError("Serial diagnostic lacks finite losses/gradients through the full prefix")
    return dict(stop, manifest_sha256=recipe["manifest_sha256"], image_prefetch=False,
                limitation="Passing one epoch does not establish the CUDA failure's cause or long-run stability")


def run_serial_candidate(args: argparse.Namespace) -> None:
    command = training_command(args)
    if not args.output.exists():
        if args.smoke_output.exists():
            raise ValueError("Serial diagnostic receipt exists without its training output")
        subprocess.run([*command, "--stop-after-epoch", "0"], check=True)
    # A successful process exit alone is insufficient; never restart a failed prefix silently.
    receipt = validate_serial_prefix(args)
    args.smoke_output.mkdir(parents=True, exist_ok=True)
    (args.smoke_output / "SERIAL_PREFIX_PASSED.json").write_text(json.dumps(receipt, indent=2))
    if (args.output / "COMPLETED.json").exists():
        completed_pretraining(args.output, args.manifest)
        return
    checkpoints = sorted(args.output.glob("epoch-*.pt"))
    subprocess.run([*command, "--resume", str(checkpoints[-1])], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model-config", "output", "smoke-output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--prefetch-mode", choices=("overlap", "serial"), required=True)
    args = p.parse_args()
    names = ("manifest", "model_config", "output", "smoke_output")
    paths = PATH_BOUNDARY.validate({k: getattr(args, k) for k in names}, resolver=resolver(args.model_config.parent,
        (args.manifest,), (args.output, args.smoke_output), args.output.parent))
    for key in names:
        setattr(args, key, paths.declared(key).path)
    if any(not getattr(args, key).is_absolute() for key in ("manifest", "model_config", "output", "smoke_output")):
        p.error("Candidate paths must be absolute")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.prefetch_mode == "serial":
        run_serial_candidate(args)
        return
    smoke = args.smoke_output
    smoke.parent.mkdir(parents=True, exist_ok=True)
    if not (smoke / "training/COMPLETED.json").exists():
        if smoke.exists():
            raise ValueError("Incomplete candidate GPU smoke requires inspection; refusing a silent restart")
        subprocess.run([sys.executable, "tests/benchmarks/ball_dpt_preflight.py", "--manifest", str(args.manifest),
            "--model-config", str(args.model_config), "--output", str(smoke), "--windows", "96", "--batch-size", "1"], check=True)
    rows = [json.loads(s) for s in (smoke / "training/train.jsonl").read_text().splitlines()]
    settings = json.loads((smoke / "training/config.json").read_text())
    declared = json.loads(json.dumps(asdict(MDDPretrainConfig.load(args.model_config))))
    if settings["recipe"]["model"] != declared or settings["code"]["source_sha256"] != source_identity()["source_sha256"]:
        raise ValueError("Candidate GPU smoke used a different model or implementation")
    if not all(all(v is not None and v > 0 for v in r["temporal_gradient_norms"]) for r in rows):
        raise ValueError("GPU smoke did not demonstrate gradients in both temporal stages")
    output = args.output
    if (output / "COMPLETED.json").exists():
        return
    command = training_command(args)
    if output.exists():
        checkpoints = sorted(output.glob("epoch-*.pt"))
        if not checkpoints:
            raise ValueError("Incomplete pretraining without checkpoint; inspect and use a new run directory")
        command += ["--resume", str(checkpoints[-1])]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
