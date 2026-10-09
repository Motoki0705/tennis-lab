"""Queue payload: GPU smoke must pass before a candidate's scratch pretraining."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from src.tasks.ball_detection.models.mdd_pretrain import MDDPretrainConfig
from src.tasks.ball_detection.training.heatmap_pretraining.runner import source_identity
from src.tasks.ball_detection.training.posttraining.paths import resolver
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model-config", "output", "smoke-output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    names = ("manifest", "model_config", "output", "smoke_output")
    paths = PATH_BOUNDARY.validate({k: getattr(args, k) for k in names}, resolver=resolver(args.model_config.parent,
        (args.manifest,), (args.output, args.smoke_output), args.output.parent))
    for key in names:
        setattr(args, key, paths.declared(key).path)
    if any(not getattr(args, key).is_absolute() for key in ("manifest", "model_config", "output", "smoke_output")):
        p.error("Candidate paths must be absolute")
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
    command = [sys.executable, "-m", "src.tasks.ball_detection.scripts.pretrain_mdd_dpt",
        "--manifest", str(args.manifest), "--model-config", str(args.model_config), "--output", str(output),
        "--epochs", "10", "--windows-per-epoch", "6000", "--learning-rate", ".0002", "--warmup-updates", "500",
        "--seed", "42", "--device", "cuda", "--precision", "bf16", "--batch-size", "1", "--num-workers", "8",
        "--prefetch-factor", "4", "--cpu-threads", "2", "--pin-memory", "--jpeg-decoder", "nvjpeg",
        "--input-verification", "upfront", "--image-prefetch", "--compile-mode", "default", "--selection-scope", "common",
        "--log-every", "50", "--preview-clips", "3"]
    if output.exists():
        checkpoints = sorted(output.glob("epoch-*.pt"))
        if not checkpoints:
            raise ValueError("Incomplete pretraining without checkpoint; inspect and use a new run directory")
        command += ["--resume", str(checkpoints[-1])]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
