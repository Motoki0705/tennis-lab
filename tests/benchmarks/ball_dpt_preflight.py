"""Real 720p smoke through the production trainer; invoke only via training queue."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--windows", type=int, default=96)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    selected = []
    for split in ("train", "val"):
        records = [r for r in manifest["clips"] if r["clip"]["split"] == split]
        # Require each diagnostic clip to cover all declared FPS conditions.
        records = [r for r in records if {w["frame_step"] for w in r["windows"]} == {1, 2, 4}]
        for record in records[:3]:
            record["windows"] = [next(w for w in record["windows"] if w["frame_step"] == step) for step in (1, 2, 4)]
            selected.append(record)
    manifest["clips"] = selected
    path = args.output / "smoke-manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    command = [sys.executable, "-m", "src.tasks.ball_detection.scripts.pretrain_mdd_dpt",
        "--manifest", str(path), "--model-config", str(args.model_config), "--output", str(args.output / "training"),
        "--epochs", "1", "--windows-per-epoch", str(args.windows), "--warmup-updates", "4",
        "--learning-rate", ".0002", "--device", "cuda", "--precision", "bf16", "--batch-size", str(args.batch_size),
        "--jpeg-decoder", "nvjpeg", "--image-prefetch", "--num-workers", "8", "--pin-memory",
        "--compile-mode", "default", "--selection-scope", "full", "--log-every", "8", "--preview-clips", "1"]
    subprocess.run(command, check=True)
    (args.output / "command.json").write_text(json.dumps(command, indent=2))


if __name__ == "__main__":
    main()
