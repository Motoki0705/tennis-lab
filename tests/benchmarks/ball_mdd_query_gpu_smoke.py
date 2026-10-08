"""Production CLI train/save/resume/val smoke on explicit real-data excerpts.

Submit this entire command via training queue. This is a small GPU integration
diagnostic, not a model-quality experiment or the requested full training run.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from src.utils.checksum import dual_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.manifest.read_text())
    excerpt = copy.deepcopy(original)
    selected = []
    counts = {}
    for split in ("train", "val"):
        for source in ("tracknet", "meiji", "chat_annotation"):
            candidates = [r for r in original["clips"] if r["clip"]["source"] == source
                          and r["clip"]["split"] == split
                          and {w["frame_step"] for w in r["windows"]} == {1, 2, 4}
                          and (split != "val" or r["common_evaluation"])]
            # Include pose-free clips in train wherever available.
            candidates.sort(key=lambda r: (r["common_evaluation"], r["clip"]["clip_id"]))
            record = copy.deepcopy(candidates[0])
            record["windows"] = [next(w for w in record["windows"] if w["frame_step"] == s) for s in (1, 2, 4)]
            record["windows_by_step"] = {str(s): 1 for s in (1, 2, 4)}
            selected.append(record)
            counts[f"{source}/{split}"] = dict(clips=1, frames=record["clip"]["frame_count"],
                                               common_clips=int(record["common_evaluation"]),
                                               windows_by_step=record["windows_by_step"])
    excerpt.update(clips=selected, counts=counts, skipped=[],
                   diagnostic=dict(purpose="GPU CLI smoke only; not full validation",
                                   parent_manifest_sha256=dual_sha256(args.manifest), test_clips=0))
    manifest = args.output / "diagnostic-windows.json"
    manifest.write_text(json.dumps(excerpt, indent=2) + "\n")
    training = args.output / "training"
    command = [sys.executable, "-m", "src.tasks.ball_detection.scripts.train_mdd_pose",
               "--manifest", str(manifest), "--model-config", str(args.model_config),
               "--output", str(training), "--learning-rate", "0.0001", "--seed", "42",
               "--device", "cuda", "--precision", "bf16", "--batch-size", str(args.batch_size),
               "--num-workers", str(args.workers), "--pin-memory", "--windows-per-epoch", "12",
               "--selection-scope", "common", "--log-every", "1"]
    subprocess.run([*command, "--epochs", "1"], check=True)
    subprocess.run([*command, "--epochs", "2", "--resume", str(training / "epoch-000.pt")], check=True)
    evaluation = args.output / "validation.json"
    subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.evaluate_mdd_coordinates",
                    "--checkpoint", str(training / "epoch-001.pt"), "--manifest", str(manifest),
                    "--output", str(evaluation), "--split", "val", "--device", "cuda",
                    "--artifact-root", str(args.output), "--output-root", str(args.output),
                    "--batch-size", str(args.batch_size), "--num-workers", str(args.workers),
                    "--pin-memory"], check=True)
    saved = torch.load(training / "epoch-001.pt", map_location="cpu", weights_only=True)
    report = json.loads(evaluation.read_text())
    expected_updates = 2 * ((12 + args.batch_size - 1) // args.batch_size)
    assert saved["training_state"]["global_step"] == expected_updates
    assert saved["training_state"]["cuda_rng"] is not None
    assert report["precision"] == saved["validation"]["precision"] == "bf16"
    assert all(torch.isfinite(t).all() for t in saved["state_dict"].values())
    assert set(report["scopes"]["common"]["by_frame_step"]) == {"1", "2", "4"}
    result = dict(status="ok", purpose="integration diagnostic, not model quality",
                  queue_run_id=os.environ.get("TENNIS_RUN_ID"), train_clips=3, val_clips=3, test_clips=0,
                  train_windows=9, val_windows=9, optimizer_updates=expected_updates,
                  precision=report["precision"], batch_size=args.batch_size, workers=args.workers,
                  cuda_rng_saved=True, resumed=True, external_eval_restored_precision=True,
                  parent_manifest_sha256=dual_sha256(args.manifest),
                  excerpt_sha256=dual_sha256(manifest), checkpoint_sha256=dual_sha256(training / "epoch-001.pt"))
    (args.output / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
