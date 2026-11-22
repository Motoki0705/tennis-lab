"""One authorized single-clip reference after the completed three-clip attempt."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from doubles_acceptance import (  # noqa: E402
    CLIPS,
    REPO,
    capture_diagnostics,
    run_variant,
    sha,
    utc,
    verify,
    write,
)
from doubles_analysis import clip_summary, lifetimes, read  # noqa: E402

CUDA_WORDS = ("cuda", "cublas", "cudnn", "nvml", "illegal memory access", "device-side assert")


def review(batch):
    """Narrow CPU audit of this SystemExit false alarm; keep original evidence."""
    entry = read(batch / "batch/entry.json")
    terminal = read(batch / "terminal.json")
    outcome = terminal["outcomes"]["batch"]
    alarm = read(batch / "anomaly.json")
    progress = read(Path(entry["progress"]))
    life = lifetimes(entry["runtime_events"])
    log = (batch / "batch.log").read_text()
    checks = {
        "engine_returned": entry.get("batch_returned") is True,
        "only_expected_system_exit": entry.get("error_type") == "SystemExit" and entry.get("error") == "1" and outcome["returncode"] == 1,
        "traceback_has_no_cuda_error": not any(word in entry["traceback"].lower() for word in CUDA_WORDS),
        "all_runtime_resources_closed": life["each_target_loaded_once"],
        "known_guard_fallback": alarm["kind"] == "cuda_process_failure" and "Traceback (most recent call last):" not in log and '"device": "cuda"' in log[-8192:],
        "nvml_queries_succeeded": outcome["device_memory_samples"] > 0 and "error_kind" not in outcome,
        "gpu_diagnostic_query_succeeded": outcome["diagnostics"]["query_returncode"] == 0,
        "exact_expected_clip_outcomes": progress["clips"][CLIPS[0]]["status"] == "generated" and all(
            progress["clips"][clip].get("reason") == "court_side_ambiguous_margin" for clip in CLIPS[1:]),
    }
    if not all(checks.values()):
        raise ValueError(f"The original alarm is not the audited business-exit case: {checks}")
    paths = ["anomaly.json", "batch/entry.json", "batch.log", "terminal.json", "health/batch.json",
             "health/batch-threads.log", "health/batch-nvidia-smi-q.stdout.log", "health/batch-nvidia-smi-q.stderr.log"]
    result = {"classification": "guard_false_positive_expected_clip_failure_exit", "checks": checks,
              "actual_cuda_exception_observed": False, "original_alarm_preserved": True,
              "mechanism": "SystemExit(1) prints no traceback. C3 guard scanned prior normal cuda model events and flagged the expected CLI failure exit.",
              "raw_evidence_sha256": {name: sha(batch / name) for name in paths}, "reviewed_at_utc": utc()}
    write(batch / "guard-review.json", result)
    return result


def prepare(batch):
    verify(batch)
    review(batch)
    clip = clip_summary(batch / "datasets/batch", CLIPS[0])
    if not clip.get("marker_exists") or not clip["body_exports_present"] or clip["scene_players"] != 4:
        raise ValueError("The single reference requires the completed doubles body scene")
    root = batch / "reference"
    root.mkdir(exist_ok=False)
    (root / "input").symlink_to(batch / "input", target_is_directory=True)
    for name in ("input-manifest.json", "source-dataset.json"):
        shutil.copyfile(batch / name, root / name)
    (batch / "single").symlink_to(root / "single", target_is_directory=True)
    write(root / "preflight.json", {"batch_root": str(batch), "reference_clip": CLIPS[0],
          "measurement_commit": read(batch / "terminal.json")["commit"], "verified_scene": clip,
          "original_input_manifest_sha256": sha(batch / "input-manifest.json"),
          "data_root": str(batch), "output_root": str(root), "additional_batch_attempts": 0})
    print(json.dumps({"prepared": str(root), "reference_clip": CLIPS[0]}))


def run(batch, expected):
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("The reference must run through the shared queue")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if actual != expected:
        raise ValueError("Reference measurement HEAD changed")
    original = read(batch / "terminal.json")["commit"]
    subprocess.run(["git", "diff", "--exit-code", original, "HEAD", "--", "src"], cwd=REPO, check=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "knowledge/runs/run-video-batch-runtime-20261117",
                    "knowledge/runs/run-i1751-runtime-gpu-a5"], cwd=REPO, check=True)
    verify(batch)
    review(batch)
    root = batch / "reference"
    if (root / "terminal.json").exists():
        raise FileExistsError("No automatic reference retry")
    result = run_variant(root, "single", [sys.executable, "-B", "-u", str(Path(__file__).with_name("doubles_acceptance.py")),
        "worker", "--root", str(root), "--data-root", str(batch), "--phase", "single", "--clip-ids", CLIPS[0]], REPO)
    if (root / "anomaly.json").exists() and "diagnostics" not in result:
        class Exited:
            def __init__(self, outcome):
                self.pid = outcome.get("pid")
                self.returncode = outcome.get("returncode", 1)

            def poll(self):
                return self.returncode
        result["diagnostics"] = capture_diagnostics(root, "single", Exited(result))
    terminal = {"status": result["status"], "commit": actual, "batch_commit": original,
                "source_tree_unchanged": True, "reference_clip": CLIPS[0], "outcomes": {"single": result},
                "queue_job": os.environ["TENNIS_RUN_ID"], "finished_at_utc": utc()}
    write(root / "terminal.json", terminal)
    write(batch / "reference-terminal.json", terminal)
    verify(batch)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--expect-commit")
    args = parser.parse_args()
    batch = args.batch_root.resolve()
    if not batch.is_relative_to((REPO / "outputs").resolve()):
        raise ValueError("Reference outputs must stay in this worktree")
    if args.action == "prepare":
        prepare(batch)
    else:
        raise SystemExit(run(batch, args.expect_commit))
