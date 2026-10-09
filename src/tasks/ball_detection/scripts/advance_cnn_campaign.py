"""Idempotent FIFO submission of two CNN candidates, then posttrain only the best."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.training.posttraining.comparison import (
    compare_pretraining,
    save_comparison,
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
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_detection.advance_cnn_campaign", fields=(
    BoundaryPathField("plan", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("campaign", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
))
CAMPAIGN_PATHS = NonHydraPathBoundary(name="ball_detection.cnn_campaign_plan", fields=(
    BoundaryPathField("code_root", PathRole.PROJECT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
    BoundaryPathField("queue_directory", PathRole.CACHE, PathDirection.OUTPUT, PathKind.DIRECTORY),
    BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("baseline_run", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
    BoundaryPathField("training_root", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    BoundaryPathField("smoke_root", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    BoundaryPathField("posttraining_run", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
))


def normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    required = {"schema", "code_root", "code_commit", "queue_directory", "manifest", "manifest_sha256", "baseline_run",
                "training_root", "smoke_root", "posttraining_run", "thread_id", "model_config_sha256"}
    if plan.get("schema") == "mdd_cnn_campaign.v1" and set(plan) == required:
        # The v1 schema fixes the original attempts and overlapping decode.
        plan = dict(plan, schema="mdd_cnn_campaign.v2", posttraining_prefetch_mode="overlap", candidates={
            v: dict(run_id=f"{v}-s42-v1", job_name=f"i1050-{v}-s42-u60000", smoke_id=v, prefetch_mode="overlap")
            for v in ("convnext_v2", "fasternet")})
    elif plan.get("schema") != "mdd_cnn_campaign.v2" or set(plan) != required | {"candidates", "posttraining_prefetch_mode"}:
        raise ValueError("Campaign plan must declare the complete v1 or v2 contract")
    if set(plan["candidates"]) != {"convnext_v2", "fasternet"}:
        raise ValueError("Campaign requires exactly the two additional CNNs")
    if plan["posttraining_prefetch_mode"] not in {"overlap", "serial"}:
        raise ValueError("Campaign must explicitly select the posttraining prefetch mode")
    for spec in plan["candidates"].values():
        if set(spec) != {"run_id", "job_name", "smoke_id", "prefetch_mode"} or spec["prefetch_mode"] not in {"overlap", "serial"}:
            raise ValueError("Candidate must declare its run/job/smoke identity and prefetch mode")
        for key in ("run_id", "job_name", "smoke_id"):
            if not isinstance(spec[key], str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec[key]) is None:
                raise ValueError("Candidate identifiers must be plain names, not paths")
    for key in ("run_id", "job_name", "smoke_id"):
        if len({spec[key] for spec in plan["candidates"].values()}) != 2:
            raise ValueError("Candidate identities must be distinct")
    return plan


def queue_once(plan: dict[str, Any], *, name: str, argv: list[str], issue: int) -> str:
    queue = Path(plan["queue_directory"])
    found = [p for state in ("jobs", "running", "done", "failed", "cancelled") for p in (queue / state).glob(f"*_{name}.job")]
    if len(found) > 1:
        raise ValueError(f"Ambiguous existing campaign jobs: {name}")
    if found:
        if found[0].parent.name in {"failed", "cancelled"}:
            raise RuntimeError(f"Campaign job needs inspection before a new attempt: {found[0]}")
        return found[0].name
    code = Path(plan["code_root"])
    env = dict(os.environ, TRAINING_QUEUE_DIR=str(queue))
    command = shlex.join(["env", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "TORCHINDUCTOR_COMPILE_THREADS=2",
                          "PYTHONUNBUFFERED=1", "CUDA_LOG_FILE=stderr", *argv])
    result = subprocess.run(["bash", str(code / ".agents/skills/training-queue/scripts/training_queue.sh"), "add", command,
        "--name", name, "--resource", "all", "--provider", "codex", "--session", plan["thread_id"], "--issue", str(issue)],
        cwd=code, env=env, check=True, capture_output=True, text=True)
    matched = re.search(r"queued: (\S+\.job)", result.stdout)
    if matched is None:
        raise ValueError("Queue acknowledgement uncertain; inspect queue before retrying")
    return matched.group(1)


def advance(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    code = Path(plan["code_root"])
    result: dict[str, Any] = dict(stage="pretraining", candidate_jobs={})
    for variant in ("convnext_v2", "fasternet"):
        spec = plan["candidates"][variant]
        candidate = Path(plan["training_root"]) / spec["run_id"]
        result["candidate_jobs"][variant] = queue_once(plan, name=spec["job_name"], issue=1050,
            argv=[str(code / ".venv/bin/python"), "-m", "src.tasks.ball_detection.scripts.train_cnn_candidate",
                  "--manifest", plan["manifest"], "--model-config", str(code / f"src/tasks/ball_detection/configs/model/mdd_dpt_{variant}.yaml"),
                  "--output", str(candidate), "--smoke-output", str(Path(plan["smoke_root"]) / spec["smoke_id"]),
                  "--prefetch-mode", spec["prefetch_mode"]])
    runs = {"residual": Path(plan["baseline_run"]),
            **{v: Path(plan["training_root"]) / spec["run_id"] for v, spec in plan["candidates"].items()}}
    pending = [name for name, run in runs.items() if not (run / "COMPLETED.json").exists()]
    if pending:
        result["pending"] = pending
        return result
    report = compare_pretraining(runs, Path(plan["manifest"]))
    save_comparison(report, root)
    result.update(stage="posttraining", winner=report["winner"], selected_run=report["selected_run"])
    output = Path(plan["posttraining_run"])
    if (output / "COMPLETED.json").exists():
        completed = json.loads((output / "COMPLETED.json").read_text())
        selected = json.loads((output / "best.json").read_text())
        parent = json.loads((Path(report["selected_run"]) / "best.json").read_text())
        stress = json.loads((output / "stress.json").read_text())
        if (completed.get("stage") != "query_posttraining" or completed["global_step"] != 60000
                or completed["pretrained_checkpoint"] != parent["sha256"] or completed["best_epoch"] != selected["epoch"]
                or Path(selected["checkpoint"]).name != selected["checkpoint"]
                or dual_sha256(output / selected["checkpoint"]) != selected["sha256"]
                or set(stress) != {"camera", "occlusion", "combined"}):
            raise ValueError("Posttraining completion artifacts disagree with the selected CNN")
        result["stage"] = "complete"
        result["posttraining"] = completed
        return result
    result["posttraining_job"] = queue_once(plan, name="i1049-best-cnn-posttrain-s42", issue=1049,
        argv=[str(code / ".venv/bin/python"), "-m", "src.tasks.ball_detection.scripts.train_best_mdd_posttraining",
              "--manifest", plan["manifest"], "--pretraining-run", report["selected_run"],
              "--augmentation-config", str(code / "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"),
              "--output", str(output), "--device", "cuda", "--precision", "bf16", "--epochs", "10", "--freeze-epochs", "1",
              "--windows-per-epoch", "6000", "--learning-rate", ".0001", "--encoder-lr-ratio", ".1", "--warmup-updates", "500",
              "--seed", "42", "--batch-size", "1", "--jpeg-decoder", "nvjpeg", "--pin-memory",
              "--num-workers", "8", "--prefetch-factor", "4", "--compile-mode", "default", "--stress-evaluation",
              *(["--image-prefetch"] if plan["posttraining_prefetch_mode"] == "overlap" else [])])
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    args = p.parse_args()
    if not args.plan.is_absolute():
        p.error("Campaign plan must be absolute")
    paths = PATH_BOUNDARY.validate(dict(plan=args.plan, campaign=args.plan.parent),
        resolver=resolver(PROJECT_ROOT, (args.plan,), (args.plan.parent,), args.plan.parent))
    args.plan, root = paths.declared("plan").path, paths.declared("campaign").path
    with (root / "advance.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = normalize_plan(json.loads(args.plan.read_text()))
        for key in ("code_root", "queue_directory", "manifest", "baseline_run", "training_root", "smoke_root", "posttraining_run"):
            if not Path(plan[key]).is_absolute():
                raise ValueError(f"Campaign {key} must be absolute")
        declared = {key: Path(plan[key]) for key in ("code_root", "queue_directory", "manifest", "baseline_run",
                                                    "training_root", "smoke_root", "posttraining_run")}
        validated = CAMPAIGN_PATHS.validate(declared, resolver=resolver(declared["code_root"].parent,
            (declared["manifest"], declared["baseline_run"]),
            (declared["training_root"], declared["smoke_root"], declared["posttraining_run"]), declared["queue_directory"].parent))
        for key in declared:
            plan[key] = str(validated.declared(key).path)
        code = Path(plan["code_root"])
        commit = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
        common = subprocess.check_output(["git", "-C", str(code), "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True).strip()
        if commit != plan["code_commit"] or Path(plan["queue_directory"]) != Path(common).parent / ".training_queue":
            raise ValueError("Campaign must use its frozen commit and shared main-repo queue")
        if dual_sha256(Path(plan["manifest"])) != plan["manifest_sha256"]:
            raise ValueError("Campaign manifest changed")
        for variant in ("convnext_v2", "fasternet"):
            config_path = code / f"src/tasks/ball_detection/configs/model/mdd_dpt_{variant}.yaml"
            if dual_sha256(config_path) != plan["model_config_sha256"][variant]:
                raise ValueError("Campaign model configuration changed")
        result = advance(plan, root)
        temporary = root / "progress.json.tmp"
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(root / "progress.json")
        if result["stage"] != "complete":
            code = Path(plan["code_root"])
            started = subprocess.run(["bash", str(code / ".agents/skills/training-queue/scripts/training_queue.sh"), "start"],
                cwd=code, env=dict(os.environ, TRAINING_QUEUE_DIR=plan["queue_directory"]), capture_output=True, text=True)
            if started.returncode and "worker already running" not in started.stderr:
                raise RuntimeError(f"Could not start training worker: {started.stderr}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
