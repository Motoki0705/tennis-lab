from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

from .dataset import update_entry
from .runner import review_worker
from .selection import load_campaign
from .storage import clip_root, lock, read_json, verify_record, write_json


def enqueue_clip(
    config: dict[str, Any],
    campaign: Path,
    index: int,
    attempt: int,
    stage: str = "tracking",
) -> str:
    if stage not in ("tracking", "pose"):
        raise ValueError("Unknown GPU generation stage")
    token = hashlib.sha256(str(campaign.resolve()).encode()).hexdigest()[:16]
    name = f"i988-{token}-{stage}-{index:05d}-a{attempt}"
    extension = str(Path(config["assets"]["dino_extension"]).parent)
    command = shlex.join(
        [
            "env",
            f"PYTHONPATH={extension}:{config['project_root']}",
            "OMP_NUM_THREADS=2",
            "OPENBLAS_NUM_THREADS=2",
            "MKL_NUM_THREADS=2",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            config["python"],
            "-u",
            "-m",
            "src.tennis_scene.chat_annotation.player_pose",
            "generate-clip" if stage == "tracking" else "pose-clip",
            "--campaign",
            str(campaign),
            "--index",
            str(index),
        ]
    )
    env = dict(os.environ, TRAINING_QUEUE_DIR=config["queue_dir"])
    with lock(clip_root(campaign, index) / f"{stage}-submit.lock") as fd:
        # Recover the job even if the coordinator died after atomic queue admission
        # but before writing its attempt receipt. The child retains the lock.
        matches = {
            p.name
            for state in ("jobs", "running", "done", "failed", "cancelled")
            for p in (Path(config["queue_dir"]) / state).glob(f"*_{name}.job")
        }
        if len(matches) > 1:
            raise RuntimeError(f"Duplicate owned queue jobs: {sorted(matches)}")
        if matches:
            return matches.pop()
        result = subprocess.run(
            [
                "bash",
                config["queue_script"],
                "add",
                command,
                "--name",
                name,
                "--provider",
                "codex",
                "--session",
                config["session_id"],
                "--issue",
                "988",
                "--resource",
                "all",
            ],
            cwd=config["project_root"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
            pass_fds=(fd,),
        )
    matched = re.search(r"queued: ([A-Za-z0-9_.-]+\.job)", result.stdout)
    if matched is None:
        raise RuntimeError(f"Queue did not return a job ID: {result.stdout}")
    return matched.group(1)


def queue_state(queue: Path, name: str) -> str:
    for state in ("jobs", "running", "done", "failed", "cancelled"):
        if (queue / state / name).exists():
            return state
    raise RuntimeError(f"Owned queue job disappeared: {name}")


def generate_through_queue(campaign: Path, stop: Event | None = None) -> None:
    config, plan, _ = load_campaign(campaign)
    failures: list[int] = []
    consecutive_failures = 0
    queue = Path(config["queue_dir"])
    entries = sorted(
        (
            c
            for c in plan["clips"]
            if c["selected"] and c.get("action", "generate") == "generate"
        ),
        key=lambda c: (c["frame_count"], c["index"]),
    )
    with lock(campaign / "queue_coordinator.lock", blocking=False):
        for entry in entries:
            if stop is not None and stop.is_set():
                raise RuntimeError(
                    "Tracking coordinator interrupted after another stage failed"
                )
            index = entry["index"]
            root = clip_root(campaign, index)
            if (root / "generation.json").exists():
                verify_record(root / "generation.json")
                continue
            succeeded = False
            for attempt in range(config["generation_attempts"]):
                record = root / f"queue-attempt-{attempt:02d}.json"
                if record.exists():
                    job = read_json(record)["job"]
                else:
                    job = enqueue_clip(config, campaign, index, attempt)
                    write_json(record, {"job": job, "index": index, "attempt": attempt})
                while True:
                    if stop is not None and stop.is_set():
                        raise RuntimeError("Tracking job remains owned for resume")
                    state = queue_state(queue, job)
                    write_json(
                        campaign / "generation_status.json",
                        {
                            "status": "running",
                            "current_clip": index,
                            "clip_id": entry["clip_id"],
                            "queue_job": job,
                            "queue_state": state,
                            "failed": failures,
                        },
                    )
                    if state in ("done", "failed", "cancelled"):
                        break
                    time.sleep(15)
                if state == "cancelled":
                    raise RuntimeError(f"Owned pose job was cancelled: {job}")
                if state == "done":
                    verify_record(root / "generation.json")
                    succeeded = True
                    break
                write_json(
                    root / "failure.json",
                    {
                        "status": "failed",
                        "job": job,
                        "attempt": attempt,
                        "last_progress": read_json(root / "progress.json")
                        if (root / "progress.json").exists()
                        else None,
                    },
                )
                time.sleep(10)
            if succeeded:
                consecutive_failures = 0
                (root / "failure.json").unlink(missing_ok=True)
            else:
                failures.append(index)
                consecutive_failures += 1
            if consecutive_failures >= 3:
                raise RuntimeError(
                    "Three consecutive clip failures; cached chunks are retained"
                )
        write_json(
            campaign / "generation_status.json",
            {"status": "complete" if not failures else "partial", "failed": failures},
        )


def orchestrate(campaign: Path) -> None:
    from .expansion import reuse_approved
    from .reporting import save_metrics

    stop = Event()

    def supervise(stage: str, run: Callable[[Path, Event | None], None]) -> None:
        try:
            run(campaign, stop)
        except Exception as exc:
            write_json(
                campaign / f"{stage}_status.json",
                {"status": "failed", "error": repr(exc)},
            )
            stop.set()
            raise

    with (
        lock(campaign / "orchestrator.lock", blocking=False),
        ThreadPoolExecutor(max_workers=3) as pool,
    ):
        reuse_approved(campaign)
        # Clear terminal coordinator state before starting consumers on resume.
        write_json(campaign / "generation_status.json", {"status": "running"})
        write_json(campaign / "review_status.json", {"status": "running"})
        tracking = pool.submit(supervise, "generation", generate_through_queue)
        reviewer = pool.submit(supervise, "review", review_worker)
        poser = pool.submit(supervise, "pose", pose_worker)
        try:
            tracking.result()
            reviewer.result()
            poser.result()
        finally:
            config, plan, _ = load_campaign(campaign)
            save_metrics(campaign, config, plan)


def pose_worker(campaign: Path, stop: Event | None = None) -> None:
    from .publication import publish
    from .reporting import save_metrics

    config, plan, _ = load_campaign(campaign)
    queue = Path(config["queue_dir"])
    entries = [c for c in plan["clips"] if c["selected"] and c["action"] == "generate"]
    failures: list[int] = []
    consecutive = 0
    with lock(campaign / "pose_worker.lock", blocking=False):
        while True:
            if stop is not None and stop.is_set():
                raise RuntimeError(
                    "Pose coordinator interrupted after another stage failed"
                )
            for entry in entries:
                index = entry["index"]
                root = clip_root(campaign, index)
                if (root / "publication.json").exists():
                    verify_record(root / "publication.json")
                    continue
                if not (root / "review.json").exists() or index in failures:
                    continue
                if (root / "pose.json").exists():
                    publish(campaign, index)
                    continue
                succeeded = False
                for attempt in range(
                    config.get("pose_attempts", config["generation_attempts"])
                ):
                    record = root / f"pose-queue-attempt-{attempt:02d}.json"
                    if record.exists():
                        job = read_json(record)["job"]
                    else:
                        job = enqueue_clip(config, campaign, index, attempt, "pose")
                        write_json(
                            record,
                            {
                                "job": job,
                                "index": index,
                                "attempt": attempt,
                                "stage": "pose",
                            },
                        )
                    update_entry(
                        Path(config["dataset"]), index, pose_status="pose_pending"
                    )
                    while True:
                        if stop is not None and stop.is_set():
                            raise RuntimeError("Pose job remains owned for resume")
                        state = queue_state(queue, job)
                        write_json(
                            campaign / "pose_status.json",
                            {
                                "status": "running",
                                "current_clip": index,
                                "queue_job": job,
                                "queue_state": state,
                                "failed": failures,
                            },
                        )
                        save_metrics(campaign, config, plan)
                        if state in ("done", "failed", "cancelled"):
                            break
                        time.sleep(15)
                    if state == "cancelled":
                        raise RuntimeError(f"Owned pose job was cancelled: {job}")
                    # Publication can be recovered on the CPU after a completed GPU
                    # receipt even when the queue process failed during publication.
                    if (root / "pose.json").exists():
                        publish(campaign, index)
                        succeeded = True
                        break
                    if state == "done":
                        raise RuntimeError("Pose job completed without a pose receipt")
                if succeeded:
                    consecutive = 0
                else:
                    failures.append(index)
                    consecutive += 1
                    update_entry(
                        Path(config["dataset"]), index, pose_status="pose_failed"
                    )
                if consecutive >= 3:
                    write_json(
                        campaign / "pose_status.json",
                        {"status": "failed", "failed": failures},
                    )
                    raise RuntimeError("Three consecutive selected-pose failures")
            terminal = read_json(campaign / "review_status.json")["status"] in (
                "complete",
                "partial",
                "failed",
            )
            save_metrics(campaign, config, plan)
            pending = [
                c
                for c in entries
                if c["index"] not in failures
                and (clip_root(campaign, c["index"]) / "review.json").exists()
                and not (clip_root(campaign, c["index"]) / "publication.json").exists()
            ]
            if terminal and not pending:
                write_json(
                    campaign / "pose_status.json",
                    {
                        "status": "partial" if failures else "complete",
                        "failed": failures,
                    },
                )
                return
            time.sleep(15)
