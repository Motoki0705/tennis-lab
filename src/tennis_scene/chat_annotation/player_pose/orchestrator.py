from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .runner import review_worker
from .selection import load_campaign
from .storage import clip_root, lock, read_json, verify_record, write_json


def enqueue_clip(
    config: dict[str, Any], campaign: Path, index: int, attempt: int
) -> str:
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
            "generate-clip",
            "--campaign",
            str(campaign),
            "--index",
            str(index),
        ]
    )
    env = dict(os.environ, TRAINING_QUEUE_DIR=config["queue_dir"])
    result = subprocess.run(
        [
            "bash",
            config["queue_script"],
            "add",
            command,
            "--name",
            f"player-pose-v2-{index:05d}-a{attempt}",
            "--provider",
            "codex",
            "--session",
            config["session_id"],
            "--resource",
            "all",
        ],
        cwd=config["project_root"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    matched = re.search(r"queued: ([A-Za-z0-9_.-]+\.job)", result.stdout)
    if matched is None:
        raise RuntimeError(f"Queue did not return a job ID: {result.stdout}")
    return matched.group(1)


def queue_state(queue: Path, name: str) -> str:
    for state in ("done", "failed", "cancelled", "running", "jobs"):
        if (queue / state / name).exists():
            return state
    raise RuntimeError(f"Owned queue job disappeared: {name}")


def generate_through_queue(campaign: Path) -> None:
    config, plan, _ = load_campaign(campaign)
    failures: list[int] = []
    consecutive_failures = 0
    queue = Path(config["queue_dir"])
    entries = sorted(
        (c for c in plan["clips"] if c["selected"]),
        key=lambda c: (c["frame_count"], c["index"]),
    )
    with lock(campaign / "queue_coordinator.lock", blocking=False):
        for entry in entries:
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
    with (
        lock(campaign / "orchestrator.lock", blocking=False),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        reviewer = pool.submit(review_worker, campaign)
        try:
            generate_through_queue(campaign)
        except Exception as exc:
            write_json(
                campaign / "generation_status.json",
                {"status": "failed", "error": repr(exc)},
            )
            reviewer.result()
            raise
        reviewer.result()
