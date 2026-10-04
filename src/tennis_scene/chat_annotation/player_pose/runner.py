from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

from .evidence import prepare
from .reviews import REVIEW_SCHEMA, accept_review
from .selection import load_campaign
from .storage import clip_root, digest, lock, read_json, verify_record, write_json


def codex_command(
    config: dict[str, Any], attempt: Path, images: list[Path]
) -> list[str]:
    return [
        config["codex_binary"],
        "--disable",
        "daemon_auto_start",
        "--disable",
        "multi_agent",
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "-m",
        config["model"],
        "-c",
        f"model_reasoning_effort={json.dumps(config['effort'])}",
        "-c",
        'approval_policy="never"',
        "-s",
        "read-only",
        "-C",
        str(attempt),
        "--skip-git-repo-check",
        "--color",
        "never",
        "--json",
        "--output-schema",
        str(attempt / "schema.json"),
        "-o",
        str(attempt / "result.json"),
        *[argument for path in images for argument in ("--image", str(path))],
        "--",
        "-",
    ]


def _invoke(
    config: dict[str, Any], attempt: Path, images: list[Path], prompt: str
) -> int:
    env = dict(os.environ, CODEX_HOME=config["codex_home"])
    env.pop("CODEX_THREAD_ID", None)
    env.pop("CODEX_SESSION_ID", None)
    started = time.monotonic()
    command = codex_command(config, attempt, images)
    write_json(
        attempt / "launch.json",
        {
            "argv": command,
            "model": config["model"],
            "images": {str(p): digest(p) for p in images},
            "started_at": time.time(),
        },
    )
    (attempt / "prompt.txt").write_text(prompt)
    with (
        (attempt / "events.jsonl").open("w") as events,
        (attempt / "stderr.log").open("w") as errors,
    ):
        child = subprocess.Popen(
            command,
            cwd=attempt,
            env=env,
            stdin=subprocess.PIPE,
            stdout=events,
            stderr=errors,
            start_new_session=True,
        )
        try:
            child.communicate(prompt.encode(), timeout=config["review_timeout_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            write_json(
                attempt / "exit.json",
                {
                    "exit_code": child.returncode,
                    "timed_out": True,
                    "seconds": time.monotonic() - started,
                },
            )
            return 124
    write_json(
        attempt / "exit.json",
        {
            "exit_code": child.returncode,
            "timed_out": False,
            "seconds": time.monotonic() - started,
        },
    )
    return int(child.returncode)


def review_clip(campaign: Path, index: int) -> dict[str, Any]:
    config, _, _ = load_campaign(campaign)
    root = clip_root(campaign, index)
    with lock(root / "review.lock", blocking=False):
        if (root / "review.json").exists():
            receipt = verify_record(root / "review.json")
            return accept_review(
                campaign, index, root / "decision.json", receipt["required_sheets"]
            )
        status_path = root / "review_status.json"
        if status_path.exists() and read_json(status_path)["status"] in (
            "approved",
            "needs_review",
        ):
            saved_status: dict[str, Any] = read_json(status_path)
            return saved_status
        verify_record(root / "generation.json")
        evidence = root / "evidence"
        packet = prepare(campaign, index, evidence)
        images = [evidence / name for name in packet["required_frame_sheets"]]
        instruction = Path(__file__).with_name("WORKER.md").read_text()
        previous_error = ""
        attempts = root / "review"
        attempts.mkdir(exist_ok=True)
        existing = len(list(attempts.glob("attempt-*")))
        for number in range(existing, existing + config["review_attempts"]):
            attempt = attempts / f"attempt-{number:03d}"
            attempt.mkdir()
            write_json(attempt / "schema.json", REVIEW_SCHEMA)
            prompt = (
                instruction
                + "\n\n"
                + f"packet: {evidence / 'packet.json'}\n"
                + json.dumps(packet, ensure_ascii=False)
                + "\n"
                + f"全frame一覧 {len(images)}枚を添付しました。人物crop一覧の絶対pathは {evidence} にあります。\n"
                + f"前回の検証結果: {previous_error}\n"
            )
            code = _invoke(config, attempt, images, prompt)
            try:
                if code != 0:
                    raise RuntimeError(f"Codex exited {code}")
                result = accept_review(
                    campaign,
                    index,
                    attempt / "result.json",
                    packet["required_frame_sheets"],
                )
                write_json(attempt / "validation.json", result)
                return result
            except (
                ValueError,
                RuntimeError,
                FileNotFoundError,
                json.JSONDecodeError,
            ) as exc:
                previous_error = str(exc)
                write_json(
                    attempt / "validation.json",
                    {"status": "failed", "error": previous_error},
                )
                log = (attempt / "stderr.log").read_text() + (
                    attempt / "events.jsonl"
                ).read_text()
                if any(
                    s in log.lower()
                    for s in (
                        "usage limit",
                        "rate limit",
                        "rate_limit",
                        "usage_limit",
                        "quota",
                    )
                ):
                    write_json(
                        campaign / "review_pause.json",
                        {"until": time.time() + 1800, "reason": "model_usage_limit"},
                    )
                    return {"status": "deferred_usage_limit"}
        result = {
            "status": "review_failed",
            "error": previous_error,
            "reason": "review_attempts_exhausted",
        }
        write_json(status_path, result)
        return result


def review_worker(campaign: Path, stop: Event | None = None) -> None:
    config, plan, _ = load_campaign(campaign)
    with lock(campaign / "review_worker.lock", blocking=False):
        selected = [
            c
            for c in plan["clips"]
            if c["selected"] and c.get("action", "generate") == "generate"
        ]
        selected.sort(key=lambda c: (c["frame_count"], c["index"]))
        with ThreadPoolExecutor(max_workers=config["review_parallel"]) as pool:
            active: dict[int, Any] = {}
            while True:
                if stop is not None and stop.is_set():
                    raise RuntimeError(
                        "Review coordinator interrupted after another stage failed"
                    )
                for index, future in list(active.items()):
                    if not future.done():
                        continue
                    try:
                        future.result()
                    except Exception as exc:
                        write_json(
                            clip_root(campaign, index) / "review_status.json",
                            {"status": "review_failed", "error": repr(exc)},
                        )
                    del active[index]
                pause_path = campaign / "review_pause.json"
                paused = (
                    pause_path.exists() and read_json(pause_path)["until"] > time.time()
                )
                if not paused:
                    for entry in selected:
                        if len(active) >= config["review_parallel"]:
                            break
                        index = entry["index"]
                        root = clip_root(campaign, index)
                        if index in active or not (root / "generation.json").exists():
                            continue
                        if (root / "review_status.json").exists() and read_json(
                            root / "review_status.json"
                        )["status"] in ("approved", "needs_review", "review_failed"):
                            continue
                        active[index] = pool.submit(review_clip, campaign, index)
                generation = read_json(campaign / "generation_status.json")["status"]
                completed: list[int] = []
                unresolved: list[int] = []
                for entry in selected:
                    p = clip_root(campaign, entry["index"]) / "review_status.json"
                    if p.exists():
                        (
                            completed
                            if read_json(p)["status"] == "approved"
                            else unresolved
                        ).append(entry["index"])
                pending_ready = [
                    c
                    for c in selected
                    if (clip_root(campaign, c["index"]) / "generation.json").exists()
                    and c["index"] not in completed + unresolved
                ]
                terminal = (
                    generation in ("complete", "partial", "failed")
                    and not active
                    and not pending_ready
                )
                status = (
                    "complete"
                    if terminal and len(completed) == len(selected)
                    else "partial"
                    if terminal
                    else "paused_usage"
                    if paused
                    else "running"
                )
                write_json(
                    campaign / "review_status.json",
                    {
                        "status": status,
                        "approved": completed,
                        "needs_review": unresolved,
                        "active": list(active),
                        "selected": len(selected),
                        "generation_status": generation,
                    },
                )
                if terminal:
                    break
                time.sleep(15)


def status(campaign: Path) -> dict[str, Any]:
    config, plan, _ = load_campaign(campaign)
    from .reporting import campaign_metrics

    return {
        **campaign_metrics(campaign, config, plan),
        "selected_clips": plan["selected_clips"],
        "selected_frames": plan["selected_frames"],
        "skipped_clips": len(plan["clips"]) - plan["selected_clips"],
        "generation": read_json(campaign / "generation_status.json"),
        "review": read_json(campaign / "review_status.json"),
        "pose": read_json(campaign / "pose_status.json"),
        "dataset": config["dataset"],
    }
