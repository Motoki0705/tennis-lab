"""Keep up to N Codex CLI workers busy; one clip x one target per worker.

Run detached (setsid). Single instance via flock. Control: control.json (mode pilot|run|drain|stop,
max_parallel, model, effort, ...). State: state.json (see campaign_state.py). Notable events are
appended to logs/events.log (LAUNCH, DONE, CONTINUE, FAIL, QUOTA_PAUSE, QUOTA_RESUME, SLOW, ERROR,
IDLE, EXIT) so the orchestrator can watch it with a Monitor.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.tennis_scene.chat_annotation.runtime.validation import validate_annotation

from .campaign_state import (
    locked_state,
    log_event,
    next_candidates,
    processed_path,
    read_control,
)
from .common import (
    atomic_write_json,
    atomic_write_text,
    load_annotation,
    load_manifest,
    locate_video,
    utc_now,
)
from .configuration import file_sha256, json_object, paths
from .launcher import MODULE, worker_command, worker_environment

SLOW_SECONDS = 3 * 3600
KILL_SECONDS = 6 * 3600
MAX_FAILURES = 2  # non-quota failures per task before status "failed"
REVIEW_BATCH = 20  # REVIEW_BATCH event every N tasks waiting for orchestrator QA

children: dict[str, subprocess.Popen[bytes]] = {}


def now() -> datetime:
    return datetime.now(UTC)


def parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------------------- launch


def build_prompt(
    task_id: str,
    task: dict[str, Any],
    attempt: int,
    attempt_dir: Path,
    previous: Path | None,
    control: dict[str, Any],
) -> str:
    manifest = load_manifest(Path(task["manifest"]))
    frames = manifest.frames
    from fractions import Fraction

    duration = float(
        (frames[-1].clip_pts + frames[-1].duration_pts) * Fraction(manifest.time_base)
    )
    lines = [
        Path(__file__).with_name("WORKER.md").read_text(encoding="utf-8").rstrip(),
        "",
        "=== TASK ===",
        f"TASK_ID: {task_id}",
        f"TARGET: {task['target']}",
        f"CLIP_ID: {task['clip_id']}",
        f"ATTEMPT: {attempt}",
        f"ATTEMPT_DIR: {attempt_dir}",
        f"CT: {worker_command()}",
        f"PROTOCOL: {paths().project_root / 'src/tennis_scene/chat_annotation/resources/PROTOCOL.md'}",
        f"REQUEST: {paths().project_root / 'src/tennis_scene/chat_annotation/resources/ball_detection/REQUEST.txt'}",
        f"CONTRACTS: {paths().project_root / 'src/tennis_scene/chat_annotation/runtime/contracts.py'}",
        f"VIDEO: {locate_video(manifest)}",
        f"MANIFEST: {task['manifest']}",
        f"FRAMES: {len(frames)} (nominal fps {manifest.nominal_fps}, {duration:.2f} s, 0..{len(frames) - 1})",
        f"SOURCE_TITLE: {manifest.source.title}",
        f"CONTEXT_STOP: {control['context_stop_fraction']}",
        f"PREVIOUS_ATTEMPT: {previous if previous else 'none'}",
    ]
    review = task.get("parent_review")
    if review:
        lines += ["PARENT_REVIEW:", review]
    lines += [
        "",
        "まず `CT info ATTEMPT_DIR` と `CT init ATTEMPT_DIR` を実行してから始めてください。",
    ]
    return "\n".join(lines) + "\n"


def prompt_version() -> str:
    """Identifies the worker instructions + toolkit + codex launch flags an attempt ran with
    (for before/after comparisons; versions.json maps hashes to labels)."""
    import hashlib

    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.read_bytes())
    digest.update(Path(__file__).with_name("WORKER.md").read_bytes())
    return digest.hexdigest()[:12]


def latest_annotation(task: dict[str, Any]) -> Path | None:
    """Newest attempt annotation that parses (checkpoints are written atomically by ct)."""
    for record in reversed(task["attempts"]):
        attempt_dir = Path(record["dir"])
        for path in attempt_dir.glob("annotation_*.json"):
            load_annotation(
                path
            )  # corrupted latest checkpoints must not silently roll back
            return attempt_dir
    return None


def choose_variant(task_id: str, control: dict[str, Any]) -> tuple[str, list[str]]:
    """A/B arm for a task: stable across its attempts (hash of the task id), weights from
    control.json "variants" {name: {"weight": w, "codex_config": ["key=value", ...]}}.
    No "variants" key -> ("base", [])."""
    import hashlib

    variants = control.get("variants") or {}
    if not variants:
        return "base", []
    names = sorted(variants)
    total = sum(float(variants[n]["weight"]) for n in names)
    point = (
        int(hashlib.sha1(task_id.encode()).hexdigest()[:8], 16) / 0x100000000 * total
    )
    for name in names:
        point -= float(variants[name]["weight"])
        if point < 0:
            return name, list(variants[name].get("codex_config", []))
    return names[-1], list(variants[names[-1]].get("codex_config", []))


def launch(state: dict[str, Any], task_id: str, control: dict[str, Any]) -> None:
    task = state["tasks"][task_id]
    attempt = len(task["attempts"]) + 1
    attempt_dir = paths().tasks / task_id / f"attempt_{attempt:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    previous_dir = latest_annotation(task)
    previous_annotation = None
    if previous_dir is not None:
        previous_annotation = str(next(previous_dir.glob("annotation_*.json")))
    manifest = load_manifest(Path(task["manifest"]))
    task_json = {
        "task_id": task_id,
        "attempt": attempt,
        "clip_id": task["clip_id"],
        "target": task["target"],
        "video": str(locate_video(manifest)),
        "manifest": task["manifest"],
        "annotation": str(attempt_dir / f"annotation_{task['clip_id']}.json"),
        "attempt_dir": str(attempt_dir),
        "previous_attempt_dir": str(previous_dir) if previous_dir else None,
        "previous_annotation": previous_annotation,
        "context_stop_fraction": control["context_stop_fraction"],
        "launched_at": utc_now(),
        "campaign_dir": str(paths().campaign_dir),
        "unreview_frames": task.get("unreview_frames", []),
    }
    task_json["prompt_version"] = prompt_version()
    task_json["variant"], codex_config = choose_variant(task_id, control)
    atomic_write_json(attempt_dir / "task.json", task_json)
    atomic_write_json(
        attempt_dir / "launch.json",
        {
            "model": control["model"],
            "effort": control["effort"],
            "codex_config": codex_config,
        },
    )
    atomic_write_text(
        attempt_dir / "prompt.md",
        build_prompt(task_id, task, attempt, attempt_dir, previous_dir, control),
    )
    with (attempt_dir / "launcher.log").open("wb") as log:
        child = subprocess.Popen(
            [
                str(paths().python_executable),
                "-m",
                MODULE,
                "--campaign",
                str(paths().campaign_dir),
                "launch-worker",
                str(attempt_dir),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(attempt_dir),
            env=worker_environment(),
        )
    children[str(attempt_dir)] = child
    task["status"] = "running"
    task["attempts"].append(
        {
            "n": attempt,
            "dir": str(attempt_dir),
            "launched_at": task_json["launched_at"],
            "launcher_pid": child.pid,
            "model": control["model"],
            "effort": control["effort"],
            "prompt_version": task_json["prompt_version"],
            "variant": task_json["variant"],
        }
    )
    log_event(
        "LAUNCH",
        f"{task_id} attempt={attempt} previous={'yes' if previous_dir else 'no'} variant={task_json['variant']}",
    )


# --------------------------------------------------------------------------- finish


USAGE_RE = re.compile(r"usage limit", re.IGNORECASE)
# Server-side transient errors: not the task's fault, not counted as failures; short global backoff.
TRANSIENT_RE = re.compile(
    r"at capacity|overloaded|temporarily unavailable|service unavailable|internal server error|stream disconnected|timed out|rate limit|too many requests|\b429\b",
    re.IGNORECASE,
)
# Server refusing more work from us: drives the adaptive concurrency limit down.
REJECT_RE = re.compile(
    r"at capacity|rate limit|too many requests|\b429\b|concurren", re.IGNORECASE
)
AGAIN_RE = re.compile(
    r"try again at ([A-Z][a-z]{2} \d{1,2}(?:st|nd|rd|th)?, \d{4},? \d{1,2}:\d{2} ?[AP]M)"
)


def scan_events(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "thread_id": None,
        "failed": None,
        "usage": None,
        "errors": [],
        "completed": False,
    }
    if not path.exists():
        return info
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "thread.started":
                info["thread_id"] = event.get("thread_id")
            elif kind == "turn.completed":
                info["usage"] = event.get("usage")
                info["completed"] = True
            elif kind == "turn.failed":
                info["failed"] = (event.get("error") or {}).get("message")
            elif kind == "error":
                info["errors"].append(str(event.get("message"))[:500])
    return info


def rollout_rate_limits(
    thread_id: str | None,
) -> tuple[dict[str, Any] | None, float | None]:
    if not thread_id:
        return None, None
    home = Path(os.environ["CODEX_HOME"])
    for day in sorted((home / "sessions").glob("*/*/*"), reverse=True)[:3]:
        for path in day.glob(f"*{thread_id}*.jsonl"):
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 400_000))
                tail = handle.read().decode("utf-8", errors="ignore").splitlines()
            for line in reversed(tail):
                if '"token_count"' not in line:
                    continue
                try:
                    payload = json.loads(line).get("payload", {})
                except json.JSONDecodeError:
                    continue
                info = payload.get("info") or {}
                last = info.get("last_token_usage") or {}
                window = info.get("model_context_window")
                fraction = last["total_tokens"] / window if last and window else None
                return payload.get("rate_limits"), fraction
    return None, None


def quota_resume_time(message: str, rate_limits: dict[str, Any] | None) -> datetime:
    primary = (rate_limits or {}).get("primary") or {}
    if primary.get("resets_at"):
        return datetime.fromtimestamp(int(primary["resets_at"]), UTC) + timedelta(
            minutes=3
        )
    match = AGAIN_RE.search(message or "")
    if match:
        text = re.sub(r"(\d)(st|nd|rd|th)", r"\1", match.group(1)).replace(",", "")
        for fmt in ("%b %d %Y %I:%M %p", "%b %d %Y %I:%M%p"):
            try:
                local = datetime.strptime(
                    text, fmt
                ).astimezone()  # message uses local time
                return local.astimezone(UTC) + timedelta(minutes=3)
            except ValueError:
                continue
    return now() + timedelta(hours=1)


def finalize(state: dict[str, Any], task_id: str) -> None:
    task = state["tasks"][task_id]
    record = task["attempts"][-1]
    attempt_dir = Path(record["dir"])
    exit_code_path = attempt_dir / "exit_code"
    record["exit_code"] = (
        int(exit_code_path.read_text().strip()) if exit_code_path.exists() else None
    )
    record["ended_at"] = (
        (attempt_dir / "ended_at").read_text().strip()
        if (attempt_dir / "ended_at").exists()
        else utc_now()
    )
    events = scan_events(attempt_dir / "events.jsonl")
    rate_limits, fraction = rollout_rate_limits(events["thread_id"])
    record.update(
        {
            "thread_id": events["thread_id"],
            "usage": events["usage"],
            "context_fraction": fraction,
            "rate_limits_after": rate_limits,
        }
    )
    if rate_limits:
        state["quota"]["last_rate_limits"] = rate_limits
    message = events["failed"] or " ".join(events["errors"])
    last_message = (
        (attempt_dir / "last_message.md").read_text(encoding="utf-8", errors="ignore")
        if (attempt_dir / "last_message.md").exists()
        else ""
    )
    outcome_match = re.search(r"^OUTCOME:\s*(\w+)", last_message, re.MULTILINE)
    record["reported_outcome"] = outcome_match.group(1) if outcome_match else None
    annotation_path = attempt_dir / f"annotation_{task['clip_id']}.json"
    reviewed = frames = errors = None
    if annotation_path.exists():
        try:
            annotation = load_annotation(annotation_path)
            report = validate_annotation(
                annotation, load_manifest(Path(task["manifest"]))
            )
            reviewed, frames, errors = (
                report.reviewed_frames,
                report.target_frames,
                len(report.errors),
            )
        except Exception as error:
            errors = -1
            record["annotation_error"] = str(error)[:500]
    record.update({"reviewed": reviewed, "frames": frames, "validation_errors": errors})
    result_path = attempt_dir / "result.json"
    result = json_object(result_path) if result_path.exists() else None
    if result and (
        result.get("task_id") != task_id
        or result.get("attempt") != record["n"]
        or not annotation_path.exists()
        or result.get("annotation_sha256") != file_sha256(annotation_path)
    ):
        record["failure"] = "result identity or annotation hash mismatch"
        record["kind"] = "failure"
        task["status"] = "failed"
        log_event("FAIL", f"{task_id} result identity or annotation hash mismatch")
        return
    record["result_outcome"] = result.get("outcome") if result else None

    if message and USAGE_RE.search(message):
        until = quota_resume_time(message, rate_limits)
        state["quota"]["paused_until"] = until.isoformat(timespec="seconds")
        state["quota"]["reason"] = message[:300]
        record["kind"] = "quota"
        task["status"] = "continue" if reviewed else "pending"
        log_event(
            "QUOTA_PAUSE",
            f"{task_id} attempt={record['n']} until={state['quota']['paused_until']} reviewed={reviewed}/{frames}",
        )
        return
    if (
        message
        and TRANSIENT_RE.search(message)
        and not (attempt_dir / "result.json").exists()
    ):
        streak = int(state.get("transient_streak", 0)) + 1
        state["transient_streak"] = streak
        wait = min(60, 5 * 2 ** (streak - 1))
        until = now() + timedelta(minutes=wait)
        state["backoff_until"] = until.isoformat(timespec="seconds")
        record["kind"] = "transient"
        if REJECT_RE.search(message):
            state.setdefault("concurrency", {})["pending_rejection"] = message[:200]
        transient = sum(1 for r in task["attempts"] if r.get("kind") == "transient")
        task["status"] = (
            "failed" if transient > 6 else ("continue" if reviewed else "pending")
        )
        log_event(
            "TRANSIENT",
            f"{task_id} attempt={record['n']} streak={streak} backoff={wait}min reviewed={reviewed}/{frames}: {message[:120]!r}",
        )
        if streak == 5:
            log_event(
                "ERROR",
                f"transient server errors 5 in a row (latest: {message[:120]!r}); dispatcher keeps backing off",
            )
        return
    ok_annotation = errors == 0 and reviewed is not None
    outcome = record["result_outcome"]
    exit_ok = record["exit_code"] == 0
    if not exit_ok and events["completed"] and not events["failed"]:
        # codex completed its turn; only the launcher's bookkeeping after it died (e.g. the launcher
        # script was edited while running). The worker's own outcome is judged as usual below.
        record["exit_inferred"] = (
            f"turn.completed in events.jsonl (launcher exit_code={record['exit_code']})"
        )
        exit_ok = True
    note = " exit_inferred=turn.completed" if record.get("exit_inferred") else ""
    if (
        exit_ok
        and result
        and ok_annotation
        and outcome in ("completed", "partial")
        and reviewed == frames
    ):
        record["kind"] = "done"
        state["transient_streak"] = 0
        task["status"] = "review"
        log_event(
            "DONE",
            f"{task_id} attempt={record['n']} outcome={outcome} reviewed={reviewed}/{frames} ctx={fraction and round(fraction, 3)}{note}",
        )
        return
    if exit_ok and result and ok_annotation and outcome == "needs_continuation":
        record["kind"] = "continuation"
        state["transient_streak"] = 0
        task["status"] = "continue"
        log_event(
            "CONTINUE",
            f"{task_id} attempt={record['n']} reviewed={reviewed}/{frames} ctx={fraction and round(fraction, 3)}",
        )
        return
    record["kind"] = "failure"
    record["failure"] = (message or last_message[-500:] or "no result.json")[:800]
    failures = sum(1 for r in task["attempts"] if r.get("kind") == "failure")
    task["status"] = "failed" if failures >= MAX_FAILURES else "continue"
    log_event(
        "FAIL",
        f"{task_id} attempt={record['n']} exit={record['exit_code']} outcome={outcome} reviewed={reviewed}/{frames} -> {task['status']}: {record['failure'][:160]!r}",
    )


def running_state(task: dict[str, Any]) -> str:
    """'finished' | 'alive' | 'dead' for the latest attempt of a running task."""
    record = task["attempts"][-1]
    attempt_dir = Path(record["dir"])
    child = children.get(record["dir"])  # keyed by attempt dir
    if (attempt_dir / "exit_code").exists():
        return "finished"
    pid_file = attempt_dir / "pid"
    pid = (
        int(pid_file.read_text().strip())
        if pid_file.exists()
        else record.get("launcher_pid")
    )
    if pid and pid_alive(pid):
        return "alive"
    if child is not None and child.poll() is None:
        return "alive"
    return "dead"


# --------------------------------------------------------------------------- adaptive concurrency


def resources() -> dict[str, float]:
    available_mb = 0.0
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) / 1024
    load1, load5, _ = os.getloadavg()
    free_gb = shutil.disk_usage(paths().campaign_dir).free / 1e9
    return {
        "available_mb": available_mb,
        "load1": load1,
        "load5": load5,
        "cpus": os.cpu_count() or 1,
        "disk_free_gb": free_gb,
    }


def concurrency_limit(
    state: dict[str, Any], control: dict[str, Any], running: int
) -> int:
    """AIMD: +step every interval while the limit is saturated and resources allow; on a server
    rejection drop to 80% of what was running, cool down, and keep a lower ceiling for a while."""
    cfg = control.get("adaptive") or {}
    if not cfg.get("enabled"):
        return int(control["max_parallel"])
    lo, hi, step = (
        int(cfg.get("min", 10)),
        int(cfg.get("max", 50)),
        int(cfg.get("step", 5)),
    )
    conc = state.setdefault("concurrency", {})
    conc.setdefault("limit", int(control["max_parallel"]))
    conc.setdefault("last_change", utc_now())
    t = now()
    res = resources()
    conc["resources"] = {k: round(v, 2) for k, v in res.items()}
    rejection = conc.pop("pending_rejection", None)
    cooldown_until = parse_time(conc.get("cooldown_until"))
    if rejection:
        if cooldown_until is None or t >= cooldown_until:
            level = max(running, 1)
            conc["limit"] = max(lo, int(level * float(cfg.get("decrease_factor", 0.8))))
            conc["ceiling"] = max(lo, level - step)
            conc["ceiling_until"] = (
                t + timedelta(minutes=float(cfg.get("ceiling_minutes", 120)))
            ).isoformat(timespec="seconds")
            conc["cooldown_until"] = (
                t + timedelta(minutes=float(cfg.get("cooldown_minutes", 30)))
            ).isoformat(timespec="seconds")
            conc["last_change"] = t.isoformat(timespec="seconds")
            log_event(
                "CONCURRENCY_DOWN",
                f"limit={conc['limit']} rejected_at={level} reason={rejection[:120]!r}",
            )
        return int(conc["limit"])
    if (
        res["available_mb"] < float(cfg.get("min_available_mb", 6000)) / 2
        or res["load5"] > float(cfg.get("max_load_per_core", 1.5)) * 5 / 3 * res["cpus"]
    ):
        if (
            t - datetime.fromisoformat(conc["last_change"])
        ).total_seconds() >= 60 * float(cfg.get("interval_minutes", 5)) and conc[
            "limit"
        ] > lo:
            conc["limit"] = max(lo, conc["limit"] - step)
            conc["last_change"] = t.isoformat(timespec="seconds")
            log_event(
                "CONCURRENCY_DOWN",
                f"limit={conc['limit']} reason=local resources {conc['resources']}",
            )
        return int(conc["limit"])
    ceiling_until = parse_time(conc.get("ceiling_until"))
    ceiling = (
        hi
        if not ceiling_until or t >= ceiling_until
        else min(hi, int(conc.get("ceiling", hi)))
    )
    ready = (
        t - datetime.fromisoformat(conc["last_change"])
    ).total_seconds() >= 60 * float(cfg.get("interval_minutes", 5))
    cooled = cooldown_until is None or t >= cooldown_until
    roomy = (
        res["available_mb"] >= float(cfg.get("min_available_mb", 6000))
        and res["load5"] <= float(cfg.get("max_load_per_core", 1.5)) * res["cpus"]
        and res["disk_free_gb"] >= float(cfg.get("min_disk_free_gb", 20))
    )
    if (
        ready
        and cooled
        and roomy
        and running >= conc["limit"]
        and conc["limit"] < ceiling
    ):
        conc["limit"] = min(ceiling, conc["limit"] + step)
        conc["last_change"] = t.isoformat(timespec="seconds")
        conc.pop("hold_reported", None)
        log_event(
            "CONCURRENCY_UP" if conc["limit"] < hi else "CONCURRENCY_MAX",
            f"limit={conc['limit']} resources={conc['resources']}",
        )
    elif (
        ready
        and cooled
        and not roomy
        and running >= conc["limit"]
        and conc["limit"] < ceiling
        and not conc.get("hold_reported")
    ):
        conc["hold_reported"] = True  # once per hold episode
        log_event(
            "CONCURRENCY_HOLD",
            f"limit={conc['limit']} held by local resources {conc['resources']}",
        )
    return int(conc["limit"])


# --------------------------------------------------------------------------- loop


def tick() -> bool:
    """One scheduling pass. Returns False when the dispatcher should exit."""
    control = read_control()
    for key, child in list(children.items()):
        if (
            child.poll() is not None
        ):  # reap; exit_code file written by run_worker.sh is authoritative
            children.pop(key)
    with locked_state() as state:
        running = [tid for tid, t in state["tasks"].items() if t["status"] == "running"]
        for task_id in running:
            try:
                status = running_state(state["tasks"][task_id])
                record = state["tasks"][task_id]["attempts"][-1]
                if status == "finished":
                    finalize(state, task_id)
                elif status == "dead":
                    marker = Path(record["dir"]) / "exit_code"
                    if not marker.exists():
                        marker.write_text(
                            "-9\n"
                        )  # launcher vanished (reboot/kill) without recording an exit
                    finalize(state, task_id)
                else:
                    elapsed = (
                        now() - datetime.fromisoformat(record["launched_at"])
                    ).total_seconds()
                    if elapsed > control["slow_seconds"] and not record.get(
                        "slow_reported"
                    ):
                        record["slow_reported"] = True
                        log_event(
                            "SLOW",
                            f"{task_id} attempt={record['n']} running {elapsed / 3600:.1f} h",
                        )
                    if elapsed > control["timeout_seconds"] and not record.get(
                        "killed"
                    ):
                        pid = int((Path(record["dir"]) / "pid").read_text().strip())
                        record["killed"] = True
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        log_event(
                            "ERROR",
                            f"{task_id} attempt={record['n']} killed after {elapsed / 3600:.1f} h",
                        )
            except Exception:
                log_event(
                    "ERROR",
                    f"{task_id} finalize/monitor: {traceback.format_exc()[-600:]!r}",
                )
        # One notice per 20 finished-and-unadopted tasks, so the orchestrator batches its QA.
        review_count = sum(
            1 for t in state["tasks"].values() if t["status"] == "review"
        )
        noticed = state.get("review_notice_at", 0)
        if review_count < noticed:
            state["review_notice_at"] = noticed = review_count
        if review_count >= noticed + REVIEW_BATCH:
            state["review_notice_at"] = review_count
            log_event("REVIEW_BATCH", f"review={review_count}")
        # External adoption (e.g. ChatGPT route) removes pending work.
        for task_id, task in state["tasks"].items():
            if (
                task["status"] in ("pending",)
                and task.get("phase", 1) == 1
                and processed_path(task["target"], task["clip_id"]).exists()
            ):
                task["status"] = "skipped"
                log_event("SKIP", f"{task_id} processed file appeared externally")
        running_now = [t for t in state["tasks"].values() if t["status"] == "running"]
        paused_until = parse_time(state["quota"].get("paused_until"))
        if paused_until and now() >= paused_until:
            state["quota"]["paused_until"] = None
            log_event("QUOTA_RESUME", "pause window ended")
            paused_until = None
        mode = control["mode"]
        if mode == "stop" and not running_now:
            log_event("EXIT", "mode=stop and no running workers")
            return False
        backoff_until = parse_time(state.get("backoff_until"))
        if backoff_until and now() >= backoff_until:
            state["backoff_until"] = backoff_until = None
        # Optional reserve for other work sharing the weekly quota (control.json quota_stop_percent;
        # unset = use the quota up to the limit). Uses the latest rate_limits seen at finalize.
        reserve = control.get("quota_stop_percent")
        used = (
            (state["quota"].get("last_rate_limits") or {}).get("primary") or {}
        ).get("used_percent")
        reserve_hit = reserve is not None and used is not None and used >= reserve
        if reserve_hit and not state["quota"].get("reserve_reported"):
            state["quota"]["reserve_reported"] = True
            log_event(
                "QUOTA_RESERVE",
                f"weekly used {used}% >= quota_stop_percent {reserve}; no new launches",
            )
        elif not reserve_hit:
            state["quota"]["reserve_reported"] = False
        if (
            mode in ("pilot", "run")
            and paused_until is None
            and backoff_until is None
            and not reserve_hit
        ):
            slots = concurrency_limit(state, control, len(running_now)) - len(
                running_now
            )
            # Spread launches out: every new worker starts with a whole-clip check and sheets, and a
            # burst of 25+ starts drove load past 120 and MemAvailable to 5 GB (2026-10-01).
            slots = min(slots, int(control.get("max_launch_per_tick", 3)))
            busy_clips = {t["clip_id"] for t in running_now}
            for task_id in next_candidates(state, control):
                if slots <= 0:
                    break
                if state["tasks"][task_id]["clip_id"] in busy_clips:
                    continue
                try:
                    launch(state, task_id, control)
                    busy_clips.add(state["tasks"][task_id]["clip_id"])
                    slots -= 1
                except Exception:
                    log_event(
                        "ERROR", f"{task_id} launch: {traceback.format_exc()[-600:]!r}"
                    )
                    state["tasks"][task_id]["status"] = "failed"
        if not running_now and not [
            t for t in state["tasks"].values() if t["status"] == "running"
        ]:
            if not state.get("idle_reported"):
                state["idle_reported"] = True
                log_event(
                    "IDLE",
                    f"mode={mode} no running workers; paused_until={state['quota'].get('paused_until')}",
                )
        else:
            state["idle_reported"] = False
    (paths().logs / "dispatcher.heartbeat").write_text(utc_now() + "\n")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show launch candidates without writing files or starting Codex",
    )
    parser.add_argument("--once", action="store_true", help="Run one dispatch tick")
    parser.add_argument(
        "--exit-when-idle",
        action="store_true",
        help="Exit after all eligible workers finish; review remains a separate step",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        from .campaign_state import read_state

        state, control = read_state(), read_control()
        print(
            json.dumps(
                {
                    "mode": control["mode"],
                    "model": control["model"],
                    "candidates": next_candidates(state, control),
                    "max_parallel": control["max_parallel"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    paths().logs.mkdir(parents=True, exist_ok=True)
    with (paths().campaign_dir / ".dispatcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("dispatcher already running", file=sys.stderr)
            return 1
        if not paths().control.exists() or not paths().state.exists():
            raise ValueError("initialize the campaign before running the dispatcher")
        (paths().logs / "dispatcher.pid").write_text(f"{os.getpid()}\n")
        log_event("START", f"pid={os.getpid()}")
        while True:
            if not tick():
                return 0
            if args.once:
                return 0
            if args.exit_when_idle:
                from .campaign_state import read_state

                state, control = read_state(), read_control()
                if not any(
                    t["status"] == "running" for t in state["tasks"].values()
                ) and not next_candidates(state, control):
                    log_event(
                        "EXIT",
                        "no runnable tasks; completed results await orchestrator QA",
                    )
                    return 0
            time.sleep(float(read_control()["poll_seconds"]))


if __name__ == "__main__":
    raise SystemExit(main())
