"""Foreground full240 queue wrapper; canonical completion evidence only."""

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess

BASE = Path("outputs/tennis_scene/visualize/player_view/stabilization-comparison")
OPS = BASE / "operations/attempt2"
ARTIFACTS = BASE / "attempt2/full"
SCRATCH = BASE / "attempt2/full-render"
REF = "d133993421c206af28340f4ffe082c65b70fe7a7"
KEY = "issue1776/player-view-stabilization-20261109-a2/full-a2/attempt-1"


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    operation = read(OPS / "gpu-operation.json")
    assert operation["deadline_utc"] == "2026-11-09T15:22:50.337862Z"
    assert operation["command_deadline_utc"] == "2026-11-09T15:06:50.337862Z"
    assert read(OPS / "parent-preview-review.json")["technical_pass"] is True
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == REF
    helpers = read(OPS / "full-driver-source-digests.json")
    for name, expected in helpers.items():
        assert sha(OPS / name) == expected, name
    assert sha(BASE / "inputs/manifest.json") == operation["input_manifest_sha256"]
    inputs = read(BASE / "inputs/manifest.json")
    assert inputs["git_ref"] == REF and inputs["source_frame_interval"] == [0, 240]
    assert inputs["player_ids"] == [0, 1]
    assert [p["sigma_seconds"] for p in inputs["packages"]] == [0.0, 0.06, 0.08, 0.12]
    if (OPS / "full-queue-registration.json").exists():
        raise RuntimeError("Full already registered; inspect the existing job")
    if ARTIFACTS.exists() or SCRATCH.exists():
        raise FileExistsError("Full artifacts or scratch already exist")
    queue_dir = (OPS / "training-queue").resolve()
    active = [str(p) for s in ["jobs", "running"] for p in (queue_dir / s).glob("*.job")]
    assert not active, active
    os.environ["TRAINING_QUEUE_DIR"] = str(queue_dir)
    deadline = dt.datetime.fromisoformat(operation["command_deadline_utc"].replace("Z", "+00:00"))
    remaining = math.floor((deadline - dt.datetime.now(dt.timezone.utc)).total_seconds())
    timeout = remaining - 60  # 20s timeout escalation plus queue teardown/assertion margin.
    assert timeout > 60
    driver = [".venv/bin/python", str(OPS / "render_full_comparison.py"), "--phase", "full", "--inputs", str(BASE / "inputs"), "--output", str(SCRATCH), "--artifacts", str(ARTIFACTS), "--blender", "/content/blender/blender-5.2.1-linux-x64/blender"]
    command = shlex.join(["timeout", "--signal=TERM", "--kill-after=20s", f"{timeout}s", *driver])
    q = ".agents/skills/training-queue/scripts/training_queue.sh"
    add = ["bash", q, "add", command, "--name", "issue1776_full_a2", "--provider", "codex", "--session", operation["agent_session"], "--issue", "1776", "--resource", "all", "--idempotency-key", KEY]
    write(OPS / "full-queue-command.json", {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "command": command, "driver_argv": driver, "add_argv": add, "timeout_seconds": timeout, "timeout_kill_after_seconds": 20, "remaining_command_budget_seconds": remaining, "teardown_margin_seconds": 60, "command_deadline_utc": operation["command_deadline_utc"], "hard_deadline_utc": operation["deadline_utc"], "source_ref": REF, "helper_sha256": helpers, "queue_dir": str(queue_dir), "raw_scratch_persisted": False})
    receipt = subprocess.check_output(add, text=True)
    (OPS / "full-queue-add.txt").write_text(receipt)
    matches = re.findall(r"^queued: (\S+\.job)$", receipt, flags=re.MULTILINE)
    assert len(matches) == 1, receipt
    job = matches[0]
    write(OPS / "full-queue-registration.json", {"registered_at": dt.datetime.now(dt.timezone.utc).isoformat(), "job": job, "idempotency_key": KEY, "provider": "codex", "session": operation["agent_session"], "issue": 1776, "resource": "all", "source_ref": REF, "command": command, "timeout_seconds": timeout, "helper_sha256": helpers})
    print("FULL_QUEUED " + job, flush=True)
    serve = subprocess.run(["bash", q, "serve", "--idle-timeout", "1"])
    states = [s for s in ["jobs", "running", "done", "failed", "cancelled"] if (queue_dir / s / job).exists()]
    result = {"completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "job": job, "states": states, "serve_returncode": serve.returncode, "source_ref": REF, "completion_proof": "Canonical exact done state and completion receipt; no exit_code log line assumption"}
    write(OPS / "full-queue-result.json", result)
    assert serve.returncode == 0 and states == ["done"], result
    stem = job[:-4]
    completion = read(queue_dir / "control/completions" / (stem + ".json"))
    assert completion["state"] == "done" and completion["job"] == job
    state = dict(row.split("=", 1) for row in (queue_dir / "state" / (stem + ".state")).read_text().splitlines())
    assert state["state"] == "done" and state["wait"] == "none" and state["resource"] == "all"
    repro = read(queue_dir / "repro" / stem / "run.json")
    assert repro["commit"] == REF and repro["provider"] == "codex" and repro["session"] == operation["agent_session"]
    summary = read(ARTIFACTS / "manifest.json")
    assert summary["phase"] == "full" and summary["source_frames"] == list(range(240))
    assert summary["git_ref"] == REF and summary["archived_source_pngs"] == 1920
    assert summary["driver_sha256"] == helpers["render_full_comparison.py"]
    assert summary["archive_helper_sha256"] == helpers["archive_render_frames.py"]
    result.update(completion=completion, full_manifest_sha256=sha(ARTIFACTS / "manifest.json"))
    write(OPS / "full-queue-result.json", result)
    print("FULL_QUEUE_COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
