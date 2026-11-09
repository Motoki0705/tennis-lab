"""Repair the wrapper's log-format assumption; validate existing renders only."""

import importlib.util
import json
from pathlib import Path

OPS = Path("outputs/tennis_scene/visualize/player_view/stabilization-comparison/operations/attempt2")
spec = importlib.util.spec_from_file_location("preview_worker_runtime", OPS / "worker_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
runtime.source()
registration = runtime.read(OPS / "queue-registration.json")
job = registration["job"]
stem = job[:-4]
queue = OPS / "training-queue"
states = [s for s in ["jobs", "running", "done", "failed", "cancelled"] if (queue / s / job).exists()]
assert states == ["done"], states
completion = runtime.read(queue / "control/completions" / (stem + ".json"))
assert completion["state"] == "done" and completion["job"] == job
state = dict(line.split("=", 1) for line in (queue / "state" / (stem + ".state")).read_text().splitlines())
assert state["state"] == "done" and state["resource"] == "all" and state["wait"] == "none", state
result = runtime.read(OPS / "queue-result.json")
assert result["states"] == ["done"] and result["serve_returncode"] == 0
repro = runtime.read(queue / "repro" / stem / "run.json")
operation = runtime.read(OPS / "gpu-operation.json")
assert repro["commit"] == runtime.REF
assert repro["provider"] == "codex" and repro["session"] == operation["agent_session"]
assert repro["resource"] == "all" and str(repro["issue"]) == "1776"
log = (queue / "logs" / (stem + ".log")).read_text()
terminal = [json.loads(line.removeprefix("COMPARISON_COMPLETE ")) for line in log.splitlines() if line.startswith("COMPARISON_COMPLETE ")]
assert len(terminal) == 1 and terminal[0]["phase"] == "preview", terminal
preview = runtime.OUTPUT / "preview"
images = [*preview.glob("renders/**/frames/*.png"), *preview.glob("player_*_contact.png")]
assert len(images) == 26
before = {str(p): runtime.digest(p) for p in images}
runtime.audit()
after = {str(p): runtime.digest(p) for p in images}
assert before == after, "Validation must not rewrite images"
runtime.write(OPS / "preview-repair-result.json", {
    "schema_version": 1,
    "validated_at": runtime.stamp(),
    "queue_job": job,
    "queue_states": states,
    "completion": completion,
    "renderer_completion": terminal[0],
    "source_ref": runtime.REF,
    "repair": "The queue log has no exit_code line. Use exact done state, completion receipt, serve returncode and renderer completion marker.",
    "images_before_after_sha256": before,
    "renderer_rerun": False,
    "additional_source_frames_rendered": 0,
    "audit": str(preview / "validation.json"),
})
print("EXISTING_PREVIEW_VALIDATION_COMPLETE renderer_rerun=false", flush=True)
