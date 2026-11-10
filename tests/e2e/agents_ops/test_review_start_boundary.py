"""Observe the real common wrapper before its first durable runtime result."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
from pathlib import Path

from tests.e2e.agents_ops.test_agent_runtime import Harness
from tests.e2e.agents_ops.test_agent_runtime import h as h


def test_common_runner_can_exist_before_result_and_stop_cannot_confirm_it(
    h: Harness,
) -> None:
    ready, release = h.root / "save-ready.pipe", h.root / "save-release.pipe"
    for path in (ready, release):
        os.mkfifo(path)
    hook = h.root / "startup-hook"
    hook.mkdir()
    # agent_exec.sh really execs agent_runtime.py. Intercept only its first atomic
    # result save, before capture() can launch even the offline Codex fixture.
    (hook / "sitecustomize.py").write_text("""import os
original = os.replace
def before_first_result(source, target, *args, **kwargs):
    if os.fspath(target) == os.environ["REVIEW_RESULT_PATH"]:
        with open(os.environ["REVIEW_READY_PIPE"], "w") as signal:
            signal.write("ready\\n")
        with open(os.environ["REVIEW_RELEASE_PIPE"]) as signal:
            signal.read(1)
    return original(source, target, *args, **kwargs)
os.replace = before_first_result
""")
    ready_fd = os.open(ready, os.O_RDWR | os.O_NONBLOCK)
    proc = subprocess.Popen(
        h.cmd(),
        env={
            **h.env,
            "PYTHONPATH": str(hook),
            "REVIEW_RESULT_PATH": str(h.root / "initial/result.json"),
            "REVIEW_READY_PIPE": str(ready),
            "REVIEW_RELEASE_PIPE": str(release),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(ready_fd, selectors.EVENT_READ)
            assert selector.select(timeout=30), "common runner never reached first save"
            assert os.read(ready_fd, 100) == b"ready\n"
        assert proc.poll() is None
        assert (h.root / "initial/session.lock").exists()
        assert not (h.root / "initial/result.json").exists()
        assert not Path(h.env["FAKE_ARGS"]).exists()
        confirmation = h.root / "supervisor.json"
        confirmation.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": "not-yet-recorded",
                    "runner": None,
                    "process": None,
                    "cleanup_complete": True,
                    "evidence": "initial-save boundary fixture; no CLI launched",
                }
            )
        )
        stop = h.query("stop", extra=("--cleanup-confirmation", str(confirmation)))
        assert stop.returncode == 2 and "unavailable" in stop.stderr
        assert not (h.root / "initial/result.json").exists()
    finally:
        proc.kill()
        proc.communicate(timeout=15)
        os.close(ready_fd)
