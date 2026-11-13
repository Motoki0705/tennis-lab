"""CPU contracts for the only metadata changes allowed during migration."""

import copy
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from desktop import (
    HISTORICAL_STORE,
    normalize_report_snapshot,
    relocate_manifest,
    relocate_recipe,
)


def test_path_relocation_keeps_teacher_ids_and_all_training_fields(tmp_path):
    recipe = dict(
        data=dict(
            snapshot=dict(directory=HISTORICAL_STORE, hashes={"index.npz": "original"}),
            summaries={"val": {"observed": 85473}},
        ),
        manifest_sha256="old_train",
        evaluation_manifest_sha256="old_val",
        model=dict(frames=64, heads=8),
        budget=dict(seed=42, schedule_offset=0),
        runtime=dict(precision="bf16", microbatch_windows=1),
        initialization=dict(sample_position=10000),
    )
    original = copy.deepcopy(recipe)
    actual = relocate_recipe(recipe, tmp_path, "new_train", "new_val")
    expected = copy.deepcopy(original)
    expected["data"]["snapshot"]["directory"] = str(tmp_path)
    expected["manifest_sha256"] = "new_train"
    expected["evaluation_manifest_sha256"] = "new_val"
    assert actual == expected
    assert recipe == original
    manifest = dict(
        ball_store=dict(directory=HISTORICAL_STORE, hashes={"index.npz": "abc"}),
        clips=[dict(clip_id="a", labels=["observed", "masked"])],
    )
    relocated = relocate_manifest(manifest, tmp_path)
    relocated["ball_store"]["directory"] = HISTORICAL_STORE
    assert relocated == manifest


def test_unexpected_snapshot_path_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="historical data root"):
        relocate_manifest(dict(ball_store=dict(directory="/different/data")), tmp_path)
    with pytest.raises(ValueError, match="resume snapshot root"):
        relocate_recipe(
            dict(data=dict(snapshot=dict(directory="/different/data"))),
            tmp_path,
            "a",
            "b",
        )


def test_report_normalization_retains_snapshot_hashes(tmp_path):
    report = dict(
        snapshot=dict(directory=str(tmp_path), hashes={"metadata.json": "bad"})
    )
    actual = normalize_report_snapshot(report, tmp_path)
    assert actual == dict(directory=HISTORICAL_STORE, hashes={"metadata.json": "bad"})
    assert report["snapshot"]["directory"] == str(tmp_path)
    with pytest.raises(ValueError, match="evaluation snapshot"):
        normalize_report_snapshot(
            dict(snapshot=dict(directory=str(Path("/unknown")))), tmp_path
        )


def test_deferred_registration_waits_for_real_runtime_exit(tmp_path, monkeypatch):
    import desktop
    import service

    for name in ("probe-acceptance", "science-audit"):
        desktop.write(tmp_path / (name + ".json"), dict(complete=True))
    waiting = threading.Event()
    original_write = desktop.write

    def write(path, value):
        original_write(path, value)
        if value.get("state") == "waiting_for_worker_runtime_exit":
            waiting.set()

    monkeypatch.setattr(desktop, "write", write)
    requests = []

    def run(command, **kwargs):
        requests.append((command, kwargs))
        return SimpleNamespace(
            returncode=0, stdout="queued: test-production.job\n", stderr=""
        )

    monkeypatch.setattr(service.subprocess, "run", run)
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
    )
    stat = Path(f"/proc/{child.pid}/stat").read_text()
    args = SimpleNamespace(
        root=tmp_path,
        queue=tmp_path / "queue",
        checkout=tmp_path / "checkout",
        source=tmp_path / "frozen",
        session="actual-session",
        rclone_config=tmp_path / "config",
        wait_pid=child.pid,
        wait_start_token=stat[stat.rindex(")") + 2 :].split()[19],
        old_probe_pgid=1_000_000_000,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(service.wait_and_register, args)
            assert waiting.wait(5)
            assert not requests
            child.stdin.write(b"x")
            child.stdin.flush()
            child.wait(timeout=5)
            future.result(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        child.stdin.close()
    command, options = requests[0]
    assert options["cwd"] == args.source
    assert command[command.index("--resource") + 1] == "all"
    assert command[-2:] == ["--idempotency-key", "i1899/desktop-20261112/production-a1"]
    assert (
        desktop.read(tmp_path / "production-registration.json")["job"]
        == "test-production"
    )


def test_deferred_registration_refuses_existing_previous_group(tmp_path):
    import service

    args = SimpleNamespace(
        root=tmp_path,
        wait_pid=1_000_000_000,
        wait_start_token="unused",
        old_probe_pgid=os.getpgrp(),
    )
    with pytest.raises(RuntimeError, match="PGID still exists; no lock bypass"):
        service.wait_and_register(args)


@pytest.mark.parametrize("state", ["done", "failed"])
def test_terminal_notice_uses_queue_state_and_never_retries_upload(
    tmp_path, monkeypatch, state
):
    import desktop
    import service

    queue = tmp_path / "queue"
    (queue / state).mkdir(parents=True)
    (queue / state / "actual.job").touch()
    args = SimpleNamespace(
        root=tmp_path,
        queue=queue,
        checkout=tmp_path / "checkout",
        job="actual",
        rclone_config=tmp_path / "config",
    )
    calls = []

    def failed_upload(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="connection failed")

    monkeypatch.setattr(service.subprocess, "run", failed_upload)
    with pytest.raises(RuntimeError, match="persistence failed"):
        service.notify(args)
    assert desktop.read(tmp_path / "notifications/actual.json")["state"] == state
    with pytest.raises(RuntimeError, match="no automatic retry"):
        service.notify(args)
    assert len(calls) == 1
