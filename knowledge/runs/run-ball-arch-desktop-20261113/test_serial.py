"""Durable completion, rather than process return, permits the next condition."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import desktop  # noqa: E402
from desktop import retire_weights, run_ordered  # noqa: E402
from rebuild_dataset import sha, write  # noqa: E402
from service import notify  # noqa: E402


def test_failed_stage_stops_the_serial_order_and_preserves_the_failed_state():
    events, executed = [], []
    def execute(key):
        executed.append(key)
        raise OSError("Drive save failed")
    with pytest.raises(OSError, match="Drive save failed"):
        run_ordered(["T1", "T2", "C1"], lambda _: False, execute, lambda k, v: events.append((k,v)))
    assert executed == ["T1"]
    assert events == [("T1","running"),("T1","failed")]


def test_successful_process_without_verified_completion_does_not_start_the_next_stage():
    executed=[]
    with pytest.raises(RuntimeError,match="durable completion"):
        run_ordered(["T1","T2"],lambda _:False,executed.append,lambda *_:None)
    assert executed == ["T1"]


def test_durable_completed_stages_are_skipped_and_the_remaining_order_is_fixed():
    complete={"T1"}
    executed=[]
    def execute(key):
        executed.append(key)
        complete.add(key)
    run_ordered(["T1","T2","C1","C2","C3"],lambda key:key in complete,execute,lambda *_:None)
    assert executed == ["T2","C1","C2","C3"]


@pytest.mark.parametrize("terminal", ["done", "failed", None])
def test_terminal_event_distinguishes_scientific_job_failure_from_queue_process_exit(tmp_path, terminal):
    args = SimpleNamespace(root=tmp_path/"run", queue=tmp_path/"queue", job="job-a1", kind="qualification")
    if terminal:
        path = args.queue/terminal/(args.job+".job")
        path.parent.mkdir(parents=True)
        path.write_text("captured job")
    code = notify(args, queue_returncode=0)
    path = args.root/"notifications/job-a1.json"
    original = path.read_bytes()
    record = json.loads(original)
    assert code == (0 if terminal == "done" else 1)
    assert record["state"] == (terminal or "process_ended_without_terminal_receipt")
    notify(args)
    assert path.read_bytes() == original


def test_retirement_requires_the_exact_verified_tensor_and_publishes_its_proof_first(tmp_path, monkeypatch):
    directory = tmp_path/"production/T1-q06000-q12000"
    directory.mkdir(parents=True)
    checkpoint = directory/"checkpoint-update-00005500.pt"
    latest = directory/"checkpoint-update-00006000.pt"
    checkpoint.write_bytes(b"verified full state")
    latest.write_bytes(b"latest full state")
    write(directory/"latest.json", dict(update=6000, checkpoint=latest.name))
    published=[]
    monkeypatch.setenv("RCLONE_CONFIG", str(tmp_path/"private.conf"))
    def publish(root, path, config):
        assert checkpoint.exists()
        published.append(json.loads(path.read_text()))
    monkeypatch.setattr(desktop, "publish_index", publish)
    with pytest.raises(ValueError, match="exact verified snapshot"):
        retire_weights(tmp_path, directory, include_latest=False)
    assert checkpoint.exists() and not published
    proof=dict(receipt=dict(status="verified",generation="old-checkpoint",remote="durable"),
               readback=dict(status="verified"),checkpoint_state_equal=True,
               checkpoint_file=checkpoint.name,checkpoint_sha256=sha(checkpoint),
               execution_commit="pushed",updated_at=1)
    write(tmp_path/"persistence"/directory.name/"old.json", proof)
    write(tmp_path/"persistence"/directory.name/"completed.json",
          dict(proof,receipt=dict(status="verified",generation="completed",remote="durable"),updated_at=2))
    retire_weights(tmp_path,directory,include_latest=False)
    assert not checkpoint.exists() and latest.exists()
    assert published[0]["files"][0]["protected_generation"] == "completed"


def test_failed_retirement_ledger_publication_keeps_local_state(tmp_path, monkeypatch):
    directory=tmp_path/"production/T1"
    directory.mkdir(parents=True)
    checkpoint=directory/"checkpoint-update-00005500.pt"
    checkpoint.write_bytes(b"full state")
    write(directory/"latest.json",dict(update=5500,checkpoint=checkpoint.name))
    write(tmp_path/"persistence/T1/verified.json",
          dict(receipt=dict(status="verified",generation="verified",remote="durable"),
               readback=dict(status="verified"),checkpoint_state_equal=True,checkpoint_file=checkpoint.name,
               checkpoint_sha256=sha(checkpoint),updated_at=1,execution_commit="pushed"))
    monkeypatch.setenv("RCLONE_CONFIG",str(tmp_path/"private.conf"))
    def fail(*args):
        raise OSError("Drive unavailable")
    monkeypatch.setattr(desktop,"publish_index",fail)
    with pytest.raises(OSError,match="Drive unavailable"):
        retire_weights(tmp_path,directory,include_latest=True)
    assert checkpoint.read_bytes() == b"full state"
