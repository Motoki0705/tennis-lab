"""CPU contracts for the experiment-specific queue and continuation adapter."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign
import train_model
from train_model import audit_optimizer


def example_plan():
    old = dict(
        stages=[
            dict(
                name=f"query-d{dim}-l{layers}-T64",
                model=dict(dim=dim, layers=layers, input_channels=5, frames=64),
                budget=dict(updates=6000, schedule_offset=0, schedule_updates=12000),
            )
            for dim, layers in campaign.CONDITIONS[:7]
        ]
    )
    parents = [
        dict(name=s["name"], checkpoint=s["name"] + ".pt", sha256="a" * 64, update=6000)
        for s in old["stages"]
    ]
    parents += [
        dict(
            name="common-dpt-T32", checkpoint="common.pt", sha256="b" * 64, update=10000
        )
    ]
    return dict(jobs=campaign.definitions(old, parents))


def test_nine_model_order_eleven_stages_and_exact_parent_boundaries():
    plan = example_plan()
    assert [j["model"] for j in plan["jobs"]] == [
        "d256-l4",
        "d384-l4",
        "d512-l4",
        "d768-l4",
        "d1024-l4",
        "d512-l6",
        "d512-l8",
        "d512-l12",
        "d512-l24",
    ]
    assert [j["updates"] for j in plan["jobs"]] == [6000] * 7 + [12000] * 2
    assert sum(j["updates"] for j in plan["jobs"]) == 66000
    assert [
        s["budget"]["schedule_offset"] for j in plan["jobs"] for s in j["stages"]
    ] == [6000] * 7 + [0, 6000, 0, 6000]
    for job in plan["jobs"][7:]:
        assert job["stages"][0]["parent"]["update"] == 10000
        assert job["stages"][1]["parent"]["previous_stage"] == job["stages"][0]["name"]
        assert all(
            s["model"]["frames"] == 64 and s["model"]["input_channels"] == 5
            for s in job["stages"]
        )


def test_real_queue_submission_is_idempotent_and_persists_all_nine(tmp_path):
    plan = example_plan()
    root, queue = tmp_path / "campaign", tmp_path / "queue"
    campaign.write(root / "plan.json", plan)
    cli = Path.cwd() / ".agents/skills/training-queue/scripts/training_queue.sh"
    first = campaign.enqueue_jobs(plan, "test-session", 1, queue, cli, root)
    second = campaign.enqueue_jobs(plan, "test-session", 1, queue, cli, root)
    assert [j["queue_id"] for j in first] == [j["queue_id"] for j in second]
    assert len(list((queue / "jobs").glob("*.job"))) == 9
    assert campaign.read(root / "queue-registration.json")["complete"]
    for job in first:
        text = (queue / "jobs" / job["queue_file"]).read_text()
        assert "# resource: all" in text and str(Path.cwd()) in text
    with pytest.raises(subprocess.CalledProcessError):
        campaign.enqueue_jobs(plan, "different-session", 1, queue, cli, root)


def test_recorded_failure_holds_successor_in_actual_queue(tmp_path):
    queue, root = tmp_path / "queue", tmp_path / "campaign"
    cli = Path.cwd() / ".agents/skills/training-queue/scripts/training_queue.sh"
    env = dict(
        os.environ,
        TRAINING_QUEUE_DIR=str(queue),
        TRAINING_QUEUE_LOCK_FILE=str(tmp_path / "gpu.lock"),
    )
    script = (
        "import sys;sys.path.insert(0,"
        + repr(str(Path(__file__).resolve().parent))
        + ");from pathlib import Path;from campaign import stop_failed;stop_failed(Path("
        + repr(str(root))
        + "),Path("
        + repr(str(queue))
        + '),"first","deliberate failure");sys.exit(17)'
    )
    first = shlex.join([sys.executable, "-c", script])
    sentinel = tmp_path / "second-started"
    second = shlex.join(["touch", str(sentinel)])
    for command in (first, second):
        subprocess.run(
            ["bash", str(cli), "add", command, "--resource", "all"],
            env=env,
            check=True,
            capture_output=True,
        )
    subprocess.run(
        ["bash", str(cli), "serve", "--idle-timeout", "0"],
        env=env,
        check=True,
        timeout=45,
        capture_output=True,
    )
    assert not sentinel.exists()
    assert len(list((queue / "failed").glob("*.job"))) == 1
    assert len(list((queue / "jobs").glob("*.job"))) == 1
    assert campaign.read(root / "FAILED.json")["successors_held"]
    assert len(list((root / "notifications").glob("failed-*.json"))) == 1


def test_named_adamw_audit_rejects_an_incompatible_moment():
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2)
    model.head = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(
        [dict(params=model.encoder.parameters()), dict(params=model.head.parameters())]
    )
    model.head(model.encoder(torch.ones(1, 2))).sum().backward()
    optimizer.step()
    names = [
        [n for n, _ in model.named_parameters() if n.startswith("encoder.")],
        [n for n, _ in model.named_parameters() if not n.startswith("encoder.")],
    ]
    import copy

    saved = dict(
        optimizer=copy.deepcopy(optimizer.state_dict()), optimizer_parameter_names=names
    )
    assert (
        audit_optimizer(saved, optimizer, model, set(dict(model.named_parameters())))[
            "inherited_states"
        ]
        == 4
    )
    saved["optimizer"]["state"][0]["exp_avg"][0, 0] += 1
    with pytest.raises(ValueError, match="AdamW state"):
        audit_optimizer(saved, optimizer, model, set(dict(model.named_parameters())))


def test_only_verified_new_run_local_cache_is_released(tmp_path, monkeypatch):
    import hashlib

    root = tmp_path / "s42-20261110-12k"
    directory = root / "query-stage"
    directory.mkdir(parents=True)
    remote = tmp_path / "drive"
    monkeypatch.setattr(train_model, "DRIVE", remote)
    monkeypatch.setattr(train_model, "RELATIVE", Path("campaign"))
    files = {}
    for kind in ("checkpoint", "weights"):
        name = f"{kind}-update-00000500.pt"
        (directory / name).write_bytes(b"verified cache")
        files[name] = dict(
            size_bytes=14, sha256=hashlib.sha256(b"verified cache").hexdigest()
        )
    (directory / "checkpoint-update-00001000.pt").write_bytes(b"latest kept")
    marker = (
        remote
        / "campaign"
        / directory.name
        / "checkpoint-00000500-test"
        / "_SNAPSHOT.json"
    )
    campaign.write(marker, dict(files=files))
    receipt = dict(generation="checkpoint-00000500-test", manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="unchanged Drive marker"):
        train_model.release_local_cache(
            root, directory, dict(update=500, receipt=receipt)
        )
    assert (directory / "checkpoint-update-00000500.pt").exists()
    receipt["manifest_sha256"] = campaign.sha(marker)
    train_model.release_local_cache(root, directory, dict(update=500, receipt=receipt))
    assert not (directory / "checkpoint-update-00000500.pt").exists()
    assert (directory / "checkpoint-update-00001000.pt").read_bytes() == b"latest kept"
    assert marker.is_file()
    with pytest.raises(ValueError, match="restricted to this new campaign"):
        train_model.release_local_cache(
            tmp_path / "s42-20261103-main", directory, dict(update=500, receipt=receipt)
        )
