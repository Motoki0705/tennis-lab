"""Recovery must keep committed state and reject a silent rollback or reset."""

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils.checksum import dual_sha256

BUNDLE = Path("knowledge/runs/run-ball-sequential-length-20261025")


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        f"sequential_{name}", BUNDLE / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipt(directory, generation, **changes):
    saved = dict(
        schema="verified_artifact_snapshot.v1",
        status="verified",
        generation=generation,
        manifest_sha256="a" * 64,
    )
    saved.update(changes)
    path = directory / f"{generation}.json"
    path.write_text(json.dumps(saved))
    return path


def test_select_committed_completion_and_reject_corrupt_newest(tmp_path):
    recover = load_script("recover")
    receipt(tmp_path, "checkpoint-00002500-aaaaaaaa")
    receipt(tmp_path, "checkpoint-00003000-bbbbbbbb")
    completed = receipt(tmp_path, "completed-00003000-cccccccc")
    assert recover.select_snapshot(tmp_path)[0] == completed
    receipt(tmp_path, "checkpoint-00003500-dddddddd", status="partial")
    with pytest.raises(ValueError, match="Invalid committed"):
        recover.select_snapshot(tmp_path)


def test_ambiguous_generations_do_not_choose_arbitrarily(tmp_path):
    recover = load_script("recover")
    receipt(tmp_path, "checkpoint-00003000-aaaaaaaa")
    receipt(tmp_path, "checkpoint-00003000-bbbbbbbb")
    with pytest.raises(ValueError, match="Ambiguous"):
        recover.select_snapshot(tmp_path)


def test_copy_reuses_identical_bytes_and_preserves_different_local_file(tmp_path):
    recover = load_script("recover")
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"committed metadata")
    recover.copy_exact(source, target)
    recover.copy_exact(source, target)
    target.write_bytes(b"new local work")
    with pytest.raises(ValueError, match="Existing local metadata differs"):
        recover.copy_exact(source, target)
    assert target.read_bytes() == b"new local work"


def test_repeated_restore_rejects_a_stage_that_has_advanced(tmp_path):
    recover = load_script("recover")
    stage = tmp_path / "stage"
    stage.mkdir()
    log = stage / "train.jsonl"
    log.write_bytes(b"original")
    marker = tmp_path / "_SNAPSHOT.json"
    marker.write_text(
        json.dumps(
            dict(
                schema="verified_artifact_snapshot.v1",
                generation="checkpoint-00003000-aaaaaaaa",
                files={"train.jsonl": dict(size_bytes=8, sha256=dual_sha256(log))},
            )
        )
    )
    saved = dict(
        manifest_sha256=dual_sha256(marker), generation="checkpoint-00003000-aaaaaaaa"
    )
    recover.verify_existing(stage, marker, saved)
    log.write_bytes(b"new updates")
    with pytest.raises(ValueError, match="Existing local stage differs"):
        recover.verify_existing(stage, marker, saved)
    assert log.read_bytes() == b"new updates"


@pytest.mark.parametrize("changed", [None, "source", "budget", "checkpoint", "download"])
def test_campaign_resumes_without_lost_benchmark_and_keeps_stage_budgets(
    tmp_path, monkeypatch, changed
):
    controller = load_script("run")
    code = dict(source_sha256={"train.py": "a" * 64}, torch_version="original")
    monkeypatch.setattr(controller, "source_identity", lambda: code)
    calls = []

    class Store:
        def __init__(self, *args, **kwargs):
            pass

        def publish_file(self, path):
            assert path.is_file()

    def run_one(args, architecture, frames, output, bud, **kwargs):
        calls.append((output.name, bud, kwargs))
        output.mkdir(exist_ok=True, parents=True)
        checkpoint = output / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        (output / "latest.json").write_text(
            json.dumps(
                dict(
                    update=bud.updates,
                    checkpoint=checkpoint.name,
                    sha256=dual_sha256(checkpoint),
                )
            )
        )
        (output / "run.json").write_text(json.dumps(dict(code=code)))
        (output / "COMPLETED.json").write_text(
            json.dumps(
                dict(
                    complete=True,
                    updates_completed=bud.updates,
                )
            )
        )

    monkeypatch.setattr(controller, "RcloneArtifactStore", Store)
    monkeypatch.setattr(controller, "run_one", run_one)
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text('{"selected":{"microbatch_windows":1}}')
    args = SimpleNamespace(
        workspace_root=tmp_path,
        run_id="test",
        benchmark=benchmark,
        resume=False,
        rclone_config=tmp_path / "config",
    )
    download = (
        tmp_path
        / "outputs/ball_detection/precompute/sequential_length/drive-download-20261025/progress.json"
    )
    download.parent.mkdir(parents=True)
    download.write_text(
        json.dumps(
            dict(
                complete=True,
                expected_files=1,
                verified_files=1,
                expected_bytes=1,
                verified_bytes=1,
            )
        )
    )
    controller.train(args)
    root = tmp_path / controller.Path(
        "outputs/ball_detection/train/sequential_length/test"
    )
    original_plan = (root / "plan.json").read_bytes()
    (root / "query-T32/COMPLETED.json").unlink()
    shutil.rmtree(root / "query-T64")
    (root / "COMPLETED.json").unlink()
    benchmark.unlink()
    (root / "benchmark-complete.json").unlink()
    args.benchmark, args.resume = None, True
    calls.clear()
    if changed == "source":
        code = dict(source_sha256={"train.py": "b" * 64}, torch_version="original")
    elif changed == "budget":
        plan = json.loads(original_plan)
        plan["pretraining_updates"] = 9999
        (root / "plan.json").write_text(json.dumps(plan))
    elif changed == "checkpoint":
        (root / "common-dpt-T32/checkpoint.pt").write_bytes(b"corrupt")
    elif changed == "download":
        state = json.loads(download.read_text())
        state["complete"] = False
        download.write_text(json.dumps(state))
    if changed:
        with pytest.raises(ValueError):
            controller.train(args)
        assert calls == []
        return
    controller.train(args)
    assert (root / "plan.json").read_bytes() == original_plan
    assert [name for name, _, _ in calls] == ["query-T32", "query-T64"]
    assert calls[0][2]["resume"] == root / "query-T32/checkpoint.pt"
    assert calls[1][2]["parent"] == root / "query-T32"
    first, second = calls[0][1], calls[1][1]
    assert first.updates == second.updates == 6000
    assert first.freeze_updates == second.freeze_updates == 600
    assert first.schedule_offset == 0 and second.schedule_offset == 6000
    assert first.lr(5999) == second.lr(-1)
    assert second.lr(5999) == pytest.approx(1e-6)
    assert calls[1][2]["resume"] is None
