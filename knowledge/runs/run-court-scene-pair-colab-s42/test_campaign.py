"""Small CPU fixtures for campaign policy, recovery and adapter boundaries."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as c  # noqa: E402
import runtime as runtime  # noqa: E402


@pytest.mark.parametrize(
    ("condition", "version"),
    [("scene-1", 0), ("scene-2", 1), ("scene-3", 0), ("scene-4", 0)],
)
def test_legacy_final_checkpoint_uses_repository_records(
    condition: str, version: int
) -> None:
    # Read the shipped evidence, including the bundles without training-completion.
    bundle = runtime.prior_bundle(condition)
    checked = runtime.legacy_final_checkpoint(condition)
    expected = (
        "outputs/court_detection/train/scene-diversity-6000-s384-colab/"
        f"{condition}-s42/logs/version_{version}/checkpoints/update-00033160.ckpt"
    )
    assert checked["checkpoint"] == expected
    evaluation = c.read(bundle / "evaluation.json")
    assert checked["checkpoint_sha256"] == evaluation["checkpoint_sha256"]
    config = yaml.safe_load((bundle / "resolved-config.yaml").read_text())
    assert Path("outputs") / config["run"]["output_dir"] == Path(expected).parents[3]
    assert runtime.plain(config) == evaluation["config"]
    assert evaluation["data_condition"]["accepted_images"] == 5500
    verification = c.read(runtime.PRIOR / "parent-verification.json")
    parent_record = verification["conditions"][int(condition[-1]) - 1]
    assert parent_record["condition"] == checked["condition"]
    assert parent_record["global_step"] == checked["global_step"] == 33160
    assert parent_record["checkpoint_sha256"] == checked["checkpoint_sha256"]
    assert (
        evaluation["splits"]["tennis_court_detector-val"]["ordered_sample_ids_sha256"]
        == parent_record["real_order_sha256"]
        == verification["conditions"][0]["real_order_sha256"]
    )
    if condition in {"scene-1", "scene-2"}:
        completion = c.read(bundle / "training-completion.json")
        for key in ("checkpoint", "checkpoint_sha256"):
            assert checked[key] == completion[key]


@pytest.mark.parametrize(
    ("filename", "key", "value"),
    [
        ("final-checkpoint.json", "schema", "unknown_schema"),
        ("final-checkpoint.json", "global_step", 29844),
        ("final-checkpoint.json", "global_step", 33160.0),
        ("final-checkpoint.json", "condition", 4),
        ("final-checkpoint.json", "condition", "3"),
        ("evaluation.json", "checkpoint", "/content/tennis-lab/wrong.ckpt"),
        ("evaluation.json", "checkpoint_sha256", "changed"),
    ],
)
def test_legacy_final_checkpoint_rejects_inconsistent_records(
    tmp_path: Path, monkeypatch: Any, filename: str, key: str, value: Any
) -> None:
    for name in ("final-checkpoint.json", "evaluation.json"):
        record = c.read(runtime.prior_bundle("scene-3") / name)
        if name == filename:
            record[key] = value
        c.write(tmp_path / name, record)
    monkeypatch.setattr(runtime, "prior_bundle", lambda condition: tmp_path)
    with pytest.raises(ValueError, match="Legacy final checkpoint"):
        runtime.legacy_final_checkpoint("scene-3")


def test_five_configs_keep_actual_scene_two_semantics() -> None:
    allowed = {
        "data.source.manifest",
        "data.source.validation_manifest",
        "mixed.sources.synthetic_court.manifest",
        "mixed.sources.synthetic_court.validation_manifest",
        "run.output_dir",
        "run.artifact_store.remote_root",
        "run.resume",
    }
    for condition, name in {
        "P02": "pair-b00-b02",
        "P03": "pair-b00-b03",
        "P12": "pair-b01-b02",
        "P13": "pair-b01-b03",
        "P23": "pair-b02-b03",
    }.items():
        config = c.make_config(condition)
        assert c.validate_config(config, condition) == allowed
        assert config["data"]["source"]["manifest"].endswith(
            name + ".train.subset.json"
        )
        assert (
            config["run"]["output_dir"]
            == f"court_detection/train/scene-pair-5500-s384-colab/{condition}-s42"
        )
        assert config["training"]["step_schedule"] == {
            "max_steps": 33160,
            "validation_every_n_steps": 3316,
            "checkpoint_every_n_steps": 3316,
        }
        assert config["run"]["resume"] is None


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("data", "batch_size"), 16),
        (("training", "learning_rate"), 2e-4),
        (("run", "seed"), 43),
        (("paths", "data_root"), "/other/data"),
        (("model", "encoder", "lora", "enabled"), True),
        (("training", "checkpoint", "save_top_k"), 0),
    ],
)
def test_unapproved_training_changes_fail(keys: tuple[str, ...], value: Any) -> None:
    config = c.make_config("P02")
    target = config
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValueError, match="Unexpected training configuration"):
        c.validate_config(config, "P02")


def test_probe_has_only_explicit_schedule_exceptions() -> None:
    probe = c.make_config("P02", probe=True)
    c.validate_config(probe, "P02", probe=True)
    assert c.differences(c.make_config("P02"), probe) == {
        "run.output_dir",
        "run.artifact_store.remote_root",
        "training.warmup_steps",
        "training.step_schedule.max_steps",
        "training.step_schedule.validation_every_n_steps",
        "training.step_schedule.checkpoint_every_n_steps",
    }
    assert probe["training"]["warmup_steps"] == 0
    assert set(probe["training"]["step_schedule"].values()) == {3}
    probe["data"]["unexpected"] = True
    with pytest.raises(ValueError, match="keys changed"):
        c.validate_config(probe, "P02", probe=True)


@pytest.mark.parametrize(
    "path",
    [
        "court_detection/train/scene-pair-5500-s384-colab/P03-s42/logs/version_0/checkpoints/update-00003316.ckpt",
        "court_detection/train/scene-pair-5500-s384-colab/P02-s42/logs/version_0/checkpoints/update-00003316.ckpt.uploading-x",
        "outputs/court_detection/train/scene-pair-5500-s384-colab/P02-s42/logs/version_0/checkpoints/update-00003316.ckpt",
    ],
)
def test_resume_rejects_foreign_temporary_and_wrong_root(path: str) -> None:
    config = c.make_config("P02")
    config["run"]["resume"] = {"role": "artifact", "path": path}
    with pytest.raises(ValueError):
        c.validate_config(config, "P02")


def test_latest_uses_numeric_update_not_listing_order_or_last() -> None:
    prefix = "outputs/court_detection/train/scene-pair-5500-s384-colab/P02-s42/logs"
    latest = f"{prefix}/version_2/checkpoints/update-00009948.ckpt"
    paths = [
        latest,
        f"{prefix}/version_0/checkpoints/update-00003316.ckpt",
        f"{prefix}/version_3/checkpoints/update-00013264.ckpt.uploading-x",
        f"{prefix}/version_2/checkpoints/last.ckpt",
    ]
    assert c.latest_checkpoint(paths, "P02") == latest
    config = c.make_config("P02")
    config["run"]["resume"] = {
        "role": "artifact",
        "path": latest.removeprefix("outputs/"),
    }
    c.validate_config(config, "P02")
    with pytest.raises(ValueError, match="Ambiguous"):
        c.latest_checkpoint([latest, latest.replace("version_2", "version_3")], "P02")


class FakeBackend:
    """A file-backed Drive substitute; no torch, images, model or network."""

    def __init__(self, local: Path, remote: Path, *, fail: str | None = None) -> None:
        self.state, self.remote, self.fail = local, remote, fail
        self.calls: list[str] = []
        self.resumes: list[int] = []

    def restore_state(self) -> None:
        if (self.remote / "state").exists():
            shutil.copytree(self.remote / "state", self.state, dirs_exist_ok=True)

    def prepare(self) -> dict[str, Any]:
        self.calls.append("prepare")
        if self.fail == "inputs":
            raise RuntimeError("injected input verification failure")
        return {"schema": "fixture-inputs-v1", "pool": "fixture-pool"}

    def execute(self, stage: c.Stage, progress: Any) -> dict[str, Any]:
        self.calls.append(stage.key)
        if self.fail == stage.kind:
            raise RuntimeError(f"injected {stage.kind} failure")
        if stage.kind == "train":
            prefix = c.train_root(stage.condition).as_posix()  # type: ignore[arg-type]
            files = [
                p.relative_to(self.remote).as_posix()
                for p in self.remote.rglob("*.ckpt")
                if p.relative_to(self.remote).is_relative_to(prefix)
            ]
            saved = c.latest_checkpoint(files, stage.condition)  # type: ignore[arg-type]
            step = c.checkpoint_step(saved, Path(prefix)) if saved else 0
            self.resumes.append(step)
            if self.fail == "interrupt" and stage.condition == "P02":
                checkpoint = (
                    self.remote
                    / prefix
                    / "logs/version_0/checkpoints/update-00006632.ckpt"
                )
                c.write(checkpoint, {"condition": "P02", "global_step": 6632})
                progress(saved_step=6632)
                raise RuntimeError("VM lost after update 6655")
        result = {
            "stage": stage.key,
            "global_step": 3 if stage.kind == "probe" else 33160,
        }
        artifact = self.remote / "artifacts" / f"{stage.key}.json"
        c.write(artifact, result)
        return {**result, "artifact_sha256": c.sha(artifact)}

    def verify(self, stage: c.Stage, result: dict[str, Any]) -> None:
        if stage.kind == "inputs":
            assert result["pool"] == "fixture-pool"
            return
        artifact = self.remote / "artifacts" / f"{stage.key}.json"
        if c.sha(artifact) != result["artifact_sha256"]:
            raise ValueError("Completed artifact changed")

    def publish(self, path: Path) -> None:
        if self.fail == "publish" and path.name == "evaluate-scene-1.json":
            raise RuntimeError("injected Drive publication failure")
        target = self.remote / "state" / path.relative_to(self.state)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def test_vm_loss_resumes_same_pair_and_skips_verified_completed_stages(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "drive"
    first = FakeBackend(tmp_path / "vm-one", remote, fail="interrupt")
    with pytest.raises(RuntimeError, match="VM lost"):
        c.Campaign(first.state, first).run()
    assert c.read(first.state / "progress.json")["latest_saved_update"] == 6632
    second = FakeBackend(tmp_path / "vm-two", remote)
    c.Campaign(second.state, second).run()
    assert second.calls == [
        "prepare",
        "train-P02",
        "evaluate-P02",
        "train-P03",
        "evaluate-P03",
        "train-P12",
        "evaluate-P12",
        "train-P13",
        "evaluate-P13",
        "train-P23",
        "evaluate-P23",
    ]
    assert second.resumes == [6632, 0, 0, 0, 0]
    assert c.read(second.state / "progress.json")["status"] == "completed"
    third = FakeBackend(tmp_path / "vm-three", remote)
    c.Campaign(third.state, third).run()
    assert third.calls == ["prepare"]


@pytest.mark.parametrize("failure", ["inputs", "train", "evaluate", "publish"])
def test_any_failed_gate_stops_before_the_next_stage(
    tmp_path: Path, failure: str
) -> None:
    backend = FakeBackend(tmp_path / "local", tmp_path / "drive", fail=failure)
    with pytest.raises(RuntimeError, match="injected"):
        c.Campaign(backend.state, backend).run()
    forbidden = {
        "inputs": "probe-P02",
        "train": "evaluate-P02",
        "evaluate": "evaluate-scene-2",
        "publish": "evaluate-scene-2",
    }[failure]
    assert forbidden not in backend.calls
    marker = {
        "inputs": "inputs",
        "train": "train-P02",
        "evaluate": "evaluate-scene-1",
        "publish": "evaluate-scene-1",
    }[failure]
    assert not (backend.state / "done" / f"{marker}.json").exists()


def test_corrupted_completion_is_rejected_instead_of_skipped(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path / "local", tmp_path / "drive")
    c.Campaign(backend.state, backend).run()
    c.write(backend.remote / "artifacts/evaluate-scene-1.json", {"wrong": "content"})
    again = FakeBackend(tmp_path / "new-vm", backend.remote)
    with pytest.raises(ValueError, match="artifact changed"):
        c.Campaign(again.state, again).run()
    assert again.calls == ["prepare"]


@pytest.mark.parametrize(
    "fault", ["valid", "step", "config", "data", "truncated", "receipt"]
)
def test_checkpoint_adapter_rejects_newest_invalid_without_fallback(
    tmp_path: Path, monkeypatch: Any, fault: str
) -> None:
    import torch

    monkeypatch.setattr(c, "REPO", tmp_path)
    relative = c.train_root("P02") / "logs/version_1/checkpoints/update-00006632.ckpt"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    config = c.make_config("P02")
    payload = {
        "global_step": 6632,
        "hyper_parameters": {"config": config, "data_condition": {"pair": "P02"}},
        "state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer_states": [{}],
        "lr_schedulers": [{}],
        "loops": {},
        "training_random_state": {},
    }
    if fault == "step":
        payload["global_step"] = 3316
    elif fault == "config":
        config["data"]["source"]["manifest"] = "foreign.train.subset.json"
    elif fault == "data":
        payload["hyper_parameters"]["data_condition"] = {"pair": "P03"}  # type: ignore[index]
    torch.save(payload, path)
    if fault == "truncated":
        path.write_bytes(path.read_bytes()[:50])
    backend = object.__new__(runtime.ColabBackend)
    backend.state = tmp_path / c.STATE
    if fault == "receipt":
        c.write(backend.saved_record(relative), {"checkpoint_sha256": "changed"})
    monkeypatch.setattr(
        backend,
        "ready",
        lambda: {"conditions": {"P02": {"data_condition": {"pair": "P02"}}}},
    )
    monkeypatch.setattr(backend, "inventory", lambda root: [relative.as_posix()])
    monkeypatch.setattr(backend, "fetch", lambda relative: tmp_path / relative)
    if fault == "valid":
        assert backend.latest("P02")["global_step"] == 6632
        return
    with pytest.raises(
        (ValueError, RuntimeError),
        match="global_step|configuration|data condition|zip archive|saved checkpoint",
    ):
        backend.latest("P02")


def test_inventory_uses_new_exclusive_receipt_each_time(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(c, "REPO", tmp_path)
    backend = object.__new__(runtime.ColabBackend)
    monkeypatch.setattr(backend, "exists", lambda root: True)
    outputs = []

    def cli(*args: str) -> dict[str, Any]:
        target = Path(args[-1])
        assert not target.exists()
        c.write(target, {"entries": [], "ambiguous_paths": []})
        outputs.append(target)
        return {}

    monkeypatch.setattr(backend, "cli", cli)
    assert backend.inventory(c.train_root("P02")) == []
    assert backend.inventory(c.train_root("P02")) == []
    assert len(set(outputs)) == 2


def test_cli_failure_returns_nonzero_and_never_reaches_later_training(
    tmp_path: Path,
) -> None:
    # Real process/CLI boundary, with only the expensive backend substituted.
    script = """
import sys
from pathlib import Path
import campaign as c
import runtime
from test_campaign import FakeBackend
root = Path(sys.argv[1])
backend = FakeBackend(root / 'local' / c.STATE, root / 'drive', fail='evaluate')
runtime.ColabBackend = lambda: backend
sys.argv = ['campaign.py', 'run']
c.REPO = root / 'local'
c.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=c.BUNDLE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "injected evaluate failure" in result.stderr
    assert not (tmp_path / "drive/artifacts/train-P02.json").exists()


@pytest.mark.parametrize("fault", ["order", "macro"])
def test_evaluation_adapter_checks_pinned_order_and_unweighted_mean(
    monkeypatch: Any, fault: str
) -> None:
    backend = object.__new__(runtime.ColabBackend)
    ids = ["B00:a", "B00:b", "B01:c", "B02:d", "B03:e"]
    ready = {
        "synthetic_ids": ids,
        "real_ids": ["real-b", "real-a"],
        "evaluation_manifest_sha256": "manifest",
    }
    monkeypatch.setattr(backend, "ready", lambda: ready)

    def split(samples: list[str], pose: bool) -> dict[str, Any]:
        return {
            "sample_ids": samples,
            "count": len(samples),
            "batch_size": 8,
            "ordered_sample_ids_sha256": runtime.ordered_digest(samples),
            "pose_supervised": pose,
            "metrics": dict.fromkeys(runtime.METRICS, 0.5),
        }

    synthetic = split(ids, True)
    synthetic["scene_order"] = ["B00", "B01", "B02", "B03"]
    synthetic["scenes"] = {}
    for scene, value in zip(
        synthetic["scene_order"], [2.0, 4.0, 6.0, 8.0], strict=True
    ):
        samples = [i for i in ids if i.startswith(scene + ":")]
        synthetic["scenes"][scene] = {
            "sample_ids": samples,
            "count": len(samples),
            "ordered_sample_ids_sha256": runtime.ordered_digest(samples),
            "metrics": dict.fromkeys(runtime.METRICS, value),
        }
    synthetic["scene_macro_mean"] = dict.fromkeys(runtime.METRICS, 5.0)
    report = {
        "schema": "court_data_scaling_evaluation_v2",
        "series": "scene-pairs",
        "global_step": 33160,
        "checkpoint_sha256": "checkpoint",
        "data_condition": {"pair": "P02"},
        "synthetic_evaluation_manifest": {"sha256": "manifest"},
        "splits": {
            "synthetic_court-val": synthetic,
            "tennis_court_detector-val": split(ready["real_ids"], False),
        },
    }
    checked = {"checkpoint_sha256": "checkpoint", "data_condition": {"pair": "P02"}}
    backend.check_evaluation(report, checked)
    if fault == "order":
        report["splits"]["tennis_court_detector-val"]["sample_ids"] = [
            "real-a",
            "real-b",
        ]
    else:
        synthetic["scene_macro_mean"]["seg_miou"] = 4.4  # image-weighted (2*2+4+6+8)/5
    with pytest.raises(ValueError, match="order|simple four-scene mean"):
        backend.check_evaluation(report, checked)
