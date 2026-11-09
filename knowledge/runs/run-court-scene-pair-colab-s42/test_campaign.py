"""Small CPU fixtures for campaign policy, recovery and adapter boundaries."""

# Match the pytest decorator policy for tests/ in pyproject.toml for this bundle.
# mypy: disallow-untyped-decorators=False

from __future__ import annotations

import copy
import importlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as c  # noqa: E402
import runtime as runtime  # noqa: E402

TRAINING_OUTPUTS = (("P02", True), *((pair, False) for pair in c.PAIRS))


def _runbook_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], list[str]]:
    # Parse the submitted command, without allocating a session or contacting Drive.
    monkeypatch.syspath_prepend(str(c.REPO / ".agents/skills/tennis-colab/scripts"))
    colab = importlib.import_module("colab")
    script = ".agents/skills/tennis-colab/scripts/colab.py"
    commands = [
        shlex.split(block.split("```", 1)[0].replace("\\\n", ""))
        for block in (c.BUNDLE / "README.md").read_text().split("```bash\n")[1:]
    ]
    executions = [
        argv[argv.index(script) + 1 :]
        for argv in commands
        if script in argv and argv[argv.index(script) + 1] == "exec"
    ]
    assert len(executions) == 1
    declared = colab.parser().parse_args(executions[0])
    assert declared.argv == [
        "--",
        "bash",
        "knowledge/runs/run-court-scene-pair-colab-s42/bootstrap.sh",
    ]
    return declared.runner_output, declared.persist


def _build_training_store(
    condition: str,
    probe: bool,
    declarations: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    from src.tasks.base.configuration import ArtifactStoreConfig
    from src.tasks.base.training.artifact_store.factory import build_artifact_store
    from src.utils.artifact_store import RcloneArtifactStore

    config = c.make_config(condition, probe=probe)
    store_config = ArtifactStoreConfig.from_mapping(config["run"]["artifact_store"])
    assert store_config.mode == "rclone" and store_config.remote == "gdrive"
    # Preserve the declared spelling, exactly as colab_remote.run constructs URIs.
    monkeypatch.setenv(
        "TENNIS_COLAB_ARTIFACT_ROOTS",
        json.dumps([f"gdrive:tennis_lab/{fragment}" for fragment in declarations]),
    )
    secret = tmp_path / "fake-rclone.conf"
    secret.touch(mode=0o600)
    monkeypatch.setenv("RCLONE_CONFIG", str(secret))
    root: Path = Path("outputs") / config["run"]["output_dir"]
    # Construction checks exact declared URI membership; no Drive operation runs.
    store = build_artifact_store(store_config, local_root=tmp_path / root)
    assert isinstance(store, RcloneArtifactStore)
    assert store.remote_uri == f"gdrive:tennis_lab/{root.as_posix()}"
    return root


def test_runbook_gives_each_publisher_one_live_drive_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declarations, persisted = _runbook_outputs(monkeypatch)
    owners = [Path(fragment) for fragment in declarations]
    periodic = [Path(fragment) for fragment in persisted]
    # Runner stores require exact URI declarations, including each condition.
    training_roots = [
        _build_training_store(condition, probe, declarations, tmp_path, monkeypatch)
        for condition, probe in TRAINING_OUTPUTS
    ]

    # Observe the actual adapter destinations, rather than repeating its allowlist.
    bundle_roots: list[Path] = []

    class Store:
        def __init__(
            self, root: Path, *, remote: str, remote_root: str, config_path: Path
        ) -> None:
            assert remote == "gdrive"
            self.remote_root = Path(remote_root).relative_to("tennis_lab")

        def publish_file(self, path: Path) -> None:
            bundle_roots.append(self.remote_root)

        def sync_tree(self) -> None:
            bundle_roots.append(self.remote_root)

    from src.utils import artifact_store

    monkeypatch.setattr(artifact_store, "RcloneArtifactStore", Store)
    monkeypatch.setattr(c, "REPO", tmp_path)
    backend = object.__new__(runtime.ColabBackend)
    monkeypatch.setattr(
        backend,
        "drive",
        SimpleNamespace(config_path=tmp_path / "unused"),
        raising=False,
    )
    monkeypatch.setattr(backend, "verify_remote", lambda local, remote: None)
    for relative in (
        c.STATE / "progress.json",
        c.STATE / "saved" / "probe.json",
        c.STATE / "restarts" / "probe.json",
        c.STATE / "done" / "inputs.json",
    ):
        c.write(tmp_path / relative, {})
        backend.publish(tmp_path / relative)
    for relative in (
        c.STATE / "partial-evaluations" / "scene-1-attempt",
        *(c.eval_root(condition) for condition in (*c.LEGACY, *c.PAIRS)),
    ):
        (tmp_path / relative).mkdir(parents=True)
        backend.publish(tmp_path / relative)

    # Ancestor declarations also overlap: an umbrella --persist must fail here.
    for index, owner in enumerate(owners):
        assert all(
            not owner.is_relative_to(other) and not other.is_relative_to(owner)
            for other in owners[index + 1 :] + periodic
        ), f"Concurrent persistence owners overlap: {owner}"
    # Bundle state/evaluation writes may target children of the declared trees.
    for root in bundle_roots:
        assert sum(root.is_relative_to(owner) for owner in owners) == 1, (
            f"Publisher needs exactly one --runner-output: {root}"
        )
    assert all(
        any(root.is_relative_to(owner) for root in [*bundle_roots, *training_roots])
        for owner in owners
    )


@pytest.mark.parametrize(
    ("condition", "probe"),
    TRAINING_OUTPUTS,
    ids=["probe-P02", *(f"train-{pair}" for pair in c.PAIRS)],
)
@pytest.mark.parametrize("mismatch", ["ancestor", "other-condition"])
def test_training_outputs_reject_inexact_declarations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    probe: bool,
    mismatch: str,
) -> None:
    declarations, _ = _runbook_outputs(monkeypatch)
    root = c.train_root(condition, probe=probe)
    wrong = (
        root.parent
        if mismatch == "ancestor"
        else root.with_name("P03-s42" if condition == "P02" else "P02-s42")
    )
    mutated = [
        wrong.as_posix() if fragment == root.as_posix() else fragment
        for fragment in declarations
    ]
    assert root.as_posix() not in mutated
    with pytest.raises(RuntimeError, match="declared Drive root"):
        _build_training_store(condition, probe, mutated, tmp_path, monkeypatch)


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

    def restore_state(self, progress: Any) -> None:
        if (self.remote / "state").exists():
            shutil.copytree(
                self.remote / "state",
                self.state,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("progress.json"),
            )
        progress(operation="completed")

    def prepare(self, progress: Any) -> dict[str, Any]:
        progress(phase="validating_inputs", completed_items=1, total_items=1)
        self.calls.append("prepare")
        if self.fail == "inputs":
            raise RuntimeError("injected input verification failure")
        return {"schema": "fixture-inputs-v1", "pool": "fixture-pool"}

    def execute(self, stage: c.Stage, progress: Any) -> dict[str, Any]:
        self.calls.append(stage.key)
        if self.fail == stage.kind:
            raise RuntimeError(f"injected {stage.kind} failure")
        if stage.kind == "train":
            assert stage.condition is not None
            prefix = c.train_root(stage.condition).as_posix()
            files = [
                p.relative_to(self.remote).as_posix()
                for p in self.remote.rglob("*.ckpt")
                if p.relative_to(self.remote).is_relative_to(prefix)
            ]
            saved = c.latest_checkpoint(files, stage.condition)
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

    def verify(self, stage: c.Stage, result: dict[str, Any], progress: Any) -> None:
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


def test_resume_verification_updates_live_phase_counts_and_last_progress(
    tmp_path: Path,
) -> None:
    first = FakeBackend(tmp_path / "one", tmp_path / "drive")
    c.Campaign(first.state, first).run()
    ticks = [0.0]
    observed: list[dict[str, Any]] = []

    class ObservedBackend(FakeBackend):
        def restore_state(self, progress: Any) -> None:
            observed.append(c.read(self.state / "progress.json"))
            super().restore_state(progress)

        def verify(self, stage: c.Stage, result: dict[str, Any], progress: Any) -> None:
            super().verify(stage, result, progress)
            assert c.read(self.state / "progress.json")["phase"] == "verifying_existing"
            for count in (0, 1, 2):
                ticks[0] += 2
                progress(
                    operation="verifying",
                    completed_files=count,
                    total_files=2,
                    completed_bytes=count * 7,
                    total_bytes=14,
                )
                observed.append(c.read(self.state / "progress.json"))

    second = ObservedBackend(tmp_path / "two", first.remote)

    def clock() -> str:
        return c.datetime.fromtimestamp(ticks[0], c.UTC).isoformat()

    c.Campaign(second.state, second, clock=clock, monotonic=lambda: ticks[0]).run()
    assert second.calls == ["prepare"]
    assert observed[0]["phase"] == "restoring"
    checks = [value for value in observed if value["stage"] == "evaluate-scene-1"]
    assert [value["completed_files"] for value in checks] == [0, 1, 2]
    assert [value["completed_bytes"] for value in checks] == [0, 7, 14]
    assert all(
        value["total_files"] == 2 and value["phase"] == "verifying_existing"
        for value in checks
    )
    assert (
        checks[0]["last_progress_at"]
        < checks[1]["last_progress_at"]
        < checks[2]["last_progress_at"]
    )


def test_progress_limits_drive_publication_and_exposes_stalled_work(
    tmp_path: Path, monkeypatch: Any
) -> None:
    backend = FakeBackend(tmp_path / "local", tmp_path / "drive")
    publications: list[dict[str, Any]] = []
    monkeypatch.setattr(
        backend, "publish", lambda path: publications.append(c.read(path))
    )
    ticks = [0.0]
    campaign = c.Campaign(
        backend.state,
        backend,
        clock=lambda: c.datetime.fromtimestamp(ticks[0], c.UTC).isoformat(),
        monotonic=lambda: ticks[0],
    )
    stage = c.Stage("train", "P23")
    campaign.progress(
        stage, phase="training", saved_step=3316, actual_training_update=3320
    )
    for step in range(3321, 3331):
        ticks[0] += 0.1
        campaign.progress(stage, actual_training_update=step)
    assert len(publications) == 1  # Each batch must not issue a Drive operation.
    ticks[0] = 2
    campaign.progress(stage, actual_training_update=3330)
    forward = c.read(backend.state / "progress.json")
    ticks[0] = 4
    campaign.progress(stage, actual_training_update=3330)
    stalled = c.read(backend.state / "progress.json")
    assert stalled["updated_at"] > forward["updated_at"]
    assert stalled["last_progress_at"] == forward["last_progress_at"]
    assert stalled["elapsed_seconds"] == 4
    assert stalled["latest_saved_update"] == 3316
    ticks[0] = 61
    campaign.progress(stage, actual_training_update=3331)
    assert len(publications) == 2
    campaign.progress(
        stage,
        phase="verifying_drive",
        operation="checkpoint_save",
        actual_training_update=6632,
    )
    checking = c.read(backend.state / "progress.json")
    assert checking["phase"] == "verifying_drive"
    assert checking["latest_saved_update"] == 3316
    assert checking["actual_training_update"] == 6632
    campaign.progress(stage, phase="training", saved_step=6632)
    assert c.read(backend.state / "progress.json")["latest_saved_update"] == 6632


@pytest.mark.parametrize("failure", [False, True])
def test_training_subprocess_progress_survives_return_to_parent(
    tmp_path: Path,
    failure: bool,
) -> None:
    ticks = [0]
    observed: list[dict[str, Any]] = []

    def clock() -> str:
        ticks[0] += 1
        return c.datetime.fromtimestamp(ticks[0], c.UTC).isoformat()

    class ChildWriterBackend(FakeBackend):
        def execute(self, stage: c.Stage, progress: Any) -> dict[str, Any]:
            if stage == c.Stage("train", "P02"):
                child_writer = c.Campaign(self.state, self, clock=clock)
                child_writer.progress(
                    stage,
                    phase="training",
                    saved_step=3316,
                    actual_training_update=3320,
                )
                if failure:
                    raise RuntimeError("training subprocess failed")
            return super().execute(stage, progress)

        def verify(self, stage: c.Stage, result: dict[str, Any], progress: Any) -> None:
            if stage == c.Stage("train", "P02"):
                observed.append(c.read(self.state / "progress.json"))
            super().verify(stage, result, progress)

    backend = ChildWriterBackend(tmp_path / "local", tmp_path / "drive")
    campaign = c.Campaign(backend.state, backend, clock=clock)
    if failure:
        with pytest.raises(RuntimeError, match="training subprocess failed"):
            campaign.run()
        observed.append(c.read(backend.state / "progress.json"))
        assert observed[-1]["status"] == "failed"
        assert "evaluate-P02" not in backend.calls
    else:
        campaign.run()
    assert observed[0]["latest_saved_update"] == 3316
    assert observed[0]["actual_training_update"] == 3320


@pytest.mark.parametrize("fault", ["valid", "adapter", "data", "source", "config"])
def test_reviewed_adapter_upgrade_preserves_only_unchanged_scientific_identity(
    tmp_path: Path, monkeypatch: Any, fault: str
) -> None:
    old: dict[str, Any] = {
        "schema": "court_scene_pair_inputs_v1",
        "bundle_sha256": c.ORIGINAL_ADAPTERS,
        "source_objects": {"src": "pinned-science"},
        "conditions": {"P02": {"config_sha256": "pinned-config"}},
        "real_ids": ["b", "a"],
    }
    current = {
        **copy.deepcopy(old),
        "bundle_sha256": {name: c.sha(c.BUNDLE / name) for name in c.ORIGINAL_ADAPTERS},
    }
    if fault == "adapter":
        current["bundle_sha256"]["runtime.py"] = "unreviewed"
        with pytest.raises(ValueError, match="Unknown campaign operational adapter"):
            c.inputs_identity(current)
        return
    if fault == "data":
        current["real_ids"].reverse()
    elif fault == "source":
        current["source_objects"]["src"] = "changed-science"
    elif fault == "config":
        current["conditions"]["P02"]["config_sha256"] = "changed-config"
    # Original receipts used digest(old) directly: no Drive done-marker rewrite.
    assert (c.inputs_identity(current) == c.digest(old)) is (fault == "valid")
    backend = object.__new__(runtime.ColabBackend)
    monkeypatch.setattr(backend, "ready", lambda: current)
    if fault == "valid":
        backend.verify(c.Stage("inputs"), old)
    else:
        with pytest.raises(ValueError, match="Input completion disagrees"):
            backend.verify(c.Stage("inputs"), old)


def test_original_done_markers_resume_without_rewriting_their_identity(
    tmp_path: Path,
) -> None:
    class CompatibleBackend(FakeBackend):
        def prepare(self, progress: Any) -> dict[str, Any]:
            result = super().prepare(progress)
            return {
                **result,
                "schema": "court_scene_pair_inputs_v1",
                "bundle_sha256": c.ORIGINAL_ADAPTERS
                if self.state.name == "old"
                else {name: c.sha(c.BUNDLE / name) for name in c.ORIGINAL_ADAPTERS},
            }

    first = CompatibleBackend(tmp_path / "old", tmp_path / "drive")
    c.Campaign(first.state, first).run()
    old_markers = {
        path.name: path.read_bytes() for path in (first.remote / "state/done").iterdir()
    }
    second = CompatibleBackend(tmp_path / "new", first.remote)
    c.Campaign(second.state, second).run()
    assert second.calls == ["prepare"]
    assert {
        path.name: path.read_bytes() for path in (first.remote / "state/done").iterdir()
    } == old_markers


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
        latest = backend.latest("P02")
        assert latest is not None and latest["global_step"] == 6632
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


@pytest.fixture
def evaluation_fixture(monkeypatch: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    backend = object.__new__(runtime.ColabBackend)
    ids = ["B00:a", "B00:b", "B01:c", "B02:d", "B03:e"]
    ready: dict[str, Any] = {
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
    report: dict[str, Any] = {
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
    checked = {
        "checkpoint_sha256": "checkpoint",
        "data_condition": {"pair": "P02"},
        "checkpoint": (
            c.train_root("P02") / "logs/version_0/checkpoints/update-00033160.ckpt"
        ).as_posix(),
        "global_step": c.FINAL_STEP,
    }
    return backend, report, checked


@pytest.mark.parametrize("fault", ["order", "macro"])
def test_evaluation_adapter_checks_pinned_order_and_unweighted_mean(
    evaluation_fixture: tuple[Any, dict[str, Any], dict[str, Any]], fault: str
) -> None:
    backend, report, checked = evaluation_fixture
    backend.check_evaluation(report, checked)
    if fault == "order":
        report["splits"]["tennis_court_detector-val"]["sample_ids"] = [
            "real-a",
            "real-b",
        ]
    else:
        report["splits"]["synthetic_court-val"]["scene_macro_mean"]["seg_miou"] = (
            4.4  # image-weighted (2*2+4+6+8)/5
        )
    with pytest.raises(ValueError, match="order|simple four-scene mean"):
        backend.check_evaluation(report, checked)


def test_completed_evaluation_batches_download_and_checks_checkpoint_provenance(
    evaluation_fixture: tuple[Any, dict[str, Any], dict[str, Any]],
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    if shutil.which("rclone") is None:
        pytest.skip("rclone executable required for isolated alias integration")
    backend, report, checked = evaluation_fixture
    monkeypatch.setattr(c, "REPO", tmp_path / "local")
    remote = tmp_path / "remote"
    source = remote / "tennis_lab" / c.eval_root("P02")
    source.mkdir(parents=True)
    config = tmp_path / "rclone.conf"
    config.write_text(f"[test]\ntype = alias\nremote = {remote}\n")
    backend.drive = runtime.RcloneBackend(
        remote_root="test:tennis_lab", executable="rclone", config_path=config
    )
    report_path = source / "evaluation.json"
    c.write(report_path, report)
    (source / "qualitative.png").write_bytes(b"pixels")
    fetched: list[Path] = []
    monkeypatch.setattr(backend, "fetch", lambda relative: fetched.append(relative))
    monkeypatch.setattr(backend, "checkpoint", lambda *args, **kwargs: checked)
    original = runtime.download_tree
    batches: list[dict[str, str]] = []

    def download(*args: Any, **kwargs: Any) -> Any:
        batches.append(kwargs["expected_files"])
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "download_tree", download)

    def receipt() -> dict[str, Any]:
        return {
            **checked,
            "evaluation": (c.eval_root("P02") / "evaluation.json").as_posix(),
            "evaluation_sha256": c.sha(report_path),
            "evaluation_files": {
                (c.eval_root("P02") / path.name).as_posix(): c.sha(path)
                for path in source.iterdir()
            },
        }

    events: list[dict[str, Any]] = []
    backend.verify(
        c.Stage("evaluate", "P02"), receipt(), lambda **event: events.append(event)
    )
    assert fetched == [Path(checked["checkpoint"])]
    assert len(batches) == 1 and set(batches[0]) == {
        "evaluation.json",
        "qualitative.png",
    }
    assert events[-1]["operation"] == "completed" and events[-1]["completed_files"] == 2
    # Even a self-consistent file manifest cannot replace checkpoint provenance.
    report["checkpoint_sha256"] = "different-checkpoint"
    c.write(report_path, report)
    shutil.rmtree(c.REPO / c.eval_root("P02"))
    with pytest.raises(ValueError, match="Evaluation identity changed"):
        backend.verify(c.Stage("evaluate", "P02"), receipt())


def test_restoring_state_keeps_current_vm_progress_visible(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    if shutil.which("rclone") is None:
        pytest.skip("rclone executable required for isolated alias integration")
    remote = tmp_path / "remote"
    source = remote / "tennis_lab" / c.STATE
    c.write(source / "progress.json", {"phase": "stale_previous_vm"})
    c.write(source / "done/inputs.json", {"receipt": "previous-inputs"})
    config = tmp_path / "rclone.conf"
    config.write_text(f"[test]\ntype = alias\nremote = {remote}\n")
    monkeypatch.setattr(c, "REPO", tmp_path / "local")
    backend = object.__new__(runtime.ColabBackend)
    backend.state = c.REPO / c.STATE
    backend.drive = runtime.RcloneBackend(
        remote_root="test:tennis_lab", executable="rclone", config_path=config
    )
    c.write(backend.state / "progress.json", {"phase": "restoring", "stage": "restore"})
    observed: list[str] = []

    def progress(**fields: Any) -> None:
        observed.append(c.read(backend.state / "progress.json")["phase"])

    backend.restore_state(progress)
    assert observed and set(observed) == {"restoring"}
    assert c.read(backend.state / "done/inputs.json") == {"receipt": "previous-inputs"}
