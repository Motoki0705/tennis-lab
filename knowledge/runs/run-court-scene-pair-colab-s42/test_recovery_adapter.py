"""Fault contracts with original campaign/report validators and fake Drive I/O."""

from __future__ import annotations

import copy
import hashlib
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as c  # noqa: E402
import recovery_adapter as a  # noqa: E402
import runtime as rt  # noqa: E402


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    repo, remote = tmp_path / "repo", tmp_path / "remote"
    repo.mkdir()
    remote.mkdir()
    monkeypatch.setattr(c, "REPO", repo)
    stage = c.Stage("evaluate", "P23")
    root = c.eval_root("P23")
    checkpoint = c.train_root("P23") / "logs/version_0/checkpoints/update-00033160.ckpt"
    cp = repo / checkpoint
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b"independent complete checkpoint")
    ids = [f"B0{scene}:image-{i}" for scene in range(4) for i in range(scene + 1)]
    ready = {
        "real_ids": ["real:one", "real:two", "real:three"],
        "synthetic_ids": ids,
        "evaluation_manifest_sha256": "fixed-manifest",
    }
    checked = {
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": c.sha(cp),
        "global_step": c.FINAL_STEP,
        "data_condition": {"source": "fixed"},
    }
    report: dict[str, Any] = {
        "schema": "court_data_scaling_evaluation_v2",
        "series": "scene-pairs",
        "global_step": c.FINAL_STEP,
        "checkpoint_sha256": checked["checkpoint_sha256"],
        "data_condition": checked["data_condition"],
        "synthetic_evaluation_manifest": {"sha256": "fixed-manifest"},
        "splits": {},
    }
    for name, values in [
        ("synthetic_court-val", ids),
        ("tennis_court_detector-val", ready["real_ids"]),
    ]:
        report["splits"][name] = {
            "sample_ids": values,
            "count": len(values),
            "batch_size": 8,
            "ordered_sample_ids_sha256": rt.ordered_digest(values),
            "pose_supervised": name.startswith("synthetic"),
            "metrics": dict.fromkeys(rt.METRICS, 0.3),
        }
    synthetic = report["splits"]["synthetic_court-val"]
    synthetic["scene_order"] = [f"B0{i}" for i in range(4)]
    synthetic["scenes"] = {}
    for n, scene in enumerate(synthetic["scene_order"]):
        values = [v for v in ids if v.startswith(scene + ":")]
        synthetic["scenes"][scene] = {
            "sample_ids": values,
            "count": len(values),
            "ordered_sample_ids_sha256": rt.ordered_digest(values),
            "metrics": dict.fromkeys(rt.METRICS, float(n + 1)),
        }
    synthetic["scene_macro_mean"] = dict.fromkeys(rt.METRICS, 2.5)
    target = remote / root
    target.mkdir(parents=True)
    c.write(target / "evaluation.json", report)
    (target / "domain").mkdir()
    (target / "domain/a.png").write_bytes(b"asymmetric image one")
    (target / "domain/b.png").write_bytes(b"asymmetric image two")
    state = repo / c.STATE
    c.write(state / "inputs.json", ready)

    class Drive:
        ambiguous = False
        fault = ""
        reads = 0

        def remote_path(self, path: str) -> str:
            return path

        def resolve(self, path: str) -> dict[str, Any]:
            if self.ambiguous:
                raise ValueError("Ambiguous ancestor")
            return {"ID": "closed-root", "IsDir": True}

        def json(self, args: list[str]) -> list[dict[str, Any]]:
            self.reads += 1
            entries = []
            for p in sorted(target.rglob("*")):
                name = p.relative_to(target).as_posix()
                entries.append(
                    {
                        "Path": name,
                        "ID": "id-" + name,
                        "IsDir": p.is_dir(),
                        "Size": -1 if p.is_dir() else p.stat().st_size,
                        "Hashes": {} if p.is_dir() else {"md5": md5(p)},
                    }
                )
            if self.fault == "duplicate":
                entries.append(copy.deepcopy(entries[-1]))
            elif self.fault == "duplicate_dir":
                entries.append({**entries[0], "ID": "other-directory-id"})
            elif self.fault == "wrong_hash":
                entries[-1]["Hashes"] = {"md5": "0" * 32}
            elif self.fault == "wrong_size":
                entries[-1]["Size"] += 1
            elif self.fault == "duplicate_id":
                entries[-1]["ID"] = entries[0]["ID"]
            elif self.fault == "missing":
                entries.pop()
            elif self.fault == "extra":
                entries.append(
                    {"Path": "unexpected", "ID": "extra", "IsDir": True, "Size": -1}
                )
            elif self.fault == "unsafe":
                entries[-1]["Path"] = "../escape"
            elif self.fault == "missing_hash":
                entries[-1]["Hashes"] = {}
            elif self.fault == "changed" and self.reads > 1:
                entries[-1]["ID"] = "replaced-during-verification"
            return entries

    class Backend(a.RecoveryBackend):
        def __init__(self) -> None:
            self.state = state
            self.drive = Drive()
            self.drive_cli = Path("/versioned/current/drive.py")
            self._qualified = None
            self.calls: list[tuple[str, ...]] = []
            self.transfer_fault = ""
            self.events: list[dict[str, Any]] = []

        def cli(self, *args: str) -> dict[str, Any]:
            self.calls.append(args)
            if args[0] == "download":
                source = remote / args[1]
                dest = Path(args[2])
                if source.is_dir():
                    shutil.copytree(source, dest)
                    if self.transfer_fault == "corrupt":
                        (dest / "domain/a.png").write_bytes(b"corrupt")
                    elif self.transfer_fault == "symlink":
                        (dest / "domain/a.png").unlink()
                        (dest / "domain/a.png").symlink_to(cp)
                    elif self.transfer_fault == "mutation":
                        (source / "domain/a.png").write_bytes(b"remote mutation")
                else:
                    shutil.copyfile(source, dest)
                return {"verified": True}
            if args[0] == "verify":
                relative = Path(args[2])
                other = remote / relative
                if relative == checkpoint:
                    return {"matches": Path(args[1]).read_bytes() == cp.read_bytes()}
                return {"matches": Path(args[1]).read_bytes() == other.read_bytes()}
            raise AssertionError(args)

        def checkpoint(
            self, condition: str, relative: Path, *, probe: bool = False
        ) -> dict[str, Any]:
            assert condition == "P23" and relative == checkpoint and not probe
            return copy.deepcopy(checked)

        def event(self, value: dict[str, Any]) -> None:
            self.events.append(value)

    b = Backend()
    result = {
        **checked,
        "evaluation": (root / "evaluation.json").as_posix(),
        "evaluation_sha256": c.sha(target / "evaluation.json"),
        "evaluation_files": {
            p.relative_to(remote).as_posix(): c.sha(p)
            for p in target.rglob("*")
            if p.is_file()
        },
    }

    def marker() -> None:
        c.write(
            state / "done/evaluate-P23.json",
            {
                "schema": c.SCHEMA,
                "stage": {"kind": "evaluate", "condition": "P23"},
                "inputs_identity": c.digest(ready),
                "result": result,
            },
        )

    marker()
    return b, stage, result, target, repo, ready, marker


def test_original_guards_scope_and_rng_are_preserved(setup: Any) -> None:
    import numpy as np
    import torch

    b, stage, result, _, repo, _, _ = setup
    py, np_state, ts = random.getstate(), np.random.get_state(), torch.get_rng_state()
    b.verify(stage, result)
    assert random.getstate() == py
    assert np.random.get_state()[0] == np_state[0]
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), ts)
    assert b._qualified is None
    assert len([v for v in b.calls if v[0] == "download"]) == 1
    assert [v[2] for v in b.calls if v[0] == "verify"] == [result["checkpoint"]]
    b.fetch(Path(result["evaluation"]))
    assert b.calls[-1][0] == "verify"
    (repo / result["evaluation"]).write_bytes(b"local corruption after cache discarded")
    with pytest.raises(ValueError, match="Drive bytes differ"):
        b.fetch(Path(result["evaluation"]))


@pytest.mark.parametrize(
    "fault",
    [
        "ambiguous",
        "duplicate",
        "duplicate_dir",
        "duplicate_id",
        "missing",
        "extra",
        "unsafe",
        "missing_hash",
        "wrong_hash",
        "wrong_size",
        "changed",
    ],
)
def test_remote_inventory_faults_fail_closed(setup: Any, fault: str) -> None:
    b, stage, result, *_ = setup
    if fault == "ambiguous":
        b.drive.ambiguous = True
    else:
        b.drive.fault = fault
    with pytest.raises((ValueError, KeyError)):
        b.verify(stage, result)
    assert b._qualified is None


@pytest.mark.parametrize("fault", ["corrupt", "symlink", "mutation"])
def test_transfer_faults_are_not_qualified(setup: Any, fault: str) -> None:
    b, stage, result, *_ = setup
    b.transfer_fault = fault
    with pytest.raises(ValueError):
        b.verify(stage, result)
    assert b._qualified is None


@pytest.mark.parametrize(
    "fault",
    ["input", "checkpoint", "schema", "lineage", "order", "count", "sha", "escape"],
)
def test_actual_original_identity_report_guards_run(setup: Any, fault: str) -> None:
    b, stage, result, target, _, _, marker = setup
    report = c.read(target / "evaluation.json")
    if fault == "input":
        marker_path = b.state / "done/evaluate-P23.json"
        value = c.read(marker_path)
        value["inputs_identity"] = "wrong"
        c.write(marker_path, value)
    elif fault == "checkpoint":
        result["checkpoint_sha256"] = "wrong"
        marker()
    elif fault == "sha":
        result["evaluation_files"][result["evaluation"]] = "0" * 64
        marker()
    elif fault == "escape":
        result["evaluation_files"]["../escape"] = "0" * 64
        marker()
    else:
        if fault == "schema":
            report["schema"] = "wrong"
        elif fault == "lineage":
            report["checkpoint_sha256"] = "wrong"
        elif fault == "order":
            report["splits"]["tennis_court_detector-val"]["sample_ids"] = [
                "real:three",
                "real:two",
                "real:one",
            ]
        elif fault == "count":
            report["splits"]["synthetic_court-val"]["count"] += 1
        c.write(target / "evaluation.json", report)
        result["evaluation_sha256"] = c.sha(target / "evaluation.json")
        result["evaluation_files"][result["evaluation"]] = c.sha(
            target / "evaluation.json"
        )
        marker()
    with pytest.raises(ValueError):
        b.verify(stage, result)
    assert b._qualified is None


def test_local_parent_symlink_and_stage_reuse_rejected(setup: Any) -> None:
    b, stage, result, _, repo, _, _ = setup
    parent = repo / c.eval_root("P23").parent
    parent.parent.mkdir(parents=True, exist_ok=True)
    other = repo / "elsewhere"
    other.mkdir()
    parent.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        b.verify(stage, result)
    assert b._qualified is None


def test_refresh_during_qualified_stage_uses_original_fetch(setup: Any) -> None:
    b, stage, result, *_ = setup
    original = b.check_evaluation

    def check(report: dict[str, Any], checked: dict[str, Any]) -> None:
        original(report, checked)
        b.fetch(Path(result["evaluation"]), refresh=True)

    b.check_evaluation = check
    b.verify(stage, result)
    assert len([v for v in b.calls if v[0] == "download"]) == 2


def test_unclosed_stage_never_bulk_qualifies(setup: Any) -> None:
    b, stage, result, target, repo, _, _ = setup
    (b.state / "done/evaluate-P23.json").unlink()
    shutil.copytree(target, repo / c.eval_root("P23"))
    b.verify(stage, result)
    assert not any(v[0] == "download" for v in b.calls)
    assert len([v for v in b.calls if v[0] == "verify"]) == 5


def test_protected_source_accepts_only_frozen_bytes() -> None:
    repo = Path(__file__).resolve().parents[3]
    objects = {
        name: subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{name}"], cwd=repo, text=True
        ).strip()
        for name in a.OBJECTS
    }
    files = {
        name: c.sha(repo / "knowledge/runs/run-court-scene-pair-colab-s42" / name)
        for name in a.FILES
    }
    if objects == a.OBJECTS and files == a.FILES:
        assert (
            a.protected_source(
                repo, require_frozen_head=False, require_submodule=False
            )["protected_source_changed"]
            is False
        )
    else:
        # A PR's merged CI checkout may contain newer science; the pinned VM must reject it.
        with pytest.raises(ValueError, match="Protected"):
            a.protected_source(repo, require_frozen_head=False, require_submodule=False)


def test_mutation_during_original_validation_is_rejected(setup: Any) -> None:
    b, stage, result, target, *_ = setup
    original = b.check_evaluation

    def check(report: dict[str, Any], checked: dict[str, Any]) -> None:
        original(report, checked)
        (target / "domain/a.png").write_bytes(b"mutated during original verification")

    b.check_evaluation = check
    with pytest.raises(ValueError, match="mutated"):
        b.verify(stage, result)
    assert b._qualified is None


def test_wrong_protected_digest_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[3]
    objects = {
        name: subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{name}"], cwd=repo, text=True
        ).strip()
        for name in a.OBJECTS
    }
    monkeypatch.setattr(a, "OBJECTS", objects)
    monkeypatch.setattr(a, "FILES", {**a.FILES, "runtime.py": "0" * 64})
    with pytest.raises(ValueError, match="bytes changed"):
        a.protected_source(
            Path(__file__).resolve().parents[3],
            require_frozen_head=False,
            require_submodule=False,
        )
