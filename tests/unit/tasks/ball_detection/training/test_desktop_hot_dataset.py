"""An immutable canonical source supplies a verified independent root-filesystem copy."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BUNDLE = Path("knowledge/runs/run-ball-arch-desktop-20261113")


@pytest.fixture
def copying(monkeypatch, tmp_path):
    def load(name):
        spec = importlib.util.spec_from_file_location("hot_dataset_" + name, BUNDLE / (name + ".py"))
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monkeypatch.setitem(sys.modules, "rebuild_dataset", load("rebuild_dataset"))
    monkeypatch.setitem(sys.modules, "guard", load("guard"))
    module = load("copy_hot_dataset")
    canonical = tmp_path / "canonical"
    source = canonical / "dataset"
    source.mkdir(parents=True)
    entries = []
    for name, content in [("index.npz", b"asymmetric index"), ("videos/example.mp4", b"video payload")]:
        path = source / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        entries.append(dict(path=name, size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    declaration = source / "distribution.json"
    declaration.write_text(json.dumps(dict(files=entries)))
    monkeypatch.setattr(module, "DISTRIBUTION", hashlib.sha256(declaration.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "PAYLOAD_FILES", 2)
    monkeypatch.setattr(module, "PAYLOAD_BYTES", 29)
    monkeypatch.setattr(module, "E_DRIVE", canonical)
    monkeypatch.setattr(module, "d_drive_available", lambda: 20 * 1024**3)
    monkeypatch.setattr(module, "e_drive_available", lambda: 20 * 1024**3)
    return module, source, tmp_path / "hot", tmp_path / "receipt.json"


def test_copy_is_independent_verified_and_reuses_only_valid_completed_files(copying):
    module, source, hot, report = copying
    original = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    first = module.copy_hot(source, hot, report)
    assert first["complete"] and first["all_sha256_verified"] and first["independent_copy"]
    assert first["verified_files"] == first["copied_files"] == 2
    assert first["copied_bytes"] == first["verified_bytes"] == 29
    for name, value in original.items():
        assert (hot / name).read_bytes() == (source / name).read_bytes() == value
        assert not os.path.samefile(source / name, hot / name)
        assert (hot / name).stat().st_nlink == 1
    second = module.copy_hot(source, hot, report)
    assert second["reused_files"] == 2 and second["copied_files"] == second["copied_bytes"] == 0


def test_corrupt_copied_bytes_do_not_publish_a_complete_payload(copying, monkeypatch):
    module, source, hot, report = copying
    original = module.shutil.copyfile

    def corrupt(before, after):
        original(before, after)
        if before.name == "index.npz":
            after.write_bytes(b"corrupted index!")

    monkeypatch.setattr(module.shutil, "copyfile", corrupt)
    with pytest.raises(ValueError, match="Copied payload differs"):
        module.copy_hot(source, hot, report)
    assert not json.loads(report.read_text())["complete"]
    assert not (hot / "index.npz").exists()
    assert (source / "index.npz").read_bytes() == b"asymmetric index"


def test_wrong_existing_destination_is_preserved_and_rejected(copying):
    module, source, hot, report = copying
    hot.mkdir()
    (hot / "index.npz").write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="Copied payload differs"):
        module.copy_hot(source, hot, report)
    assert (hot / "index.npz").read_bytes() == b"do not overwrite"


@pytest.mark.parametrize("alias", [False, True])
def test_destination_on_canonical_volume_or_alias_is_rejected(copying, alias):
    module, source, hot, report = copying
    target = source.parent / "wrong"
    if alias:
        hot.symlink_to(target, target_is_directory=True)
        target = hot
    with pytest.raises(ValueError, match="real directory outside"):
        module.copy_hot(source, target, report)
    assert not report.exists()


def test_unexpected_destination_file_prevents_acceptance(copying):
    module, source, hot, report = copying
    hot.mkdir()
    (hot / "unexpected.bin").write_bytes(b"untracked")
    with pytest.raises(ValueError, match="Unexpected hot dataset files"):
        module.copy_hot(source, hot, report)
    assert not json.loads(report.read_text())["complete"]


def test_nested_alias_cannot_write_into_the_canonical_dataset(copying):
    module, source, hot, report = copying
    hot.mkdir()
    (hot / "videos").symlink_to(source / "videos", target_is_directory=True)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="parent escaped"):
        module.copy_hot(source, hot, report)
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


def test_disk_guard_stops_before_copying_payloads(copying, monkeypatch):
    module, source, hot, report = copying
    monkeypatch.setattr(module, "d_drive_available", lambda: 10 * 1024**3 - 1)
    with pytest.raises(OSError, match="Insufficient D:/E:"):
        module.copy_hot(source, hot, report)
    assert list(hot.iterdir()) == []


def test_bootstrap_keeps_input_links_on_e_and_selects_a_verified_hot_copy(copying, monkeypatch):
    module, source, hot, _ = copying
    monkeypatch.setitem(sys.modules, "copy_hot_dataset", module)
    monkeypatch.setitem(sys.modules, "state", SimpleNamespace(prepare=lambda root: None))
    spec = importlib.util.spec_from_file_location("hot_bootstrap", BUNDLE / "bootstrap_recovery.py")
    assert spec and spec.loader
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    monkeypatch.setattr(bootstrap, "rebuild", lambda *_: pytest.fail("Must not hardlink a D dataset into E"))
    monkeypatch.setattr(bootstrap.subprocess, "check_output", lambda *_args, **_kwargs: "inactive\n")
    old, root = source.parent / "previous", source.parent / "next"
    old.mkdir()
    (old / "production-registration-a4.json").write_text(json.dumps(dict(unit="stopped-a4.service")))
    for name in ("original", "original-preparation", "inputs", "env-original", "msda-build", "cache"):
        (old / name).mkdir()
        (old / name / "preserve.bin").write_bytes(b"immutable input")
    for name in ("storage-baseline.json", "inputs-verified.json", "environment-verified.json",
                 "environment-freeze.txt", "environment-exact-runtime.txt", "environment-missing-dependencies.txt",
                 "original-vm-pip-freeze.txt", "archive-acceptance.json", "msda-build-manifest.json",
                 "msda-numerical-check.json", "full-shape-equivalence.json"):
        (old / name).write_text("{}")
    bootstrap.main(old, root, hot_dataset=hot, dataset_source=source)
    status = json.loads((root / "recovery-bootstrap.json").read_text())
    assert status["next_queue_attempt"] == 5 and status["dataset_mode"] == "nvme_copy"
    assert (root / "dataset").resolve() == hot.resolve()
    assert os.path.samefile(old / "inputs/preserve.bin", root / "inputs/preserve.bin")
    assert not os.path.samefile(source / "index.npz", root / "dataset/index.npz")
