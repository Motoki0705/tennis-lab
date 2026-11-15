"""Resident input retention and explicitly allowed E: references."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from tests.support.agents_ops.storage_tier import OPS, Lab, make_lab


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lab:
    monkeypatch.syspath_prepend(str(OPS))
    return make_lab(tmp_path)


def declarations(
    lab: Lab, *, residents: tuple[str, ...] = (), links: tuple[str, ...] = ()
) -> None:
    # Inject declarations independently of parsing so regression probes also
    # exercise old discovery/deletion code. Config.load is tested separately.
    object.__setattr__(lab.config, "residents", residents)
    object.__setattr__(lab.config, "external_links", links)


def external(lab: Lab, path: str = "data/synthetic_data_generation") -> Path:
    declarations(lab, links=(path,))
    target = lab.config.mirror / path
    target.mkdir(parents=True)
    (target / "kept.bin").write_bytes(b"authoritative external input")
    link = lab.config.repo / path
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    os.utime(link, ns=(1, 1), follow_symlinks=False)
    return cast(Path, link)


def config_file(lab: Lab, tmp_path: Path, **fields: Any) -> Path:
    data = {
        "mirror_mount": str(lab.config.mount),
        "dataset_containers": ["data/task"],
        "dataset_units": [],
        "resident_units": ["data/smplx", "data/tennis_multiview/raw/meiji_4cam"],
        "unmanaged_data_links": ["data/synthetic_data_generation"],
        **fields,
    }
    path = tmp_path / "config.toml"
    path.write_text(
        "\n".join(f"{key} = {json.dumps(value)}" for key, value in data.items())
    )
    return path


def test_resident_and_reference_configuration_is_explicit(
    lab: Lab, tmp_path: Path
) -> None:
    config = lab.layout.Config.load(lab.config.repo, config_file(lab, tmp_path))
    assert config.residents == ("data/smplx", "data/tennis_multiview/raw/meiji_4cam")
    assert config.external_links == ("data/synthetic_data_generation",)


@pytest.mark.parametrize(
    "fields",
    [
        {"resident_units": ["data/task/version"]},
        {"resident_units": ["data/smplx", "data/smplx"]},
        {"unmanaged_data_links": ["data/task"]},
        {"unmanaged_data_links": ["data/smplx"]},
        {"unmanaged_data_links": ["../outside"]},
        {"unmanaged_data_links": ["outputs/other"]},
    ],
)
def test_new_boundaries_reject_overlap_or_invalid_paths(
    lab: Lab, tmp_path: Path, fields: dict[str, Any]
) -> None:
    with pytest.raises(lab.filesystem.StorageError):
        lab.layout.Config.load(lab.config.repo, config_file(lab, tmp_path, **fields))


@pytest.mark.parametrize(
    "path", ["data/smplx", "data/tennis_multiview/processed/meiji_4cam"]
)
def test_resident_roundtrip_keeps_old_inputs_and_records_hashes(
    lab: Lab, path: str
) -> None:
    declarations(lab, residents=(path,))
    lab.unit(path)
    unit = lab.layout.Unit(path, "resident")
    lab.engine.mirror(unit, dry_run=False)
    assert lab.engine.evict(unit, dry_run=False)["action"] == "kept"
    assert (lab.config.repo / path / "a.bin").read_bytes() == b"apple\n"
    found = lab.layout.discover(lab.config, lab.config.repo)
    assert unit in found.units and found.errors == []
    assert (
        lab.engine.manifest(unit)["mirror"]["a.bin"]["sha256"]
        == hashlib.sha256(b"apple\n").hexdigest()
    )
    lab.age(unit, 15)
    assert lab.run("cycle", unit)[0] == 0
    assert (lab.config.repo / path / "a.bin").exists()
    saved = lab.config.repo.parent / "retained-input"
    lab.filesystem.rename(lab.config.repo / path, saved)
    assert lab.run("hydrate", unit)[0] == 0
    assert lab.run("verify", unit)[0] == 0
    assert (lab.config.repo / path / "nested/b.bin").read_bytes() == b"\x00\x01\x04\x09"


@pytest.mark.parametrize("operation", ["scan", "cycle", "mirror", "evict", "hydrate"])
def test_allowed_reference_is_validated_once_without_discovering_payload(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    link = external(lab)
    target = lab.config.mirror / link.relative_to(lab.config.repo)
    before = lab.filesystem.snapshot(target, hashes=True)
    reads = []
    original = os.readlink

    def readlink(path: Any, **kwargs: Any) -> Any:
        if path == "":
            reads.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(os, "readlink", readlink)
    if operation == "hydrate":
        lab.filesystem.rename(
            lab.config.repo / unit.path, lab.config.repo.parent / "old"
        )
    code, report = lab.engine.run(
        operation, [] if operation in {"scan", "cycle"} else [unit.path], dry_run=True
    )
    assert code == 0, report
    assert report["layout_errors"] == []
    assert len(reads) == 1
    (record,) = report["unmanaged_links_D"]
    assert record["path"] == str(link.relative_to(lab.config.repo))
    assert record["target"] == str(target)
    assert record["atime_before_ns"] == 1
    assert record["atime_after_ns"] == link.lstat().st_atime_ns
    assert record["atime_changed"] == (
        record["atime_before_ns"] != record["atime_after_ns"]
    )
    assert before.entries == lab.filesystem.snapshot(target, hashes=True).entries
    assert all(row["unit"] != record["path"] for row in report.get("units", []))
    assert link.is_symlink()


@pytest.mark.parametrize(
    "operation", ["scan", "cycle", "mirror", "evict", "hydrate", "verify", "index"]
)
def test_wrong_reference_fails_even_for_an_explicit_unrelated_unit(
    lab: Lab, operation: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    link = external(lab)
    link.unlink()
    link.symlink_to(lab.config.mirror / "data/task/v1")
    with pytest.raises(lab.filesystem.StorageError, match="unmanaged.*link|reference"):
        lab.engine.run(
            operation,
            [] if operation in {"scan", "cycle"} else [unit.path],
            dry_run=True,
        )
    assert (lab.config.repo / unit.path / "a.bin").read_bytes() == b"apple\n"


@pytest.mark.parametrize(
    "bad",
    [
        "undeclared",
        "dangling",
        "relative",
        "real-directory",
        "target-symlink",
        "ancestor-symlink",
    ],
)
def test_external_reference_never_uses_an_unverified_target(lab: Lab, bad: str) -> None:
    link = external(lab)
    target = lab.config.mirror / link.relative_to(lab.config.repo)
    if bad == "undeclared":
        declarations(lab)
    elif bad == "dangling":
        lab.filesystem.rename(target, target.with_name("preserved"))
    elif bad == "relative":
        link.unlink()
        link.symlink_to(os.path.relpath(target, link.parent))
    elif bad == "real-directory":
        link.unlink()
        link.mkdir()
    elif bad == "target-symlink":
        other = target.with_name("preserved")
        lab.filesystem.rename(target, other)
        target.symlink_to(other)
    else:
        root = lab.config.repo / "data"
        lab.filesystem.rename(root, root.with_name("preserved-data"))
        root.symlink_to(root.with_name("preserved-data"))
        # Root migration aliases remain a layout error and must not be followed.
        code, report = lab.engine.run("scan", [], dry_run=True)
        assert code == 1 and report["layout_errors"]
        return
    with pytest.raises(lab.filesystem.StorageError):
        lab.engine.run("scan", [], dry_run=True)


def test_nested_references_keep_meiji_resident_and_other_inputs_external(
    lab: Lab,
) -> None:
    path = "data/tennis_multiview/raw/meiji_4cam"
    other = "data/tennis_multiview/raw/meiji_3cam"
    lab.unit(path)
    external(lab, other)
    declarations(lab, residents=(path,), links=(other,))
    code, report = lab.engine.run("scan", [], dry_run=True)
    assert code == 0, report
    assert {r["unit"] for r in report["units"]} == {path}
    assert report["units"][0]["kind"] == "resident"
    assert other in report["unmanaged"]["D"]


def test_missing_e_mount_rejects_links_before_readlink(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    external(lab)

    def absent(config: Any) -> None:
        raise lab.filesystem.StorageError("E: not mounted")

    def forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("readlink before mount verification")

    lab.engine.mount_check = absent
    monkeypatch.setattr(os, "readlink", forbidden)
    with pytest.raises(lab.filesystem.StorageError, match="not mounted"):
        lab.engine.run("scan", [], dry_run=True)
    assert not lab.config.state.exists()


def test_actual_deletion_audits_references_and_keeps_external_payload(lab: Lab) -> None:
    unit = lab.unit()
    link = external(lab)
    target = lab.config.mirror / link.relative_to(lab.config.repo)
    before = lab.filesystem.snapshot(target, hashes=True)
    lab.mirror(unit)
    assert lab.run("evict", unit)[1]["results"][0]["action"] == "evicted"
    deleted = next(row for row in lab.records() if row["event"] == "deleted")
    assert deleted["unmanaged_links_D"][0]["verified"] is True
    assert before.entries == lab.filesystem.snapshot(target, hashes=True).entries
    assert link.is_symlink()


def test_file_reference_is_unmanaged_and_preserves_target_metadata(lab: Lab) -> None:
    relative = "data/tennis_multiview/processed/meiji_3cam.tar"
    declarations(lab, links=(relative,))
    source = lab.config.mirror / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"existing archive")
    os.utime(source, ns=(1, 1))
    link = lab.config.repo / relative
    link.parent.mkdir(parents=True)
    link.symlink_to(source)
    before = source.stat()
    code, report = lab.engine.run("scan", [], dry_run=True)
    assert code == 0, report
    assert source.stat() == before
    assert relative in report["unmanaged"]["D"]
    assert relative in report["unmanaged"]["E"]
    assert report["units"] == []


@pytest.mark.parametrize("operation", ["mirror", "evict", "hydrate"])
def test_reference_cannot_be_selected_as_a_managed_unit(
    lab: Lab, operation: str
) -> None:
    link = external(lab)
    target = lab.config.mirror / link.relative_to(lab.config.repo)
    before = lab.filesystem.snapshot(target, hashes=True)
    with pytest.raises(lab.filesystem.StorageError, match="unknown/ambiguous"):
        lab.engine.run(
            operation, [str(link.relative_to(lab.config.repo))], dry_run=True
        )
    assert before.entries == lab.filesystem.snapshot(target, hashes=True).entries
    assert link.is_symlink()


@pytest.mark.parametrize("wrong", [False, True])
def test_activate_validates_external_references(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    wrong: bool,
) -> None:
    module = importlib.import_module("storage_tier.__main__")
    for name in ("outputs", "ckpt"):
        (lab.config.repo / name).mkdir()
    link = external(lab)
    if wrong:
        link.unlink()
        link.symlink_to(lab.config.mirror / "outside")
    monkeypatch.setattr(module, "main_checkout", lambda path: lab.config.repo)
    monkeypatch.setattr(module, "common_directory", lambda path: lab.config.repo)
    monkeypatch.setattr(module.Config, "load", lambda *args: lab.config)
    monkeypatch.setattr(module, "Engine", lambda config: lab.engine)
    monkeypatch.setenv(module.CHECKOUT_ROOT_ENV, str(lab.config.repo))
    monkeypatch.setattr("sys.argv", ["storage_tier", "activate", "--dry-run"])
    code = module.main()
    captured = capsys.readouterr()
    assert code == int(wrong), captured
    assert not lab.config.state.exists()
