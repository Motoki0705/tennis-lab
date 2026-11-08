"""Explicit OAuth Drive staging for the frozen mounted-Drive campaign.

This adapter never patches the controller, receipts, source, or manifests.
Only the alias talks to Drive; the recovery filesystem contains controls only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import configparser
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

DATA = "data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1"
CAMPAIGN = "outputs/ball_detection/train/capacity_scaling/s42-20261103-main"
DOWNLOAD = "outputs/ball_detection/precompute/sequential_length/drive-download-20261025"
EXPECTED_FILES = 2925
EXPECTED_BYTES = 157028755547
DISTRIBUTION_SHA256 = "f8425ec2cc6f4d0ae018a3c9387cb3d48b3bd8353794a82888936c3d0bb7477b"
INPUT_MARKER = "b6cab45141f1ab50019870fe5ab7ea977d57a2a0af20751b676fafd0e3667a96"
SOURCE_HEAD = "e99753cc7f2da87199965faa7cf60c64f8bd006e"


def relative(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or path.as_posix() != value
            or any(p in {"", ".", ".."} for p in value.split("/"))
            or "\\" in value or ":" in value or "\n" in value):
        raise ValueError("Unsafe relative path")
    return value


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def make_alias(source, destination):
    config = configparser.ConfigParser(interpolation=None)
    config.read_file(source.open())
    if (config["gdrive"].get("type") != "drive"
            or not config["gdrive"].get("client_id")):
        raise ValueError("Explicit dedicated OAuth gdrive backend required")
    alias = configparser.ConfigParser(interpolation=None)
    alias["gdrive"] = dict(config["gdrive"])
    alias["mounted_drive"] = {"type": "alias", "remote": "gdrive:tennis_lab"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate_alias(destination)
        return
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        alias.write(stream)
    validate_alias(destination)


def validate_alias(path):
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise ValueError("Alias config must be private and regular")
    config = configparser.ConfigParser(interpolation=None)
    config.read_file(path.open())
    if (config["mounted_drive"].get("type") != "alias"
            or config["mounted_drive"].get("remote") != "gdrive:tennis_lab"
            or config["gdrive"].get("type") != "drive"
            or not config["gdrive"].get("client_id")):
        raise ValueError("Alias must connect the original tree to dedicated OAuth")


def rclone(config, *arguments):
    validate_alias(config)
    return subprocess.check_output(
        ["rclone", "--config", str(config), *map(str, arguments)], timeout=7200
    )


def inventory(config, remote):
    entries = json.loads(rclone(config, "lsjson", remote, "--recursive", "--hash"))
    result = {}
    seen = set()
    for entry in entries:
        name = relative(entry["Path"])
        if name in seen:
            raise ValueError("Duplicate remote path: " + name)
        seen.add(name)
        if not entry["IsDir"]:
            result[name] = entry
    return result


def distribution(raw, expected_files=EXPECTED_FILES, expected_bytes=EXPECTED_BYTES,
                 expected_sha=DISTRIBUTION_SHA256):
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Frozen distribution SHA256 differs")
    manifest = json.loads(raw)
    if (manifest["schema_version"] != "ball_frame_store_distribution.v1"
            or manifest["source"]["version"] != "ball-mix-v2-existing500-720p-20261018"):
        raise ValueError("Distribution schema/source differs")
    entries = manifest["files"]
    names = [relative(e["path"]) for e in entries]
    if (len(set(names)) != len(names) or len(entries) != expected_files
            or sum(e["size_bytes"] for e in entries) != expected_bytes):
        raise ValueError("Distribution membership/count/size differs")
    return entries


def check_membership(target, entries):
    paths = list(target.rglob("*"))
    if target.is_symlink() or any(p.is_symlink() for p in paths):
        raise ValueError("Dataset contains symlink")
    names = {p.relative_to(target).as_posix() for p in paths if p.is_file()}
    if names != {e["path"] for e in entries} | {"distribution.json"}:
        raise ValueError("Dataset file membership differs")


def verify_file(item):
    target, entry = item
    path = target / relative(entry["path"])
    before = path.stat()
    if path.is_symlink() or before.st_size != entry["size_bytes"]:
        raise ValueError("Payload size/type differs: " + entry["path"])
    primary = hashlib.sha256()
    independent = importlib.import_module(
        "_sha2" if sys.version_info >= (3, 12) else "_sha256"
    ).sha256()
    count = 0
    with path.open("rb") as stream:
        descriptor = os.fstat(stream.fileno())
        while chunk := stream.read(4 * 1024 * 1024):
            primary.update(chunk)
            independent.update(chunk)
            count += len(chunk)
        after_fd = os.fstat(stream.fileno())
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if (identity(before) != identity(descriptor) or identity(before) != identity(after_fd)
            or identity(before) != identity(after) or count != entry["size_bytes"]
            or primary.hexdigest() != independent.hexdigest()
            or primary.hexdigest() != entry["sha256"]):
        raise ValueError("Payload size/dual SHA/stability differs: " + entry["path"])
    return dict(path=entry["path"], size_bytes=count, sha256=primary.hexdigest(),
                independent_sha256=independent.hexdigest(), reused=True)


def verify_dataset(target, entries, receipts, state, workers=8):
    state = dict(state, complete=False, verified_files=0, verified_bytes=0)
    atomic_json(receipts / "progress.json", state)
    (receipts / "files.json").unlink(missing_ok=True)
    check_membership(target, entries)
    records = []
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(verify_file, (target, e)) for e in entries]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            state.update(verified_files=len(records),
                         verified_bytes=state["verified_bytes"] + record["size_bytes"],
                         elapsed_seconds=time.monotonic() - started)
            if len(records) % 20 == 0:
                atomic_json(receipts / "progress.json", state)
                print(json.dumps(state), flush=True)
    check_membership(target, entries)
    if (state["verified_files"] != state["expected_files"]
            or state["verified_bytes"] != state["expected_bytes"]):
        raise ValueError("Incomplete dataset receipt")
    atomic_json(receipts / "files.json", sorted(records, key=lambda r: r["path"]))
    # The caller rechecks Drive and the immutable manifest before completion.
    return state


def download(config, workspace, workers):
    remote = "mounted_drive:" + DATA
    receipts = workspace / DOWNLOAD
    atomic_json(receipts / "progress.json", dict(complete=False, phase="transfer"))
    (receipts / "files.json").unlink(missing_ok=True)
    raw = rclone(config, "cat", remote + "/distribution.json")
    entries = distribution(raw)
    before = inventory(config, remote)
    expected = {e["path"]: e["size_bytes"] for e in entries}
    if ({n: e["Size"] for n, e in before.items()}
            != expected | {"distribution.json": len(raw)}):
        raise ValueError("Remote dataset membership/size differs")
    target = workspace / DATA
    target.mkdir(parents=True, exist_ok=True)
    existing = {p.relative_to(target).as_posix(): p.stat().st_size
                for p in target.rglob("*") if p.is_file()}
    missing = sum(size for name, size in expected.items() if existing.get(name) != size)
    if missing + 12 * 2**30 > shutil.disk_usage(workspace).free:
        raise RuntimeError("Insufficient SSD for missing payloads plus 12 GiB")
    rclone(config, "copy", remote, target, "--checksum", "--transfers", "12",
           "--checkers", "16", "--stats", "60s")
    if (target / "distribution.json").read_bytes() != raw:
        raise ValueError("Transferred distribution differs")
    state = dict(schema="ball_verified_mount_download.v1", transport="rclone-dedicated-oauth",
                 source=remote, destination=str(target), expected_files=EXPECTED_FILES,
                 expected_bytes=EXPECTED_BYTES, distribution_sha256=DISTRIBUTION_SHA256)
    state = verify_dataset(target, entries, receipts, state, workers)
    if inventory(config, remote) != before or rclone(config, "cat", remote + "/distribution.json") != raw:
        raise ValueError("Remote dataset changed during verification")
    atomic_json(receipts / "progress.json", dict(state, complete=True))
    print(json.dumps(dict(state, complete=True)), flush=True)


def controls(config, stage):
    remote = "mounted_drive:" + CAMPAIGN
    files = inventory(config, remote)
    if (stage / CAMPAIGN).exists():
        raise ValueError("Use a new control-only staging tree")
    names = [n for n in files if n == "inputs-receipt.json"
             or n.startswith("persistence/") or n.endswith("/_SNAPSHOT.json")
             or n in {"COMPLETED.json", "ANALYSIS_COMPLETED.json"}]
    if "inputs-receipt.json" not in names:
        raise ValueError("Input receipt absent")
    destination = stage / CAMPAIGN
    destination.mkdir(parents=True)
    listing = stage / "controls-files.txt"
    listing.write_text("\n".join(names) + "\n")
    rclone(config, "copy", remote, destination, "--files-from", listing, "--checksum")
    # Preserve existence evidence for orphaned/initialization-only stages.
    for name in files:
        first = name.split("/")[0]
        if first.startswith(("common-", "query-")):
            (destination / first).mkdir(exist_ok=True)
    after = inventory(config, remote)
    if {n: after.get(n) for n in names} != {n: files[n] for n in names}:
        raise ValueError("Remote controls changed while staging")
    rclone(config, "check", remote, destination, "--files-from", listing, "--one-way")
    print(json.dumps(dict(control_stage=str(stage), controls=len(names))), flush=True)


def bootstrap(config, workspace):
    sys.path.insert(0, str(workspace))
    from src.utils.artifact_store import RcloneSnapshotStore
    target = Path("/tmp/ball-capacity-inputs-v1")
    RcloneSnapshotStore(target, remote="mounted_drive", remote_root=CAMPAIGN + "/inputs",
                        config_path=config, timeout_seconds=3600).restore(
                            "inputs-v1", manifest_sha256=INPUT_MARKER)
    bundle = target / "source.bundle"
    subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=workspace, check=True)
    subprocess.run(["git", "fetch", str(bundle), "HEAD"], cwd=workspace, check=True)
    worktree = workspace / ".claude/worktrees/ball-capacity"
    if worktree.exists():
        raise ValueError("Remote experiment worktree already exists; inspect before reuse")
    subprocess.run(["git", "worktree", "add", "-b", "codex/ball-capacity",
                    str(worktree), SOURCE_HEAD], cwd=workspace, check=True)
    (worktree / ".venv").symlink_to(workspace / ".venv")
    print(json.dumps(dict(worktree=str(worktree), frozen_head=SOURCE_HEAD)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["alias", "bootstrap", "data", "controls"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--credential-source", type=Path)
    p.add_argument("--workspace", type=Path, default=Path("/content/tennis-lab"))
    p.add_argument("--stage", type=Path)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.action == "alias":
        make_alias(a.credential_source, a.config)
    elif a.action == "bootstrap":
        bootstrap(a.config, a.workspace)
    elif a.action == "controls":
        controls(a.config, a.stage)
    else:
        download(a.config, a.workspace, a.workers)


if __name__ == "__main__":
    main()
