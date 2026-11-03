"""Selectively transfer this run's inputs/outputs with the tennis-drive backend.

The stock JSON CLI has no file filter. Its inventory establishes path identity
and content; its RcloneBackend downloads those IDs and uploads a filtered copy.
No JPEG shard is downloaded and no existing remote file is replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DRIVE_SCRIPTS = REPO / ".agents/skills/tennis-drive/scripts"
sys.path.insert(0, str(DRIVE_SCRIPTS))

from drive_core import RcloneBackend  # noqa: E402

POOL = "data/court_scene_diversity_6000_v1"
REMOTE = "gdrive:tennis_lab"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(config: Path, output: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(DRIVE_SCRIPTS / "drive.py"),
            "--rclone-config",
            str(config),
            "--remote-root",
            REMOTE,
            "inventory",
            "--path",
            POOL,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"tennis-drive inventory failed ({completed.returncode}): "
            f"{completed.stdout.strip()} {completed.stderr.strip()}"
        )
    return read_inventory(output)


def read_inventory(path: Path) -> dict[str, Any]:
    document = read_json(path)
    if not isinstance(document, dict):
        raise ValueError("Inventory must be an object.")
    if (
        document["schema_version"] != 1
        or document["action"] != "inventory"
        or document["remote_root"] != REMOTE
        or document["path"] != POOL
        or document["ambiguous_paths"]
    ):
        raise ValueError("Unexpected/ambiguous tennis-drive inventory.")
    rows = {entry["path"]: entry for entry in document["entries"]}
    if len(rows) != len(document["entries"]):
        raise ValueError("Duplicate inventory path.")
    return document


def entries(document: dict[str, Any]) -> dict[str, Any]:
    return {entry["path"]: entry for entry in document["entries"]}


def verify_files(root: Path, paths: list[str], observed: dict[str, Any]) -> int:
    total = 0
    for name in paths:
        path = root / name
        entry = observed[name]
        if entry["type"] != "file" or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Size/type mismatch: {name}")
        hashes = {key.lower(): value for key, value in entry["hashes"].items()}
        common = hashes.keys() & {"md5", "sha1", "sha256"}
        if not common:
            raise ValueError(f"Unverifiable remote content: {name}")
        for algorithm in common:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, algorithm).hexdigest()
            if actual != hashes[algorithm]:
                raise ValueError(f"Content mismatch: {name}, {algorithm}")
        total += path.stat().st_size
    return total


def copy_filtered(
    config: Path, source: str, destination: str, paths: list[str], filter_path: Path
) -> None:
    # The preceding CLI inventory checks every ancestor and duplicate subtree
    # name. --immutable also refuses a conflicting concurrent destination.
    filter_path.write_text("".join(f"{name}\n" for name in paths))
    if not paths:
        return
    backend = RcloneBackend(
        executable="rclone", remote_root=REMOTE, config_path=config, timeout=1800
    )
    backend.run(
        [
            "copy",
            source,
            destination,
            "--files-from-raw",
            str(filter_path),
            "--immutable",
            "--no-traverse",
            "--stats",
            "30s",
            "--stats-log-level",
            "NOTICE",
            "--transfers",
            "4",
            "--checkers",
            "4",
        ],
        capture_output=False,
    )


def missing_files(root: Path, paths: list[str], observed: dict[str, Any]) -> list[str]:
    existing = [name for name in paths if (root / name).exists()]
    # Resume only explicitly verified bytes; a partial/corrupt file is an error.
    verify_files(root, existing, observed)
    existing_names = set(existing)
    return [name for name in paths if name not in existing_names]


def download_ids(
    config: Path, root: Path, paths: list[str], observed: dict[str, Any]
) -> None:
    """Download only IDs from the unambiguous, hashed CLI inventory.

    copyid accepts a local destination. It avoids repeated folder-name lookup
    in this deeply nested pool. Disjoint I/O batches keep RAM bounded;
    these are file transfers, not training or pytest workers.
    """
    pending = missing_files(root, paths, observed)
    if not pending:
        return
    for name in pending:
        if not isinstance(observed[name]["id"], str) or not observed[name]["id"]:
            raise ValueError(f"Inventory lacks a Drive ID: {name}")
        (root / name).parent.mkdir(parents=True, exist_ok=True)

    def batch(names: list[str]) -> None:
        if not names:
            return
        arguments = [
            "backend",
            "copyid",
            REMOTE,
            "--immutable",
            "--stats",
            "30s",
            "--stats-log-level",
            "NOTICE",
        ]
        for name in names:
            arguments.extend((observed[name]["id"], str((root / name).resolve())))
        backend = RcloneBackend(
            executable="rclone", remote_root=REMOTE, config_path=config, timeout=1800
        )
        backend.run(arguments, capture_output=False)

    # Every destination belongs to exactly one batch; there are no concurrent
    # Drive writes. Copy errors propagate and the complete inventory is checked.
    io_batches = min(16, len(pending))
    with ThreadPoolExecutor(max_workers=io_batches) as executor:
        futures = [
            executor.submit(batch, pending[offset::io_batches])
            for offset in range(io_batches)
        ]
        failures = []
        for future in futures:
            try:
                future.result()
            except Exception as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup("Metadata download batch failures", failures)


def fetch(args: argparse.Namespace) -> None:
    before = read_inventory(args.before)
    observed = entries(before)
    versions = sorted(name for name in observed if name.endswith(".dataset.json"))
    if len(versions) != 1 or "/" in versions[0]:
        raise ValueError("Expected exactly one root publication.")
    root = args.local_root
    root.mkdir(parents=True, exist_ok=True)
    initial = [
        versions[0],
        "generation-completion.json",
        "trajectory-partition.json",
        *(
            f"scene-{count}.{split}.subset.json"
            for count in range(1, 5)
            for split in ("train", "val")
        ),
    ]
    download_ids(args.rclone_config, root, initial, observed)
    verify_files(root, initial, observed)
    version = read_json(root / versions[0])
    selected = set(initial)
    for scene in version["scenes"]:
        selected.add(scene["accepted_ids_path"])
        for store in scene["stores"]:
            selected.update(
                (f"{store['path']}/dataset.json", f"{store['path']}/samples/index.npz")
            )
    paths = sorted(selected)
    if any(name not in observed for name in paths):
        raise ValueError("Publication references files absent from inventory.")
    download_ids(args.rclone_config, root, paths, observed)
    total = verify_files(root, paths, observed)
    complete_list = args.before.parent / "all-metadata-files.txt"
    complete_list.write_text("".join(f"{name}\n" for name in paths))
    if (
        sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        != paths
    ):
        raise ValueError("Metadata root contains unexpected files.")
    write_json(
        args.report,
        {
            "schema_version": 1,
            "action": "metadata-download",
            "remote_root": REMOTE,
            "pool": POOL,
            "observed_at": before["observed_at"],
            "inventory_sha256": sha256(args.before),
            "file_count": len(paths),
            "bytes": total,
            "jpeg_shards_downloaded": 0,
            "content_verified_against_tennis_drive_inventory": True,
            "transport": "tennis-drive RcloneBackend backend copyid --immutable (up to sixteen disjoint local I/O batches)",
            "files_list_sha256": sha256(complete_list),
        },
    )


def check_new_names(names: list[str], before: dict[str, Any]) -> None:
    if len(set(names)) != len(names) or any("/" in name for name in names):
        raise ValueError("Output must contain distinct root-level files.")
    if set(names) & before.keys():
        raise ValueError("Refusing to upload an existing filename.")


def publish(args: argparse.Namespace) -> None:
    if args.audit is None:
        raise ValueError("--audit is required before publishing.")
    audit = read_json(args.audit)
    if audit["status"] != "passed" or audit["stage"] != "verify":
        raise ValueError("A successful independent verify report is required.")
    for name, proof in audit["manifests"].items():
        if "/" in name or sha256(args.local_root / name) != proof["sha256"]:
            raise ValueError("Manifest changed after verification.")
    before = read_inventory(args.before)
    old = entries(before)
    current = inventory(args.rclone_config, args.before.parent / "drive-preupload.json")
    if entries(current) != old:
        raise ValueError(
            "Pool changed since the input snapshot; inspect before upload."
        )
    names = sorted(path.name for path in args.local_root.iterdir())
    if (
        names != sorted(audit["manifests"])
        or len(names) != 11
        or any(
            not (args.local_root / name).is_file() or not name.endswith(".subset.json")
            for name in names
        )
    ):
        raise ValueError("Upload root must contain exactly the eleven new manifests.")
    check_new_names(names, old)
    copy_filtered(
        args.rclone_config,
        str(args.local_root),
        f"{REMOTE}/{POOL}",
        names,
        args.before.parent / "upload-files.txt",
    )
    after = inventory(args.rclone_config, args.before.parent / "drive-after.json")
    new = entries(after)
    changed = sorted(name for name, entry in old.items() if new.get(name) != entry)
    added = sorted(new.keys() - old.keys())
    if changed or added != names:
        raise ValueError(f"Unexpected Drive changes: changed={changed}, added={added}")
    total = verify_files(args.local_root, names, new)
    write_json(
        args.report,
        {
            "schema_version": 1,
            "action": "manifest-upload",
            "remote_root": REMOTE,
            "pool": POOL,
            "observed_at": datetime.now(UTC).isoformat(),
            "before_observed_at": before["observed_at"],
            "after_observed_at": after["observed_at"],
            "before_inventory_sha256": sha256(args.before),
            "after_inventory_sha256": sha256(args.before.parent / "drive-after.json"),
            "preexisting_entries_checked": len(old),
            "preexisting_files_checked": sum(
                row["type"] == "file" for row in old.values()
            ),
            "changed_preexisting_entries": changed,
            "added_entries": [new[name] for name in names],
            "uploaded_files": len(names),
            "uploaded_bytes": total,
            "content_verified_against_tennis_drive_inventory": True,
            "transport": "tennis-drive RcloneBackend.copy --files-from-raw --immutable --no-traverse",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fetch", "publish"))
    parser.add_argument("--rclone-config", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    options = parser.parse_args()
    {"fetch": fetch, "publish": publish}[options.stage](options)
