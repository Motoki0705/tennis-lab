"""Copy the pinned canonical dataset to an independently verified NVMe work root."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from guard import D_AVAILABLE_MIN_BYTES, E_AVAILABLE_MIN_BYTES, E_DRIVE, d_drive_available, e_drive_available
from rebuild_dataset import DISTRIBUTION, sha, write

PAYLOAD_FILES = 2925
PAYLOAD_BYTES = 157028755547


def copy_hot(source, destination, report):
    started = time.monotonic()
    source = source.resolve()
    if not source.is_relative_to(E_DRIVE):
        raise ValueError("Canonical dataset must be on the data volume")
    if destination.is_symlink() or destination.resolve().is_relative_to(E_DRIVE):
        raise ValueError("Hot destination must be a real directory outside the data volume")
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    if destination.stat().st_dev != Path("/").stat().st_dev:
        raise ValueError("Hot destination must be on the WSL root filesystem")
    manifest = source / "distribution.json"
    if sha(manifest) != DISTRIBUTION:
        raise ValueError("Canonical distribution differs from the pinned snapshot")
    entries = json.loads(manifest.read_text())["files"]
    if len(entries) != PAYLOAD_FILES or sum(e["size_bytes"] for e in entries) != PAYLOAD_BYTES:
        raise ValueError("Fixed payload count/bytes differ")
    status = dict(schema_version=1, complete=False, source=str(source), destination=str(destination),
                  distribution_sha256=DISTRIBUTION, verified_files=0, verified_bytes=0,
                  copied_files=0, copied_bytes=0, reused_files=0, root_device=destination.stat().st_dev,
                  canonical_device=source.stat().st_dev, hardlinks_only=False)
    write(report, status)
    try:
        for entry in [dict(path="distribution.json", sha256=DISTRIBUTION, size_bytes=manifest.stat().st_size), *entries]:
            if d_drive_available() < D_AVAILABLE_MIN_BYTES or e_drive_available() < E_AVAILABLE_MIN_BYTES:
                raise OSError("Insufficient D:/E: space for the hot dataset copy")
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid distribution path")
            original, target = source / relative, destination / relative
            before = original.stat()
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.parent.resolve().is_relative_to(destination) or target.parent.stat().st_dev != status["root_device"]:
                raise ValueError("Hot payload parent escaped the WSL root destination")
            existed = target.exists()
            candidate = target if existed else target.with_name(target.name + ".copying")
            if existed and target.is_symlink():
                raise ValueError("Hot payload must be an independent regular file")
            if not existed:
                if candidate.exists():
                    raise ValueError("Interrupted partial copy needs inspection: " + str(candidate))
                shutil.copyfile(original, candidate)
            if (candidate.stat().st_size != entry["size_bytes"]
                    or sha(candidate) != entry["sha256"]):
                raise ValueError("Copied payload differs: " + str(candidate))
            if not candidate.resolve().is_relative_to(destination) or candidate.stat().st_dev != status["root_device"]:
                raise ValueError("Hot payload escaped the WSL root destination")
            after = original.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("Canonical payload changed while copying")
            if os.path.samefile(original, candidate) or candidate.stat().st_nlink != 1:
                raise ValueError("Hot payload must be an independent copy")
            if not existed:
                candidate.replace(target)
            if relative.as_posix() != "distribution.json":
                status["verified_files"] += 1
                status["verified_bytes"] += entry["size_bytes"]
                status["reused_files" if existed else "copied_files"] += 1
                if not existed:
                    status["copied_bytes"] += entry["size_bytes"]
            status["elapsed_seconds"] = time.monotonic() - started
            write(report, status)
            if status["verified_files"] % 100 == 0:
                print(json.dumps({k: status[k] for k in ("verified_files", "verified_bytes", "elapsed_seconds")}), flush=True)
        actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
        if actual != {entry["path"] for entry in entries} | {"distribution.json"}:
            raise ValueError("Unexpected hot dataset files")
        status.update(complete=True, all_sha256_verified=True, independent_copy=True,
                      additional_payload_bytes=status["copied_bytes"], finished_at=time.time())
        write(report, status)
        return status
    except BaseException as error:
        status.update(error=str(error), elapsed_seconds=time.monotonic() - started, failed_at=time.time())
        write(report, status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    copy_hot(args.source, args.destination, args.report)
