"""Hardlink the exact distribution; read/hash every payload before training."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

DISTRIBUTION = "f8425ec2cc6f4d0ae018a3c9387cb3d48b3bd8353794a82888936c3d0bb7477b"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def rebuild(source, destination, report):
    manifest = source / "distribution.json"
    if sha(manifest) != DISTRIBUTION:
        raise ValueError("Distribution differs from #1859 pinned snapshot")
    entries = json.loads(manifest.read_text())["files"]
    if len(entries) != 2925 or sum(e["size_bytes"] for e in entries) != 157028755547:
        raise ValueError("Fixed snapshot file count/bytes differ")
    started = time.monotonic()
    status = dict(schema_version=1, complete=False, distribution_sha256=DISTRIBUTION,
                  expected_files=2925, expected_bytes=157028755547,
                  verified_files=0, verified_bytes=0, source=str(source),
                  destination=str(destination), hardlinks_only=True)
    for entry in [dict(path="distribution.json", sha256=DISTRIBUTION, size_bytes=manifest.stat().st_size), *entries]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid distribution path")
        original, target = source / relative, destination / relative
        before = original.stat()
        if before.st_size != entry["size_bytes"] or sha(original) != entry["sha256"]:
            raise ValueError("Payload differs: " + str(original))
        after = original.stat()
        if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
            raise ValueError("Payload changed while hashing")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not os.path.samefile(original, target):
                raise ValueError("Destination is not the expected hardlink")
        else:
            os.link(original, target)
        if relative.as_posix() != "distribution.json":
            status["verified_files"] += 1
            status["verified_bytes"] += before.st_size
        status["elapsed_seconds"] = time.monotonic() - started
        write(report, status)
        if status["verified_files"] % 100 == 0:
            print(json.dumps({k: v for k, v in status.items() if k.startswith("verified") or k == "elapsed_seconds"}), flush=True)
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    if actual != {e["path"] for e in entries} | {"distribution.json"}:
        raise ValueError("Unexpected dataset files")
    status.update(complete=True, all_sha256_verified=True, additional_payload_bytes=0)
    write(report, status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.source, args.destination, args.report)
