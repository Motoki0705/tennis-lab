"""Verify every pinned RGB shard before hardlinking the read-only local source."""

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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.manifest) != DISTRIBUTION:
        raise ValueError("Untrusted distribution manifest")
    manifest = json.loads(args.manifest.read_text())
    for name, expected in manifest["source"]["hashes"].items():
        if sha(args.source / name) != expected:
            raise ValueError("Original source identity differs: " + name)
    mapping = {clip["payload"]: clip["source_payload"] for clip in manifest["clips"]}
    shards = [item for item in manifest["files"] if item["path"].startswith("shards/")]
    records = []
    started = time.monotonic()
    for item in shards:
        source = args.source / mapping[item["path"]]
        target = args.destination / item["path"]
        stat = source.stat()
        digest = sha(source)
        if stat.st_size != item["size_bytes"] or digest != item["sha256"]:
            raise ValueError("RGB shard differs: " + str(source))
        after = source.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ino) != (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ino,
        ):
            raise ValueError("Source changed while hashing: " + str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not os.path.samefile(source, target):
                raise ValueError("Existing destination is not the expected hardlink")
        else:
            os.link(source, target)
        records.append(
            dict(
                path=item["path"],
                source=str(source),
                sha256=digest,
                size_bytes=stat.st_size,
                inode=stat.st_ino,
            )
        )
        write(
            args.report,
            dict(
                complete=len(records) == len(shards),
                verified_files=len(records),
                total_files=len(shards),
                verified_bytes=sum(r["size_bytes"] for r in records),
                seconds=time.monotonic() - started,
                files=records,
            ),
        )
        if len(records) % 100 == 0 or len(records) == len(shards):
            print(
                json.dumps(
                    dict(
                        verified=len(records),
                        total=len(shards),
                        seconds=time.monotonic() - started,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
