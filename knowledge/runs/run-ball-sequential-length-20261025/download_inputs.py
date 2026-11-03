import _sha256
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

source = Path(
    "/content/drive/MyDrive/tennis_lab/data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1"
)
target = Path(
    "/content/tennis-lab/data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1"
)
receipts = Path(
    "/content/tennis-lab/outputs/ball_detection/precompute/sequential_length/drive-download-20261025"
)
receipts.mkdir(parents=True, exist_ok=True)
raw = (source / "distribution.json").read_bytes()
manifest = json.loads(raw)
assert manifest["schema_version"] == "ball_frame_store_distribution.v1"
assert manifest["source"]["version"] == "ball-mix-v2-existing500-720p-20261018"
entries = manifest["files"]
assert len({e["path"] for e in entries}) == len(entries)
total = sum(e["size_bytes"] for e in entries)
if total + 12 * 2**30 > shutil.disk_usage("/content").free:
    raise RuntimeError(f"Insufficient space for {total} bytes plus 12 GiB reserve")
target.mkdir(parents=True, exist_ok=True)
if (target / "distribution.json").exists() and (
    target / "distribution.json"
).read_bytes() != raw:
    raise RuntimeError("Existing local distribution manifest differs")
(target / "distribution.json").write_bytes(raw)
start = time.time()
records = []


def transfer(entry):
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe distribution path")
    src, dst = source / relative, target / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    before = src.stat()
    if before.st_size != entry["size_bytes"] or src.is_symlink():
        raise RuntimeError(f"Source metadata mismatch: {relative}")
    exists = dst.exists()
    candidate = dst if exists else dst.with_name(dst.name + ".partial")
    primary, independent = hashlib.sha256(), _sha256.sha256()
    count = 0
    reader = dst if exists else src
    with reader.open("rb") as stream:
        output = None if exists else candidate.open("wb")
        try:
            while chunk := stream.read(4 * 1024 * 1024):
                primary.update(chunk)
                independent.update(chunk)
                count += len(chunk)
                if output is not None:
                    output.write(chunk)
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
        finally:
            if output is not None:
                output.close()
    if (
        count != entry["size_bytes"]
        or primary.hexdigest() != independent.hexdigest()
        or primary.hexdigest() != entry["sha256"]
    ):
        raise RuntimeError(f"Size/SHA-256 mismatch: {relative}")
    after = src.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed: {relative}")
    if not exists:
        candidate.replace(dst)
    return dict(
        path=str(relative), size_bytes=count, sha256=primary.hexdigest(), reused=exists
    )


def publish(complete=False):
    state = dict(
        schema="ball_verified_mount_download.v1",
        source_folder_url="https://drive.google.com/drive/folders/1-JFH7qD9vrlvKU7aNiOqFcQnG6IiFkiP",
        source=str(source),
        destination=str(target),
        expected_files=len(entries),
        expected_bytes=total,
        verified_files=len(records),
        verified_bytes=sum(r["size_bytes"] for r in records),
        elapsed_seconds=time.time() - start,
        complete=complete,
        distribution_sha256=hashlib.sha256(raw).hexdigest(),
    )
    temp = receipts / "progress.tmp"
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(receipts / "progress.json")
    print(json.dumps(state), flush=True)
    if complete:
        (receipts / "files.json").write_text(json.dumps(records, indent=2) + "\n")


publish()
# Materialize the small metadata and annotations first; RGB follows in parallel.
ordered = sorted(entries, key=lambda e: (e["path"].startswith("shards/"), e["path"]))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for future in concurrent.futures.as_completed(
        [pool.submit(transfer, e) for e in ordered]
    ):
        records.append(future.result())
        if len(records) % 20 == 0:
            publish()
if (source / "distribution.json").read_bytes() != raw:
    raise RuntimeError("Remote distribution manifest changed during download")
actual = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
if actual != {e["path"] for e in entries} | {"distribution.json"}:
    raise RuntimeError("Unexpected local file membership")
publish(True)
