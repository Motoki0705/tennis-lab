import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.utils.checksum import dual_sha256

src = Path(
    "/content/drive/MyDrive/tennis_lab/data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1"
)
dst = Path(
    "/content/tennis-lab/data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1"
)
out = Path(
    "/content/tennis-lab/outputs/ball_detection/train/arch_efficiency/s42-20261111"
)
out.mkdir(parents=True, exist_ok=True)
raw = (src / "distribution.json").read_bytes()
assert (
    hashlib.sha256(raw).hexdigest()
    == "f8425ec2cc6f4d0ae018a3c9387cb3d48b3bd8353794a82888936c3d0bb7477b"
)
manifest = json.loads(raw)
files = manifest["files"]
assert len(files) == 2925 and sum(x["size_bytes"] for x in files) == 157028755547
assert len(set(x["path"] for x in files)) == len(files)
start = time.monotonic()


def sha(path):
    return dual_sha256(path)


def copy(entry):
    rel = Path(entry["path"])
    assert not rel.is_absolute() and ".." not in rel.parts
    s, d = src / rel, dst / rel
    if d.exists():
        assert d.stat().st_size == entry["size_bytes"] and sha(d) == entry["sha256"], (
            str(rel)
        )
        return entry
    d.parent.mkdir(parents=True, exist_ok=True)
    initial = s.stat()
    assert initial.st_size == entry["size_bytes"], str(s)
    partial = d.with_suffix(d.suffix + ".partial")
    h = hashlib.sha256()
    with s.open("rb") as source, partial.open("wb") as target:
        while chunk := source.read(8 * 1024 * 1024):
            target.write(chunk)
            h.update(chunk)
        target.flush()
        os.fsync(target.fileno())
    assert (
        h.hexdigest() == entry["sha256"]
        and partial.stat().st_size == entry["size_bytes"]
    ), str(s)
    assert s.stat().st_mtime_ns == initial.st_mtime_ns, str(s)
    assert sha(partial) == entry["sha256"], str(partial)
    partial.replace(d)
    return entry


dst.mkdir(parents=True, exist_ok=True)
(dst / "distribution.json").write_bytes(raw)
progress = {
    "schema": "ball_arch_input_verification.v1",
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "expected_files": len(files),
    "expected_bytes": sum(x["size_bytes"] for x in files),
    "verified_files": 0,
    "verified_bytes": 0,
    "complete": False,
}


def save():
    progress["elapsed_seconds"] = time.monotonic() - start
    p = out / "download-progress.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.replace(p)


save()
try:
    with (
        (out / "verified-files.jsonl").open("a") as receipts,
        concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool,
    ):
        for e in pool.map(copy, files):
            receipts.write(json.dumps(e) + "\n")
            receipts.flush()
            progress["verified_files"] += 1
            progress["verified_bytes"] += e["size_bytes"]
            save()
            if progress["verified_files"] % 100 == 0:
                print(progress, flush=True)
    assert (src / "distribution.json").read_bytes() == raw
    actual = {str(x.relative_to(dst)) for x in dst.rglob("*") if x.is_file()}
    assert actual == {x["path"] for x in files} | {"distribution.json"}, (
        "unexpected files"
    )
    progress["complete"] = True
    save()
    (out / "data-verified.json").write_text(json.dumps(progress, indent=2) + "\n")
except BaseException as e:
    progress["error"] = str(e)
    save()
    raise
