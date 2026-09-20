"""Verify the uploaded archive manifest and safely extract the fixed training set."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).parent


def validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not (member.isfile() or member.isdir())
    ):
        raise ValueError(f"Unsafe dataset archive member: {member.name}")


def main() -> None:
    expected = json.loads((HERE / "archives.json").read_text())
    actual = json.loads(Path("data/court_training_archives.json").read_text())
    if actual != expected:
        raise ValueError(
            "Drive training archive manifest differs from the inspected baseline"
        )
    required = sum(a["uncompressed_bytes"] for a in expected["archives"]) + 25 * 1024**3
    if shutil.disk_usage("data").free < required:
        raise RuntimeError(
            f"Insufficient extraction/checkpoint space; need {required} bytes"
        )
    for entry in expected["archives"]:
        path = Path("data") / entry["name"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != entry["size_bytes"] or digest != entry["sha256"]:
            raise ValueError(f"Archive integrity mismatch: {path}")
        print(f"Verified {path}; extracting", flush=True)
        with subprocess.Popen(
            ["zstd", "-dc", str(path)], stdout=subprocess.PIPE
        ) as process:
            assert process.stdout is not None
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    validate_member(member)
                    archive.extract(member, path="data", filter="data")
            if process.wait() != 0:
                raise RuntimeError(f"zstd extraction failed: {path}")
    print("All six dataset archives verified and extracted", flush=True)


if __name__ == "__main__":
    main()
