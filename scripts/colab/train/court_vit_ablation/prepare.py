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
        not path.parts
        or path.parts[0] != "data"
        or path.is_absolute()
        or ".." in path.parts
        or not (member.isfile() or member.isdir())
    ):
        raise ValueError(f"Unsafe dataset archive member: {member.name}")


def extract_members(archive: tarfile.TarFile, repository: Path) -> None:
    """Uploaded archives already contain the repository-relative data/ prefix."""
    for member in archive:
        validate_member(member)
        archive.extract(member, path=repository, filter="data")


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
                extract_members(archive, Path.cwd())
            if process.wait() != 0:
                raise RuntimeError(f"zstd extraction failed: {path}")
    required_paths = [
        Path(
            f"data/synthetic_data_generation/scenes/B0{index}/datasets/court/dataset.json"
        )
        for index in range(4)
    ]
    required_paths.extend(
        [Path("data/court"), Path("data/court_detection/derived_targets")]
    )
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Archive did not materialize the required dataset path: {required_path}"
            )
    print("All six dataset archives verified and extracted", flush=True)


if __name__ == "__main__":
    main()
