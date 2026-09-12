"""Stage a Drive tar and locate its one dataset.json without assuming tar prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def prepare_archive(archive: Path, destination: Path) -> Path:
    """Extract ordinary files/directories only and atomically install the dataset."""
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    receipt = destination / ".archive-sha256"
    if destination.exists():
        if (
            destination.is_symlink()
            or not receipt.is_file()
            or receipt.read_text().strip() != digest
        ):
            raise ValueError(
                f"Existing dataset is not from this archive; choose a fresh destination: {destination}"
            )
        if not (destination / "dataset.json").is_file():
            raise ValueError(f"Incomplete extracted dataset: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".meiji-extract-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        # Copy Drive input once before random access/extraction on VM-local disk.
        local_archive = staging / "input.tar"
        shutil.copyfile(archive, local_archive)
        with local_archive.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise ValueError("Archive changed while being copied")
        extracted = staging / "extracted"
        extracted.mkdir()
        with tarfile.open(local_archive, "r:*") as bundle:
            members = bundle.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f"Unsupported archive member: {member.name}")
            bundle.extractall(extracted, members=members, filter="data")
        candidates = [
            p.parent
            for p in extracted.rglob("dataset.json")
            if (p.parent / "clips").is_dir()
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one dataset.json with clips/, found {len(candidates)}"
            )
        source = candidates[0]
        manifest = json.loads((source / "dataset.json").read_text())
        if manifest["version"] != 1 or not manifest["clips"]:
            raise ValueError("Empty or unsupported multiview dataset manifest")
        (source / ".archive-sha256").write_text(digest + "\n")
        source.rename(destination)
    return destination


def main() -> None:
    """Command-line staging entry point, also usable from an ordinary Colab cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_archive(args.archive, args.destination))


if __name__ == "__main__":
    main()
