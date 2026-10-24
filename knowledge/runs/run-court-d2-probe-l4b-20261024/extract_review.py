"""Extract exactly the hash-pinned JPEG/JSON review inventory into a new root."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        assert all(member.isfile() for member in members)
        names = [member.name for member in members]
        assert len(names) == len(set(names))
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and ".." not in path.parts
            assert path.as_posix() == name
        source = archive.extractfile("review-manifest.json")
        assert source is not None
        manifest_bytes = source.read()
        manifest = json.loads(manifest_bytes)
        assert manifest["schema"] == "court_d2_review_bundle_v1"
        entries = {entry["path"]: entry for entry in manifest["files"]}
        assert len(entries) == len(manifest["files"])
        assert set(names) == set(entries) | {"review-manifest.json"}
        args.output.mkdir(parents=True, exist_ok=False)
        for member in members:
            source = archive.extractfile(member)
            assert source is not None
            content = source.read()
            if member.name != "review-manifest.json":
                entry = entries[member.name]
                assert len(content) == entry["bytes"]
                assert hashlib.sha256(content).hexdigest() == entry["sha256"]
            else:
                assert content == manifest_bytes
            target = args.output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print(
        f"Verified and extracted {len(names)} review files; raw arrays remain on Drive"
    )


if __name__ == "__main__":
    main()
