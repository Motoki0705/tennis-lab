"""Construct archive expectations from prior source records, never archive bytes."""
import argparse
import hashlib
import json
from pathlib import Path

from rebuild_dataset import sha, write


def expected(inventory, markers, uncovered, output):
    raw = json.loads(inventory.read_text())
    files = {entry["path"]: entry for entry in raw["entries"] if entry["type"] == "file"}
    if len(files) != 1917 or sum(entry["size_bytes"] for entry in files.values()) != 19243387824:
        raise ValueError("Original closed architecture tree changed")
    known = {}
    for marker in markers.rglob("_SNAPSHOT.json"):
        name = marker.relative_to(markers).as_posix()
        contents = marker.read_bytes()
        if hashlib.md5(contents).hexdigest() != files[name]["hashes"]["md5"]:
            raise ValueError("Source marker differs from preflight inventory")
        known[name] = sha(marker)
        for path, identity in json.loads(contents)["files"].items():
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid source snapshot path")
            target = (marker.parent.relative_to(markers) / relative).as_posix()
            if files[target]["size_bytes"] != identity["size_bytes"]:
                raise ValueError("Snapshot size differs from inventory")
            if target in known and known[target] != identity["sha256"]:
                raise ValueError("Overlapping trusted source records disagree")
            known[target] = identity["sha256"]
    for source in uncovered.rglob("*"):
        if not source.is_file():
            continue
        name = source.relative_to(uncovered).as_posix()
        if name in known:
            raise ValueError("Unexpected uncovered duplicate")
        contents = source.read_bytes()
        if len(contents) != files[name]["size_bytes"] or hashlib.md5(contents).hexdigest() != files[name]["hashes"]["md5"]:
            raise ValueError("Source metadata differs from preflight inventory")
        known[name] = hashlib.sha256(contents).hexdigest()
    if set(known) != set(files):
        raise ValueError("Archive expectations do not cover the complete tree")
    result = dict(schema_version=1, files=known)
    if output.exists() and json.loads(output.read_text()) != result:
        raise ValueError("Existing archive expectations differ")
    write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "markers", "uncovered", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    expected(args.inventory, args.markers, args.uncovered, args.output)
