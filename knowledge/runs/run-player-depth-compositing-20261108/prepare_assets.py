"""Fetch the adopted manifest and B00 only, then verify every B00 input digest."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REMOTE = (
    "data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    args.receipts.mkdir(parents=True, exist_ok=False)
    cli = [sys.executable, ".agents/skills/tennis-drive/scripts/drive.py", "download"]
    for source, destination, receipt in (
        (
            REMOTE + "/assets.json",
            args.destination / "assets.json",
            "manifest-download.json",
        ),
        (REMOTE + "/scenes/B00", args.destination / "scenes/B00", "b00-download.json"),
    ):
        result = subprocess.run(
            [*cli, source, str(destination), "--verify"],
            check=True,
            capture_output=True,
            text=True,
        )
        (args.receipts / receipt).write_text(result.stdout)
    manifest = json.loads((args.destination / "assets.json").read_text())
    records = []
    for record in manifest["files"]:
        if not record["path"].startswith("scenes/B00/"):
            continue
        path = args.destination / record["path"]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if path.stat().st_size != record["size_bytes"] or digest != record["sha256"]:
            raise ValueError(f"Adopted input mismatch: {record['path']}")
        records.append(record)
    if len(records) != 6:
        raise ValueError("Expected exactly six adopted B00 input files")
    result = {
        "schema_version": 1,
        "remote": REMOTE,
        "verified_files": records,
        "available_scenes": [scene["scene_id"] for scene in manifest["scenes"]],
    }
    (args.receipts / "input-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
