"""Archive selected weights without rewriting the original experiment receipts."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if not args.destination.is_absolute():
        parser.error("--destination must be an explicit absolute path")
    rows = json.loads(args.receipts.read_text())
    if len(rows) != 10:
        raise ValueError("Expected the ten completed runs")
    archived, entries = [], []
    for row in rows:
        receipt = row["receipt"]
        source = Path(receipt["checkpoint"])
        directory = args.destination / row["name"]
        destination = directory / source.name
        directory.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(".copying")
            if temporary.exists():
                raise FileExistsError(temporary)
            shutil.copy2(source, temporary)
            if checksum(temporary) != receipt["checkpoint_sha256"]:
                raise ValueError(f"Copied checkpoint differs: {source}")
            temporary.rename(destination)
        actual_sha = checksum(destination)
        if actual_sha != receipt["checkpoint_sha256"]:
            raise ValueError(f"Archive checksum mismatch: {destination}")
        relocated = {**receipt, "checkpoint": str(destination)}
        (directory / "selection.json").write_text(
            json.dumps(relocated, indent=2) + "\n"
        )
        archived.append({"name": row["name"], "receipt": relocated})
        entries.append(
            {
                "name": row["name"],
                "arm": receipt["arm"],
                "seed": receipt["seed"],
                "original_checkpoint": str(source),
                "checkpoint": str(destination),
                "size_bytes": destination.stat().st_size,
                "sha256": actual_sha,
                "relocated_receipt": str(directory / "selection.json"),
            }
        )
    (args.destination / "completed-training.json").write_text(
        json.dumps(archived, indent=2) + "\n"
    )
    manifest = {
        "schema": "blcs.checkpoint_storage.v1",
        "source_receipts": str(args.receipts),
        "source_receipts_sha256": checksum(args.receipts),
        "archive_root": str(args.destination),
        "checkpoint_count": len(entries),
        "total_bytes": sum(row["size_bytes"] for row in entries),
        "checkpoints": entries,
        "original_receipts_unchanged": True,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "checkpoint_count": len(entries),
                "total_bytes": manifest["total_bytes"],
                "archive_root": str(args.destination),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
