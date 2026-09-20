"""Preserve and quarantine the 12 recorded suspect files before regeneration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.checksum import dual_sha256
from src.utils.io import save_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preservation", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.observations.resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.is_absolute():
        raise ValueError("Use a new absolute output directory")
    document = json.loads(args.preservation.read_text())
    expected = {
        f"video_002/clip_{clip:03d}/cam2_{suffix}"
        for clip in (5, 6, 12)
        for suffix in (
            "people.npz",
            "people.metadata.json",
            "detections.npz",
            "detections.metadata.json",
        )
    }
    rows = document["inputs"]
    relative = [str(Path(row["source"]).relative_to(root)) for row in rows]
    if len(rows) != 12 or set(relative) != expected:
        raise ValueError("Preservation inventory differs from the fixed repair scope")
    # Validate every original and independent copy before moving any original.
    for row in rows:
        source, preserved = Path(row["source"]), Path(row["preserved"])
        if source.is_symlink() or preserved.is_symlink():
            raise ValueError("Expected regular observation files")
        if not dual_sha256(source) == dual_sha256(preserved) == row["sha256"]:
            raise ValueError(f"Preservation mismatch: {source}")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "moving",
        "preservation_sha256": dual_sha256(args.preservation),
        "source_root": str(root),
        "planned": rows,
        "moved": [],
    }
    receipt_path = args.output / "quarantine.json"
    save_json_atomic(receipt, receipt_path)
    try:
        for row, relative_path in zip(rows, relative, strict=True):
            source = Path(row["source"])
            target = args.output / "quarantined" / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or dual_sha256(source) != row["sha256"]:
                raise ValueError(f"Input changed or destination exists: {source}")
            source.rename(target)
            if dual_sha256(target) != row["sha256"]:
                raise ValueError(f"Quarantined bytes differ: {target}")
            receipt["moved"].append({**row, "quarantined": str(target)})
            save_json_atomic(receipt, receipt_path)
        receipt["status"] = "quarantined"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_json_atomic(receipt, receipt_path)
    print(f"Quarantined {len(rows)} files; receipt: {receipt_path}")


if __name__ == "__main__":
    main()
