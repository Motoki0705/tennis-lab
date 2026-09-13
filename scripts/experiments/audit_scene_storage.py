"""Read-only scene accounting and exact duplicate audit (already shared inodes excluded)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    by_size: dict[int, list[Path]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    fields: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    logical = physical = allocated = count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        count += 1
        logical += stat.st_size
        key = path.name if "datasets/court/samples" in str(path) else "other"
        fields[key][0] += 1
        fields[key][1] += stat.st_size
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            physical += stat.st_size
            allocated += stat.st_blocks * 512
            seen.add(identity)
            by_size[stat.st_size].append(path)
    duplicates = []
    for size, paths in by_size.items():
        if size == 0 or len(paths) < 2:
            continue
        partial: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            with path.open("rb") as stream:
                prefix = stream.read(65536)
                stream.seek(max(0, size - 65536))
                suffix = stream.read(65536)
            partial[hashlib.sha256(prefix + suffix).hexdigest()].append(path)
        for candidates in partial.values():
            if len(candidates) < 2:
                continue
            full: dict[str, list[str]] = defaultdict(list)
            for path in candidates:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                full[digest].append(str(path.relative_to(root)))
            for digest, members in full.items():
                if len(members) > 1:
                    duplicates.append(
                        {
                            "sha256": digest,
                            "bytes_per_copy": size,
                            "reclaimable_bytes": size * (len(members) - 1),
                            "paths": members,
                        }
                    )
    result = {
        "root": str(root),
        "files": count,
        "logical_bytes": logical,
        "unique_inode_bytes": physical,
        "allocated_unique_inode_bytes": allocated,
        "already_shared_bytes": logical - physical,
        "by_field": fields,
        "exact_duplicate_reclaimable_bytes": sum(
            g["reclaimable_bytes"] for g in duplicates
        ),
        "duplicates": sorted(
            duplicates, key=lambda g: g["reclaimable_bytes"], reverse=True
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["duplicates", "by_field"]}
        )
    )


if __name__ == "__main__":
    main()
