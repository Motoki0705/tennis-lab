"""Publish completed JSON snapshots outside the live generation directories."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.utils.data.image_record_store import file_sha256


def main() -> None:
    evidence = Path(
        "outputs/synthetic_data_generation/profile/court-scale-smoke/evidence"
    )
    fetched = Path("tmp/court-scale-from-drive-smoke")
    resumed = Path("tmp/court-scale-resumed-smoke")
    verification = json.loads((evidence / "verification.json").read_text())
    version_name = verification["dataset_version_path"]
    request_sha = verification["request_sha256"]
    copies = [
        (fetched / version_name, Path(version_name)),
        (fetched / f"reports/{request_sha}.json", Path(f"reports/{request_sha}.json")),
        (
            fetched / f"completed-requests/{request_sha}.json",
            Path(f"completed-requests/{request_sha}.json"),
        ),
    ]
    copies.extend(
        (path, Path("cache-metrics") / path.name)
        for path in sorted((fetched / "cache-metrics").glob("*.json"))
    )
    attempts = list((resumed / "attempts").glob("*.json"))
    assert len(attempts) == 1
    copies.append((attempts[0], Path("resumed-attempt.json")))
    manifest = []
    with tempfile.TemporaryDirectory(
        prefix=".court-scale-evidence-", dir="/content"
    ) as temporary:
        for index, (source, relative) in enumerate(copies):
            assert source.is_file() and source.stat().st_size <= 256 * 1024
            target = evidence / relative
            assert not target.exists()
            completed = Path(temporary) / str(index)
            completed.write_bytes(source.read_bytes())
            target.parent.mkdir(parents=True, exist_ok=True)
            completed.replace(target)
            manifest.append(
                {
                    "path": relative.as_posix(),
                    "bytes": target.stat().st_size,
                    "sha256": file_sha256(target),
                }
            )
        target = evidence / "snapshot-inventory.json"
        assert not target.exists()
        completed = Path(temporary) / "inventory.json"
        completed.write_text(
            json.dumps(
                {"schema": "court_scale_smoke_snapshots_v1", "files": manifest},
                indent=2,
            )
            + "\n"
        )
        completed.replace(target)
    print(
        json.dumps(
            {
                "snapshot_file_count": len(manifest),
                "snapshot_bytes": sum(entry["bytes"] for entry in manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
