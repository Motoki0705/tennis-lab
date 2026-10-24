"""Bundle existing JPEGs and records for CPU review; never invoke a renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    probe = root / "probe"
    state = json.loads((probe / "state.json").read_text())
    assert state["outcome"] == "completed"
    assert state["draw_attempts"] == state["completed_frames"] == 192
    inventory = json.loads((probe / "inventory.json").read_text())["files"]
    originals = {row["path"]: row for row in inventory}
    assert len(list((probe / "reports").glob("*.json"))) == 16
    assert len(list((probe / "frames").glob("*/*/*/rgb.jpg"))) == 192
    assert len(list((probe / "frames").glob("*/*/*/raw.npz"))) == 192
    selected = [
        path
        for path in sorted(probe.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".jpg"}
    ]
    selected.extend(sorted((root / "setup").glob("*.json")))
    selected.extend([root / "render-timing.json", root / "profile-summary.json"])
    files: list[dict[str, Any]] = []
    for path in selected:
        content_hash = sha256(path)
        if path.is_relative_to(probe) and path.name != "inventory.json":
            entry = originals[path.relative_to(probe).as_posix()]
            assert content_hash == entry["sha256"]
            assert path.stat().st_size == entry["bytes"]
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": content_hash,
            }
        )
    manifest = root / "review-manifest.json"
    assert not manifest.exists()
    manifest.write_text(
        json.dumps(
            {
                "schema": "court_d2_review_bundle_v1",
                "source_state_sha256": sha256(probe / "state.json"),
                "files": files,
                "raw_arrays_retained_on_drive": 192,
                "render_calls": 0,
            },
            indent=2,
        )
        + "\n"
    )
    archive = root / "review.tar"
    assert not archive.exists()
    temporary = root / "review.partial"
    with tarfile.open(temporary, "w") as bundle:
        for path in [*selected, manifest]:
            bundle.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    temporary.rename(archive)
    print(
        json.dumps(
            {
                "bundle": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
                "review_files": len(files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
