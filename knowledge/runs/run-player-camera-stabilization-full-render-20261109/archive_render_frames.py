"""Losslessly archive one completed render; publish only after byte verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
import zipfile
from pathlib import Path


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def archive_render(render: Path, archive: Path, *, expected_frames: int) -> dict:
    sidecar = archive.with_suffix(".manifest.json")
    if os.path.lexists(archive) or os.path.lexists(sidecar):
        raise FileExistsError(archive)
    report_path = render / "render.json"
    report = json.loads(report_path.read_text())
    rows = report["frames"]
    if not report["complete"] or len(rows) != expected_frames or expected_frames < 1:
        raise ValueError("Archive requires a completed render and exact frame count")
    sources = [r["source_frame"] for r in rows]
    if sources != list(range(sources[0], sources[0] + expected_frames)):
        raise ValueError("Archive requires the contiguous full comparison timeline")
    paths = sorted((render / "frames").iterdir())
    if [p.name for p in paths] != [f"{i:05d}.png" for i in range(expected_frames)]:
        raise ValueError("Unexpected or missing source PNG")
    entries = []
    for i, (path, row) in enumerate(zip(paths, rows, strict=True)):
        if path.is_symlink() or not path.is_file() or row["output_index"] != i:
            raise ValueError("Invalid source frame")
        with path.open("rb") as stream:
            header = stream.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise ValueError("Source frame is not PNG")
        if list(struct.unpack(">II", header[16:24])) != report["camera"]["resolution"]:
            raise ValueError("Source PNG and camera audit resolution disagree")
        entries.append(
            {
                "path": "frames/" + path.name,
                "source_frame": sources[i],
                "camera_valid": row["camera_valid"],
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    manifest = {
        "schema": "player_view_lossless_render_archive_v1",
        "player_id": report["player_id"],
        "source_frames": sources,
        "resolution": report["camera"]["resolution"],
        "render_report_sha256": digest(report_path),
        "files": entries,
        "compression": "ZIP_STORED; original PNG bytes retained",
    }
    payload = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=archive.parent, prefix=".frames-", suffix=".partial", delete=False
    ) as temporary:
        staging = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            staging, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as output:
            for path, entry in zip(paths, entries, strict=True):
                output.write(path, arcname=entry["path"])
            output.writestr("manifest.json", payload)
        with zipfile.ZipFile(staging) as saved:
            if saved.namelist() != [e["path"] for e in entries] + ["manifest.json"]:
                raise ValueError("Archive member set differs")
            if saved.read("manifest.json") != payload:
                raise ValueError("Archive manifest differs")
            for entry in entries:
                with saved.open(entry["path"]) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if (
                    actual != entry["sha256"]
                    or saved.getinfo(entry["path"]).file_size != entry["bytes"]
                ):
                    raise ValueError("Archive PNG bytes differ")
        record = {
            "schema": "player_view_lossless_render_archive_receipt_v1",
            "archive": archive.name,
            "bytes": staging.stat().st_size,
            "sha256": digest(staging),
            "md5": digest(staging, "md5"),
            "members": expected_frames,
            "source_frames": sources,
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "render_report_sha256": manifest["render_report_sha256"],
            "all_uncompressed_png_sha256_verified": True,
        }
        if os.path.lexists(archive) or os.path.lexists(sidecar):
            raise FileExistsError(archive)
        staging.replace(archive)
        sidecar_staging = sidecar.with_suffix(".json.partial")
        sidecar_staging.write_text(json.dumps(record, indent=2) + "\n")
        sidecar_staging.replace(sidecar)
        return record
    finally:
        staging.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            archive_render(args.render, args.archive, expected_frames=args.frames)
        )
    )
