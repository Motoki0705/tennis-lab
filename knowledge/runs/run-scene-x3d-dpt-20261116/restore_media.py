"""Copy recorded comparison videos from old E: outputs to new D: outputs.

The original prediction reports and stores remain in the explicitly supplied
archive. Refuse different existing destination bytes or a missing checkpoint.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required scene checkpoint is absent: {checkpoint}")
    checkpoint_sha = sha(checkpoint)
    if checkpoint_sha != args.checkpoint_sha256:
        raise ValueError("Scene checkpoint differs from the original exported artifact")
    archive = args.archive_root.resolve(strict=True)
    destination = args.destination.resolve()
    if archive == destination:
        raise ValueError("Archive and destination must be distinct")
    original_prefix = Path("outputs/tennis_scene/evaluate/x3d-dpt-20261116")
    manifest = json.loads(args.manifest.read_text())
    records = []
    for entry in manifest["videos"]:
        relative = Path(entry["path"]).relative_to(original_prefix)
        if ".." in relative.parts:
            raise ValueError("Video path escapes the recorded run")
        source = archive / relative
        target = args.destination / relative
        expected = entry["sha256"]
        source_sha = sha(source)
        if source.stat().st_size != entry["size_bytes"] or source_sha != expected:
            raise ValueError(f"Archived video differs from recorded evidence: {relative}")
        reused = target.exists()
        if reused:
            if not target.is_file() or sha(target) != expected:
                raise ValueError(f"Destination already has different evidence: {relative}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
        target_sha = sha(target)
        if target_sha != expected:
            raise ValueError(f"Copied video differs from recorded evidence: {relative}")
        records.append(dict(relative_path=str(relative), source=str(source), destination=str(target),
                            bytes=target.stat().st_size, expected_sha256=expected,
                            source_sha256=source_sha, destination_sha256=target_sha,
                            reused_existing_equal_copy=reused))
    report = dict(schema="scene_dpt_storage_relocation.v1", verified_on="2026-11-16",
                  archive_storage="旧outputs（E:）", archive_root=str(archive),
                  destination_storage="新persistent outputs（D:）", destination=str(args.destination),
                  destination_resolved=str(destination), original_manifest_sha256=sha(args.manifest),
                  checkpoint=dict(path=str(checkpoint), resolved=str(checkpoint.resolve()),
                                  sha256=checkpoint_sha, expected_sha256=args.checkpoint_sha256),
                  raw_reports_and_stores="Retained in old E: outputs; historical commands and paths are unchanged.",
                  videos=records)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(videos_verified=len(records), bytes_verified=sum(r["bytes"] for r in records),
                          checkpoint_sha256=checkpoint_sha)), flush=True)


if __name__ == "__main__":
    main()
