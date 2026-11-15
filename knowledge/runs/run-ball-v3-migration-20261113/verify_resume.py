"""Verify relocated legacy inputs and the completed preparations after WSL recovery."""

import argparse
import datetime
import json
import sqlite3
from pathlib import Path

from src.tasks.ball_detection.annotation.video import VideoAnnotationInput, read_annotation
from src.tasks.ball_detection.annotation.video.inputs import validate_video_input
from src.tasks.ball_detection.annotation.video.migration import MigrationRecord, convert_v2_annotation
from src.tasks.base.annotation.contracts import ClipManifest, content_digest
from src.tasks.base.annotation.media import check_clip
from src.tasks.base.annotation.paths import AnnotationLayout

from inventory import digest, write_once


def legacy_baseline(source_root, image_inventory):
    result = []
    hashes = {}
    for number, image in enumerate(image_inventory, 1):
        path = source_root / image["path"]
        stat = path.stat()
        if stat.st_size != image["size"]:
            raise ValueError(f"legacy image size changed: {image['path']}")
        physical = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if physical not in hashes:
            hashes[physical] = digest(path)
        result.append({"path": image["path"], "size_bytes": stat.st_size, "sha256": hashes[physical]})
        if number % 50 == 0:
            print(json.dumps({"legacy_image_hashes": number, "total": len(image_inventory)}), flush=True)
    return result


def preparation(layout, entry):
    root = layout.path("migration_archive", "ball-v2-v3", entry["original_sha256"])
    if not root.exists():
        return {"key": entry["key"], "state": "not_prepared"}
    expected = VideoAnnotationInput.model_validate_json((root / "input.json").read_bytes())
    source = expected.source
    if (source.recording_id, source.camera_id, source.media_sha256, source.frame_count) != (
        entry["recording_id"], entry["camera"], entry["media_sha256"], entry["frames"]
    ):
        raise ValueError("saved input identity differs from the fixed selection")
    original = root / "original-v2.json"
    if digest(original) != entry["original_sha256"]:
        raise ValueError("retained original v2 bytes changed")
    record = MigrationRecord.model_validate_json((root / "migration.json").read_bytes())
    converted, regenerated = convert_v2_annotation(original, expected_input=expected, evidence=record.evidence)
    saved = read_annotation(root / "annotation-v3.json", expected_input=expected, require_complete=True)
    if regenerated != record or converted != saved:
        raise ValueError("saved conversion differs from its original v2")
    retained_source = layout.path("sources", "blobs", source.media_sha256)
    validate_video_input(retained_source, expected)
    clip_id = content_digest([content_digest(["local", source.sha256]), expected.frame_range.model_dump()])
    clip_root = layout.path("clips", clip_id)
    manifest = ClipManifest.model_validate_json((clip_root / "manifest.json").read_bytes())
    if (
        manifest.preparation_digest != expected.sha256
        or manifest.source.video_id != source.sha256
        or manifest.source.channel_id != content_digest(source.recording_id)
        or manifest.source_range != expected.frame_range
        or tuple((f.source_frame_index, f.source_pts) for f in manifest.frames)
        != tuple((f.frame_index, f.pts) for f in expected.frames)
    ):
        raise ValueError("prepared image identity/frame/PTS differs from its input")
    media = layout.artifact_path(manifest.media)
    check_clip(media, manifest)
    return {
        "key": entry["key"], "state": "verified", "frames": len(expected.frames),
        "manifest": (clip_root / "manifest.json").relative_to(layout.root).as_posix(),
        "archive_sha256": manifest.media.digest, "archive_size_bytes": manifest.media.size_bytes,
        "migration_record_sha256": digest(root / "migration.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    storage = args.storage_root.resolve()
    layout = AnnotationLayout(storage, storage / "outputs/chat_annotation", "production")
    if (storage / "data").stat().st_dev != layout.root.stat().st_dev:
        raise ValueError("publication data and annotation images must share a filesystem")
    with sqlite3.connect(layout.registry.as_uri() + "?mode=ro", uri=True) as database:
        checks = database.execute("PRAGMA quick_check").fetchall()
    if checks != [("ok",)]:
        raise ValueError(f"annotation registry integrity check failed: {checks}")
    selection = json.loads(args.selection.read_bytes())
    images = json.loads(args.selection.with_name("legacy-images.json").read_bytes())
    baseline = {
        "schema_version": "ball.relocated-legacy-images.v1",
        "observation_date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        "previous_image_inventory_sha256": digest(args.selection.with_name("legacy-images.json")),
        "files": legacy_baseline(args.source_root, images),
    }
    write_once(args.baseline_output, baseline)
    results = []
    for entry in selection["entries"]:
        if entry["decision"] != "convert":
            continue
        try:
            result = preparation(layout, entry)
        except (ValueError, OSError, KeyError) as error:
            result = {"key": entry["key"], "state": "invalid", "error": str(error)}
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {
        "schema_version": "ball.migration-resume-audit.v1",
        "registry_quick_check": "ok",
        "legacy_image_baseline_sha256": digest(args.baseline_output),
        "verified": sum(r["state"] == "verified" for r in results),
        "not_prepared": sum(r["state"] == "not_prepared" for r in results),
        "invalid": sum(r["state"] == "invalid" for r in results),
        "results": results,
    }
    write_once(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
