"""Freeze the finite legacy set before conversion. No writes to source assets."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_once(path, value):
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"frozen record differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive-inventory", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root
    relative = "data/tennis_multivew/processed/meiji_3cam/dataset"
    base = root / relative
    dataset = json.loads((base / "dataset.json").read_bytes())
    snapshots = []
    splits = {}
    bindings = {}
    images = []
    for directory in sorted((root / "data/ball_detection").iterdir()):
        metadata = directory / "metadata.json"
        if not metadata.is_file():
            continue
        data = json.loads(metadata.read_bytes())
        if not any(c["source"] == "meiji" for c in data["clips"]):
            continue
        snapshots.append(dict(path=str(directory.relative_to(root)), schema=data["schema_version"],
                              metadata_sha256=digest(metadata), index_sha256=digest(directory / "index.npz"),
                              source_counts=dict(Counter(c["source"] for c in data["clips"]))))
        for clip in data["clips"]:
            if clip["source"] != "meiji":
                continue
            old = splits.setdefault(clip["group_id"], clip["split"])
            if old != clip["split"]:
                raise ValueError("legacy snapshots disagree on group/split")
            key = clip["clip_id"].removeprefix("meiji/")
            identity = (clip["annotation_sha256"], clip["media_sha256"])
            if bindings.setdefault(key, identity) != identity:
                raise ValueError("legacy snapshots disagree on annotation/media version")
            shard = directory / "shards" / f"clip-{clip['index']:05d}.bin"
            stat = shard.stat()
            images.append(dict(path=str(shard.relative_to(root)), size=stat.st_size,
                               mtime_ns=stat.st_mtime_ns, inode=stat.st_ino))
    drive = json.loads(args.drive_inventory.read_bytes())
    by_md5 = {}
    for entry in drive["entries"]:
        if entry["type"] == "file":
            by_md5.setdefault(entry["hashes"].get("md5"), []).append(dict(path=entry["path"], id=entry["id"]))
    entries = []
    totals = Counter()
    decisions = Counter()
    for clip in dataset["clips"]:
        directory = base / clip["path"]
        clip_path = directory / "clip.json"
        manifest = json.loads(clip_path.read_bytes())
        if (manifest["video_id"], manifest["clip_id"]) != (clip["video_id"], clip["clip_id"]):
            raise ValueError("clip manifest differs from dataset index")
        for camera in manifest["cameras"]:
            name = camera["camera_id"]
            path = directory / "outsource" / f"{name}_annotations.json"
            payload = path.read_bytes()
            annotation = json.loads(payload)
            original_sha = hashlib.sha256(payload).hexdigest()
            key = f"{clip['clip_id']}/{name}"
            if bindings[key] != (original_sha, annotation["source"]["sha256"]):
                raise ValueError(f"snapshot/input identity mismatch: {key}")
            reasons = []
            if annotation["target"]["scope"] != "main-court rally ball":
                reasons.append("target_includes_held_ball")
            if annotation["summary"].get("all_frames_reviewed") is not True:
                # Individual image review is not required by v3. This conservative
                # first selection defers files without that sufficient evidence;
                # it does NOT relabel their unresolved rows as unprocessed.
                reasons.append("processing_completion_requires_report_interpretation")
            decision = "hold" if reasons else "convert"
            entries.append(dict(
                key=key, clip=clip["path"], camera=name, video=camera["video"],
                original_sha256=original_sha, original_size_bytes=len(payload),
                media_sha256=annotation["source"]["sha256"], media_size_bytes=(directory / camera["video"]).stat().st_size,
                clip_manifest_sha256=digest(clip_path), recording_id=f"{dataset['dataset_id']}/{clip['video_id']}",
                legacy_group=clip["video_id"], split=splits[clip["video_id"]],
                counts=annotation["summary"]["counts"], frames=len(annotation["frames"]),
                decision=decision, reasons=reasons,
                drive_copies=by_md5.get(hashlib.md5(payload).hexdigest(), []),
            ))
            totals.update(annotation["summary"]["counts"])
            decisions[decision] += 1
    selection = dict(schema_version="ball.v3-migration-selection.v1", issue=1714,
                     legacy_root=relative, dataset_sha256=digest(base / "dataset.json"),
                     code_before_migration=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                     snapshots=snapshots, entries=entries,
                     image_inventory_sha256=hashlib.sha256(json.dumps(images, sort_keys=True).encode()).hexdigest(),
                     counts=dict(totals), decisions=dict(decisions),
                     drive_annotation_root=drive["path"], drive_observed_at=drive["observed_at"],
                     excluded_teachers={"chat_annotation": "all-ball schema; no active-ball equivalence evidence",
                                        "tracknet": "legacy point/visibility labels; no active-ball equivalence evidence"})
    write_once(args.output, selection)
    write_once(args.output.with_name("legacy-images.json"), images)
    print(json.dumps(dict(entries=len(entries), decisions=decisions, counts=totals,
                          selected_frames=sum(e["frames"] for e in entries if e["decision"] == "convert"),
                          drive_matched=sum(bool(e["drive_copies"]) for e in entries), bytes=args.output.stat().st_size)))


if __name__ == "__main__":
    main()
