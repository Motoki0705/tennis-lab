"""Finish the closed distribution audit without rereading already hashed RGB."""

import argparse
import collections
import json
import os
import shutil
import time
from pathlib import Path

from rebuild_dataset import DISTRIBUTION, sha, write


def counts(metadata):
    result = {}
    for clip in metadata["clips"]:
        row = result.setdefault(clip["split"], dict(clips=0, frames=0))
        row["clips"] += 1
        row["frames"] += clip["frame_count"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root, source = args.root.resolve(), args.source.resolve()
    dataset = root / "dataset"
    if sha(args.manifest) != DISTRIBUTION:
        raise ValueError("Distribution manifest changed")
    distribution = json.loads(args.manifest.read_text())
    report = json.loads((root / "shard-verification.json").read_text())
    if not report["complete"] or report["verified_files"] != 1537:
        raise ValueError("Whole-source RGB hash audit required")
    shards = {item["path"]: item for item in report["files"]}
    for item in distribution["files"]:
        path = dataset / item["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item["size_bytes"]
        ):
            raise ValueError("Missing/invalid payload: " + str(path))
        if item["path"].startswith("shards/"):
            record = shards[item["path"]]
            if record["sha256"] != item["sha256"] or not os.path.samefile(
                record["source"], path
            ):
                raise ValueError("Verified hardlink identity changed")
        elif sha(path) != item["sha256"]:
            raise ValueError("Annotation/index/metadata payload differs")
    shutil.copyfile(args.manifest, dataset / "distribution.json")
    expected = {item["path"] for item in distribution["files"]} | {"distribution.json"}
    actual = {
        path.relative_to(dataset).as_posix()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    if expected != actual:
        raise ValueError("Complete dataset file set differs")
    before = json.loads((source / "metadata.json").read_text())
    after = json.loads((dataset / "metadata.json").read_text())
    old = {clip["clip_id"]: clip for clip in before["clips"]}
    changed = collections.Counter()
    for clip in after["clips"]:
        origin = old[clip["clip_id"]]
        for key, value in clip.items():
            if value != origin.get(key):
                changed[key] += 1
                if key not in {"index", "annotation_path"}:
                    raise ValueError("Scientific clip metadata differs: " + key)
    selected = {clip["clip_id"] for clip in after["clips"]}
    excluded = [clip for clip in before["clips"] if clip["clip_id"] not in selected]
    if any(
        clip["split"] in {"train", "val"} and clip["frame_count"] >= 128
        for clip in excluded
    ):
        raise ValueError("Unexpected omission from the train/val distribution")
    summary = dict(
        complete=True,
        payload_files=len(distribution["files"]),
        complete_file_set=len(actual),
        payload_bytes=sum(item["size_bytes"] for item in distribution["files"]),
        hardlinked_shards=1537,
        hardlinked_bytes=report["verified_bytes"],
        additional_payload_bytes=sum(
            item["size_bytes"] for item in distribution["files"]
        )
        - report["verified_bytes"],
        source_split_counts=counts(before),
        distribution_split_counts=counts(after),
        omitted_train_val_clips=sum(
            clip["split"] in {"train", "val"} for clip in excluded
        ),
        omissions="test split and train/val clips shorter than 128 frames",
        changed_clip_metadata_fields=dict(changed),
        scientific_clip_metadata_exact=True,
        all_sha256_verified=True,
        verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write(root / "dataset-verified.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
