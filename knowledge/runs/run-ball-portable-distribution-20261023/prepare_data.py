"""Reproduce the pinned train/validation and separate test distribution plans."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.distribution import (
    prepare_distribution,
    sample_windows,
    source_payloads,
    verify_distribution,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.utils.checksum import dual_sha256
from src.utils.data.verification import file_version

PROTOCOL = Path("knowledge/runs/run-ball-all-frame-protocol-20261022")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def prepare(source_repo: Path, output: Path) -> None:
    checkout = Path.cwd().resolve()
    if not output.is_relative_to(checkout) or output.is_relative_to(source_repo / "data") or output.is_relative_to(source_repo / "outputs"):
        raise ValueError("Write only below the active worktree, outside source data/outputs")
    output.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    free_before = shutil.disk_usage(output)._asdict()
    protocol = checkout / PROTOCOL
    pinned = json.loads((protocol / "inputs.json").read_text())
    snapshot = source_repo / pinned["snapshot"]
    for name, digest in pinned["snapshot_hashes"].items():
        if dual_sha256(snapshot / name) != digest:
            raise ValueError("The fixed protocol snapshot changed")
    if dual_sha256(source_repo / pinned["source_manifest"]) != pinned["source_manifest_sha256"]:
        raise ValueError("The protocol's original manifest changed")
    store = BallFrameStore(snapshot)
    with (protocol / "clip-catalog.csv").open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        assert fields is not None
        catalog = list(reader)
    known = {int(row["clip_index"]): row for row in catalog}
    if len(known) != len(store.clips):
        raise ValueError("Protocol catalog does not cover the source snapshot")
    for clip in store.clips:
        row = known[clip.index]
        if (row["clip_id"], row["source"], row["group_id"], row["split"], row["native_fps"], int(row["frames"])) != (
            clip.clip_id, clip.source, clip.group_id, clip.split, clip.fps, clip.frame_count
        ):
            raise ValueError("Protocol catalog differs from the source clip identity")
    intervals = {}
    for split in ("train", "val", "test"):
        rows = [json.loads(line) for line in (protocol / f"common-intervals-{split}.jsonl").read_text().splitlines()]
        intervals[split] = dict(rows)
        if len(intervals[split]) != len(rows):
            raise ValueError("Duplicate protocol clip intervals")
        for index, spans in rows:
            clip = store.clips[index]
            if clip.split != split or not spans or any(not 0 <= a < b <= clip.frame_count or b - a < 128 for a, b in spans):
                raise ValueError("Invalid protocol interval or split")
    versions = {path: file_version(path) for path in [snapshot / "index.npz", snapshot / "metadata.json"]}
    versions.update({store.image_path(clip): file_version(store.image_path(clip)) for clip in store.clips})
    versions.update({store.annotation_path(clip): file_version(store.annotation_path(clip))
                     for clip in store.clips if clip.source == "chat_annotation"})
    result = dict(schema_version=1, started_at=started_at, source_snapshot=pinned["snapshot"], units={})
    for name, splits in (("train-val", ("train", "val")), ("test", ("test",))):
        unit_started = time.monotonic()
        indices = {i for split in splits for i in intervals[split]}
        selected = [clip for clip in store.clips if clip.index in indices]
        input_root = output / (name + "-window-inputs")
        input_root.mkdir()
        with (input_root / "clip-catalog.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(row for row in catalog if int(row["clip_index"]) in indices)
        inputs = {"inputs/clip-catalog.csv": input_root / "clip-catalog.csv"}
        for split in splits:
            for kind in ("common-intervals", "train-packs"):
                file = f"{kind}-{split}.jsonl"
                inputs["inputs/" + file] = protocol / file
        inputs["inputs/assignment-rules.json"] = protocol / "assignment-rules.json"
        write_json(input_root / "protocol-source.json", dict(
            run_id="run-ball-all-frame-protocol-20261022",
            protocol_inputs_sha256=dual_sha256(protocol / "inputs.json"),
            input_identities={path.name: dual_sha256(path) for path in inputs.values()},
            clip_index_basis="source snapshot; join by clip_id through distribution.json, never reuse as distribution index",
            lengths=[32, 64, 128], frame_step=1, splits=list(splits),
        ))
        inputs["inputs/protocol-source.json"] = input_root / "protocol-source.json"
        directory = output / name
        manifest = prepare_distribution(snapshot, directory,
                                        version=f"ball-mix1800-720p-all-frame-{name}-v1",
                                        clip_ids=[c.clip_id for c in selected], inputs=inputs)
        prepared_seconds = time.monotonic() - unit_started
        verification_started = time.monotonic()
        receipt = verify_distribution(directory, source_directory=snapshot)
        verification_seconds = time.monotonic() - verification_started
        windows: list[tuple[str, int, int]] = []
        for split in splits:
            for source in ("tracknet", "meiji", "chat_annotation"):
                clip = next(c for c in selected if c.split == split and c.source == source)
                first = intervals[split][clip.index][0][0]
                windows.extend((clip.clip_id, first, length) for length in (32, 64, 128))
        samples_started = time.monotonic()
        reference = sample_windows(store, windows)
        view = BallFrameStore(directory, external_payloads=source_payloads(directory, snapshot))
        if sample_windows(view, windows) != reference:
            raise ValueError("Source and distribution teacher/RGB/PTS samples differ")
        write_json(output / f"{name}-sample-windows.json", windows)
        write_json(output / f"{name}-local-samples.json", reference)
        write_json(output / f"{name}-local-verification.json", receipt)
        result["units"][name] = dict(
            prepare_seconds=prepared_seconds, verification_seconds=verification_seconds,
            sample_seconds=time.monotonic() - samples_started,
            payload_files=len(manifest["clips"]), files=receipt["files"], size_bytes=receipt["size_bytes"],
            distribution_sha256=receipt["distribution_sha256"],
            split_clips=dict(Counter(c.split for c in selected)),
            common_frames=sum(b - a for split in splits for spans in intervals[split].values() for a, b in spans),
            local_generated_bytes=sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
            sampled_windows=len(windows), local_samples_equal=True,
        )
        write_json(output / "local-preparation.json", result)
        print(json.dumps(dict(unit=name, **result["units"][name])), flush=True)
    result.update(
        completed_at=datetime.now(UTC).isoformat(), elapsed_seconds=time.monotonic() - started,
        disk_before=free_before, disk_after=shutil.disk_usage(output)._asdict(),
        source_files_stat_checked=len(versions),
        source_versions_unchanged=all(file_version(path) == value for path, value in versions.items()),
    )
    if not result["source_versions_unchanged"]:
        raise ValueError("Source inode/size/mtime/ctime changed")
    write_json(output / "local-preparation.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_repo_root.resolve(), args.output_root.resolve())
