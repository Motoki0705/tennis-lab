"""Check the portable-store/window boundary without copying or linking RGB."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.distribution import (
    load_distribution,
    source_payloads,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.all_frame_manifest import (
    build_all_frame_manifests,
)
from src.tasks.ball_detection.data.windows.all_frames import (
    AllFrameWindowConfig,
    common_intervals,
)
from src.tasks.ball_detection.data.windows.hybrid import reviewed_empty_evidence
from src.utils.checksum import dual_sha256
from src.utils.data.verification import file_version


def capture(call: Callable[[], Any], *, roots: dict[Path, str]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        call()
    except (ValueError, FileNotFoundError) as error:
        message = str(error)
        for root, replacement in sorted(roots.items(), key=lambda item: -len(str(item[0]))):
            message = message.replace(str(root), replacement)
        return dict(outcome="raised", exception=type(error).__name__, message=message,
                    seconds=time.monotonic() - started)
    return dict(outcome="returned", seconds=time.monotonic() - started)


def probe(distribution: Path, source_directory: Path | None) -> dict[str, Any]:
    started = time.monotonic()
    distribution = distribution.resolve(strict=True)
    manifest = load_distribution(distribution)
    payloads = None if source_directory is None else source_payloads(distribution, source_directory)
    store = BallFrameStore(distribution, external_payloads=payloads)
    roots = {distribution: "<distribution>"}
    if source_directory is not None:
        source_directory = source_directory.resolve(strict=True)
        roots[source_directory] = "<source-store>"
    roots[Path.cwd()] = "<worktree>"
    watched = {distribution / name for name in ("metadata.json", "index.npz", "distribution.json")}
    watched.update(store.image_path(clip) for clip in store.clips)
    watched.update(store.annotation_path(clip) for clip in store.clips if clip.source == "chat_annotation")
    if source_directory is not None:
        watched.update(source_directory / name for name in ("metadata.json", "index.npz"))
    before = {path: file_version(path) for path in watched}

    eligible = []
    config = AllFrameWindowConfig(32)
    for clip in store.clips:
        if clip.source != "chat_annotation":
            continue
        states = annotation_states(store, clip)
        empty = states.reviewed & states.target & (states.count == 0)
        if empty.any() and common_intervals(states, config):
            eligible.append(clip)
    if not eligible:
        raise ValueError("The probe needs a common-interval Chat clip with reviewed empty frames")
    clip = sorted(eligible, key=lambda item: item.clip_id)[0]
    states = annotation_states(store, clip)
    entry = store.metadata["clips"][clip.index]
    raw_path = Path(entry["annotation_path"])
    resolved = store.annotation_path(clip)
    annotation_matches = dual_sha256(resolved) == clip.annotation_sha256
    if not annotation_matches:
        raise ValueError("The distribution's resolved original annotation changed")
    checks: dict[str, Any] = dict(
        preparation_file_version=capture(lambda: file_version(raw_path), roots=roots),
        reviewed_empty_evidence=capture(lambda: reviewed_empty_evidence(store, clip, states), roots=roots),
    )
    file_hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    shard_hashes = {item["clip_id"]: file_hashes[item["payload"]] for item in manifest["clips"]}
    checks["build_all_frame_manifests"] = capture(
        lambda: build_all_frame_manifests(store, [config], shard_hashes=shard_hashes), roots=roots,
    )
    source_control = None
    if source_directory is not None:
        source = BallFrameStore(source_directory)
        original = source.clip_by_id(clip.clip_id)
        original_states = annotation_states(source, original)
        original_mask, _ = reviewed_empty_evidence(source, original, original_states)
        if not (np.array_equal(states.reviewed, original_states.reviewed)
                and np.array_equal(states.target, original_states.target)
                and np.array_equal(states.count, original_states.count)):
            raise ValueError("Normalized teacher inputs differ from the pinned source")
        source_control = dict(outcome="returned", same_reviewed_target_instance_counts=True,
                              accepted_empty_frames=int(original_mask.sum()))
    unchanged = all(file_version(path) == version for path, version in before.items())
    if not unchanged:
        raise ValueError("An input file changed during the read-only probe")
    blocked = any(check["outcome"] == "raised" for check in checks.values())
    return dict(
        schema_version=1, observed_at=datetime.now(UTC).isoformat(),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        status="blocked" if blocked else "boundary_passed",
        mode="source_payloads" if payloads is not None else "self_contained",
        distribution_sha256=dual_sha256(distribution / "distribution.json"),
        version=store.metadata["version"], clips=len(store.clips),
        eligible_chat_clips=len(eligible), clip_id=clip.clip_id,
        raw_annotation_path=raw_path.as_posix(), raw_annotation_path_is_absolute=raw_path.is_absolute(),
        resolved_annotation_path=resolved.relative_to(distribution).as_posix(),
        resolved_annotation_sha256_matches=annotation_matches,
        source_control=source_control, checks=checks, input_file_versions_checked=len(before),
        input_file_versions_unchanged=unchanged, rgb_payload_bytes_copied_or_linked=0,
        elapsed_seconds=time.monotonic() - started,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--source-store", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous receipt; select a new output file")
    receipt = probe(args.distribution, args.source_store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    raise SystemExit(2 if receipt["status"] == "blocked" else 0)
