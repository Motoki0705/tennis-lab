"""Verify all test T against the protocol and pin both portable distributions."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.checkpoints.temporal_heatmap import save_json
from src.tasks.ball_detection.data.distribution import (
    load_distribution,
    source_payloads,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.all_frame_preparation import (
    prepare_all_frame_training,
)
from src.tasks.ball_detection.data.windows.all_frames import AllFrameWindowConfig
from src.tasks.ball_detection.evaluation.temporal_heatmap_contract import (
    build_evaluation_contract,
)
from src.utils.checksum import dual_sha256
from src.utils.data.verification import file_version

PROTOCOL = Path("knowledge/runs/run-ball-all-frame-protocol-20261022")


def read(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text())
    return result


def prepare(args: argparse.Namespace) -> None:
    root = Path.cwd().resolve()
    if not args.output.is_relative_to(root / "tmp") or args.output.exists():
        raise ValueError("Use a new output directory below this checkout's tmp")
    started, started_at = time.monotonic(), datetime.now(UTC).isoformat()
    pinned = read(PROTOCOL / "inputs.json")
    source = args.source_repo_root / pinned["snapshot"]
    if any(dual_sha256(source / name) != value for name, value in pinned["snapshot_hashes"].items()):
        raise ValueError("Protocol source snapshot changed")
    store = BallFrameStore(source)
    versions = {p: file_version(p) for p in (source / "index.npz", source / "metadata.json")}
    versions.update({store.image_path(c): file_version(store.image_path(c)) for c in store.clips})
    versions.update({store.annotation_path(c): file_version(store.annotation_path(c)) for c in store.clips if c.source == "chat_annotation"})
    args.output.mkdir(parents=True)
    test_output = args.test_manifests if args.test_manifests is not None else args.output / "test-manifests"
    preparation_seconds = 0.
    if args.test_manifests is None:
        print(json.dumps(dict(stage="prepare_test_manifests", status="starting")), flush=True)
        prepare_started = time.monotonic()
        prepare_all_frame_training(args.test_distribution, test_output, configs=[AllFrameWindowConfig(t) for t in (32, 64, 128)],
            external_payloads=source_payloads(args.test_distribution, source))
        preparation_seconds = time.monotonic() - prepare_started
    distribution = load_distribution(args.test_distribution)
    mapping = {c["clip_id"]: c for c in distribution["clips"]}
    with (PROTOCOL / "clip-catalog.csv").open() as stream:
        catalog = {int(r["clip_index"]): r for r in csv.DictReader(stream)}
    intervals = dict(json.loads(line) for line in (PROTOCOL / "common-intervals-test.jsonl").read_text().splitlines())
    with (PROTOCOL / "evaluation-support.csv").open() as stream:
        owners = {int(r["frames_per_window"]): r for r in csv.DictReader(stream) if r["split"] == "test"}
    summary = read(PROTOCOL / "summary.json")
    expected = next(r for r in summary["rows"] if (r["split"], r["scope"], r["dimension"], r["name"]) == ("test", "common", "all", "all"))
    comparisons = {}
    for length in (32, 64, 128):
        path = test_output / f"evaluation_T{length}.json"
        manifest = read(path)
        actual_indices = set()
        for record in manifest["clips"]:
            clip = record["clip"]
            entry = mapping[clip["clip_id"]]
            original = entry["source_index"]
            row = catalog[original]
            if (entry["index"] != clip["index"] or original in actual_indices
                    or (clip["clip_id"], clip["source"], clip["group_id"], clip["split"], clip["fps"], clip["frame_count"]) !=
                    (row["clip_id"], row["source"], row["group_id"], row["split"], row["native_fps"], int(row["frames"]))
                    or record["common_intervals"] != intervals[original]):
                raise ValueError("Test protocol clip correspondence/common intervals changed")
            actual_indices.add(original)
        if actual_indices != set(intervals):
            raise ValueError("Test protocol clip membership changed")
        actual = manifest["summaries"]["test"]
        for name in ("target_identity_sha256", "teacher_identity_sha256"):
            if actual[name] != expected[name]:
                raise ValueError("Test protocol frame/teacher identity changed")
        if (actual["owner_identity_sha256"] != owners[length]["owner_sha256"]
                or actual["counts"]["windows"] != int(owners[length]["windows"])
                or actual["counts"]["frames"] != int(owners[length]["assigned_frames"])
                or actual["counts"]["scored_frames"] != int(owners[length]["assigned_labeled_frames"])):
            raise ValueError("Test protocol ownership/counts changed")
        comparisons[str(length)] = dict(manifest_sha256=dual_sha256(path), summary=actual,
                                       clip_correspondence_equal=True, common_intervals_equal=True,
                                       frame_teacher_identity_equal=True, owner_identity_equal=True)
    print(json.dumps(dict(stage="build_contract", status="starting", preparation_seconds=preparation_seconds)), flush=True)
    contract_started = time.monotonic()
    contract = build_evaluation_contract(source_store=source,
        training_distribution=args.training_distribution / "distribution.json",
        test_distribution=args.test_distribution / "distribution.json",
        validation_manifests=[args.training_manifests / f"evaluation_T{t}.json" for t in (32, 64, 128)],
        test_manifests=[test_output / f"evaluation_T{t}.json" for t in (32, 64, 128)])
    save_json(args.output / "evaluation-contract.json", contract)
    if any(file_version(p) != v for p, v in versions.items()):
        raise ValueError("Existing snapshot inode/size/mtime/ctime changed")
    receipt = dict(schema_version=1, status="verified_local_contract", started_at=started_at,
                   completed_at=datetime.now(UTC).isoformat(), elapsed_seconds=time.monotonic() - started,
                   preparation_seconds=preparation_seconds, contract_seconds=time.monotonic() - contract_started,
                   reused_explicit_test_manifests=args.test_manifests is not None,
                   protocol_inputs_sha256=dual_sha256(PROTOCOL / "inputs.json"),
                   contract_sha256=dual_sha256(args.output / "evaluation-contract.json"),
                   test_distribution_sha256=dual_sha256(args.test_distribution / "distribution.json"),
                   training_distribution_sha256=dual_sha256(args.training_distribution / "distribution.json"),
                   comparisons=comparisons, source_versions_unchanged=True, source_files_stat_checked=len(versions),
                   rgb_payload_bytes_copied_or_linked=0, model_test_inference=False, colab_used=False)
    save_json(args.output / "local-verification.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "comparisons"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-repo-root", "training-distribution", "test-distribution", "training-manifests", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--test-manifests", type=Path, help="Explicit existing prepared manifests; still verify every identity and source byte")
    prepare(parser.parse_args())
