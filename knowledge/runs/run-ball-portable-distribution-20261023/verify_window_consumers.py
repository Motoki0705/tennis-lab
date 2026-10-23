"""Compare the real all-frame CLI on the pinned source and portable source view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.distribution import (
    load_distribution,
    source_payloads,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.utils.checksum import dual_sha256
from src.utils.data.verification import file_version


def compare_manifests(source: dict[str, Any], portable: dict[str, Any]) -> dict[str, Any]:
    if any(source[key] != portable[key] for key in ("schema", "mode", "config", "semantics", "label_kinds")):
        raise ValueError("Source and distribution all-frame contracts differ")
    expected = {r["clip"]["clip_id"]: r for r in source["clips"] if r["clip"]["split"] in {"train", "val"}}
    actual = {r["clip"]["clip_id"]: r for r in portable["clips"]}
    if (len(actual) != len(portable["clips"]) or set(expected) != set(actual)
            or any(r["clip"]["split"] not in {"train", "val"} for r in actual.values())):
        raise ValueError("Source and distribution common clip memberships differ")
    remapped = 0
    for clip_id, reference in expected.items():
        record = actual[clip_id]
        before, after = reference["clip"].copy(), record["clip"].copy()
        remapped += before.pop("index") != after.pop("index")
        if before != after:
            raise ValueError(f"Clip identity differs: {clip_id}")
        for key in ("common_intervals", "label_runs", "semantic_sha256", "rgb_shard_sha256",
                    "packs" if source["mode"] == "train" else "windows"):
            if reference[key] != record[key]:
                raise ValueError(f"All-frame membership/teacher/owner differs: {clip_id}/{key}")
        evidence = [{k: v for k, v in r["negative_evidence"].items() if k != "annotation_path"}
                    for r in (reference, record)]
        if evidence[0] != evidence[1]:
            raise ValueError(f"Verified original annotation evidence differs: {clip_id}")
    summaries = {}
    for split in ("train", "val"):
        if source["summaries"][split] != portable["summaries"][split]:
            raise ValueError(f"Common interval, pack, teacher or ownership identities differ: {split}")
        summaries[split] = portable["summaries"][split]
    return dict(clips=len(actual), remapped_clip_indices=remapped, summaries=summaries,
                common_intervals_equal=True, packs_or_owner_windows_equal=True,
                teacher_kinds_equal=True, negative_evidence_equal=True)


def verify(source_repo: Path, distribution: Path, output: Path) -> dict[str, Any]:
    root = Path.cwd().resolve()
    if not all(p.is_absolute() for p in (source_repo, distribution, output)):
        raise ValueError("Source repository, distribution and output must be explicit absolute paths")
    if not output.is_relative_to(root / "tmp") or output.exists():
        raise ValueError("Use a new output below this worktree's tmp directory")
    protocol = json.loads(Path("knowledge/runs/run-ball-all-frame-protocol-20261022/inputs.json").read_text())
    snapshot = source_repo / protocol["snapshot"]
    rgb_manifest = source_repo / protocol["source_manifest"]
    for name, digest in protocol["snapshot_hashes"].items():
        if dual_sha256(snapshot / name) != digest:
            raise ValueError("The protocol source snapshot changed")
    if dual_sha256(rgb_manifest) != protocol["source_manifest_sha256"]:
        raise ValueError("The source RGB manifest changed")
    manifest = load_distribution(distribution)
    payloads = source_payloads(distribution, snapshot)
    source = BallFrameStore(snapshot)
    portable = BallFrameStore(distribution, external_payloads=payloads)
    paths = {p for p in distribution.rglob("*") if p.is_file()}
    paths.update(snapshot / name for name in ("metadata.json", "index.npz"))
    paths.add(rgb_manifest)
    paths.update(source.image_path(c) for c in source.clips)
    paths.update(source.annotation_path(c) for c in source.clips if c.source == "chat_annotation")
    versions = {p: file_version(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    started, started_at = time.monotonic(), datetime.now(UTC).isoformat()

    def portable_text(value: str) -> str:
        return value.replace(str(root), "<worktree>").replace(str(source_repo), "<source-repo>")

    stages = {}
    for name, store in (("source", snapshot), ("distribution", distribution)):
        command = [sys.executable, "-m", "src.tasks.ball_detection.scripts.prepare_data", "--mode", "all_frames",
                   "--ball-store", str(store), "--output", str(output / name),
                   "--window-lengths", "32", "64", "128", "--max-window-length", "128",
                   "--pack-length", "128", "--pack-stride", "16", "--timestamp-gap-factor", "2.5"]
        command += ["--rgb-manifest", str(rgb_manifest)] if name == "source" else ["--distribution-source-store", str(snapshot)]
        stage_started, stage_at = time.monotonic(), datetime.now(UTC).isoformat()
        print(json.dumps(dict(stage=name, status="starting", started_at=stage_at)), flush=True)
        with (output / f"{name}.stdout.log").open("x") as stdout, (output / f"{name}.stderr.log").open("x") as stderr:
            subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, check=True,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                                    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        stages[name] = dict(argv=[portable_text(v) for v in command], started_at=stage_at,
                            elapsed_seconds=time.monotonic() - stage_started, returncode=0)
        print(json.dumps(dict(stage=name, **stages[name])), flush=True)
    plans = {name: json.loads((output / name / "experiments.json").read_text()) for name in stages}
    code = plans["source"]["code"]
    if code["source_sha256"] != plans["distribution"]["code"]["source_sha256"]:
        raise ValueError("Implementation changed between source and distribution preparation")
    comparisons: dict[str, dict[str, Any]] = {}
    for length in (32, 64, 128):
        comparisons[str(length)] = {}
        for mode in ("train", "evaluation"):
            records = [json.loads(Path(plans[name]["datasets"][str(length)][mode]["path"]).read_text())
                       for name in stages]
            comparisons[str(length)][mode] = compare_manifests(*records)
    if any(file_version(path) != version for path, version in versions.items()):
        raise ValueError("Source or distribution inputs changed during consumer verification")
    relevant = ("data/store.py", "data/windows/all_frame_preparation.py", "data/windows/hybrid.py", "scripts/prepare_data.py")
    result = dict(
        schema_version=1, status="verified", started_at=started_at, completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.monotonic() - started, source_snapshot=protocol["snapshot"],
        distribution=distribution.relative_to(root).as_posix(), output=output.relative_to(root).as_posix(),
        distribution_version=portable.metadata["version"], distribution_sha256=dual_sha256(distribution / "distribution.json"),
        source_snapshot_hashes=manifest["source"]["hashes"], stages=stages, comparisons=comparisons,
        source_files_stat_checked=len(versions), source_versions_unchanged=True,
        rgb_payload_bytes_copied_or_linked=0, preparation_api="prepare_all_frame_training via the public prepare_data CLI",
        payload_mode=plans["distribution"]["payload_mode"], dataset_fixture_verification="test_relocated_distribution_preserves_all_frame_manifests_and_dataset",
        code=dict(base_commit=code["base_commit"],
                  source_identity_sha256=hashlib.sha256(json.dumps(code["source_sha256"], sort_keys=True).encode()).hexdigest(),
                  source_files_sha256={k: v for k, v in code["source_sha256"].items()
                                       if k.removeprefix("src/tasks/ball_detection/") in relevant}),
    )
    (output / "consumer-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status="verified", elapsed_seconds=result["elapsed_seconds"],
                         source_files_stat_checked=len(versions))), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-root", type=Path, required=True)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.source_repo_root, args.distribution, args.output)
