"""Read back the two GPU pilots in separate stores and check the live references."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from shutil import copy2, copytree
from typing import Any

import numpy as np
from finish import load_gpu_record, publish_clip
from frozen_clip import SCENE_FIELDS, FrozenClip, player_counts, read_json
from prepare import ReplayRun, read_preparation

from src.tennis_scene.generate_dataset.manifest import (
    DatasetClipRecord,
    load_dataset_manifest,
)
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.body_placement import BodyPlacementOutput
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.gvhmr import GVHMROutput
from src.tennis_scene.pipeline.dataset_publication import (
    ANNOTATION_RELATIVE_DIR,
    validate_dataset_result,
)
from src.tennis_scene.pipeline.imports.external_ball_publication import ScenePointers
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import dual_sha256

PILOTS = ("video_000/clip_000", "video_000/clip_020")
LEDGER = (
    Path(__file__).resolve().parent.parent
    / "run-meiji-external-ball-20261124/clips.csv"
)


def verify_baseline(dataset: Path, records: dict[str, DatasetClipRecord]) -> int:
    with LEDGER.open(newline="") as handle:
        baseline = {row["clip"]: row for row in csv.DictReader(handle)}
    if len(baseline) != 69 or set(records) != {
        f"video_000/{clip}" for clip in baseline
    }:
        raise ValueError("Expected the complete #2458 clip ledger and dataset")
    for record in records.values():
        root = dataset / record.path / ANNOTATION_RELATIVE_DIR
        row = baseline[record.clip_name]
        if (
            dual_sha256(root / "scene.json") != row["after_scene_sha256"]
            or dual_sha256(root / "annotation.json") != row["after_annotation_sha256"]
        ):
            raise ValueError(
                f"{record.clip_id}: live pointers differ from the #2458 ledger"
            )
    return len(baseline)


def verify_pilot(
    run: ReplayRun,
    isolated: ReplayRun,
    record: DatasetClipRecord,
    job_id: str,
) -> dict[str, Any]:
    clip = FrozenClip.bind(run.dataset, record, run.recipe_for(record))
    preparation = read_preparation(run, record, clip)
    script_sha = dual_sha256(Path(__file__).with_name("finish.py"))
    gpu = load_gpu_record(run, record, clip, script_sha)
    if gpu["queue_job_id"] != job_id.removesuffix(".job"):
        raise ValueError(
            f"{record.clip_id}: saved body belongs to a different queue job"
        )
    recovered = clip.load(gpu["recovered"], GVHMROutput, "body_parameters", 2)
    selection = clip.load(
        preparation["body_view_selection"],
        BodyViewSelectionOutput,
        "body_view_selection",
    )
    if len(recovered.bodies) != preparation["cpu_player_count"]:
        raise ValueError(
            f"{record.clip_id}: GVHMR player count differs from the pilot preparation"
        )
    for person, view in zip(recovered.bodies, selection.selections, strict=True):
        if (person.person_id, person.camera_id) != (view.person_id, view.camera_id):
            raise ValueError(
                f"{record.clip_id}: GVHMR person or camera differs from the selected view"
            )
        for segment, request, observed in zip(
            person.segments, view.requests, view.observed_samples, strict=True
        ):
            if segment.observed_samples != observed or not np.array_equal(
                segment.source_frames, request.source_frames
            ):
                raise ValueError(
                    f"{record.clip_id}: GVHMR segment differs from the source frames"
                )
    placed = clip.load(gpu["placement"], BodyPlacementOutput, "placed_bodies").players
    if placed is None or not placed.root_valid.any() or not placed.smpl_valid.any():
        raise ValueError(f"{record.clip_id}: expected valid body placements")

    local_preparation = isolated.preparation_path(record)
    local_preparation.parent.mkdir()
    for name in ("cpu.json", "gpu.json"):
        copy2(
            run.preparation_path(record).with_name(name),
            local_preparation.with_name(name),
        )
    root = local_preparation.parent / "tennis_scene"
    copytree(clip.store.root, root)
    local = replace(
        clip, store=ClipStore(root, json_value(clip.source), memory_entries=0)
    )
    publish_clip(isolated, record, local, preparation, gpu, run.config(), script_sha)

    report = read_json(local_preparation.with_name("publication.json"))
    revised = ScenePointers.read(root).index()
    reference = ArtifactRef(**revised["artifacts"]["scene_assembly"])
    scene = local.store.load(reference, ArtifactCodec(SceneResult))
    validate_dataset_result(scene, record)
    counts = player_counts({name: getattr(scene, name) for name in SCENE_FIELDS})
    if (
        counts != report["after"]
        or scene.player_valid is None
        or scene.player_smpl_valid is None
        or not np.array_equal(scene.player_valid, placed.root_valid)
        or not np.array_equal(scene.player_smpl_valid, placed.smpl_valid)
    ):
        raise ValueError(
            f"{record.clip_id}: scene, export and body-placement masks disagree"
        )
    return {
        **{key: value for key, value in report.items() if key != "upstream"},
        "gvhmr_artifact_id": clip.ref(gpu["recovered"]).artifact_id,
        "body_placement_artifact_id": clip.ref(gpu["placement"]).artifact_id,
        "gvhmr_segments": sum(len(person.segments) for person in recovered.bodies),
        "scene_path": f"{record.clip_name}/tennis_scene/exports/{reference.artifact_id}/scene.npz",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory holding the CPU and GPU receipts",
    )
    parser.add_argument(
        "--readback",
        type=Path,
        required=True,
        help="New directory for the two verified local stores",
    )
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    dataset, readback = args.dataset.resolve(strict=True), args.readback.resolve()
    if readback.is_relative_to(dataset):
        parser.error("--readback must lie outside the shared dataset")
    records = load_dataset_manifest(dataset).clips
    before = verify_baseline(dataset, records)
    run = ReplayRun.bind(
        dataset, args.output.resolve(strict=True), args.comparison.resolve()
    )
    readback.mkdir(parents=True)
    isolated = replace(run, output=readback)
    clips = [verify_pilot(run, isolated, records[clip], args.job_id) for clip in PILOTS]
    after = verify_baseline(dataset, records)
    write_json_atomic(
        readback / "pilot-summary.json",
        {
            "queue_job": args.job_id,
            "live_pointer_pairs_verified_before": before,
            "live_pointer_pairs_verified_after": after,
            "ball_arrays_byte_matched_per_pilot": 5,
            "clips": clips,
        },
    )
    print(
        f"Read back {len(clips)} pilot scenes and verified {after} live #2458 pointer pairs."
    )


if __name__ == "__main__":
    main()
