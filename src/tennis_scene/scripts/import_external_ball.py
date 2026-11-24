"""Attach a saved external ball trajectory to a structured clip dataset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src.tennis_scene.generate_dataset.manifest import (
    load_dataset_manifest,
    validate_id_component,
)
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.pipeline.clip_inputs import select_dataset_clips
from src.tennis_scene.pipeline.dataset_publication import ANNOTATION_RELATIVE_DIR
from src.tennis_scene.pipeline.imports.external_ball_geometry import PinnedBallCourt
from src.tennis_scene.pipeline.imports.external_ball_publication import (
    restore_publication,
)
from src.tennis_scene.pipeline.imports.external_ball_update import update_clip
from src.tennis_scene.pipeline.source_identity import source_digests
from src.utils.checksum import FileIntegrityError, dual_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser(
        "publish", help="Triangulate and switch the selected clip pointers"
    )
    restore = commands.add_parser(
        "restore", help="Restore the recorded previous clip pointers"
    )
    for command in (publish, restore):
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--video", required=True)
        command.add_argument(
            "--clip",
            action="append",
            metavar="NAME",
            help="Select by clip directory name, repeat as needed",
        )
    publish.add_argument("--positions", type=Path, required=True)
    publish.add_argument("--side-store", type=Path, required=True)
    publish.add_argument("--side-digest", required=True)
    publish.add_argument("--court-store", type=Path, required=True)
    publish.add_argument("--drift-store", type=Path)
    publish.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    start = time.perf_counter()
    dataset = args.dataset.resolve()
    dataset_sha256 = dual_sha256(dataset / "dataset.json")
    video = validate_id_component(args.video, field_name="video")
    requested = (
        tuple(
            f"{video}/{validate_id_component(c, field_name='clip')}" for c in args.clip
        )
        if args.clip
        else None
    )
    selected = tuple(
        sorted(
            key
            for key, record in load_dataset_manifest(dataset).clips.items()
            if record.video_id == video
        )
    )
    if not selected or (
        requested
        and (
            len(set(requested)) != len(requested)
            or not set(requested).issubset(selected)
        )
    ):
        raise ValueError(
            "Select one or more distinct clips from the declared dataset video"
        )
    records = select_dataset_clips(dataset, requested or selected)
    if args.command == "restore":
        for record in records:
            restored = restore_publication(
                (dataset / record.path / ANNOTATION_RELATIVE_DIR).resolve()
            )
            print(
                f"{record.clip_id} {'restored' if restored else 'unchanged'}",
                flush=True,
            )
        return 0

    court = PinnedBallCourt.load(
        args.side_store, args.side_digest, args.court_store, args.drift_store
    )
    if requested is None and set(selected) != set(court.side.clips):
        raise ValueError(
            "The selected dataset video differs from the pinned side clip membership"
        )
    implementation = document_digest(
        source_digests(Path(__file__).resolve().parents[2])
    )
    outputs: list[dict[str, Any]] = []
    for record in records:
        path = args.positions / f"{record.clip_name}_ball_positions.json"
        try:
            result = update_clip(dataset, record, path, court, implementation)
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            RuntimeError,
            FileIntegrityError,
        ) as error:
            result = {
                "clip_id": record.clip_id,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"{record.clip_id} failed: {result['error']}", flush=True)
        else:
            result["status"] = "published"
            print(
                f"{record.clip_id} {result['valid_3d']}/{result['frames']} valid 3D ({result['elapsed_seconds']:.2f}s)",
                flush=True,
            )
        outputs.append(result)

    dataset_unchanged = dual_sha256(dataset / "dataset.json") == dataset_sha256
    write_json_atomic(
        args.report,
        {
            "dataset_sha256": dataset_sha256,
            "dataset_unchanged": dataset_unchanged,
            "video": video,
            "side_digest": court.side.digest,
            "code_sha256": implementation,
            "wall_seconds": time.perf_counter() - start,
            "clips": outputs,
        },
    )
    return int(
        not dataset_unchanged or any(output["status"] == "failed" for output in outputs)
    )


if __name__ == "__main__":
    raise SystemExit(main())
