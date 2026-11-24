"""Reconstruct bodies from pinned 3D tracks and preserve the adopted ball and cameras."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.multiview_scene.inputs import RECIPE, read_inputs
from src.tennis_scene.multiview_scene.recipe import (
    adapter_settings,
    make_clip_execution,
    prepare_store,
)
from src.tennis_scene.multiview_scene.report import (
    completion,
    report_clip,
    scene_fields,
    write_csv,
)
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.pipeline.clip_inputs import (
    resolve_dataset_clip,
    select_dataset_clips,
)
from src.tennis_scene.pipeline.dataset_publication import ANNOTATION_RELATIVE_DIR
from src.tennis_scene.pipeline.errors import is_clip_failure
from src.tennis_scene.pipeline.execution_plan import ClipExecution, ExecutionPlan
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.source_identity import source_digests
from src.tennis_scene.pipeline.storage.clip_store import clip_store_lock
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.inference_recording import InferenceRecorder
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.multiview_scene",
    fields=(
        BoundaryPathField(
            "dataset",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "progress",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "tracks",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            required=False,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "candidates",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            required=False,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)
BALL_FIELDS = ("ball_uv", "ball_vis", "ball_3d", "ball_3d_valid", "ball_rejection_code")


def run_recipe(
    args: argparse.Namespace,
    records: Any,
    progress: dict[str, Any],
    *,
    prepare_only: bool,
) -> int:
    cfg = PipelineRuntimeConfig.from_config(
        OmegaConf.create(progress["pipeline_config_yaml"]), bind_inputs=False
    )
    implementation = document_digest(source_digests(PROJECT_ROOT / "src"))
    settings = adapter_settings(cfg, implementation)
    players_per_side = args.players_per_side
    executions: list[ClipExecution] = []
    prepared: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def failed(clip_id: str, error: Exception) -> None:
        reason = getattr(error, "reason", type(error).__name__)
        errors[clip_id] = f"{reason}: {error}"
        print(f"{clip_id} stopped: {errors[clip_id]}", flush=True)

    for record in records:
        key = record.clip_id
        try:
            start = perf_counter()
            clip = resolve_dataset_clip(args.dataset, record)
            old = StoreReader(clip.clip_dir / ANNOTATION_RELATIVE_DIR / "scene.json")
            tracks = StoreReader(args.tracks / key / "scene.json")
            source, frozen = read_inputs(
                clip,
                old,
                tracks,
                args.candidates / key,
                players_per_side=players_per_side,
                threshold=cfg.human_vis_threshold,
                crops=cfg.tracking.features,
            )
            new = prepare_store(
                args.output / key, source, frozen, old.index_path, settings
            )
            executions.append(
                make_clip_execution(new, cfg, implementation, prepare_only=prepare_only)
            )
            prepared[key] = (new, clip.snapshot, perf_counter() - start)
            print(
                f"{key} prepared IDs {frozen.people.grouped.identities.tolist()}",
                flush=True,
            )
            del frozen
        except Exception as exc:
            if not is_clip_failure(exc):
                raise
            failed(key, exc)

    def finalize(key: str) -> None:
        selected, media, _ = prepared[key]
        media.assert_unchanged()
        if dual_sha256(selected.legacy_index) != selected.pins["legacy_index_sha256"]:
            raise ValueError(f"{key}: legacy scene changed during reconstruction")
        scene = executor.output((key, "scene_assembly"))
        old_ball = scene_fields(StoreReader(selected.legacy_index), BALL_FIELDS)
        if any(
            not np.array_equal(getattr(scene, name), value)
            for name, value in old_ball.items()
        ):
            raise ValueError(f"{key}: original ball arrays do not match")
        executor.finalize_clip(key)
        print(f"{key} exported {scene.metadata['status']}", flush=True)

    timing: dict[str, Any] = {"phase": args.phase, "clips": {}, "errors": errors}
    if executions:
        plan = ExecutionPlan(executions)
        executor = PlanExecutor(plan)
        executor.run(
            continue_on_error=True,
            on_clip_error=failed,
            on_clip_complete=None if prepare_only else finalize,
        )
        for key, (_, _, bind_seconds) in prepared.items():
            timing["clips"][key] = {
                "bind_seconds": bind_seconds,
                "nodes": {
                    node: {"seconds": seconds, "status": plan.statuses[(cid, node)]}
                    for (cid, node), seconds in plan.seconds.items()
                    if cid == key
                },
            }
    write_json_atomic(args.output / f"{args.phase}.json", timing)
    return int(bool(errors))


def write_completion(
    args: argparse.Namespace, records: Any, progress: dict[str, Any]
) -> list[dict[str, Any]]:
    old_rows = []
    new_rows = []
    run_path = args.output / "run.json"
    state_path = run_path if run_path.is_file() else args.output / "prepare.json"
    timing = json.loads(state_path.read_text()) if state_path.is_file() else {}

    for record in records:
        key = record.clip_id
        old = StoreReader(
            args.dataset / record.path / ANNOTATION_RELATIVE_DIR / "scene.json"
        )
        old_state = progress["clips"][key]
        stop = (
            str(old_state.get("reason") or old_state.get("error") or "")
            if old_state["status"] == "failed"
            else ""
        )
        before = report_clip(
            old, players_per_side=args.players_per_side, new=False, reason=stop
        )
        new_index = args.output / key / "scene.json"
        error = timing.get("errors", {}).get(key, "")
        if new_index.is_file():
            result = report_clip(
                StoreReader(new_index),
                players_per_side=args.players_per_side,
                new=True,
                reason=error,
            )
        else:
            result = completion(
                key,
                record.num_frames,
                (),
                {},
                args.players_per_side,
                None,
                {},
                reason=error or "not_prepared",
            )
            result.update(
                unused_view_points=0,
                unused_view_median_px=None,
                unused_view_p95_px=None,
            )
        before["seconds"] = sum(
            float(node["seconds"])
            for node in progress["nodes"][key].values()
            if node["status"] == "executed"
        )
        current = timing.get("clips", {}).get(key, {})
        result["seconds"] = (
            sum(
                node["seconds"]
                for node in current.get("nodes", {}).values()
                if node["status"] == "executed"
            )
            if current
            else None
        )
        old_rows.append(before)
        new_rows.append(result)
    write_csv(args.output / "legacy-completion.csv", old_rows)
    write_csv(args.output / "multiview-completion.csv", new_rows)
    return [row for row in new_rows if not row["complete"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "run", "report", "publish", "restore")
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Archived dataset with adopted external ball",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        required=True,
        help="Original video batch progress.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Separate multiview scene store and completion tables",
    )
    parser.add_argument(
        "--tracks", type=Path, help="Integrated multiview tracking stores"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        help="Frame pose candidates pinned by the integrated tracks",
    )
    parser.add_argument("--players-per-side", type=int, required=True, choices=(1, 2))
    parser.add_argument(
        "--clip", action="append", help="Select full clip ID; repeat to select several"
    )
    args = parser.parse_args()
    if args.phase in {"prepare", "run"} and (
        args.tracks is None or args.candidates is None
    ):
        parser.error("prepare/run require --tracks and --candidates")
    if args.phase == "run" and not all(
        os.environ.get(name) for name in ("TENNIS_RUN_ID", "TENNIS_GPU_RESOURCE")
    ):
        parser.error("Run the GPU phase through the shared training queue")
    paths = {
        field.name: getattr(args, field.name).expanduser().absolute()
        for field in PATH_BOUNDARY.fields
        if getattr(args, field.name) is not None
    }
    output = paths["output"].resolve()
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT,
        output_root=output,
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    resolved = PATH_BOUNDARY.validate(
        paths, resolver=PathResolver(roots), independent_artifact_inputs=True
    )
    for name in paths:
        setattr(args, name, resolved.declared(name).path)
    inputs = [args.dataset]
    if args.tracks is not None:
        inputs.append(args.tracks)
    if args.candidates is not None:
        inputs.append(args.candidates)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
        parser.error("--output must be separate from the input dataset and stores")
    progress = json.loads(args.progress.read_text())
    records = select_dataset_clips(args.dataset, args.clip)
    if any(record.clip_id not in progress["clips"] for record in records):
        raise ValueError("Selected clips must be in the original batch record")
    with clip_store_lock(output):
        if args.phase in {"publish", "restore"}:
            from src.tennis_scene.multiview_scene.publication import change_publication

            change_publication(
                args.dataset, output, records, restore=args.phase == "restore"
            )
            return 0
        if args.phase == "report":
            write_completion(args, records, progress)
            return 0
        if args.phase == "prepare":
            code = run_recipe(args, records, progress, prepare_only=True)
        else:
            receipt = (
                output / f"recording-{Path(os.environ['TENNIS_RUN_ID']).name}.json"
            )
            with InferenceRecorder(
                receipt,
                metadata={"recipe": RECIPE, "clips": [r.clip_id for r in records]},
                devices=("cuda:0",),
            ):
                code = run_recipe(args, records, progress, prepare_only=False)
        report_records = (
            select_dataset_clips(args.dataset) if args.phase == "run" else records
        )
        incomplete = write_completion(args, report_records, progress)
        requested = {record.clip_id for record in records}
        incomplete = [row for row in incomplete if row["clip_id"] in requested]
        if args.phase == "run" and incomplete:
            for row in incomplete:
                print(f"{row['clip_id']} incomplete: {row['reason']}", flush=True)
            return 1
        return code


if __name__ == "__main__":
    raise SystemExit(main())
