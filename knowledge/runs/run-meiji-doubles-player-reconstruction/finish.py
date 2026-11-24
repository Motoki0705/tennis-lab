"""Run body inference and publish the prepared Meiji player revisions."""

from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any
from zipfile import ZipFile

import numpy as np
from frozen_clip import (
    BALL_FIELDS,
    PLAYERS,
    SCENE_FIELDS,
    FrozenClip,
    player_counts,
    read_json,
)
from prepare import ReplayRun, read_preparation, replay_node

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import DatasetClipRecord
from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.assembly import assemble_automatic_scene
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.body_placement import (
    BodyPlacementModule,
    BodyPlacementOutput,
)
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.components.hmr2_features import HMR2FeaturesModule
from src.tennis_scene.pipeline.components.identity import PlayerIdentitiesOutput
from src.tennis_scene.pipeline.components.scene_assembly import SceneAssemblyModule
from src.tennis_scene.pipeline.components.triangulation import (
    BallTriangulationOutput,
    PlayerTriangulationOutput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.dataset_publication import (
    ANNOTATION_RELATIVE_DIR,
    DECLARED_PIPELINE_CONTRACT,
    validate_dataset_result,
)
from src.tennis_scene.pipeline.imports.external_ball_publication import (
    ScenePointers,
    current_publication,
    publish_scene_revision,
    restore_publication,
)
from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime, ModelSpec
from src.tennis_scene.pipeline.input_assembly.body import (
    BodyPlacementInputAssembler,
    GVHMRInputAssembler,
    HMR2FeaturesInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.scene import SceneAssemblyInputAssembler
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
)
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import dual_sha256
from src.utils.video import VideoInfo

NEEDS_PLACEMENT = {"recover_bodies", "place_saved_bodies"}
ARCHIVED_BODY_MODELS = {
    "hmr2_features": "hmr2",
    "gvhmr": "gvhmr",
    "body_placement": "body_geometry",
}


def gpu_record(run: ReplayRun, record: DatasetClipRecord) -> Path:
    return run.preparation_path(record).with_name("gpu.json")


def reconstruction_values(clip: FrozenClip, bindings: dict[str, str]) -> dict[str, Any]:
    return {
        "calibration": clip.load(
            bindings["calibration"], CourtCalibrationOutput, "local_court_calibration"
        ),
        "alignment": clip.load(
            bindings["alignment"], CameraAlignmentOutput, "aligned_cameras"
        ),
        "identities": clip.load(
            bindings["identities"], PlayerIdentitiesOutput, "person_identities", 3
        ),
        "skeleton": clip.load(
            bindings["skeleton"], PlayerTriangulationOutput, "player_skeletons"
        ),
        **{
            f"pose_{camera}": clip.load(
                bindings[f"pose_{camera}"], ObjectObservations, "person_poses", 2
            )
            for camera in clip.source.camera_ids
        },
    }


def load_gpu_record(
    run: ReplayRun,
    record: DatasetClipRecord,
    clip: FrozenClip,
    script_sha: str,
) -> dict[str, Any]:
    saved = read_json(gpu_record(run, record))
    if (
        saved["cpu_sha256"] != dual_sha256(run.preparation_path(record))
        or saved["script_sha256"] != script_sha
    ):
        raise ValueError(
            f"{record.clip_id}: body results do not match the prepared CPU inputs and script"
        )
    staged = {name: ArtifactRef(**ref) for name, ref in saved["staged"].items()}
    if any(staged.get(name) != ref for name, ref in clip.staged.items()):
        raise ValueError(
            f"{record.clip_id}: body results changed the prepared CPU artifacts"
        )
    clip.check_staged(staged)
    clip.staged = staged
    return saved


def archived_body_assets(dataset: Path) -> dict[str, dict[str, dict[str, str]]]:
    (archive,) = (dataset.parents[1] / "all/progress").glob("*/progress.json")
    models = read_json(archive)["models"]
    assets = {}
    for stage, model in ARCHIVED_BODY_MODELS.items():
        (loaded,) = (
            row["attributes"]["weights"]
            for row in models
            if row["stage"] == "model_load"
            and row["status"] == "ok"
            and row["attributes"]["model"] == model
        )
        assets[stage] = loaded
    return assets


def recover_body_batch(
    run: ReplayRun,
    records: Sequence[DatasetClipRecord],
    cfg: PipelineRuntimeConfig,
    script_sha: str,
) -> None:
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_GPU_RESOURCE"):
        raise ValueError("Run the gpu phase through the shared training queue")
    pending = []
    for record in records:
        clip = FrozenClip.bind(run.dataset, record, run.recipe_for(record))
        preparation = read_preparation(run, record, clip)
        if preparation["mode"] in NEEDS_PLACEMENT:
            if gpu_record(run, record).exists():
                load_gpu_record(run, record, clip, script_sha)
            else:
                pending.append((record, clip, preparation))
    if not pending:
        return

    cameras = pending[0][1].source.camera_ids
    if any(clip.source.camera_ids != cameras for _, clip, _ in pending):
        raise ValueError("The body queue batch must have one camera layout")
    hmr2 = HMR2FeaturesModule(cfg.people)
    gvhmr = GVHMRModule(cfg.people)
    placement = BodyPlacementModule(
        cameras,
        cfg.people,
        cfg.player_placement,
        reprojection_px=cfg.player_reprojection_px,
    )
    specs: dict[str, Mapping[str, ModelSpec[Any]]] = {
        "body_placement": placement.model_specs(),
    }
    if any(p["mode"] == "recover_bodies" for _, _, p in pending):
        specs.update(hmr2_features=hmr2.model_specs(), gvhmr=gvhmr.model_specs())
    declarations = {
        (record.clip_id, stage): spec
        for record, _, p in pending
        for stage, spec in specs.items()
        if stage == "body_placement" or p["mode"] == "recover_bodies"
    }
    assets = archived_body_assets(run.dataset)
    expected_assets = {}
    for node, aliases in declarations.items():
        declared = {
            name: path
            for spec in aliases.values()
            for name, path in spec.weights.items()
        }
        archived = assets[node[1]]
        if declared.keys() != archived.keys():
            raise ValueError(
                f"{node[1]} model assets differ from the archived #2363 declaration"
            )
        expected_assets[node] = {
            path: archived[name]["sha256"] for name, path in declared.items()
        }
    with InferenceRuntime(declarations, expected_assets=expected_assets) as runtime:
        for record, clip, preparation in pending:
            started = perf_counter()
            context = AssemblyContext(clip.source)
            selected_node = preparation["body_view_selection"]
            selected = clip.load(
                selected_node, BodyViewSelectionOutput, "body_view_selection"
            )
            selected_values = {"selection": selected}
            hmr_seconds = gvhmr_seconds = 0.0
            if preparation["mode"] == "recover_bodies":
                stage_start = perf_counter()
                with runtime.node((record.clip_id, "hmr2_features")) as models:
                    features = hmr2.process_with_models(
                        HMR2FeaturesInputAssembler().assemble(context, selected_values),
                        models,
                    )
                    feature_node = clip.stage(
                        replay_node("hmr2_features"),
                        features,
                        hmr2.io,
                        {"selection": selected_node},
                        settings={
                            "script_sha256": script_sha,
                            "model_identity": models.identity("features"),
                        },
                    )
                hmr_seconds = perf_counter() - stage_start

                stage_start = perf_counter()
                with runtime.node((record.clip_id, "gvhmr")) as models:
                    recovered = gvhmr.process_with_models(
                        GVHMRInputAssembler().assemble(
                            context, {**selected_values, "features": features}
                        ),
                        models,
                    )
                    recovered_node = clip.stage(
                        replay_node("gvhmr"),
                        recovered,
                        gvhmr.io,
                        {"selection": selected_node, "features": feature_node},
                        settings={
                            "script_sha256": script_sha,
                            "model_identity": models.identity("mesh"),
                        },
                    )
                gvhmr_seconds = perf_counter() - stage_start
            else:
                recovered_node = "gvhmr"
                recovered = clip.load_output(recovered_node, gvhmr.io)

            stage_start = perf_counter()
            bindings = preparation["reconstruction"]
            values = {**reconstruction_values(clip, bindings), "recovered": recovered}
            with runtime.node((record.clip_id, "body_placement")) as models:
                placed = placement.process_with_models(
                    BodyPlacementInputAssembler(cfg.human_vis_threshold).assemble(
                        context, values
                    ),
                    models,
                )
                placed_node = clip.stage(
                    replay_node("body_placement"),
                    placed,
                    placement.io,
                    {
                        **bindings,
                        "recovered": recovered_node,
                        "body_views": selected_node,
                    },
                    settings={
                        "script_sha256": script_sha,
                        "model_identity": models.identity("body"),
                    },
                )
            clip.check_staged(clip.staged)
            clip.clip.snapshot.assert_unchanged()
            if ScenePointers.read(clip.store.root) != clip.current:
                raise ValueError(
                    f"{record.clip_id}: published pointers changed during body reconstruction"
                )
            write_json_atomic(
                gpu_record(run, record),
                {
                    "cpu_sha256": dual_sha256(run.preparation_path(record)),
                    "script_sha256": script_sha,
                    "queue_job_id": os.environ["TENNIS_RUN_ID"],
                    "staged": json_value(clip.staged),
                    "placement": placed_node,
                    "recovered": recovered_node,
                    "seconds": {
                        "hmr2_features": hmr_seconds,
                        "gvhmr": gvhmr_seconds,
                        "body_placement": perf_counter() - stage_start,
                        "gpu": perf_counter() - started,
                    },
                },
            )
            print(
                f"{record.clip_id} {preparation['mode']} placement prepared", flush=True
            )


def assemble_revision(
    clip: FrozenClip,
    preparation: dict[str, Any],
    gpu: dict[str, Any] | None,
    cfg: PipelineRuntimeConfig,
) -> tuple[SceneResult, SceneAssemblyModule, dict[str, str]]:
    source = clip.source
    module = SceneAssemblyModule(
        source.camera_ids, require_people=True, require_ball=True, require_body=True
    )
    prior = clip.descriptor("scene_assembly", "scene_result", 2)
    mode = preparation["mode"]
    ball_node = "ball_triangulation"
    if mode == "reuse_scene_players":
        scene = clip.load_output("scene_assembly", module.io)
        names = {ref: name for name, ref in clip.refs.items()}
        bindings = {
            port: names[ArtifactRef(**ref)]
            for port, ref in prior["dependencies"].items()
        }
        bindings.update(preparation["reconstruction"])
        if preparation["body_view_selection"] is not None:
            bindings["body_views"] = preparation["body_view_selection"]
    else:
        ball = clip.load(
            ball_node, BallReconstructionResult, "external_ball_trajectory"
        )
        bindings = preparation["reconstruction"].copy()
        if mode == "stopped":
            calibration = clip.load(
                bindings["calibration"],
                CourtCalibrationOutput,
                "local_court_calibration",
            )
            alignment = clip.load(
                bindings["alignment"], CameraAlignmentOutput, "aligned_cameras"
            )
            scene = assemble_automatic_scene(
                video_paths=source.paths,
                camera_ids=source.camera_ids,
                info=VideoInfo(source.fps, *source.size, source.num_frames),
                court=calibration.court,
                active_indices=tuple(
                    view.source_index for view in calibration.calibration.views
                ),
                geometry=alignment.geometry,
                grouped_people=None,
                skeleton=None,
                players=None,
                ball=ball,
                metadata={
                    "status": "partial",
                    "dataset_clip_id": source.clip_id,
                    "pipeline_contract": DECLARED_PIPELINE_CONTRACT,
                },
            )
        else:
            if gpu is None:
                raise ValueError(f"{source.clip_id}: body placement is pending")
            bindings["placement"] = gpu["placement"]
            values = {
                **reconstruction_values(clip, bindings),
                "ball": BallTriangulationOutput(ball),
                "placement": clip.load(
                    bindings["placement"], BodyPlacementOutput, "placed_bodies"
                ),
            }
            scene = module.process(
                SceneAssemblyInputAssembler(cfg.human_vis_threshold).assemble(
                    AssemblyContext(source), values
                )
            )
    metadata = dict(scene.metadata)
    metadata["external_ball"] = prior["payload"]["metadata"]["external_ball"]
    metadata["player_reconstruction"] = {
        "status": "unavailable" if mode == "stopped" else mode,
        "selection_rule": preparation["selection_rule"],
        "previous_scene_assembly": clip.pinned["previous_scene_assembly"],
        "camera_geometry": clip.pinned["alignment"],
        "previous": preparation["before_stop"],
        **(preparation["stop"] or {}),
    }
    scene = replace(scene, metadata=metadata)
    clip.check_ball(scene)
    validate_dataset_result(scene, clip.clip.record)
    return scene, module, {**bindings, **preparation["selection"], "ball": ball_node}


def verify_revision(clip: FrozenClip) -> dict[str, Any]:
    receipt = clip.player_receipt
    if receipt is None or receipt.after != clip.current:
        raise ValueError(
            f"{clip.source.clip_id}: the revised scene and annotation pointers are incomplete"
        )
    path = clip.current.verify(clip.store.root)
    if path is None:
        raise ValueError("Revised scene is missing")
    revised_index = clip.current.index()
    original_index = clip.previous.index()
    for node, ref in original_index["artifacts"].items():
        if node != "scene_assembly" and revised_index["artifacts"].get(node) != ref:
            raise ValueError(
                f"{clip.source.clip_id}: preserved reference changed: {node}"
            )
    required = {
        key: original_index["artifacts"][key]
        for key in ("ball_triangulation", clip.pinned["alignment_node"])
    }
    lineage = assert_current_component_lineage(
        revised_index, clip.store.root, revised_index["exports"]["scene"]["inputs"]
    )
    if not required.keys() <= lineage:
        raise ValueError(
            "Revised export does not depend on the fixed camera and ball artifacts"
        )
    old = clip.descriptor("scene_assembly", "scene_result", 2)
    with ZipFile(path) as archive:
        for name in BALL_FIELDS:
            with archive.open(f"{name}.npy") as component:
                digest = hashlib.sha256()
                for chunk in iter(lambda: component.read(1024 * 1024), b""):
                    digest.update(chunk)
                if (
                    digest.hexdigest()
                    != old["arrays"][old["payload"][name]["array"]]["sha256"]
                ):
                    raise ValueError(
                        f"{clip.source.clip_id}: published ball bytes changed: {name}"
                    )
    with np.load(path, allow_pickle=False) as values:
        counts = player_counts({field: values[field] for field in SCENE_FIELDS})
    if (
        counts != receipt.statistics["after"]
        or clip.before_counts() != receipt.statistics["before"]
    ):
        raise ValueError(
            f"{clip.source.clip_id}: published player masks disagree with the receipt"
        )
    return {
        **receipt.statistics,
        "pointer_digests": receipt.pointer_digests(),
        "scene_assembly_id": path.parent.name,
        "previous_scene_assembly_id": clip.refs["scene_assembly"].artifact_id,
        "alignment_artifact_id": clip.refs[clip.pinned["alignment_node"]].artifact_id,
        "ball_artifact_id": clip.refs["ball_triangulation"].artifact_id,
        "upstream": clip.pinned["upstream"],
    }


def publish_clip(
    run: ReplayRun,
    record: DatasetClipRecord,
    clip: FrozenClip,
    preparation: dict[str, Any],
    gpu: dict[str, Any] | None,
    cfg: PipelineRuntimeConfig,
    script_sha: str,
) -> None:
    start = perf_counter()
    report_path = run.preparation_path(record).with_name("publication.json")
    identity = {
        **clip.pinned,
        "cpu_sha256": dual_sha256(run.preparation_path(record)),
        "publication_script_sha256": script_sha,
    }
    if gpu is not None:
        identity["gpu_sha256"] = dual_sha256(gpu_record(run, record))
    if clip.player_receipt is not None:
        if clip.player_receipt.inputs != identity:
            raise ValueError(
                f"{record.clip_id}: published scene input identity differs"
            )
        if clip.current == clip.player_receipt.after:
            verified = verify_revision(clip)
            if not report_path.exists():
                write_json_atomic(report_path, verified)
            return
    scene, module, bindings = assemble_revision(clip, preparation, gpu, cfg)
    counts = player_counts({name: getattr(scene, name) for name in SCENE_FIELDS})
    if counts["joint_player_frames"] != preparation["cpu_joint_player_frames"] or (
        counts["players"] != preparation["cpu_player_count"]
    ):
        raise ValueError(
            f"{record.clip_id}: assembly changed the prepared player masks or axis"
        )
    clip.stage("scene_assembly", scene, module.io, bindings, settings=identity)
    stats = {
        "clip_id": record.clip_id,
        "mode": preparation["mode"],
        "before": preparation["before"],
        "after": counts,
        "before_stop": preparation["before_stop"].get("reason_code"),
        "stop": (preparation["stop"] or {}).get("reason_code"),
        "seconds": {
            **preparation["seconds"],
            **({} if gpu is None else gpu["seconds"]),
            "assembly": perf_counter() - start,
        },
    }
    receipt = publish_scene_revision(
        clip.store.root,
        clip.current,
        clip.previous,
        clip.staged,
        scene,
        clip.clip,
        clip.clip.snapshot,
        identity,
        stats,
        kind=PLAYERS,
    )
    published = replace(clip, current=receipt.after, player_receipt=receipt, staged={})
    verified = verify_revision(published)
    verified["seconds"]["publish_and_verify"] = perf_counter() - start
    write_json_atomic(report_path, verified)
    print(
        f"{record.clip_id} {verified['scene_assembly_id']} {stats['after']}", flush=True
    )


def complete_phase(
    phase: str, run: ReplayRun, records: Sequence[DatasetClipRecord]
) -> None:
    if phase == "cpu-report":
        report_cpu(run, records)
        return
    if phase == "restore":
        for record in records:
            root = (run.dataset / record.path / ANNOTATION_RELATIVE_DIR).resolve(
                strict=True
            )
            print(
                f"{record.clip_id} restore={restore_publication(root, PLAYERS)}",
                flush=True,
            )
        return
    script_sha = dual_sha256(Path(__file__))
    if phase == "gpu":
        recover_body_batch(run, records, run.config(), script_sha)
        return
    if phase == "publish":
        cfg = run.config()
        ready = []
        for record in records:
            clip = FrozenClip.bind(run.dataset, record, run.recipe_for(record))
            preparation = read_preparation(run, record, clip)
            gpu = (
                load_gpu_record(run, record, clip, script_sha)
                if preparation["mode"] in NEEDS_PLACEMENT
                else None
            )
            ready.append((record, clip, preparation, gpu))
        for record, clip, preparation, gpu in ready:
            publish_clip(run, record, clip, preparation, gpu, cfg, script_sha)
        return
    report = [
        verify_revision(FrozenClip.bind(run.dataset, record, run.recipe_for(record)))
        for record in records
    ]
    write_json_atomic(run.output / "verification.json", report)
    write_csv(
        run.output / "clips.csv",
        [
            {
                "clip_id": row["clip_id"],
                "mode": row["mode"],
                "stop_before": row["before_stop"],
                "stop_after": row["stop"],
                **{
                    f"{name}_{era}": value
                    for era in ("before", "after")
                    for name, value in row[era].items()
                },
                **{
                    f"{name}_seconds": row["seconds"].get(name, 0.0)
                    for name in (
                        "cpu",
                        "association",
                        "triangulation",
                        "body_view_selection",
                        "hmr2_features",
                        "gvhmr",
                        "body_placement",
                        "gpu",
                        "assembly",
                    )
                },
                **{
                    name: row[name]
                    for name in (
                        "previous_scene_assembly_id",
                        "scene_assembly_id",
                        "alignment_artifact_id",
                        "ball_artifact_id",
                    )
                },
            }
            for row in report
        ],
    )
    write_csv(
        run.output / "upstream.csv",
        [
            {
                "clip_id": row["clip_id"],
                "node": node,
                "artifact_id_before_and_after": ref["artifact_id"],
            }
            for row in report
            for node, ref in row["upstream"].items()
        ],
    )
    print(
        f"Read back {len(report)} revisions, fixed camera and ball lineage, and all ball arrays",
        flush=True,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report_cpu(run: ReplayRun, records: Sequence[DatasetClipRecord]) -> None:
    observations = []
    upstream_rows = []
    for record in records:
        cpu = read_json(run.preparation_path(record))
        root = (run.dataset / record.path / ANNOTATION_RELATIVE_DIR).resolve(
            strict=True
        )
        current, receipt = current_publication(root, PLAYERS)
        baseline = current if receipt is None else receipt.before
        pins = cpu["pinned"]
        if (
            cpu["clip_id"] != record.clip_id
            or any(
                pins.get(key) != value for key, value in run.recipe_for(record).items()
            )
            or document_digest(baseline) != pins["previous_pointer_sha256"]
        ):
            raise ValueError(
                f"{record.clip_id}: CPU evidence does not match the #2458 inputs"
            )
        masks = cpu["masks"]
        observations.append(
            {
                "clip_id": record.clip_id,
                "mode": cpu["mode"],
                "geometry": pins["alignment_node"],
                "players_before": cpu["before"]["players"],
                "players_after_cpu": cpu["cpu_player_count"],
                "joint_player_frames_before": cpu["before"]["joint_player_frames"],
                "joint_player_frames_after_cpu": cpu["cpu_joint_player_frames"],
                "root_player_frames_before": cpu["before"]["root_player_frames"],
                "mesh_player_frames_before": cpu["before"]["mesh_player_frames"],
                "stop_before": cpu["before_stop"].get("reason_code"),
                "stop_after_cpu": (cpu["stop"] or {}).get("reason_code"),
                "changed_selection_views": sum(
                    not row["same_mask_as_before"] for row in masks.values()
                ),
                "observations_before": sum(
                    row["before_observations"] for row in masks.values()
                ),
                "observations_after": sum(
                    row["after_observations"] for row in masks.values()
                ),
                **{
                    f"{name}_seconds": round(value, 4)
                    for name, value in cpu["seconds"].items()
                },
                "previous_scene_assembly_id": pins["previous_scene_assembly"][
                    "artifact_id"
                ],
                "alignment_artifact_id": pins["alignment"]["artifact_id"],
                "ball_artifact_id": pins["ball"]["artifact_id"],
                "comparison_sha256": cpu["selection_comparison_sha256"],
                "missing_raw_appearance": sum(
                    row.get("missing_raw_appearance", 0) for row in masks.values()
                ),
                "prepare_sha256": pins["cpu_script_sha256"]["prepare.py"],
            }
        )
        for node, ref in pins["upstream"].items():
            upstream_rows.append(
                {
                    "clip_id": record.clip_id,
                    "node": node,
                    "artifact_id": ref["artifact_id"],
                }
            )
    write_csv(run.output / "cpu-clips.csv", observations)
    write_csv(run.output / "cpu-upstream.csv", upstream_rows)
    write_json_atomic(
        run.output / "cpu-summary.json",
        {
            "recipe": run.recipe,
            "clips": len(observations),
            "mode": dict(Counter(row["mode"] for row in observations)),
            "geometry": dict(Counter(row["geometry"] for row in observations)),
            "changed_selection_views": sum(
                row["changed_selection_views"] for row in observations
            ),
            "joint_player_frames_before": sum(
                row["joint_player_frames_before"] for row in observations
            ),
            "joint_player_frames_after_cpu": sum(
                row["joint_player_frames_after_cpu"] for row in observations
            ),
            "stop_after_cpu": {
                row["clip_id"]: row["stop_after_cpu"]
                for row in observations
                if row["stop_after_cpu"]
            },
            "recovered_clips": [
                row["clip_id"]
                for row in observations
                if row["stop_before"] and not row["stop_after_cpu"]
            ],
            "cpu_seconds_sum": sum(row["cpu_seconds"] for row in observations),
            "upstream_references": len(upstream_rows),
            "prepare_versions": dict(
                Counter(row["prepare_sha256"] for row in observations)
            ),
        },
    )
