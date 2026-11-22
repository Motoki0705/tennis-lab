"""Replay the player-only geometry on CPU with committed GPU player observations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path.cwd() / "knowledge/runs/run-i1757-video-batch-meiji"))

from benchmark import CLIPS, SPEC, read, verify_dataset  # noqa: E402
from report import compare_value  # noqa: E402

from src.tennis_scene.archive import load_scene_result  # noqa: E402
from src.tennis_scene.configuration import PipelineRuntimeConfig  # noqa: E402
from src.tennis_scene.pipeline.clip_inputs import prepare_video_inputs  # noqa: E402
from src.tennis_scene.pipeline.components.body_placement import (
    BodyPlacementOutput,  # noqa: E402
)
from src.tennis_scene.pipeline.components.body_view_selection import (  # noqa: E402
    BodyViewSelectionModule,
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.camera_alignment import (  # noqa: E402
    CameraAlignmentModule,
    CameraAlignmentOutput,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,  # noqa: E402
)
from src.tennis_scene.pipeline.components.court_side import (
    CourtSideOutput,  # noqa: E402
)
from src.tennis_scene.pipeline.components.identity import (
    PlayerIdentitiesOutput,  # noqa: E402
)
from src.tennis_scene.pipeline.components.scene_assembly import (
    SceneAssemblyModule,  # noqa: E402
)
from src.tennis_scene.pipeline.components.triangulation import (  # noqa: E402
    PlayerTriangulationModule,
    PlayerTriangulationOutput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext  # noqa: E402
from src.tennis_scene.pipeline.dataset_publication import (  # noqa: E402
    ANNOTATION_RELATIVE_DIR,
    validate_dataset_result,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable  # noqa: E402
from src.tennis_scene.pipeline.input_assembly.body import (
    BodyViewSelectionInputAssembler,  # noqa: E402
)
from src.tennis_scene.pipeline.input_assembly.reconstruction import (  # noqa: E402
    CameraAlignmentInputAssembler,
    PlayerTriangulationInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.scene import (
    SceneAssemblyInputAssembler,  # noqa: E402
)
from src.tennis_scene.pipeline.observation_types import ObjectObservations  # noqa: E402
from src.tennis_scene.pipeline.storage.codec import (  # noqa: E402
    restore_type,
    unpack_value,
)
from src.tennis_scene.pipeline.storage.scene_index import (  # noqa: E402
    annotation_scene_path,
    read_component_descriptor,
)
from src.tennis_scene.schema import SceneResult  # noqa: E402


def read_output(store, index, name, output_type):
    ref = index["artifacts"][name]
    descriptor = read_component_descriptor(
        store, ref, node=name, source_sha256=index["source_sha256"]
    )
    payload = unpack_value(
        descriptor["payload"], (store / ref["path"]).parent, descriptor["arrays"]
    )
    return restore_type(payload, output_type)


def comparable(value):
    if isinstance(value, dict):
        return {key: comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [comparable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    return value


def player_arrays(scene):
    return {
        field.name: getattr(scene, field.name)
        for field in fields(SceneResult)
        if field.name.startswith(("player_", "human_kp_", "smpl_", "gvhmr_aligned_"))
    }


def replay(clip, record, runtime, dataset):
    context = AssemblyContext(clip.media.source)
    cameras = context.source.camera_ids
    store = clip.inputs.clip_dir / ANNOTATION_RELATIVE_DIR
    index = read(store / "scene.json")
    ports = {
        "calibration": read_output(
            store, index, "court_calibration", CourtCalibrationOutput
        ),
        "side": read_output(store, index, "court_side", CourtSideOutput),
        "identities": read_output(
            store, index, "player_association", PlayerIdentitiesOutput
        ),
        **{
            f"pose_{camera}": read_output(
                store, index, f"pose_estimation/{camera}", ObjectObservations
            )
            for camera in cameras
        },
    }
    saved_alignment = read_output(
        store, index, "camera_alignment", CameraAlignmentOutput
    )
    if saved_alignment.geometry is None:
        raise ValueError("Expected committed camera geometry for the matched scene")
    module = CameraAlignmentModule(
        cameras,
        runtime.camera_geometry,
        player_reprojection_px=runtime.player_reprojection_px,
        ball_reprojection_px=runtime.ball_reprojection_px,
        joint_confidence=runtime.joint_confidence,
        max_frames=runtime.sampling_max_frames,
        include_ball=False,
    )
    aligned = module.process(
        CameraAlignmentInputAssembler(runtime.human_vis_threshold, None).assemble(
            context, ports
        )
    )
    if aligned.geometry is None:
        return {"status": "no_player_geometry"}
    geometry, saved = asdict(aligned.geometry), asdict(saved_alignment.geometry)
    validation = [
        value.pop("document")["side_validation"] for value in (saved, geometry)
    ]
    roots = (dataset, dataset)
    geometry_exact = compare_value(
        comparable(saved), comparable(geometry), "camera_alignment", roots
    )["exact"]

    ports["alignment"] = aligned
    triangulate = PlayerTriangulationModule(
        cameras,
        reprojection_px=runtime.player_reprojection_px,
        joint_confidence=runtime.joint_confidence,
        enabled=runtime.enabled["player_reconstruction"],
    )
    skeleton = triangulate.process(
        PlayerTriangulationInputAssembler(runtime.human_vis_threshold).assemble(
            context, ports
        )
    )
    saved_skeleton = read_output(
        store, index, "player_triangulation", PlayerTriangulationOutput
    )
    skeleton_diff = compare_value(
        comparable(asdict(saved_skeleton)),
        comparable(asdict(skeleton)),
        "player_triangulation",
        roots,
    )

    selector = BodyViewSelectionModule(
        cameras,
        max_frames=runtime.sampling_max_frames,
        enabled=runtime.enabled["gvhmr"],
    )
    selection = selector.process(
        BodyViewSelectionInputAssembler(runtime.human_vis_threshold).assemble(
            context, ports
        )
    )
    saved_selection = read_output(
        store, index, "body_view_selection", BodyViewSelectionOutput
    )
    selection_exact = compare_value(
        comparable(asdict(saved_selection)),
        comparable(asdict(selection)),
        "body_view_selection",
        roots,
    )["exact"]
    measurements = {
        "with_ball_validation": validation[0],
        "player_only_validation": validation[1],
        "camera_geometry_exact": geometry_exact,
        "player_triangulation": skeleton_diff,
        "gpu_body_request_exact": selection_exact,
    }
    if not (geometry_exact and skeleton_diff["exact"] and selection_exact):
        return {"status": "saved_body_inputs_changed", **measurements}

    ports.update(
        skeleton=skeleton,
        placement=read_output(store, index, "body_placement", BodyPlacementOutput),
    )
    scene = SceneAssemblyModule(
        cameras,
        require_people=runtime.enabled["player_reconstruction"],
        require_ball=False,
        require_body=runtime.enabled["gvhmr"],
        include_ball=False,
    ).process(
        SceneAssemblyInputAssembler(
            runtime.human_vis_threshold, include_ball=False
        ).assemble(context, ports)
    )
    validate_dataset_result(scene, record)
    archive = annotation_scene_path(store, read(store / "annotation.json"))
    published = load_scene_result(archive)
    outcome = compare_value(
        comparable(player_arrays(published)),
        comparable(player_arrays(scene)),
        "scene_assembly",
        roots,
    )
    return {
        "status": scene.metadata["status"],
        "players": len(scene.player_position),
        "ball_observed": bool(scene.ball_3d_valid.any()),
        "scene_players_vs_published": outcome,
        **measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip", choices=CLIPS)
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        help="Explicit compatible config for artifacts from an older schema",
    )
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    config_path = args.pipeline_config or attempt.parents[1] / "pipeline.yaml"
    config = OmegaConf.load(config_path)
    config.ball_detection.enabled = False
    config.ball_reconstruction.enabled = False
    runtime = PipelineRuntimeConfig.from_config(config, bind_inputs=False)
    if not runtime.omit_ball_components:
        raise ValueError("The pinned config did not select player-only reconstruction")
    dataset = attempt / "dataset"
    manifest = verify_dataset(dataset)
    _, clips = prepare_video_inputs(
        dataset, SPEC["video_id"], [args.clip] if args.clip else CLIPS
    )
    results = {}
    for clip in clips:
        name = clip.media.source.clip_id
        try:
            result = replay(clip, manifest.clips[name], runtime, dataset)
        except ReconstructionUnavailable as error:
            result = {
                "status": "reconstruction_unavailable",
                "reason": error.reason,
                "diagnostics": error.diagnostics,
            }
        results[name] = result
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for clip, row in results.items():
        print(
            clip, row["status"], row.get("scene_players_vs_published", {}).get("exact")
        )


if __name__ == "__main__":
    main()
