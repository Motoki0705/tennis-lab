"""Real-clip qualification of the default pipeline with confirmed-data imports.

The default ``pipeline.yaml`` runs on one structured clip. The only overrides
are root paths, the device, and ``execution.ball_detection=load`` for the ball
import. Nodes without a merged model (``court_side``, ``player_association``)
and the ball are filled by ``src/tennis_scene/pipeline/imports``; the receipt
lists them under ``imported_nodes`` so that nothing imported is mistaken for
model output. GPU execution goes through the shared training queue.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.imports.ball_annotations import import_ball_annotations
from src.tennis_scene.pipeline.imports.court_side import import_ball_confirmed_sides
from src.tennis_scene.pipeline.imports.person_association import (
    import_confirmed_person_association,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.utils.configuration import PathRole

CONFIG_DIR = Path(__file__).resolve().parents[2] / "src/tennis_scene/configs"


def compose_default(repo: Path, report: Path, device: str) -> Any:
    overrides = [f"paths.project_root={CONFIG_DIR.parents[2]}", f"paths.data_root={repo / 'data'}",
        f"paths.checkpoint_root={repo / 'ckpt'}", f"paths.external_asset_root={repo / 'third_party'}",
        f"paths.artifact_root={report}", f"paths.output_root={report}", f"paths.cache_root={report / 'cache'}",
        f"device={device}", "output_directory=run", "execution.ball_detection=load"]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="pipeline", overrides=overrides), overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Root holding data/, ckpt/ and third_party/")
    parser.add_argument("--clip", type=Path, required=True, help="Structured clip directory (clip.json, media/, outsource/, annotations/)")
    parser.add_argument("--report", type=Path, required=True, help="Run-owned directory for the store and receipts")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo, clip, report = args.repo.resolve(), args.clip.resolve(), args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    config, overrides = compose_default(repo, report, args.device)
    (report / "pipeline_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    runtime = PipelineRuntimeConfig.from_config(config, bind_inputs=False)
    manifest = ClipManifest.load(clip)
    camera_ids = tuple(manifest.camera_ids)
    videos = tuple(manifest.media_path(camera) for camera in camera_ids)
    source = build_clip_source(videos, camera_ids, clip_id=manifest.clip_id)
    store_root = report / "store"
    store = ClipStore(store_root, json_value(source))
    application = TennisSceneOrchestrator(runtime)
    nodes = standard_definition(runtime, source, code_identity=application.code_identity)
    receipt: dict[str, Any] = {"schema": "component_pipeline_qualification_v2", "status": "running",
        "clip": str(clip), "clip_id": source.clip_id, "config_overrides": overrides,
        "publication_identity": application.publication_identity(), "store": str(store_root)}
    try:
        balls = import_ball_annotations(nodes, store, source, clip / "outsource")
        upstream = ComponentRunner(nodes, store)
        upstream.run(targets=("court_calibration", *(f"pose_estimation/{c}" for c in camera_ids)))
        receipt["upstream"] = {"status": upstream.statuses, "seconds": upstream.seconds}
        side, side_confirmation = import_ball_confirmed_sides(nodes, store, source,
            ball_threshold=runtime.ball_detection.score_threshold, ball_reprojection_px=runtime.ball_reprojection_px,
            max_frames=runtime.sampling_max_frames, config=runtime.camera_geometry)
        people, person_confirmation = import_confirmed_person_association(nodes, store, source,
            historical_association=clip / "annotations/player_association_result.json",
            legacy_gvhmr_directory=clip / "annotations")
        receipt["imported_nodes"] = {**{name: json_value(ref) for name, ref in balls.items()},
            "court_side": json_value(side), "player_association": json_value(people)}
        receipt["side_confirmation"], receipt["person_confirmation"] = side_confirmation, person_confirmation
        write_json_atomic(report / "side_confirmation.json", side_confirmation)
        write_json_atomic(report / "person_confirmation.json", person_confirmation)

        scene = application.run(videos, video_role=PathRole.DATA, camera_ids=camera_ids, store_root=store_root,
                                clip_id=source.clip_id)
        runner = application.last_runner
        assert runner is not None
        imported = [name for name in receipt["imported_nodes"] if runner.statuses[name] != "loaded"]
        if imported:
            raise AssertionError(f"Imported nodes must be loaded, not recomputed: {imported}")
        expected = sorted(set(person_confirmation["player_ids"]))
        players = [] if scene.player_track_ids is None else scene.player_track_ids.tolist()
        if players != expected:
            raise AssertionError(f"Scene players {players} differ from the confirmed {expected}")
        for node, field in (("body_view_selection", "selections"), ("gvhmr", "bodies")):
            ids = sorted(item.person_id for item in getattr(runner.output(node), field))
            if ids != expected:
                raise AssertionError(f"{node} must contain exactly the confirmed players, not {ids}")
        restored = load_scene_result(store.index_path)
        np.testing.assert_array_equal(scene.ball_3d, restored.ball_3d)
        np.testing.assert_array_equal(scene.player_valid, restored.player_valid)
        # A load-only resume from disk must not execute any component.
        resume = ComponentRunner([replace(node, source="load") for node in nodes], ClipStore(store_root, json_value(source), memory_entries=0))
        resume.run()
        np.testing.assert_array_equal(resume.output("scene_assembly").ball_3d, scene.ball_3d)
        receipt.update(status=scene.metadata["status"], run=application.last_receipt, load_only_resume=resume.statuses,
            validity=scene.metadata["validity_statistics"], player_ids=players,
            half_turns=scene.metadata["court_reference"]["view_half_turns"], frame_count=source.num_frames,
            ball_annotation_counts={name: store.descriptor(ref)["provenance"]["counts"] for name, ref in balls.items()})
    except Exception as error:
        receipt.update(status="failed", error=str(error), error_type=type(error).__name__, run=application.last_receipt)
        raise
    finally:
        write_json_atomic(report / "evaluation.json", receipt)
    print(json.dumps({"status": receipt["status"], "validity": receipt["validity"], "scene_index": str(store.index_path)}))


if __name__ == "__main__":
    main()
