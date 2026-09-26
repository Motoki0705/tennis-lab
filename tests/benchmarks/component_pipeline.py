"""Real-clip qualification of the default pipeline with confirmed-data imports.

The default ``pipeline.yaml`` runs on one structured clip. The only overrides
are root paths, the device, and ``execution.ball_detection=load`` for the ball
import. The node without a merged model (``player_association``) and the ball
are filled by ``src/tennis_scene/pipeline/imports``; the receipt lists them
under ``imported_nodes`` so that nothing imported is mistaken for model output.
``court_side`` is decided by its component from the imported ball. GPU
execution goes through the shared training queue.

With ``--dataset`` the clip belongs to a structured dataset: the store is the
clip's ``annotations/tennis_scene`` and the production
``generate_pseudo_annotations`` publishes ``annotation.json``; the dataset
readers (SLCS, PLCS residual) then read the publication back. ``--seed-from``
first creates that clip from a source clip (media hard-linked, import inputs
copied) in a dataset directory that must not contain it yet.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    register_exported_clip,
)
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    ANNOTATION_RELATIVE_DIR,
    generate_pseudo_annotations,
)
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.imports.ball_annotations import import_ball_annotations
from src.tennis_scene.pipeline.imports.person_association import (
    import_confirmed_person_association,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.schema import SceneResult
from src.utils.configuration import PathRole

CONFIG_DIR = Path(__file__).resolve().parents[2] / "src/tennis_scene/configs"


def compose_default(repo: Path, report: Path, device: str) -> Any:
    overrides = [f"paths.project_root={CONFIG_DIR.parents[2]}", f"paths.data_root={repo / 'data'}",
        f"paths.checkpoint_root={repo / 'ckpt'}", f"paths.external_asset_root={repo / 'third_party'}",
        f"paths.artifact_root={report}", f"paths.output_root={report}", f"paths.cache_root={report / 'cache'}",
        f"device={device}", "output_directory=run", "execution.ball_detection=load"]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="pipeline", overrides=overrides), overrides


def seed_clip(source: Path, dataset: Path) -> Path:
    """Create ``source``'s clip in ``dataset``: media hard-linked, import inputs copied, registered."""
    manifest = ClipManifest.load(source)
    destination: Path = dataset / "videos" / manifest.video_id / "clips" / manifest.clip_name
    if destination.exists():
        raise FileExistsError(f"Seed destination already exists: {destination}")
    (destination / "media").mkdir(parents=True)
    (destination / "outsource").mkdir()
    (destination / "annotations").mkdir()
    shutil.copy2(source / "clip.json", destination / "clip.json")
    for camera in manifest.camera_ids:
        media = manifest.media_path(camera)
        os.link(media, destination / media.relative_to(source))
        shutil.copy2(source / "outsource" / f"{camera}_annotations.json", destination / "outsource")
        shutil.copy2(source / "annotations" / f"gvhmr_result_{camera}.json", destination / "annotations")
    shutil.copy2(source / "annotations/player_association_result.json", destination / "annotations")
    register_exported_clip(dataset, destination / "clip.json")
    return destination


def verify_dataset_readers(clip: Path) -> dict[str, Any]:
    """Read the publication back through every dataset reader."""
    from src.tasks.plcs.inference.residual_clip_io import load_real_clip
    from src.tasks.slcs.data.annotation import load_slcs_annotation

    scene = load_slcs_annotation(ClipManifest.load(clip))
    residual = load_real_clip(clip)
    return {"slcs": {"schema_version": scene.schema_version, "num_frames": scene.num_frames},
            "plcs_residual": {"player_ids": [item.metadata["player_id"] for item in residual],
                              "calibration_frame_index": residual[0].metadata["calibration_frame_index"]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Root holding data/, ckpt/ and third_party/")
    parser.add_argument("--clip", type=Path, required=True, help="Structured clip directory (clip.json, media/, outsource/, annotations/)")
    parser.add_argument("--report", type=Path, required=True, help="Run-owned directory for the store and receipts")
    parser.add_argument("--dataset", type=Path, default=None, help="Publish into this structured dataset (the clip must belong to it)")
    parser.add_argument("--seed-from", type=Path, default=None, help="With --dataset: create --clip from this source clip first")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo, clip, report = args.repo.resolve(), args.clip.resolve(), args.report.resolve()
    dataset = None if args.dataset is None else args.dataset.resolve()
    if dataset is None and args.seed_from is not None:
        raise ValueError("--seed-from requires --dataset")
    if dataset is not None and not clip.is_relative_to(dataset):
        raise ValueError(f"--clip {clip} is not inside --dataset {dataset}")
    report.mkdir(parents=True, exist_ok=True)
    if args.seed_from is not None:
        assert dataset is not None
        if seed_clip(args.seed_from.resolve(), dataset) != clip:
            raise ValueError("--clip must be the seeded clip directory")
    config, overrides = compose_default(repo, report, args.device)
    pipeline_yaml = OmegaConf.to_yaml(config, resolve=True)
    (report / "pipeline_config.yaml").write_text(pipeline_yaml)
    runtime = PipelineRuntimeConfig.from_config(config, bind_inputs=False)
    manifest = ClipManifest.load(clip)
    camera_ids = tuple(manifest.camera_ids)
    videos = tuple(manifest.media_path(camera) for camera in camera_ids)
    source = build_clip_source(videos, camera_ids, clip_id=manifest.clip_id)
    store_root = report / "store" if dataset is None else clip / ANNOTATION_RELATIVE_DIR
    store = ClipStore(store_root, json_value(source))
    application = TennisSceneOrchestrator(runtime)
    nodes = standard_definition(runtime, source, code_identity=application.code_identity)
    receipt: dict[str, Any] = {"schema": "component_pipeline_qualification_v2", "status": "running",
        "clip": str(clip), "clip_id": source.clip_id, "config_overrides": overrides,
        "publication_identity": application.publication_identity(), "store": str(store_root),
        "dataset": None if dataset is None else str(dataset),
        "seeded_from": None if args.seed_from is None else str(args.seed_from.resolve())}

    def qualify() -> SceneResult:
        balls = import_ball_annotations(nodes, store, source, clip / "outsource")
        upstream = ComponentRunner(nodes, store)
        upstream.run(targets=("court_calibration", *(f"pose_estimation/{c}" for c in camera_ids)))
        receipt["upstream"] = {"status": upstream.statuses, "seconds": upstream.seconds}
        people, person_confirmation = import_confirmed_person_association(nodes, store, source,
            historical_association=clip / "annotations/player_association_result.json",
            legacy_gvhmr_directory=clip / "annotations")
        receipt["imported_nodes"] = {**{name: json_value(ref) for name, ref in balls.items()},
            "player_association": json_value(people)}
        receipt["person_confirmation"] = person_confirmation
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
        side = runner.output("court_side")
        receipt["court_side"] = {"view_half_turns": list(side.view_half_turns), "margin": side.margin, "frames": side.frames,
            "hypotheses": [json_value(h) for h in side.hypotheses]}
        receipt.update(status=scene.metadata["status"], run=application.last_receipt, load_only_resume=resume.statuses,
            validity=scene.metadata["validity_statistics"], player_ids=players,
            half_turns=scene.metadata["court_reference"]["view_half_turns"], frame_count=source.num_frames,
            ball_annotation_counts={name: store.descriptor(ref)["provenance"]["counts"] for name, ref in balls.items()})
        return scene

    try:
        if dataset is None:
            qualify()
        else:
            def run_clip(video_paths: object, cameras: object, clip_directory: Path) -> SceneResult:
                if clip_directory.resolve() != clip:
                    raise AssertionError(f"Generator asked for {clip_directory}, not {clip}")
                return qualify()

            outcomes = generate_pseudo_annotations(dataset, run_clip, pipeline_config_yaml=pipeline_yaml,
                publication_identity=application.publication_identity(), clip_ids=[source.clip_id],
                continue_on_error=False)
            if [outcome.status for outcome in outcomes] != ["generated"]:
                raise AssertionError(f"Dataset publication did not generate the clip: {outcomes}")
            receipt["annotation"] = str(outcomes[0].annotation_path)
            receipt["dataset_readers"] = verify_dataset_readers(clip)
    except Exception as error:
        receipt.update(status="failed", error=str(error), error_type=type(error).__name__, run=application.last_receipt)
        raise
    finally:
        write_json_atomic(report / "evaluation.json", receipt)
    print(json.dumps({"status": receipt["status"], "validity": receipt["validity"], "scene_index": str(store.index_path)}))


if __name__ == "__main__":
    main()
