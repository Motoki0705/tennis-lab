"""Qualify real-clip ball evidence publication and load-only restart.

Runs only the production ball_detection nodes, with the shipped pipeline
settings and explicitly recorded overrides. GPU execution must use the shared
training queue. All output lives below --report; source clips are read-only.
This checks the evidence contract, not detector accuracy against ground truth.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore


def qualify(
    runtime: PipelineRuntimeConfig, source: ClipSource, report: Path, *, code_identity: str,
) -> dict[str, Any]:
    """Execute the production camera nodes and restore every array from disk."""
    all_nodes = standard_definition(runtime, source, code_identity=code_identity)
    nodes = tuple(node for node in all_nodes if node.io.name == "ball_detection")
    if {node.context.camera_id for node in nodes} != set(source.camera_ids):
        raise ValueError("Ball qualification must cover every camera")
    store = ClipStore(report / "store", json_value(source), memory_entries=0)
    runner = ComponentRunner(nodes, store)
    runner.run()
    resumed = ComponentRunner(
        [replace(node, source="load") for node in nodes],
        ClipStore(store.root, json_value(source), memory_entries=0),
    )
    resumed.run()
    if any(status != "loaded" for status in resumed.statuses.values()):
        raise ValueError("Ball qualification did not restart in load-only mode")
    cameras: dict[str, Any] = {}
    for node in nodes:
        value: BallDetectionOutput = resumed.output(node.name)
        evidence = value.evidence
        if evidence is None or value.score_semantics != "model_score":
            raise ValueError("Real detector qualification requires model evidence")
        reference = resumed.references[node.name]
        if reference.schema != "ball_detections" or reference.version != 2:
            raise ValueError("Ball qualification requires schema v2")
        if len(evidence.heatmaps) != source.num_frames or evidence.source_size_wh != source.size:
            raise ValueError("Ball qualification changed the timeline or coordinate frame")
        candidates_per_frame = evidence.candidate_valid.sum(axis=-1)
        cameras[value.camera_id] = {
            "artifact": json_value(reference),
            "artifact_path": str(store.root / reference.path),
            "source_frames": len(value.frame_indices),
            "observed_frames": int(value.observed.sum()),
            "native_heatmap_shape": list(evidence.heatmaps.shape),
            "candidate_shape": list(evidence.candidate_uv_px.shape),
            "patch_shape": list(evidence.patches.shape),
            "candidate_config": json_value(evidence.config),
            "candidate_counts": {
                "min": int(candidates_per_frame.min()), "max": int(candidates_per_frame.max()),
                "total": int(candidates_per_frame.sum()),
            },
            "unobserved_frames_with_raw_candidates": int(((~value.observed) & (candidates_per_frame > 0)).sum()),
            "candidate_score_min": float(evidence.candidate_scores[evidence.candidate_valid].min()) if evidence.candidate_valid.any() else None,
            "out_of_image_patch_cells": int((evidence.candidate_valid[..., None, None] & ~evidence.patch_valid).sum()),
            "dense_heatmap_bytes": int(evidence.heatmaps.nbytes),
            "arrays_read_only": all(not array.flags.writeable for array in (
                evidence.heatmaps, evidence.candidate_uv_px, evidence.patches,
            )),
            "window_starts": [int(x) for x in np.unique(evidence.selected_window_start)],
        }
    return {
        "schema": "ball_evidence_qualification_v1", "clip_id": source.clip_id,
        "store": str(store.root), "code_sha256": code_identity,
        "statuses": runner.statuses, "seconds": runner.seconds,
        "load_only_statuses": resumed.statuses, "cameras": cameras,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Root containing data/ and ckpt/")
    parser.add_argument("--clip", type=Path, required=True, help="Structured clip directory")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    overrides = [
        f"paths.project_root={project}", f"paths.data_root={repo / 'data'}",
        f"paths.checkpoint_root={repo / 'ckpt'}", f"paths.external_asset_root={repo / 'third_party'}",
        f"paths.artifact_root={report}", f"paths.output_root={report}", f"paths.cache_root={report / 'cache'}",
        f"device={args.device}", "output_directory=run", "person_observations.enabled=false",
        "player_reconstruction.enabled=false", "gvhmr.enabled=false", *args.override,
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(project / "src/tennis_scene/configs")):
        config = compose(config_name="pipeline", overrides=overrides)
    (report / "qualification.pipeline_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    runtime = PipelineRuntimeConfig.from_config(config, bind_inputs=False)
    manifest = ClipManifest.load(args.clip.resolve())
    source = build_clip_source(tuple(manifest.media_path(camera) for camera in manifest.camera_ids),
                               tuple(manifest.camera_ids), clip_id=manifest.clip_id)
    record = qualify(runtime, source, report, code_identity=TennisSceneOrchestrator(runtime).code_identity)
    record["config_overrides"] = overrides
    write_json_atomic(report / "qualification.json", record)
    print(report / "qualification.json", flush=True)


if __name__ == "__main__":
    main()
