"""Run a short pinned clip through the standard player reconstruction recipe."""

import argparse
import os
from collections import Counter
from pathlib import Path

from hydra import compose, initialize_config_dir

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.clip_inputs import (
    resolve_dataset_clip,
    select_dataset_clips,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.utils.configuration import PathRole
from src.utils.inference_recording import InferenceRecorder
from src.utils.io import write_json_atomic
from src.utils.paths import PROJECT_ROOT

MODELS = {
    "dino": ("dino_detector", "DINO"),
    "vitpose": ("vitpose", "ViTPose"),
    "clip": ("clip_reid", "CLIP-ReID"),
    "hmr2": ("hmr2", "HMR2"),
    "gvhmr": ("gvhmr", "GVHMR-network"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory under the artifact root"
    )
    parser.add_argument(
        "--engine",
        action="append",
        required=True,
        help="MODEL=artifact-relative-directory",
    )
    args = parser.parse_args()
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("Run the pipeline smoke through the shared training queue")
    engines = dict(item.split("=", 1) for item in args.engine)
    if engines.keys() != MODELS.keys() or len(args.engine) != len(MODELS):
        raise ValueError("Select one bundle for each of the five models")
    overrides = [
        "device=cuda",
        "court_side.evidence=player",
        "ball_detection.enabled=false",
        "ball_reconstruction.enabled=false",
        "player_association.players_per_side=2",
        f"output_directory={args.output}",
    ]
    for name, directory in engines.items():
        prefix = f"people_models.runtime.{MODELS[name][0]}"
        overrides += [
            f"{prefix}.backend=tensorrt",
            f"{prefix}.precision=fp16",
            f"{prefix}.batch_size=1",
            f"{prefix}.engines=[{directory}]",
        ]
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        raw = compose(config_name="pipeline", overrides=overrides)
    cfg = PipelineRuntimeConfig.from_config(raw, bind_inputs=False)
    dataset = cfg.resolver.resolve(
        PathRole.DATA, "tennis_multiview/processed/meiji_4cam/dataset"
    )
    (record,) = select_dataset_clips(dataset, ("video_000/clip_037",))
    pinned = resolve_dataset_clip(dataset, record)
    output = cfg.resolver.resolve(PathRole.ARTIFACT, args.output)
    output.mkdir(parents=True, exist_ok=False)
    pipeline = TennisSceneOrchestrator(cfg)
    identity = pipeline.publication_identity()
    write_json_atomic(output / "publication.json", identity)
    recorder = InferenceRecorder(output / "recording.json", devices=("cuda:0",))
    with recorder:
        scene = pipeline.run(
            pinned.video_paths,
            video_role=PathRole.DATA,
            camera_ids=pinned.camera_ids,
            clip_id=record.clip_id,
            store_root=output / "store",
        )
    pinned.snapshot.assert_unchanged()
    executed = Counter(
        event["attributes"].get("model")
        for event in recorder.events
        if event["stage"] == "inference" and event["status"] == "ok"
    )
    wanted = {model for _, model in MODELS.values()}
    missing = wanted - executed.keys()
    write_json_atomic(
        output / "smoke.json",
        {
            "queue_job": os.environ["TENNIS_RUN_ID"],
            "clip_id": record.clip_id,
            "media_sha256": pinned.media_sha256,
            "frames": scene.num_frames,
            "status": scene.metadata["status"],
            "validity": scene.metadata["validity_statistics"],
            "forward_calls": {name: executed[name] for name in sorted(wanted)},
            "nodes": pipeline.last_runner.statuses,
        },
    )
    if missing:
        raise ValueError(f"Selected models did not run: {sorted(missing)}")


if __name__ == "__main__":
    main()
