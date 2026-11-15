"""Run the canonical single-clip pipeline for a fixed model and readonly inputs.

Queue this script with resource=all. The experiment disables people/GVHMR in
both variants to compare ball-only camera alignment and triangulation.
"""

import argparse
import json
import logging
from pathlib import Path
import subprocess
import time

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

from src.tennis_scene.archive import save_scene_result
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline import TennisSceneOrchestrator
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionInput, BallDetectionModule
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from src.utils.video import probe_video_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-assets", type=Path, required=True)
    parser.add_argument("--variant", choices=("i935", "x3d"), required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.70)
    config_name = "pipeline" if args.variant == "i935" else "pipeline_x3d_dpt"
    with initialize_config_dir(version_base="1.3", config_dir=str(Path.cwd() / "src/tennis_scene/configs")):
        cfg = compose(config_name=config_name, overrides=[
            "person_observations.enabled=false", "player_reconstruction.enabled=false", "gvhmr.enabled=false",
            f"paths.external_asset_root={args.external_assets}",
        ])
    runtime = PipelineRuntimeConfig.from_config(cfg, bind_inputs=False)
    assert runtime.ball_detection is not None
    for clip in args.clips:
        source_root = args.dataset / "videos/video_000/clips" / clip
        marker = source_root / "clip.json"
        metadata = json.loads(marker.read_text())
        if not metadata.get("exported_at") or metadata["camera_ids"] != ["cam0", "cam1", "cam2", "cam3"]:
            raise ValueError("Use only completed four-camera exports")
        videos = tuple((source_root / path).resolve() for path in metadata["video_paths"])
        for video in videos:
            info = probe_video_info(video)
            if (info.num_frames, info.width, info.height) != (metadata["num_frames"], metadata["width"], metadata["height"]):
                raise ValueError("Export marker/video dimensions differ")
        output = args.output / args.variant / clip
        if output.resolve().is_relative_to(args.dataset.resolve()):
            raise ValueError("Experiment must not write into the dataset")
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2) + "\n")
        orchestrator = TennisSceneOrchestrator.from_runtime_config(runtime)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        scene = None
        failure = None
        try:
            scene = orchestrator.run(videos, video_role=PathRole.DATA, camera_ids=metadata["camera_ids"],
                                     store_root=output / "store", clip_id=metadata["clip_id"])
            save_scene_result(scene, output / "scene.npz")
        except ReconstructionUnavailable as exc:
            failure = dict(reason=exc.reason, message=str(exc), diagnostics=exc.diagnostics)
        torch.cuda.synchronize()
        record = dict(schema="scene_dpt_meiji_comparison.v1", variant=args.variant, clip_id=metadata["clip_id"],
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_marker_sha256=dual_sha256(marker), checkpoint_sha256=dual_sha256(runtime.ball_detection.checkpoint),
            status=scene.metadata["status"] if scene is not None else "unavailable", failure=failure,
            num_frames=metadata["num_frames"], fps=metadata["fps"],
            pipeline_seconds=time.perf_counter() - start, pipeline_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            pipeline_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            comparison_scope="People, player reconstruction and GVHMR disabled equally; canonical single-clip orchestrator. No GT; reprojection measures geometric consistency only.",
            memory_fraction=.70, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
            receipt=json_value(orchestrator.last_receipt))
        runner = orchestrator.last_runner
        if runner is None:
            raise ValueError("Pipeline did not create a runner")
        detection = [runner.output(f"ball_detection/{camera}") for camera in metadata["camera_ids"]]
        points = [runner.output(f"ball_points/{camera}") for camera in metadata["camera_ids"]]
        record["ball_stage_seconds"] = {camera: runner.seconds[f"ball_detection/{camera}"] for camera in metadata["camera_ids"]}
        record["ball_detection_frames"] = {camera: int(out.observed.sum()) for camera, out in zip(metadata["camera_ids"], detection)}
        record["ball_points_frames"] = {camera: int(out.observed.sum()) for camera, out in zip(metadata["camera_ids"], points)}
        record["any_camera_2d_frames"] = int(np.any(np.stack([out.observed for out in detection]), axis=0).sum())
        data = dict(uv_px=np.stack([out.uv_px for out in detection]), observed=np.stack([out.observed for out in detection]),
                    confidence=np.stack([out.confidence for out in detection]))
        if scene is not None:
            data.update(ball_3d=scene.ball_3d, ball_3d_valid=scene.ball_3d_valid)
            ball = runner.output("ball_triangulation").ball
            errors = ball.trajectory.reprojection_px[ball.trajectory.inliers & ball.trajectory.valid[None]]
            record.update(ball_3d_valid_frames=int(scene.ball_3d_valid.sum()),
                          reprojection_inlier_count=len(errors),
                          reprojection_mean_px=float(errors.mean()) if len(errors) else None,
                          reprojection_median_px=float(np.median(errors)) if len(errors) else None,
                          reprojection_p95_px=float(np.quantile(errors, .95)) if len(errors) else None,
                          court_reference=scene.metadata.get("court_reference"))
        else:
            record.update(ball_3d_valid_frames=None, reprojection_mean_px=None)
        np.savez_compressed(output / "comparison-arrays.npz", **data)
        (output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k not in {"receipt", "failure"}}), flush=True)
        # A separate same-component probe gives ball-only peak VRAM, while the
        # standard run above keeps unmodified definitions/cache identities.
        first = videos[0]
        source = SourceVideo("cam0", first, dual_sha256(first), metadata["num_frames"], metadata["fps"], metadata["width"], metadata["height"])
        module = BallDetectionModule(runtime.ball_detection)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        probe = module.process(BallDetectionInput(source))
        torch.cuda.synchronize()
        if not np.array_equal(probe.observed, detection[0].observed) or not np.array_equal(probe.uv_px, detection[0].uv_px):
            raise ValueError("Repeated cam0 component output changed")
        record["ball_cam0_probe"] = dict(seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            includes="model load, decode/resize, inference, candidates, gates and unload; torch.compile cache already populated")
        (output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
