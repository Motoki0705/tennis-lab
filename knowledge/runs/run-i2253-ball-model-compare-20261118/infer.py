"""Run just the detector in a fresh process, using its pinned model source."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def save_npz(path, arrays):
    import numpy as np

    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def scene_configs():
    from hydra import compose, initialize_config_dir

    from src.tennis_scene.configuration import PipelineRuntimeConfig

    prior = HERE.parent / "run-i1752-shared-fp16-20261118"
    overrides = read(prior / "settings.json")["overrides"]
    with initialize_config_dir(
        config_dir=str(REPO / "src/tennis_scene/configs"), version_base="1.3"
    ):
        return {
            profile: PipelineRuntimeConfig.from_config(
                compose(config_name=profile, overrides=overrides), bind_inputs=False
            )
            for profile in ("pipeline_x3d_dpt", "pipeline")
        }


def scene_detector(model, *, preflight):
    from dataclasses import replace

    from src.tennis_scene.pipeline.components.ball_detection import BallDetectionModule
    from src.tennis_scene.pipeline.contracts import SourceVideo
    from src.utils.geometry.keypoints import denormalize_grid_keypoints

    cfg = scene_configs()[model["profile"]].ball_detection
    assert cfg is not None
    if (
        cfg.num_frames,
        list(cfg.image_size),
        cfg.precision,
        cfg.compile_mode,
        cfg.score_threshold,
    ) != (
        model["frames"],
        model["image_size_hw"],
        model["precision"],
        model["compile_mode"],
        model["threshold"],
    ):
        raise ValueError("Model selection diverges from the standard profile")
    if cfg.checkpoint.resolve() != Path(model["checkpoint"]):
        raise ValueError("Standard profile checkpoint differs from manifest")
    if preflight:
        cfg = replace(cfg, device="cpu", precision="fp32", compile_mode="off")
    detector = BallDetectionModule(cfg)
    predictor = detector._load_predictor()

    def detect(record, heartbeat):
        video = SourceVideo(**dict(record, path=Path(record["path"])))
        heartbeat(video.camera_id)
        coords, score, _ = detector._predict_video(video, predictor)
        return denormalize_grid_keypoints(coords, video.width, video.height), score

    return detect


def temporal_detector(model, *, preflight):
    import cv2
    import numpy as np
    import torch

    from src.tasks.ball_detection.data.windows.all_frames import owner_starts
    from src.tasks.ball_detection.model_io.mdd_temporal import (
        restore_temporal_model,
        temporal_forward,
    )
    from src.tasks.ball_detection.training.runtime.compilation import (
        compile_coordinate_model,
        coordinate_compile_scope,
    )
    from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic
    from src.utils.geometry.keypoints import denormalize_grid_keypoints
    from src.utils.video import (
        FramePacket,
        OpenCVVideoFrameReader,
        iter_temporal_windows,
    )

    training = read(Path(model["training_record"]))
    recipe = training["recipe"]
    saved = torch.load(model["checkpoint"], map_location="cpu", weights_only=True)
    if (
        training["code"]["base_commit"] != model["model_source_commit"]
        or json.dumps(saved["model_config"], sort_keys=True)
        != json.dumps(recipe["model"], sort_keys=True)
        or saved["input_contract"] != recipe["input_contract"]
        or saved["model_config"]["frames"] != model["frames"]
        or recipe["runtime"]["precision"] != model["precision"]
        or recipe["runtime"]["compile_mode"] != model["compile_mode"]
        or torch.__version__ != training["code"]["torch_version"]
    ):
        raise ValueError("Saved temporal checkpoint/recipe/runtime contract changed")
    network = restore_temporal_model(saved).eval()
    if preflight:
        return None
    device = torch.device("cuda")
    network.to(device)
    compile_coordinate_model(network, mode=model["compile_mode"])
    height, width = model["image_size_hw"]
    frames = model["frames"]
    heatmap = model["threshold"] is not None

    def decode_video(video):
        reader = OpenCVVideoFrameReader(
            Path(video["path"]), max_frames=video["num_frames"]
        )
        for packet in reader:
            if packet.original_size != (video["width"], video["height"]):
                raise ValueError("Decoded camera dimensions changed")
            resized = cv2.resize(
                packet.frame, (width, height), interpolation=cv2.INTER_LINEAR
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            yield FramePacket(
                packet.index,
                torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
                packet.original_size,
            )

    def detect(video, heartbeat):
        count = video["num_frames"]
        owners = owner_starts(0, count, frames)
        coords = np.full((count, 2), np.nan, dtype=np.float32)
        scores = np.full(count, np.nan, dtype=np.float32)
        with torch.inference_mode(), coordinate_compile_scope(network):
            for window in iter_temporal_windows(
                decode_video(video),
                sequence_length=frames,
                stride=frames // 2,
                tail_policy="backfill",
            ):
                heartbeat(f"{video['camera_id']} window {window.start_index}")
                idx = np.asarray(window.frame_indices)
                selected = owners[idx] == window.start_index
                rgb = torch.stack(window.frames).unsqueeze(0).to(device)
                times = torch.tensor(
                    idx / video["fps"], dtype=torch.float32, device=device
                )[None]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = temporal_forward(network, rgb, times)
                if heatmap:
                    shape = (1, frames, (height + 3) // 4, (width + 3) // 4)
                    if tuple(prediction.shape) != shape:
                        raise ValueError(
                            f"Unexpected DPT logits shape: {prediction.shape}"
                        )
                    probability = prediction.float().sigmoid()
                    argmax, score = heatmaps_to_argmax(probability)
                    prediction = refine_peaks_log_parabolic(probability, argmax)
                    scores[idx[selected]] = score[0].cpu().numpy()[selected]
                if tuple(prediction.shape) != (1, frames, 2):
                    raise ValueError(f"Unexpected coordinate shape: {prediction.shape}")
                coords[idx[selected]] = prediction[0].float().cpu().numpy()[selected]
        if not np.isfinite(coords).all() or (heatmap and not np.isfinite(scores).all()):
            raise ValueError("Incomplete temporal detector timeline")
        return denormalize_grid_keypoints(
            coords, video["width"], video["height"]
        ), scores

    return detect


def execute(root, name, *, preflight):
    work = root / "work" / name
    phase = work / "gpu"
    if not preflight:
        phase.mkdir(parents=True, exist_ok=True)

    def heartbeat(operation):
        if not preflight:
            write(
                phase / "heartbeat.json",
                {"monotonic": time.monotonic(), "operation": operation},
            )

    heartbeat("imports")
    import numpy as np
    import torch

    torch.manual_seed(42)
    plan = read(root / "plan.json")
    model = next(item for item in plan["models"] if item["name"] == name)
    heartbeat("load_weights")
    loader = scene_detector if "profile" in model else temporal_detector
    detect = loader(model, preflight=preflight)
    if preflight:
        print(f"{name}: strict CPU weight load, torch={torch.__version__}", flush=True)
        return
    assert detect is not None
    started = time.monotonic()
    for clip in plan["clips"]:
        rows = []
        for video in clip["inference_videos"]:
            heartbeat(f"{clip['clip_id']} {video['camera_id']}")
            rows.append(detect(video, heartbeat))
            print(
                f"{name} {clip['clip_id']} {video['camera_id']} {time.monotonic() - started:.1f}s",
                flush=True,
            )
        uv, score = zip(*rows, strict=True)
        save_npz(
            work / f"{clip['clip_id']}.npz",
            {"raw_uv_px": np.stack(uv), "raw_score": np.stack(score)},
        )
    torch.cuda.synchronize()
    write(
        work / "inference.json",
        {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "seconds": time.monotonic() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("model")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        execute(args.root, args.model, preflight=True)
        return
    if "TENNIS_RUN_ID" not in os.environ:
        raise RuntimeError("GPU inference must run through the shared training queue")
    with open(os.environ["TENNIS_STACK_FILE"], "a") as stack:
        faulthandler.enable(file=stack, all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=stack, all_threads=True)
        try:
            execute(args.root, args.model, preflight=False)
        except BaseException:
            faulthandler.dump_traceback(file=stack, all_threads=True)
            raise
        finally:
            faulthandler.unregister(signal.SIGUSR1)


if __name__ == "__main__":
    main()
