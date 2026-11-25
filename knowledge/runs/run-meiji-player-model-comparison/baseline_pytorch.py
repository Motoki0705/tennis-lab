"""Score DINO and DINO plus ViTPose-H on the fixed Meiji PT16 condition."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from baseline import write_baseline_outputs
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray

from src.submodules.models import (
    DinoPersonDetector,
    PersonDetectionRequest,
    ViTPosePose2D,
)
from src.tasks.person_tracking.features import DEFAULT_CONFIG, infer_poses
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, Predictions
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.utils.checksum import dual_sha256
from src.utils.video.reader import RandomAccessVideoReader

DEVICE = "cuda:0"


def infer_frame(
    frame: int,
    image: NDArray[np.uint8],
    detector: DinoPersonDetector,
    pose: ViTPosePose2D,
) -> Predictions:
    detection = detector.predict(PersonDetectionRequest(image))
    joints = infer_poses(frame, image, detection.boxes_xyxy, pose)
    return Predictions(
        detection.boxes_xyxy, detection.scores, np.where(joints[..., 2:] > 0, joints, 0)
    )


def pin_configuration(
    args: argparse.Namespace, source_sha: str
) -> tuple[PeopleModelConfig, dict[str, Any]]:
    receipt_path = pinned_file(
        args.speed_root,
        {
            "path": f"results/dino-vitpose/{args.backend.upper()}-b1/report.json",
            "sha256": args.speed_receipt_sha256,
        },
    )
    receipt = json.loads(receipt_path.read_bytes())
    fixed = json.loads(
        pinned_file(
            args.speed_root,
            {"path": "inputs/manifest.json", "sha256": args.speed_manifest_sha256},
        ).read_bytes()
    )
    case = receipt["case"]
    if (
        (case["name"], case["backend"], case["batch_size"], case["item_unit"])
        != ("dino-vitpose", args.backend.upper(), 1, "image")
        or case["input_manifest_sha256"] != args.speed_manifest_sha256
        or case["settings"]["dataset_id"] != fixed["id"]
        or fixed["source_benchmark_sha256"] != source_sha
        or receipt["status"] not in {"ok", "parity_failed"}
    ):
        raise ValueError(
            "The speed receipt must describe the same batch-one workload and media"
        )
    with initialize_config_dir(
        config_dir=str(Path("src/tennis_scene/configs").resolve()), version_base="1.3"
    ):
        config = PipelineRuntimeConfig.from_config(
            compose(
                config_name="pipeline",
                overrides=[
                    f"paths.external_asset_root={args.external_assets.resolve()}",
                    f"device={DEVICE}",
                ],
            ),
            bind_inputs=False,
        ).people
    dino = config.runtime.dino_detector
    speed_settings = case["settings"]
    if (
        (dino.confidence, dino.short_side, dino.max_long_side)
        != tuple(
            fixed["detector"][key]
            for key in ("confidence", "short_side", "max_long_side")
        )
        or speed_settings["dino_confidence"] != dino.confidence
        or speed_settings["flip_test"] != config.runtime.vitpose.flip_test
        or set(receipt["environment"]["fp32_precision"].values()) != {"ieee"}
    ):
        raise ValueError(
            "The PyTorch configuration differs from the published speed protocol"
        )
    upstream = subprocess.run(
        ["git", "-C", str(config.dino_repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if (
        upstream != fixed["detector"]["external_source_sha"]
        or subprocess.run(
            ["git", "-C", str(config.dino_repository), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    ):
        raise ValueError("The DINO checkout differs from the pinned speed source")
    assets = {
        "dino": config.dino_checkpoint,
        "vitpose": config.vitpose_checkpoint,
        "dino_msda_cuda": Path(os.environ["TENNIS_LAB_DINO_EXTENSION"]),
    }
    for name, path in assets.items():
        pin = receipt["weights"][name]
        if path.stat().st_size != pin["bytes"] or dual_sha256(path) != pin["sha256"]:
            raise ValueError(f"{name} differs from the pinned speed assets: {path}")
    return config, {
        "upstream_dino_commit": upstream,
        "speed_case": case,
        "speed_receipt_sha256": args.speed_receipt_sha256,
        "speed_status": receipt["status"],
        "assets_sha256": {name: receipt["weights"][name]["sha256"] for name in assets},
        "speed_environment": receipt["environment"],
        "speed_input_opencv": fixed["opencv"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "source", "output", "speed-root", "external-assets"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in (
        "manifest-sha256",
        "speed-manifest-sha256",
        "speed-receipt-sha256",
        "scorer-commit",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--backend", required=True, choices=("pt16",))
    parser.add_argument("--check-assets", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.benchmark, expected_sha256=args.manifest_sha256)
    source_sha = manifest["source_benchmark"]["manifest_sha256"]
    pinned_file(args.source, {"path": "manifest.json", "sha256": source_sha})
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            args.scorer_commit,
            "--",
            "src/tennis_scene/evaluation/player_pose",
        ],
        check=True,
    )
    config, provenance = pin_configuration(args, source_sha)
    if args.check_assets:
        print(json.dumps(provenance, sort_keys=True))
        return

    gpu, torch_version = torch.cuda.get_device_name(DEVICE), str(torch.__version__)
    environment = provenance["speed_environment"]
    if (gpu, torch_version, torch.version.cuda, cv2.__version__) != (
        environment["gpu"],
        environment["torch"],
        environment["cuda"],
        provenance["speed_input_opencv"],
    ):
        raise ValueError(
            "The scoring device or runtime differs from the pinned speed run"
        )
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if repro := os.environ.get("TENNIS_REPRO_DIR"):
        (Path(repro) / "output_dir.txt").write_text(str(out.resolve()) + "\n")
    references = load_reference_frames(
        args.benchmark, expected_sha256=args.manifest_sha256
    )
    det, pose = config.runtime.dino_detector, config.runtime.vitpose
    detector = DinoPersonDetector(
        config.dino_checkpoint,
        config.dino_repository,
        device=DEVICE,
        precision="fp16",
        confidence=det.confidence,
        short_side=det.short_side,
        max_long_side=det.max_long_side,
    )
    estimator = ViTPosePose2D(
        config.vitpose_checkpoint,
        device=DEVICE,
        precision="fp16",
        flip_test=pose.flip_test,
        batch_size=provenance["speed_case"]["settings"]["pose_crop_batch"],
        head_config=pose.head,
        show_progress=False,
    )
    paired = {}
    with torch.cuda.stream(torch.cuda.Stream(device=DEVICE)):
        for clip in manifest["clips"]:
            clip_id = clip["clip_id"]
            for media in clip["source_videos"]:
                camera = media["camera_id"]
                with RandomAccessVideoReader(pinned_file(args.source, media)) as reader:
                    for frame in clip["evaluation_frames"]:
                        key = FrameKey(clip_id, camera, frame)
                        image = reader.read(frame)
                        if image.shape != (clip["height"], clip["width"], 3):
                            raise ValueError(f"Unexpected native image shape: {key}")
                        paired[key] = infer_frame(frame, image, detector, estimator)
                print(f"Completed {args.backend} {clip_id}/{camera}", flush=True)
    provenance.update(
        source_manifest_sha256=source_sha,
        bbox_enlarge=DEFAULT_CONFIG.bbox_enlarge,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    write_baseline_outputs(
        out,
        manifest["benchmark_id"],
        args.manifest_sha256,
        args.backend,
        references,
        paired,
        provenance,
        scorer_commit=args.scorer_commit,
    )


if __name__ == "__main__":
    main()
