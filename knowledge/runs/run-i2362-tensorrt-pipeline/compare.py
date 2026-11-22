"""Compare production TensorRT adapters on the locked Phase 1 observations."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.submodules.models import (
    DinoPersonDetector,
    GvhmrMeshRecovery,
    GvhmrRequest,
    Hmr2FeatureExtractor,
    SmplCoco17Reconstructor,
    ViTPosePose2D,
)
from src.tasks.player_association.appearance.encoders import (
    CLIP_REID_PROFILE,
    ClipReIDEncoder,
)
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.clip_inputs import select_dataset_clips
from src.tennis_scene.pipeline.model_resources import (
    clip_tensorrt,
    dino_tensorrt,
    gvhmr_tensorrt,
    hmr2_tensorrt,
    vitpose_tensorrt,
)
from src.utils.io import write_json_atomic
from src.utils.paths import PROJECT_ROOT


def runtime():
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        raw = compose(config_name="pipeline", overrides=["device=cuda"])
    return PipelineRuntimeConfig.from_config(raw, bind_inputs=False)


def clip(args, cfg):
    from benchmark import accuracy, verify_inputs

    _, addresses = verify_inputs(args.inputs)
    checkpoint = cfg.tracking_encoder_weights
    bundle = clip_tensorrt(checkpoint).bind(
        args.engine, device="cuda", batch_size=args.batch_size
    )
    # The locked graph input is [-1, 1]. The public encoder accepts RGB [0, 1].
    crops = (
        torch.from_numpy(np.load(args.inputs / "eval.npy", allow_pickle=False))
        .mul_(0.5)
        .add_(0.5)
    )

    def infer(engine, batch):
        encoder = ClipReIDEncoder(CLIP_REID_PROFILE, checkpoint, "cuda", engine=engine)
        return torch.cat([encoder.embed(chunk) for chunk in crops.split(batch)]).numpy()

    before, after = infer(None, 1), infer(bundle, args.batch_size)
    metrics = accuracy(after, before, addresses)
    return metrics, metrics["cosine_p05"] >= 0.995 and metrics["id_matches_fp32"] / len(
        addresses
    ) >= 0.99


def images(args, cfg):
    from fixed_crops import verify
    from vit_models import distribution, feature_accuracy

    people = cfg.people
    pose = args.model == "vitpose"
    verify("vitpose_h" if pose else "hmr2", args.inputs)
    crops = torch.from_numpy(np.load(args.inputs / "eval.npy", allow_pickle=False))
    boxes = torch.from_numpy(
        np.load(args.inputs / "eval-boxes.npy", allow_pickle=False)
    )
    spec = vitpose_tensorrt(people) if pose else hmr2_tensorrt(people)
    bundle = spec.bind(args.engine, device="cuda", batch_size=args.batch_size)
    if pose:
        options = people.runtime.vitpose
        # Restore the crop width expected by predict_crops; ViTPose reads columns 32:224.
        crops = torch.nn.functional.pad(crops, (32, 32))

    def infer(engine, batch):
        if pose:
            model = ViTPosePose2D(
                people.vitpose_checkpoint,
                device="cuda",
                flip_test=options.flip_test,
                batch_size=batch,
                head_config=options.head,
                show_progress=False,
                engine=engine,
            )
        else:
            model = Hmr2FeatureExtractor(
                people.hmr2_checkpoint,
                device="cuda",
                batch_size=batch,
                mean_params_path=people.smpl_assets.hmr2_mean_params,
                engine=engine,
            )
        try:
            if pose:
                return torch.cat(
                    [
                        model.predict_crops(x, b).keypoints
                        for x, b in zip(
                            crops.split(batch), boxes.split(batch), strict=True
                        )
                    ]
                ).numpy()
            return torch.cat(
                [model.predict_crops(x).features for x in crops.split(batch)]
            ).numpy()
        finally:
            model.unload()

    before, after = infer(None, 1), infer(bundle, args.batch_size)
    if not np.isfinite(after).all() or not np.isfinite(before).all():
        raise ValueError("Nonfinite image model output")
    if not pose:
        metrics, _ = feature_accuracy(after, before)
        return metrics, metrics["cosine"]["p05"] >= 0.995 and metrics["relative_l2"][
            "p95"
        ] <= 0.05
    before_mask, after_mask = before[..., 2] > 0, after[..., 2] > 0
    valid = before_mask & after_mask
    xy = distribution(np.linalg.norm((after - before)[..., :2], axis=-1)[valid])
    score = distribution(np.abs((after - before)[..., 2])[valid])
    metrics = {
        "crops": len(before),
        "common_valid_joints": int(valid.sum()),
        "xy_l2_px": xy,
        "confidence_abs": score,
        "mask_mismatches": int(np.count_nonzero(before_mask != after_mask)),
    }
    return metrics, bool(
        xy
        and score
        and xy["p95"] <= 1
        and xy["max"] <= 5
        and score["p95"] <= 0.02
        and metrics["mask_mismatches"] == 0
    )


def dino(args, cfg):
    from benchmark_dino import compare, verify_inputs
    from fixed_inputs import load_sample, rows_for

    people = cfg.people
    bucket = "750x1333"
    lock = verify_inputs(args.inputs, bucket, people.dino_checkpoint)
    rows = rows_for(lock, bucket, "eval")
    bundle = dino_tensorrt(people, 750, 1333).bind(
        args.engine, device="cuda", batch_size=args.batch_size
    )
    settings = people.runtime.dino_detector

    def infer(engines, batch):
        detector = DinoPersonDetector(
            people.dino_checkpoint,
            people.dino_repository,
            device="cuda",
            confidence=settings.confidence,
            short_side=settings.short_side,
            max_long_side=settings.max_long_side,
            engines=engines,
        )
        results = []
        try:
            detector.load()
            for start in range(0, len(rows), batch):
                chunk = rows[start : start + batch]
                pixels = [
                    torch.from_numpy(load_sample(args.inputs, row)[0]) for row in chunk
                ]
                image_sizes = [tuple(row["original_hw"][::-1]) for row in chunk]
                outputs = detector._predict_compatible_batch(pixels, image_sizes)
                results.extend(
                    {
                        "file": row["file"],
                        "boxes": result.boxes_xyxy.tolist(),
                        "scores": result.scores.tolist(),
                    }
                    for row, result in zip(chunk, outputs, strict=True)
                )
        finally:
            detector.unload()
        return results

    metrics = compare(infer((), 1), infer((bundle,), args.batch_size), rows)
    overall = metrics["overall"]
    passed = (
        overall["iou"]["p05"] >= 0.98
        and overall["score_abs_error"]["p95"] <= 0.01
        and overall["missing"] / overall["reference_boxes"] <= 0.01
        and overall["extra"] / overall["candidate_boxes"] <= 0.01
    )
    return metrics, passed


def gvhmr(args, cfg):
    from accuracy_gvhmr import SPACES, compare
    from locked_inputs import checked_segments, sha256_file

    people = cfg.people
    segments, checkpoint = checked_segments(args.inputs, cfg.roots.checkpoint_root)
    lock = json.loads((args.inputs / "tensor-lock.json").read_text())
    for asset in lock["assets"].values():
        if sha256_file(cfg.roots.checkpoint_root / asset["path"]) != asset["sha256"]:
            raise ValueError(f"GVHMR asset digest mismatch: {asset['path']}")
    bundle = gvhmr_tensorrt(people).bind(
        args.engine, device="cuda", batch_size=args.batch_size
    )
    eval_segments = [
        (row, arrays) for row, arrays in segments if row["split"] == "eval"
    ]
    dataset = cfg.roots.data_root / "tennis_multiview/processed/meiji_4cam/dataset"
    records = {
        record.clip_id: record
        for record in select_dataset_clips(
            dataset.resolve(), sorted({row["clip"] for row, _ in eval_segments})
        )
    }
    reconstructor = SmplCoco17Reconstructor(
        people.body_models_dir, device="cuda", smpl_assets=people.smpl_assets
    )

    def infer(engine, selected):
        model = GvhmrMeshRecovery(
            checkpoint,
            people.body_models_dir,
            device="cuda",
            smpl_assets=people.smpl_assets,
            engine=engine,
        )
        values = defaultdict(list)
        try:
            for row, arrays in selected:
                intrinsics = arrays["K_fullimg"]
                if not np.array_equal(
                    intrinsics, np.broadcast_to(intrinsics[0], intrinsics.shape)
                ):
                    raise ValueError(
                        "The locked static-camera intrinsics vary by frame"
                    )
                record = records[row["clip"]]
                request = GvhmrRequest(
                    kp2d=torch.from_numpy(arrays["kp2d"]),
                    bbx_xys=torch.from_numpy(arrays["bbx_xys"]),
                    f_imgseq=torch.from_numpy(arrays["f_imgseq_raw"]),
                    width=record.width,
                    height=record.height,
                    static_cam=bool(arrays["static_cam"]),
                    K_fullimg=torch.from_numpy(intrinsics[0]),
                )
                result = model.predict(request)
                for space in SPACES:
                    params = getattr(result, f"smpl_params_{space}")
                    for name, value in params.items():
                        values[f"{space}_{name}"].append(value.numpy())
                    values[f"{space}_joints"].append(
                        reconstructor.reconstruct(params).numpy()
                    )
                # The TensorRT denoiser adapter checks both outputs against the complete input.
                values["mask_out"].append(arrays["sequence_mask"][0])
                values["length_out"].append(arrays["length_raw"])
        finally:
            model.unload()
        return {name: np.stack(value) for name, value in values.items()}

    metrics, passed = {}, True
    for length in sorted({row["length"] for row, _ in eval_segments}):
        selected = [
            (row, arrays) for row, arrays in eval_segments if row["length"] == length
        ]
        before, after = infer(None, selected), infer(bundle, selected)
        measured = compare(after, before, [row for row, _ in selected])
        metrics[str(length)] = measured
        passed &= (
            measured["mask_matches_fp32"]
            and measured["length_matches_fp32"]
            and all(
                measured[space][field]["p95"] <= limit
                for space in SPACES
                for field, limit in (
                    ("body_pose_deg", 1),
                    ("global_orient_deg", 1),
                    ("coco17_l2_mm", 10),
                )
            )
        )
    reconstructor.unload()
    return metrics, passed


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["clip", "vitpose", "hmr2", "dino", "gvhmr"])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError(
            "Run the TensorRT comparison through the shared training queue"
        )
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    function = {
        "clip": clip,
        "vitpose": images,
        "hmr2": images,
        "dino": dino,
        "gvhmr": gvhmr,
    }[args.model]
    metrics, passed = function(args, runtime())
    write_json_atomic(
        args.output,
        {
            "model": args.model,
            "batch_size": args.batch_size,
            "queue_job": os.environ["TENNIS_RUN_ID"],
            "passed": passed,
            "metrics": metrics,
        },
    )
    if not passed:
        raise SystemExit(f"{args.model} did not meet the Phase 1 comparison thresholds")


if __name__ == "__main__":
    main()
