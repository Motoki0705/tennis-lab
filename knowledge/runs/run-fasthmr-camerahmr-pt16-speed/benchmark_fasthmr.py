"""Measure FastHMR PT16 on the published Meiji body-recovery speed schedule."""

import argparse
import subprocess
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
from speed_inputs import BODY_LENGTH, InputSet
from ultralytics.models.yolo.segment import SegmentationPredictor

from src.submodules.models.fasthmr.assets import (
    MERGE_RATIO,
    UPSTREAM_COMMIT,
    WINDOW_LENGTH,
    FastHMRConfig,
)
from src.submodules.models.fasthmr.inference import FastHMR
from src.submodules.models.fasthmr.upstream import OfficialSMPL
from src.tennis_scene.pipeline.components.fasthmr import native_smpl_parameters
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BenchmarkAdapter,
    BenchmarkCase,
    Stage,
    benchmark,
)
from src.utils.io import write_json_atomic

HARNESS = "13e508cb78c46663fbe58ad8be03bcf9c96816a6"
PARITY_REASON = "未判定（PT32計測はスコープ外、ユーザー判断 2026-11-25）"
FEATURE_BATCH = 4
DEVICE = "cuda:0"
STAGES = (
    "src/utils/inference_benchmark.py",
    "src/utils/inference_benchmark_inputs.py",
    "knowledge/runs/run-person-model-speed-baseline/timing-protocol.json",
    "knowledge/runs/run-person-model-speed-baseline/speed_inputs.py",
)


class MeasuredFastHMR(FastHMR):
    """Expose the production model forwards and keep IO inside the wall time."""

    def _images(self, images):
        return self.stages.stage(
            Stage.H2D, partial(super()._images, images), name="bgr_or_rgb_pixels"
        )


def load_fast_hmr(config):
    if "5060 Ti" not in torch.cuda.get_device_name(config.device):
        raise RuntimeError("The fixed comparison requires the local RTX 5060 Ti")
    model = MeasuredFastHMR(config)
    model.load()
    backbone, fov, decoder, segmentor, crop = model.models
    stages = None

    def measure(name, operation):
        if stages is None:
            raise RuntimeError("FastHMR timing requires an active harness batch")
        return stages.measure_forward(Backend.PT16, operation, name=name)

    def forward(name, module):
        def run(*args, **kwargs):
            return measure(name, partial(module, *args, **kwargs))

        return run

    class TimedSegmentationPredictor(SegmentationPredictor):
        def inference(self, im, *args, **kwargs):
            if not self.model.fp16:
                raise RuntimeError("FastHMR YOLO predictor did not select FP16")
            return measure(
                "yolo11x_seg", partial(super().inference, im, *args, **kwargs)
            )

    class Segmentor:
        def predict(self, *args, **kwargs):
            return segmentor.predict(
                *args, predictor=TimedSegmentationPredictor, **kwargs
            )

    class TimedSMPL(OfficialSMPL):
        def rotations(self, pose, shape):
            return model.stages.stage(
                Stage.POSTPROCESS,
                partial(super().rotations, pose, shape),
                name="hybrik_fp32",
            )

    body = TimedSMPL(config)
    body.load()
    model.models = (
        forward("camerahmr_eclm_mask_tome", backbone),
        forward("humanfov", fov),
        forward("fast_hmr_diffusion_vae", decoder),
        Segmentor(),
        crop,
    )

    def infer(rows, batch_stages):
        nonlocal stages
        stages = model.stages = batch_stages
        try:
            return [run_sequence(model, body, row) for row in rows]
        finally:
            stages = model.stages = None

    return infer


def run_sequence(model, body, row):
    # Each fixed row is one contiguous, predecoded, 16-sample segment.
    size = tuple(row["size"])
    images, boxes = row["images"], row["boxes"]
    features = [
        model._encode(
            images[start : start + FEATURE_BATCH],
            boxes[start : start + FEATURE_BATCH],
            size,
        )
        for start in range(0, len(images), FEATURE_BATCH)
    ]
    original, flipped, betas, camera = (
        torch.cat(parts) for parts in zip(*features, strict=True)
    )
    shape = betas.float().mean(dim=0, keepdim=True)
    device = torch.device(model.config.device)
    with torch.random.fork_rng(devices=[device.index]), torch.cuda.device(device):
        torch.cuda.manual_seed(model.config.seed)
        prediction = model._decode(original, flipped, shape, camera, boxes, size, body)
    return model.stages.stage(
        Stage.POSTPROCESS,
        partial(native_smpl_parameters, prediction, BODY_LENGTH),
        name="native_smpl_parameters",
    )


def verify_harness():
    subprocess.run(["git", "merge-base", "--is-ancestor", HARNESS, "HEAD"], check=True)
    subprocess.run(
        ["git", "diff", "--quiet", HARNESS, "HEAD", "--", *STAGES], check=True
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def verify_sequence(row):
    frames = np.asarray(row["source_frames"])
    if (
        frames.shape != (BODY_LENGTH,)
        or not np.all(np.diff(frames) == 2)
        or len(row["images"]) != BODY_LENGTH
        or row["boxes"].shape != (BODY_LENGTH, 3)
        or not np.isfinite(row["boxes"]).all()
        or (row["boxes"][:, 2] <= 0).any()
        or any(
            image.shape != (row["size"][1], row["size"][0], 3)
            for image in row["images"]
        )
    ):
        raise ValueError(f"Unexpected sequence input {row['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    source_commit = verify_harness()
    fixed = InputSet(args.input_dir)
    schedule = fixed.pins.schedule("body_sequences", args.batch)
    rows = {row["id"]: row for row in fixed.rows("body_sequences")}
    if args.check_inputs:
        verify_sequence(fixed.body(rows[schedule.batch_ids[0][0]]))
        print(
            f"verified {len(rows)} pinned sequence rows and B={args.batch} input order {schedule.timed_ids_sha256}"
        )
        return 0
    if args.output is None:
        parser.error("--output is required to measure")

    cv2.setNumThreads(1)
    data = {key: fixed.body(row) for key, row in rows.items()}
    for row in data.values():
        verify_sequence(row)
    batches = [
        Batch(
            ids,
            tuple(data[key] for key in ids),
            len(ids) * BODY_LENGTH,
            len(ids) * BODY_LENGTH,
        )
        for ids in schedule.batch_ids
    ]
    schedule.require_batches([batch.ids for batch in batches])
    cfg = FastHMRConfig(
        Path("third_party/FastHMR").resolve(),
        Path("ckpt/fasthmr").resolve(),
        Path("ckpt/body_models/gvhmr/smpl_coco17_J_regressor.pt").resolve(),
        DEVICE,
        "pt16",
        FEATURE_BATCH,
        0,
    )
    weights = {
        **cfg.inference_weights,
        **cfg.geometry_weights,
        "upstream_demo_config": cfg.demo_config,
    }
    case = BenchmarkCase(
        "fasthmr-camerahmr",
        Backend.PT16,
        args.batch,
        "person-frame",
        fixed.pins.manifest_sha256,
        {
            "source_commit": source_commit,
            "harness_commit": HARNESS,
            "upstream_commit": UPSTREAM_COMMIT,
            "input_id": fixed.manifest["id"],
            "timing_protocol_sha256": fixed.pins.timing_sha256,
            "timed_input_ids_sha256": schedule.timed_ids_sha256,
            "host_threads": torch.get_num_threads(),
            "cuda_stream": "non-default",
            "feature_batch_size": FEATURE_BATCH,
            "sequence_execution": "serial",
            "window_length": WINDOW_LENGTH,
            "body_window_frames": BODY_LENGTH,
            "padding": 0,
            "merging_ratio": MERGE_RATIO,
            "flip_test": True,
            "diffusion_seed": 0,
            "boundary": "predecoded BGR plus fixed body_view_selection boxes to native SMPL body parameters",
            "fp32_ops": ["crop_camera", "hybrik", "native_smpl_parameters"],
            "forward_gpu": [
                "yolo11x_seg",
                "humanfov",
                "camerahmr_eclm_mask_tome",
                "fast_hmr_diffusion_vae",
            ],
            "wall_only": [
                "remaining crop preprocessing",
                "crop camera translation",
                "output D2H",
            ],
            "use": "speed_reference_only",
        },
    )
    report = benchmark(
        case,
        batches,
        {
            Backend.PT16: BenchmarkAdapter(
                Backend.PT16, partial(load_fast_hmr, cfg), weights
            )
        },
        device=DEVICE,
        dedicated_stream=True,
        protocol=schedule.protocol,
        parity_not_evaluated_reason=PARITY_REASON,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, report.to_dict())
    print(
        f"{case.name}/PT16/B{args.batch}: {report.status}, {report.parity_status}, {len(report.samples)} repetitions",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
