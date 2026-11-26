"""Time the current model paths on fixed images, crops and body windows."""

from __future__ import annotations

from contextlib import contextmanager
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.submodules.models import (
    DinoPersonDetector,
    GvhmrMeshRecovery,
    GvhmrRequest,
    Hmr2FeatureExtractor,
    ViTPosePose2D,
)
from src.submodules.models._base.crops import iter_frame_person_crops
from src.submodules.models.dino.architecture import (
    COCO_PERSON_CLASS_ID,
    preprocess_frame,
)
from src.submodules.models.dino.person_detector import PersonDetectionResult
from src.submodules.vendor.gvhmr.vitpose.flip_utils import flip_heatmap_coco17
from src.submodules.vendor.gvhmr.vitpose.kp2d_utils import keypoints_from_heatmaps
from src.tasks.person_tracking.features import FeatureConfig, pose_boxes
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.model_resources import (
    dino_tensorrt,
    gvhmr_tensorrt,
    hmr2_tensorrt,
    vitpose_tensorrt,
)
from src.utils.inference_benchmark import Backend, ConditionUnavailable, Stage, Status
from src.utils.models.tensorrt_runtime import TensorRTRunner, tensorrt_module

HMR2_CROP_BATCH = 16
DEVICE = "cuda:0"


def pipeline_config(external_assets: Path) -> PipelineRuntimeConfig:
    config = Path("src/tennis_scene/configs").resolve()
    with initialize_config_dir(config_dir=str(config), version_base="1.3"):
        raw = compose(
            config_name="pipeline",
            overrides=[
                f"paths.external_asset_root={external_assets}",
                f"device={DEVICE}",
            ],
        )
    return PipelineRuntimeConfig.from_config(raw, bind_inputs=False)


def dino_eager(config, backend: Backend):
    runtime = config.runtime.dino_detector
    return DinoPersonDetector(
        config.dino_checkpoint,
        config.dino_repository,
        device=DEVICE,
        confidence=runtime.confidence,
        short_side=runtime.short_side,
        max_long_side=runtime.max_long_side,
        precision="fp16" if backend is Backend.PT16 else "fp32",
    )


def engine_paths(root: Path) -> dict[str, Path]:
    shapes = {
        "dino": "750x1333",
        "vitpose": "256x192",
        "hmr2": "256x256",
        "gvhmr": "l2-177-1055",
    }
    return {
        name: root / name / shape / "i2363-qualification-v1"
        for name, shape in shapes.items()
    }


def quiet_nvtx(runner: TensorRTRunner) -> None:
    verbosity = tensorrt_module().ProfilingVerbosity.NONE
    runner._context.nvtx_verbosity = verbosity
    if runner._context.nvtx_verbosity != verbosity:
        raise RuntimeError("TensorRT execution context failed to disable NVTX")


def bind(spec, directory: Path, *, size: int):
    try:
        return spec.bind(directory, device=DEVICE, batch_size=size)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        if isinstance(error, torch.OutOfMemoryError):
            raise
        raise ConditionUnavailable(
            Status.UNSUPPORTED, f"{directory}: {error}"
        ) from error


@contextmanager
def classify_startup(backend: Backend):
    try:
        yield
    except (ConditionUnavailable, torch.OutOfMemoryError):
        raise
    except (OSError, RuntimeError, ValueError) as error:
        if backend is not Backend.TRT16:
            raise
        raise ConditionUnavailable(
            Status.UNAVAILABLE, f"TensorRT startup failed: {error}"
        ) from error


def timed_engine(runner, inputs):
    try:
        return runner(inputs)
    except (ValueError, RuntimeError) as error:
        if isinstance(error, torch.OutOfMemoryError):
            raise
        raise ConditionUnavailable(
            Status.UNSUPPORTED, f"TensorRT failed: {error}"
        ) from error


def dino_decode_gpu(output, sizes, confidence):
    """Apply the DINO decoder's score filter and scaling on the input device."""
    logits, boxes = output["pred_logits"].float(), output["pred_boxes"].float()
    if (
        logits.ndim != 3
        or boxes.shape != (*logits.shape[:2], 4)
        or len(logits) != len(sizes)
    ):
        raise ValueError(f"Invalid DINO output shapes: {logits.shape}, {boxes.shape}")
    decoded = []
    for scores, boxes_cxcywh, (height, width) in zip(
        logits[..., COCO_PERSON_CLASS_ID].sigmoid(), boxes, sizes, strict=True
    ):
        keep = scores >= confidence
        filtered = scores[keep]
        order = filtered.argsort(descending=True)
        center, radius = boxes_cxcywh[keep][order].split(2, dim=-1)
        xyxy = torch.cat((center - radius / 2, center + radius / 2), dim=-1).clamp_(
            0.0, 1.0
        )
        decoded.append(
            (xyxy * xyxy.new_tensor([width, height, width, height]), filtered[order])
        )
    return decoded


class Dino:
    def __init__(self, config, backend: Backend, directory: Path, batch: int) -> None:
        self.runtime = config.runtime.dino_detector
        self.backend = backend
        if backend is Backend.TRT16:
            bundle = bind(dino_tensorrt(config, 750, 1333), directory, size=batch)
            runner = TensorRTRunner(bundle, torch.device(DEVICE))
            quiet_nvtx(runner)
            self.predict = lambda pixels: timed_engine(runner, {"pixels": pixels})
        else:
            eager = dino_eager(config, backend)
            eager.load()
            self.predict = eager._model
            if self.predict is None:
                raise RuntimeError("DINO weights were not loaded")

    def __call__(self, images, stages):
        settings = self.runtime
        pixels = stages.stage(
            Stage.PREPROCESS,
            lambda: torch.stack(
                [
                    preprocess_frame(
                        image,
                        short_side=settings.short_side,
                        max_long_side=settings.max_long_side,
                    )
                    for image in images
                ]
            ),
            name="dino",
        )
        pixels = stages.stage(Stage.H2D, lambda: pixels.to(DEVICE), name="dino")

        def infer():
            with autocast(self.backend):
                return self.predict(pixels)

        raw = stages.measure_forward(self.backend, infer, name="dino")
        sizes = [image.shape[:2] for image in images]
        chosen = stages.stage(
            Stage.POSTPROCESS,
            lambda: dino_decode_gpu(raw, sizes, settings.confidence),
            name="dino",
        )
        return stages.stage(
            Stage.D2H,
            lambda: [
                PersonDetectionResult(boxes.cpu().numpy(), scores.cpu().numpy())
                for boxes, scores in chosen
            ],
            name="dino",
        )


def autocast(backend: Backend):
    return torch.autocast("cuda", dtype=torch.float16, enabled=backend is Backend.PT16)


def crop_groups(groups):
    boxes = torch.cat([boxes for _, boxes in groups])
    indices = torch.repeat_interleave(
        torch.arange(len(groups)), torch.tensor([len(boxes) for _, boxes in groups])
    )
    return next(
        iter_frame_person_crops(
            [(i, image) for i, (image, _) in enumerate(groups)],
            boxes,
            indices,
            batch_size=len(boxes),
        )
    )


def pose_groups(rows):
    return [
        (chunk[0][1], torch.tensor([row[2] for row in chunk], dtype=torch.float32))
        for _, group in groupby(rows, key=lambda row: row[0])
        if (chunk := list(group))
    ]


def decode_pose(heatmaps, boxes):
    scale = (torch.cat((boxes[:, [2]] * 24 / 32, boxes[:, [2]]), dim=1) / 200).numpy()
    xy, confidence = keypoints_from_heatmaps(
        heatmaps=heatmaps, center=boxes[:, :2].numpy(), scale=scale, use_udp=True
    )
    return np.concatenate((xy, confidence), axis=-1)


class Pose:
    def __init__(
        self, config, backend: Backend, directory: Path, *, flip: bool, batch: int
    ) -> None:
        self.backend, self.flip = backend, flip
        bundle = None
        if backend is Backend.TRT16:
            bundle = bind(
                vitpose_tensorrt(config),
                directory,
                size=batch if flip else max(1, batch // 2),
            )
            bundle.engine.profiles["pixels"].require(
                (batch * (2 if flip else 1), 3, 256, 192), tensor="pixels"
            )
        self.model = ViTPosePose2D(
            config.vitpose_checkpoint,
            device=DEVICE,
            batch_size=batch,
            flip_test=flip,
            precision="fp16" if backend is Backend.PT16 else "fp32",
            head_config=config.runtime.vitpose.head,
            engine=bundle,
            show_progress=False,
        )
        self.model.load()
        if self.model._runner is not None:
            quiet_nvtx(self.model._runner)

    def __call__(self, groups, stages):
        images, boxes = stages.stage(
            Stage.PREPROCESS, lambda: crop_groups(groups), name="vitpose"
        )
        dtype = torch.float16 if self.backend is Backend.TRT16 else torch.float32
        pixels = stages.stage(
            Stage.H2D,
            lambda: images[..., 32:224].to(device=DEVICE, dtype=dtype),
            name="vitpose",
        )

        def infer():
            with autocast(self.backend):
                if not self.flip:
                    return self.heatmaps(pixels.clone())
                raw, flipped = self.heatmaps(torch.cat((pixels, pixels.flip(3)))).chunk(
                    2
                )
                return (raw + flip_heatmap_coco17(flipped)) * 0.5

        heatmaps = stages.measure_forward(self.backend, infer, name="vitpose")
        values = stages.stage(
            Stage.D2H, lambda: heatmaps.float().cpu().numpy(), name="vitpose"
        )
        return stages.stage(
            Stage.POSTPROCESS, lambda: decode_pose(values, boxes), name="vitpose"
        )

    def heatmaps(self, pixels):
        if self.model._runner is not None:
            return timed_engine(self.model._runner, {"pixels": pixels})["heatmaps"]
        return self.model._forward_heatmaps(pixels)


def two_stage(dino, pose, images, stages, *, crop_batch: int, features: FeatureConfig):
    """Run a ViTPose crop for every DINO row and restore the per-image row order."""
    detections = dino(images, stages)
    crops = stages.stage(
        Stage.PREPROCESS,
        lambda: [
            (i, image, box.tolist())
            for i, (image, detection) in enumerate(zip(images, detections, strict=True))
            for box in pose_boxes(detection.boxes_xyxy, features)
        ],
        name="dino_to_pose",
    )
    poses = []
    for start in range(0, len(crops), crop_batch):
        poses.extend(pose(pose_groups(crops[start : start + crop_batch]), stages))
    if len(poses) != len(crops):
        raise ValueError("ViTPose changed the DINO crop count")
    by_image = []
    offset = 0
    for detection in detections:
        count = len(detection.boxes_xyxy)
        joints = np.asarray(poses[offset : offset + count], dtype=np.float32)
        by_image.append((detection, joints.reshape(count, 17, 3)))
        offset += count
    return by_image


class Body:
    def __init__(self, config, backend: Backend, directories, batch: int) -> None:
        self.backend = backend
        feature_bundle = recovery_bundle = None
        if backend is Backend.TRT16:
            feature_bundle = bind(
                hmr2_tensorrt(config), directories["hmr2"], size=HMR2_CROP_BATCH
            )
            recovery_bundle = bind(
                gvhmr_tensorrt(config), directories["gvhmr"], size=batch
            )
        self.hmr = Hmr2FeatureExtractor(
            config.hmr2_checkpoint,
            device=DEVICE,
            batch_size=HMR2_CROP_BATCH,
            mean_params_path=config.smpl_assets.hmr2_mean_params,
            engine=feature_bundle,
        )
        self.gvhmr = GvhmrMeshRecovery(
            config.gvhmr_checkpoint,
            config.body_models_dir,
            device=DEVICE,
            smpl_assets=config.smpl_assets,
            engine=recovery_bundle,
        )
        self.hmr.load()
        self.gvhmr.load()
        if self.hmr._runner is not None:
            quiet_nvtx(self.hmr._runner)
            quiet_nvtx(self.gvhmr._model.pipeline.denoiser3d.runner)

    def feature(self, images):
        if self.hmr._runner is not None:
            return timed_engine(self.hmr._runner, {"pixels": images})["features"]
        with autocast(self.backend):
            return self.hmr._model({"img": images})

    def __call__(self, rows, stages):
        raw = stages.stage(
            Stage.PREPROCESS,
            lambda: [
                (image, torch.from_numpy(square[None, :].copy()))
                for row in rows
                for image, square in zip(row["images"], row["boxes"], strict=True)
            ],
            name="hmr2_crop_rows",
        )
        all_features = []
        dtype = torch.float16 if self.backend is Backend.TRT16 else torch.float32
        for index in range(0, len(raw), HMR2_CROP_BATCH):
            groups = raw[index : index + HMR2_CROP_BATCH]
            pixels, _ = stages.stage(
                Stage.PREPROCESS, lambda groups=groups: crop_groups(groups), name="hmr2"
            )
            pixels = stages.stage(
                Stage.H2D,
                lambda pixels=pixels: pixels.to(device=DEVICE, dtype=dtype),
                name="hmr2",
            )
            features = stages.measure_forward(
                self.backend, lambda pixels=pixels: self.feature(pixels), name="hmr2"
            )
            all_features.append(
                stages.stage(
                    Stage.D2H,
                    lambda features=features: features.float().cpu(),
                    name="hmr2",
                )
            )
        model = self.gvhmr._model
        feature_rows = torch.cat(all_features).split(
            [len(row["boxes"]) for row in rows]
        )

        def prepare():
            prepared = [
                model._batch_inputs(
                    self.gvhmr._prepare(
                        GvhmrRequest(
                            kp2d=torch.from_numpy(row["keypoints"]),
                            bbx_xys=torch.from_numpy(row["boxes"]),
                            f_imgseq=features,
                            width=row["size"][0],
                            height=row["size"][1],
                            static_cam=True,
                            K_fullimg=torch.from_numpy(row["intrinsic"]).float(),
                        )
                    )
                )
                for row, features in zip(rows, feature_rows, strict=True)
            ]
            return {
                key: torch.cat([item[key] for item in prepared]) for key in prepared[0]
            }

        inputs = stages.stage(Stage.PREPROCESS, prepare, name="gvhmr")
        inputs = stages.stage(
            Stage.H2D,
            lambda: {key: val.to(DEVICE) for key, val in inputs.items()},
            name="gvhmr",
        )

        def predict_denoising():
            with autocast(self.backend):
                return model._predict_denoiser(inputs)

        predicted = stages.measure_forward(
            self.backend, predict_denoising, name="gvhmr_denoiser"
        )

        def recover():
            return [
                model.pipeline.decode_prediction(
                    {key: val[i : i + 1] for key, val in inputs.items()},
                    {key: val[i : i + 1].float() for key, val in predicted.items()},
                    static_cam=True,
                )["pred_smpl_params_incam"]
                for i in range(len(rows))
            ]

        params = stages.stage(
            Stage.POSTPROCESS, recover, name="gvhmr_decode_static_ik_fp32"
        )
        return stages.stage(
            Stage.D2H,
            lambda: [
                {
                    "params": {
                        key: val[0].float().cpu().numpy() for key, val in item.items()
                    },
                    "features": features.numpy(),
                }
                for item, features in zip(params, feature_rows, strict=True)
            ],
            name="gvhmr",
        )
