"""Try to capture the released CameraHMR Mask-ToMe stage for a TRT16 export."""

import json
import os
from contextlib import closing
from pathlib import Path
from typing import Any

import tensorrt
import torch
from quality import Clip, cohort_records, config, cuda, emit
from torch import nn

from src.submodules.models.fasthmr.inference import FastHMR
from src.utils.video import OpenCVVideoFrameReader


class CameraBackboneBindings(nn.Module):
    """Return precisely the feature, camera and shape bindings used by FastHMR."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, img, person_masks, box_center, box_size, img_size, cam_int):
        values = {
            "img": img,
            "person_masks": person_masks,
            "box_center": box_center,
            "box_size": box_size,
            "img_size": img_size,
            "cam_int": cam_int,
        }
        with torch.autocast("cuda", dtype=torch.float16):
            params, camera, _, tokens = self.backbone(values)
        return tokens.float(), camera.float(), params["betas"].float()


def actual_batch(model, clip):
    selected = clip.inputs.selection.selections[0].requests[0]
    indices = selected.source_frames[:4]
    video = selected.video_path
    backbone = model.models[0]
    batches = []
    handle = backbone.register_forward_pre_hook(lambda _, args: batches.append(args[0]))
    try:
        with closing(
            iter(OpenCVVideoFrameReader(video, max_frames=int(indices[-1]) + 1))
        ) as reader:
            images = model._source_frames(reader, indices, selected.size)
        model._encode(images, selected.boxes_xys[:4], selected.size)
    finally:
        handle.remove()
    batch = batches[0]
    marker = batch["person_masks"][..., 32:-32].flatten(1).any(1)
    if not bool(marker.any()):
        raise RuntimeError("No positive person mask in the four pinned source frames")
    row = int(marker.nonzero()[0, 0])
    values = tuple(tensor[row : row + 1].contiguous() for tensor in batch.values())
    if tuple(batch) != (
        "img",
        "person_masks",
        "box_center",
        "box_size",
        "img_size",
        "cam_int",
    ):
        raise ValueError("CameraHMR binding order changed")
    return int(indices[row]), values


def errors(actual, expected):
    if any(a.shape != b.shape for a, b in zip(actual, expected, strict=True)):
        return {
            "actual_shapes": [list(a.shape) for a in actual],
            "expected_shapes": [list(b.shape) for b in expected],
        }
    return {
        name: float((a - b).abs().max())
        for name, a, b in zip(
            ("tokens", "crop_camera", "betas"), actual, expected, strict=True
        )
    }


def compare(captured, masked, unmasked, eager):
    return {
        name: errors(captured(*sample), reference)
        for name, sample, reference in zip(
            ("with_mask", "without_mask"), (masked, unmasked), eager, strict=True
        )
    }


def failed(exc: Exception) -> dict[str, Any]:
    return {"status": "failed", "type": type(exc).__name__, "message": str(exc)[:3500]}


@torch.no_grad()
def main():
    cuda("trt16-export")
    output = Path(os.environ["TENNIS_REPRO_DIR"]) / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    record, source = cohort_records()[0]
    clip = Clip.read(record, source)
    model = FastHMR(config("pt16"))
    model.load()
    frame, masked = actual_batch(model, clip)
    unmasked = (masked[0], torch.zeros_like(masked[1]), *masked[2:])
    bindings = CameraBackboneBindings(model.models[0]).eval()
    eager = tuple(bindings(*sample) for sample in (masked, unmasked))
    report = {
        "phase": "camera_backbone_export",
        "clip": clip.inputs.source.clip_id,
        "source_frame": frame,
        "tensorrt": tensorrt.__version__,
        "profile": {"batch": 1, "image": list(masked[0].shape)},
        "positive_mask_pixels": int(masked[1].count_nonzero()),
        "eager_mask_effect": errors(*eager),
    }
    emit(report)
    try:
        program = torch.export.export(bindings, masked, strict=False)
        report["torch_export"] = {
            "status": "captured",
            "absolute_delta": compare(program.module(), masked, unmasked, eager),
        }
    except Exception as exc:
        report["torch_export"] = failed(exc)
    emit({"phase": "torch_export", **report["torch_export"]})
    try:
        traced = torch.jit.trace(bindings, masked, check_trace=False)
        report["torchscript_export"] = {
            "status": "captured",
            "absolute_delta": compare(traced, masked, unmasked, eager),
        }
    except Exception as exc:
        report["torchscript_export"] = failed(exc)
    emit({"phase": "torchscript_export", **report["torchscript_export"]})
    report["trt_engine"] = "not_built"
    (output / "trt-export.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    emit(report)


if __name__ == "__main__":
    main()
