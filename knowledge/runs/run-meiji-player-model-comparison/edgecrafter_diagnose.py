"""Trace the input, raw heads and batch slots of the pinned EdgeCrafter plans."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
from edgecrafter import ASSETS, HEADS, INPUT_NAME, SCORE_THRESHOLD, _load_forward
from numpy.typing import NDArray
from speed_inputs import InputSet

from src.submodules.models.ecpose.inference import (
    decode_predictions,
    preprocess_images,
)
from src.submodules.models.ecpose.releases import RELEASES
from src.utils.checksum import dual_sha256
from src.utils.geometry.bbox import match_iou, pairwise_iou
from src.utils.inference_benchmark import Backend
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import ONNX_FILE, EngineBundle
from src.utils.models.tensorrt_runtime import TensorRTRunner

ARTIFACTS = ASSETS.parent
ONNX_CPU_MODELS = ("ecdet-l-coco", "ecpose-l-o365", "ecpose-x-o365")
FRAMES = (
    "video_000/clip_003/cam0/frame_000148",
    "video_000/clip_028/cam3/frame_000513",
    "video_000/clip_057/cam3/frame_000114",
    "video_000/clip_013/cam0/frame_000096",
)
BATCHES = {
    f"pair{pair}_{name}": tuple(offset + index for index in order)
    for pair, offset in ((1, 0), (2, 2))
    for name, order in (
        ("a", (0,)),
        ("b", (1,)),
        ("ab", (0, 1)),
        ("ba", (1, 0)),
        ("ab_repeat", (0, 1)),
    )
}

FloatArray: TypeAlias = NDArray[np.floating]


@dataclass(frozen=True)
class Capture:
    frames: tuple[str, ...]
    heads: Mapping[str, FloatArray]
    boxes: tuple[FloatArray, ...]
    scores: tuple[FloatArray, ...]


def head_stats(values: FloatArray) -> dict[str, int | float | None]:
    finite = values[np.isfinite(values)]
    return {
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(values).sum()),
        "posinf_count": int(np.isposinf(values).sum()),
        "neginf_count": int(np.isneginf(values).sum()),
        "finite_min": float(finite.min()) if finite.size else None,
        "finite_max": float(finite.max()) if finite.size else None,
    }


def box_matches(first: FloatArray, second: FloatArray) -> dict[str, int | float | None]:
    iou = pairwise_iou(first, second)
    row, col = match_iou(iou, 0.5)
    return {
        "reference_rows": len(first),
        "candidate_rows": len(second),
        "matches_at_iou_0_50": len(row),
        "matched_iou_p05": float(np.percentile(iou[row, col], 5)) if len(row) else None,
    }


def compare_images(first: Capture, second: Capture) -> dict[str, object]:
    """Align slots by the source image before comparing any tensor elements."""
    common = [frame for frame in first.frames if frame in second.frames]
    if not common:
        raise ValueError("Compared batches need a shared image")
    result: dict[str, object] = {}
    for frame in common:
        i, j = first.frames.index(frame), second.frames.index(frame)
        heads = {
            name: tensor_difference(
                first.heads[name][i], second.heads[name][j], label=f"{frame}/{name}"
            )
            for name in HEADS
        }
        result[frame] = {
            "reference_slot": i,
            "candidate_slot": j,
            "heads": heads,
            "decoded": box_matches(first.boxes[i], second.boxes[j]),
        }
    return result


def tensor_difference(
    left: NDArray[Any], right: NDArray[Any], *, label: str
) -> dict[str, int | float | None]:
    if left.shape != right.shape:
        raise ValueError(f"Raw head shape differs for {label}")
    left_finite, right_finite = np.isfinite(left), np.isfinite(right)
    valid = left_finite & right_finite
    delta = np.abs(left[valid].astype(np.float64) - right[valid])
    return {
        "finite_pairs": int(valid.sum()),
        "finite_mask_difference": int((left_finite != right_finite).sum()),
        "abs_diff_p95": float(np.percentile(delta, 95)) if delta.size else None,
        "abs_diff_max": float(delta.max()) if delta.size else None,
    }


def capture(
    model: str,
    raw: tuple[torch.Tensor, torch.Tensor],
    sizes: torch.Tensor,
    frames: tuple[str, ...],
) -> Capture:
    release = RELEASES[model]
    rows = decode_predictions(
        release,
        raw,
        sizes,
        confidence=SCORE_THRESHOLD,
        pose_box_margin=0.10 if release.has_pose else None,
    )
    return Capture(
        frames=frames,
        heads={
            name: value.cpu().numpy() for name, value in zip(HEADS, raw, strict=True)
        },
        boxes=tuple(row.boxes_xyxy.cpu().numpy() for row in rows),
        scores=tuple(row.scores.cpu().numpy() for row in rows),
    )


def engine_bundle(model: str) -> EngineBundle:
    bundle = EngineBundle.read(ARTIFACTS / model / "trt16-trace2/engine")
    if bundle.export.model != model or bundle.export.sources != {
        "weights": RELEASES[model].sha256
    }:
        raise ValueError(f"Wrong plan sources for {model}")
    return bundle


def diagnose_cuda(
    model: str,
    pixels: torch.Tensor,
    sizes: torch.Tensor,
    device: torch.device,
    output: Path,
) -> dict[str, object]:
    bundle = engine_bundle(model)
    pt16 = _load_forward(model, ASSETS, Backend.PT16, device, None)
    trt16 = TensorRTRunner(bundle, device)
    batches = {}
    captures: dict[tuple[str, str], Capture] = {}
    arrays = {}
    for name, selection in BATCHES.items():
        frames = tuple(FRAMES[index] for index in selection)
        index = list(selection)
        # Share one contiguous allocation between the forward calls for each case.
        inputs, wh = pixels[index].contiguous(), sizes[index]
        input_digest = sha256(inputs.cpu().numpy().tobytes()).hexdigest()
        raw_pt = pt16(inputs)
        if sha256(inputs.cpu().numpy().tobytes()).hexdigest() != input_digest:
            raise ValueError(f"PT16 changed the shared input tensor for {name}")
        bindings = trt16({INPUT_NAME: inputs})
        context = trt16._context
        raw_trt = bindings[HEADS[0]], bindings[HEADS[1]]
        metadata: dict[str, object] = {}
        for binding, tensor in {INPUT_NAME: inputs, **bindings}.items():
            strides = tuple(context.get_tensor_strides(binding))
            metadata[binding] = {
                "context_shape": tuple(context.get_tensor_shape(binding)),
                "context_strides": strides,
                "torch_strides": tensor.stride(),
                "dtype": str(tensor.dtype),
                "strides_match": strides == tensor.stride(),
            }
        results = {}
        for backend, raw in (("pt16", raw_pt), ("trt16", raw_trt)):
            item = capture(model, raw, wh, frames)
            captures[backend, name] = item
            for head, values in item.heads.items():
                arrays[f"{backend}_{name}_{head}"] = values
            results[backend] = {
                "head_shape": {
                    head: list(values.shape) for head, values in item.heads.items()
                },
                "head_dtype": {
                    head: str(values.dtype) for head, values in item.heads.items()
                },
                "image_stats": {
                    frame: {
                        "heads": {
                            head: head_stats(values[slot])
                            for head, values in item.heads.items()
                        },
                        "person_rows": int(item.scores[slot].size),
                        "person_rows_at_0_25": int((item.scores[slot] >= 0.25).sum()),
                    }
                    for slot, frame in enumerate(frames)
                },
            }
        batches[name] = {
            "frames": frames,
            "input_tensor_sha256": input_digest,
            "trt_bindings": metadata,
            "outputs": results,
        }

    comparisons = {
        "pt16_vs_trt16": {
            name: compare_images(captures["pt16", name], captures["trt16", name])
            for name in BATCHES
        },
        **{
            backend: {
                f"pair{pair}_{first}_vs_{second}": compare_images(
                    captures[backend, f"pair{pair}_{first}"],
                    captures[backend, f"pair{pair}_{second}"],
                )
                for pair in (1, 2)
                for first, second in (
                    ("ab", "ba"),
                    ("a", "ab"),
                    ("b", "ab"),
                    ("ab", "ab_repeat"),
                )
            }
            for backend in ("pt16", "trt16")
        },
    }
    raw_path = output / f"{model}-heads.npz"
    np.savez(raw_path, **arrays)
    layers_path = output / f"{model}-engine-layers.json"
    try:
        write_json_atomic(layers_path, trt16.inspect_layers())
        layer_report = {"path": layers_path.name, "sha256": dual_sha256(layers_path)}
    except (OSError, RuntimeError, ValueError) as error:
        layer_report = {"inspection_failure": f"{type(error).__name__}: {error}"}
    return {
        "weight_sha256": RELEASES[model].sha256,
        "engine_sha256": bundle.engine.plan_sha256,
        "onnx_sha256": bundle.export.onnx_files[ONNX_FILE],
        "engine_profile": bundle.engine.profiles[INPUT_NAME].model_dump(),
        "environment": bundle.engine.environment.model_dump(),
        "batches": batches,
        "comparisons": comparisons,
        "raw_heads": {"path": raw_path.name, "sha256": dual_sha256(raw_path)},
        "engine_layers": layer_report,
    }


def stored_boxes(model: str, directory: str, frame: str) -> FloatArray:
    clip, camera, image = frame.rsplit("/", 2)
    key = (clip, camera, int(image.removeprefix("frame_")))
    source = ARTIFACTS / model / directory / "predictions.jsonl"
    prefix = (
        f'{{"clip_id":"{key[0]}","camera_id":"{key[1]}","source_frame":{key[2]},'
    ).encode()
    with source.open("rb") as handle:
        for line in handle:
            if line.startswith(prefix):
                boxes: FloatArray = np.asarray(
                    json.loads(line)["boxes_xyxy"], dtype=np.float64
                ).reshape((-1, 4))
                return boxes
    raise ValueError(f"Missing saved image {frame} in {source}")


def diagnose_onnx_cpu(
    model: str, pixels: torch.Tensor, sizes: torch.Tensor
) -> dict[str, object]:
    ort = import_module("onnxruntime")
    bundle = engine_bundle(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(bundle.root / ONNX_FILE),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("The ONNX diagnosis requires the CPU execution provider")
    # The second image in pair 1 has large saved ECDet and ECPose-X recall losses.
    frame = FRAMES[1]
    tensor = pixels[1:2]
    logits, coordinates = session.run(list(HEADS), {INPUT_NAME: tensor.numpy()})
    row = capture(
        model,
        (torch.from_numpy(logits), torch.from_numpy(coordinates)),
        sizes[1:2],
        (frame,),
    )
    return {
        "frame": frame,
        "input_tensor_sha256": sha256(tensor.numpy().tobytes()).hexdigest(),
        "onnx_sha256": bundle.export.onnx_files[ONNX_FILE],
        "engine_sha256": bundle.engine.plan_sha256,
        "onnxruntime_version": ort.__version__,
        "providers": session.get_providers(),
        "vs_saved_b1": {
            backend: box_matches(row.boxes[0], stored_boxes(model, backend, frame))
            for backend in ("pt16", "trt16-trace2")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=RELEASES, action="append")
    parser.add_argument("--onnx-cpu", action="store_true")
    args = parser.parse_args()
    if not args.onnx_cpu and len(args.model or ()) != 1:
        parser.error("CUDA diagnostics require one --model per process")
    fixed = InputSet(args.inputs)
    pixels, sizes = preprocess_images(tuple(fixed.frame(frame) for frame in FRAMES))
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "checkout_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": dual_sha256(Path(__file__)),
        "input_manifest_sha256": fixed.sha256,
        "frames": {
            frame: {
                "source_image_sha256": fixed.lookups[frame]["sha256"],
                "pixel_tensor_sha256": sha256(
                    pixels[slot].numpy().tobytes()
                ).hexdigest(),
                "width_height": sizes[slot].tolist(),
            }
            for slot, frame in enumerate(FRAMES)
        },
        "torch_version": str(torch.__version__),
        "decode": {"confidence_exclusive": SCORE_THRESHOLD, "ecpose_box_margin": 0.10},
        "models": {},
    }
    if args.onnx_cpu:
        for model in args.model or ONNX_CPU_MODELS:
            report["models"][model] = diagnose_onnx_cpu(model, pixels, sizes)
        write_json_atomic(args.output / "onnx-cpu.json", report)
    else:
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        model = args.model[0]
        with torch.no_grad(), torch.cuda.stream(torch.cuda.Stream(device=device)):
            report["models"][model] = diagnose_cuda(
                model, pixels.to(device), sizes.to(device), device, args.output
            )
        write_json_atomic(args.output / "cuda.json", report)
        print(f"{model} produced all batch diagnostics", flush=True)


if __name__ == "__main__":
    main()
