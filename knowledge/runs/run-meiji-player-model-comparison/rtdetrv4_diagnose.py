"""Compare RT-DETRv4 execution on identical inputs before and after box decoding."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rtdetrv4 import (
    OUTPUT_NAMES,
    UPSTREAM_COMMIT,
    WEIGHT_SHA256,
    Assets,
    Detector,
    ModelOutputs,
    load_detector,
    person_predictions,
    preprocess,
)

from src.tennis_scene.evaluation.player_pose.bundle import load_reference_frames
from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
    ReferenceFrame,
    score_players,
)
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import Backend
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import ONNX_FILE, PLAN_FILE

# Two views each from the first two clips of the pinned speed input index.
FRAME_INDICES = (0, 4, 16, 20)


def _onnx_cpu(onnx_path: Path) -> tuple[str, Callable[[torch.Tensor], ModelOutputs]]:
    ort = import_module("onnxruntime")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Expected the ONNX Runtime CPU provider")

    def forward(pixels: torch.Tensor) -> ModelOutputs:
        logits, boxes = session.run(
            list(OUTPUT_NAMES), {"pixels": pixels.cpu().numpy()}
        )
        return torch.from_numpy(logits), torch.from_numpy(boxes)

    return str(ort.__version__), forward


def decode_rows(
    decoder: Detector, raw: ModelOutputs, sizes: torch.Tensor
) -> list[Predictions]:
    logits, boxes = raw
    decoded = decoder.decode((logits.to(sizes.device), boxes.to(sizes.device)), sizes)
    labels, xyxy, scores = (tensor.cpu().numpy() for tensor in decoded)
    return person_predictions(labels, xyxy, scores, sizes.cpu().numpy())


def capture_batches(
    forward: Callable[[torch.Tensor], ModelOutputs],
    decoder: Detector,
    pixels: torch.Tensor,
    sizes: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[Predictions]]:
    logits, boxes, detections = [], [], []
    for start in range(0, len(pixels), batch_size):
        rows = slice(start, start + batch_size)
        raw_logits, raw_boxes = forward(pixels[rows])
        logits.append(raw_logits.cpu().numpy())
        boxes.append(raw_boxes.cpu().numpy())
        # The deployed postprocessor and pixel scale are shared by all three paths.
        detections.extend(decode_rows(decoder, (raw_logits, raw_boxes), sizes[rows]))
    arrays = {
        "pred_logits": np.concatenate(logits),
        "pred_boxes": np.concatenate(boxes),
    }
    return arrays, detections


def _compare_batches(
    forward: Callable[[torch.Tensor], ModelOutputs],
    decoder: Detector,
    pixels: torch.Tensor,
    sizes: torch.Tensor,
    keys: tuple[FrameKey, ...],
    references: dict[FrameKey, ReferenceFrame],
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays, detections = capture_batches(forward, decoder, pixels, sizes, batch_size)
    score = score_players(references, dict(zip(keys, detections, strict=True)))
    return {
        "status": "ok",
        "raw": {
            name: {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for name, values in arrays.items()
        },
        "max_person_score_by_image": [
            float(row.scores.max()) if len(row.scores) else None for row in detections
        ],
        "person_count_at_score_0_25_by_image": [
            int((row.scores >= 0.25).sum()) for row in detections
        ],
        "pseudo_gt": score["overall"]["detection"]["recall_by_iou"],
        "pseudo_gt_references": score["overall"]["detection"]["references"],
    }, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=WEIGHT_SHA256, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--speed-inputs", type=Path, required=True)
    parser.add_argument("--speed-sha256", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_REPRO_DIR"):
        raise ValueError("Run this CUDA diagnostic in the shared training queue")

    assets = Assets(args.model_id, args.upstream, args.weights)
    assets.verify()
    fixed = import_module("speed_inputs").InputSet(args.speed_inputs)
    if fixed.sha256 != args.speed_sha256:
        raise ValueError(f"Speed input SHA-256 mismatch: {args.speed_inputs}")
    selected = [fixed.rows("frames")[index] for index in FRAME_INDICES]
    keys = tuple(
        FrameKey(row["clip_id"], row["camera"], row["frame"]) for row in selected
    )
    references = load_reference_frames(
        args.benchmark, expected_sha256=args.manifest_sha256
    )
    references = {key: references[key] for key in keys}
    pixels_cpu, sizes_cpu = preprocess([fixed.frame(row["id"]) for row in selected])
    args.output.mkdir(parents=True, exist_ok=False)
    (Path(os.environ["TENNIS_REPRO_DIR"]) / "output_dir.txt").write_text(
        f"{args.output.resolve()}\n"
    )
    pixels_path = args.output / "pixels-fp32.npy"
    np.save(pixels_path, pixels_cpu.numpy(), allow_pickle=False)

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.cuda.synchronize(device)
    arrays: dict[str, Any] = {}
    results = {}
    ort_version = None
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        pixels, sizes = pixels_cpu.to(device), sizes_cpu.to(device)
        pt16 = load_detector(assets, Backend.PT16, device, None)
        trt16 = load_detector(assets, Backend.TRT16, device, args.engine_dir)
        runners: dict[str, Callable[[torch.Tensor], ModelOutputs]] = {
            "pt16": pt16.forward,
            "trt16": trt16.forward,
        }
        try:
            ort_version, runners["onnx_cpu"] = _onnx_cpu(args.engine_dir / ONNX_FILE)
        except Exception as error:
            results["onnx_cpu"] = {
                "status": "infeasible",
                "reason": f"{type(error).__name__}: {error}",
            }
        for name, forward in runners.items():
            for batch in (1, 2):
                case = f"{name}_b{batch}"
                try:
                    results[case], raw = _compare_batches(
                        forward, pt16, pixels, sizes, keys, references, batch
                    )
                    arrays.update(
                        {f"{case}_{key}": value for key, value in raw.items()}
                    )
                except Exception as error:
                    results[case] = {
                        "status": "infeasible",
                        "reason": f"{type(error).__name__}: {error}",
                    }

    raw_path = args.output / "raw-outputs.npz"
    np.savez_compressed(raw_path, allow_pickle=False, **arrays)
    receipt = {
        "model_id": args.model_id,
        "job_id": os.environ["TENNIS_RUN_ID"],
        "upstream_commit": UPSTREAM_COMMIT,
        "weight_sha256": assets.weight_sha256,
        "engine_sha256": dual_sha256(args.engine_dir / PLAN_FILE),
        "onnx_sha256": dual_sha256(args.engine_dir / ONNX_FILE),
        "speed_manifest_sha256": fixed.sha256,
        "accuracy_manifest_sha256": args.manifest_sha256,
        "input_ids": [row["id"] for row in selected],
        "pixels_sha256": dual_sha256(pixels_path),
        "raw_outputs_sha256": dual_sha256(raw_path),
        "gpu": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "onnxruntime_cpu": ort_version,
        "cases": results,
    }
    write_json_atomic(args.output / "diagnosis.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if any(result["status"] != "ok" for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
