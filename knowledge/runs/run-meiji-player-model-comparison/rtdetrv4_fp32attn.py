"""Test FP32 decoder attention scores on the four archived RT-DETRv4 images."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from shutil import copyfile
from typing import Any

import numpy as np
import onnx
import torch
from pydantic import JsonValue
from rtdetrv4 import (
    GIB,
    INPUT_NAME,
    NUM_CLASSES,
    NUM_QUERIES,
    OUTPUT_NAMES,
    PT32_NOT_MEASURED,
    WEIGHT_SHA256,
    Assets,
    Detector,
    ModelOutputs,
    _export_settings,
    detector_engine_bundle,
    preprocess,
)
from rtdetrv4_diagnose import FRAME_INDICES, capture_batches, decode_rows

from src.tennis_scene.evaluation.player_pose.bundle import load_reference_frames
from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
    score_players,
)
from src.utils.checksum import dual_sha256
from src.utils.geometry.bbox import match_iou, pairwise_iou
from src.utils.inference_benchmark import (
    Backend,
    Conversion,
    ParityStatus,
    measure_conversion,
)
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import (
    EXPORT_MANIFEST,
    ONNX_FILE,
    EngineBundle,
    ExportManifest,
)
from src.utils.models.tensorrt_build import build_bundle, export_bundle
from src.utils.models.tensorrt_runtime import TensorRTRunner

CONDITION = "trt16-fp32attn"
DECODER_PREFIX = "/model/decoder/decoder/layers."
SELF_ATTENTION_SCOPES = tuple(f"{DECODER_PREFIX}{i}/self_attn" for i in range(6))
FRAME_ORDERS = {"original": (0, 1, 2, 3), "swapped": (1, 0, 3, 2)}
RECEIPTS = Path(__file__).with_name("rtdetrv4-results.json")


def promote_decoder_attention_scores(model: onnx.ModelProto) -> onnx.ModelProto:
    """Widen only the six isolated QK-to-Softmax regions of the pinned graph."""
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph
    by_name = {node.name: node for node in graph.node}
    targets = {
        node.name
        for node in graph.node
        if node.name.startswith(DECODER_PREFIX)
        and node.name.endswith("/self_attn/Softmax")
    }
    expected = {f"{scope}/Softmax" for scope in SELF_ATTENTION_SCOPES}
    if len(by_name) != len(graph.node) or targets != expected:
        raise ValueError(
            "Expected six decoder self-attention Softmax nodes "
            f"(missing={sorted(expected - targets)}, extra={sorted(targets - expected)})"
        )

    values = {
        value.name: value for value in (*graph.input, *graph.output, *graph.value_info)
    }
    consumers: dict[str, list[str]] = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node.name)
    graph_outputs = {value.name for value in graph.output}
    used_names = {
        *values,
        *consumers,
        *(name for node in graph.node for name in node.output),
    }
    before: dict[str, list[onnx.NodeProto]] = {}
    after: dict[str, onnx.NodeProto] = {}
    half, single = onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT
    for scope in SELF_ATTENTION_SCOPES:
        score_node = by_name.get(f"{scope}/MatMul_3")
        softmax = by_name[f"{scope}/Softmax"]
        if (
            score_node is None
            or score_node.op_type != "MatMul"
            or len(score_node.input) != 2
            or len(score_node.output) != 1
            or softmax.op_type != "Softmax"
            or len(softmax.input) != 1
            or len(softmax.output) != 1
            or score_node.output[0] != softmax.input[0]
        ):
            raise ValueError(f"Expected a direct score MatMul-to-Softmax at {scope}")
        (score,) = score_node.output
        (probabilities,) = softmax.output
        if (
            consumers[score] != [softmax.name]
            or consumers[probabilities] != [f"{scope}/MatMul_4"]
            or by_name[f"{scope}/MatMul_4"].op_type != "MatMul"
            or score in graph_outputs
            or any(
                name not in values or values[name].type.tensor_type.elem_type != half
                for name in (*score_node.input, score, probabilities)
            )
        ):
            raise ValueError(f"Expected an isolated FP16 score region at {scope}")

        q_fp32, k_fp32 = (f"{scope}/q_fp32", f"{scope}/k_fp32")
        probabilities_fp32 = f"{probabilities}_fp32"
        restore = f"{scope}/restore_fp16"
        if {q_fp32, k_fp32, probabilities_fp32} & used_names or {
            q_fp32,
            k_fp32,
            restore,
        } & by_name.keys():
            raise ValueError(f"FP32 attention cast name already exists at {scope}")
        before[score_node.name] = [
            onnx.helper.make_node("Cast", [value], [target], name=target, to=single)
            for value, target in zip(score_node.input, (q_fp32, k_fp32), strict=True)
        ]
        score_node.input[:] = [q_fp32, k_fp32]
        values[score].type.tensor_type.elem_type = single
        softmax.output[0] = probabilities_fp32
        after[softmax.name] = onnx.helper.make_node(
            "Cast", [probabilities_fp32], [probabilities], name=restore, to=half
        )

    updated_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        updated_nodes.extend(before.get(node.name, ()))
        updated_nodes.append(node)
        if node.name in after:
            updated_nodes.append(after[node.name])
    del graph.node[:]
    graph.node.extend(updated_nodes)
    return onnx.shape_inference.infer_shapes(inferred, strict_mode=True)


def _attention_settings(base: EngineBundle) -> dict[str, JsonValue]:
    return {
        **base.export.settings,
        "precision_variant": CONDITION,
        "source_onnx_sha256": base.export.onnx_files[ONNX_FILE],
        "fp32_attention_scopes": list(SELF_ATTENTION_SCOPES),
    }


def export_attention_variant(base: EngineBundle, assets: Assets, target: Path) -> None:
    def write_graph(path: Path) -> None:
        original = onnx.load(base.root / ONNX_FILE, load_external_data=False)
        rewritten = promote_decoder_attention_scores(original)
        for filename in base.export.onnx_files.keys() - {ONNX_FILE}:
            copyfile(base.root / filename, path.parent / filename)
        onnx.save_model(rewritten, path)
        onnx.checker.check_model(str(path), full_check=True)

    export_bundle(
        target,
        model=assets.model_id,
        settings=_attention_settings(base),
        sources={"checkpoint": assets.weights},
        export=write_graph,
    )
    candidate = ExportManifest.model_validate_json(
        (target / EXPORT_MANIFEST).read_bytes()
    )
    if (
        candidate.inputs != base.export.inputs
        or candidate.outputs != base.export.outputs
        or candidate.sources != base.export.sources
        or {k: v for k, v in candidate.onnx_files.items() if k != ONNX_FILE}
        != {k: v for k, v in base.export.onnx_files.items() if k != ONNX_FILE}
    ):
        raise ValueError("The FP32 attention graph changed its I/O or weight contract")


def capture_in_frame_order(
    forward: Callable[[torch.Tensor], ModelOutputs],
    decoder: Detector,
    pixels: torch.Tensor,
    sizes: torch.Tensor,
    batch_size: int,
    order: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], list[Predictions]]:
    indices = list(order)
    if sorted(indices) != list(range(len(pixels))):
        raise ValueError("Expected a permutation of all diagnostic frames")
    heads, rows = capture_batches(
        forward, decoder, pixels[indices], sizes[indices], batch_size
    )
    frame_order = np.argsort(indices)
    return {name: head[frame_order] for name, head in heads.items()}, [
        rows[index] for index in frame_order
    ]


def _compare(
    reference_heads: dict[str, np.ndarray],
    reference_rows: Sequence[Predictions],
    heads: dict[str, np.ndarray],
    rows: Sequence[Predictions],
) -> dict[str, Any]:
    raw_delta = {}
    for name in OUTPUT_NAMES:
        reference, observed = reference_heads[name], heads[name]
        if reference.shape != observed.shape or not (
            np.isfinite(reference).all() and np.isfinite(observed).all()
        ):
            raise ValueError(f"Raw {name} shape or finiteness differs")
        error = np.abs(reference.astype(np.float32) - observed.astype(np.float32))
        raw_delta[name] = {
            "abs_p95": float(np.percentile(error, 95)),
            "abs_max": float(error.max()),
        }

    return {
        "raw_delta_by_query_index": raw_delta,
        "decoded_boxes_by_min_score": {
            f"{cutoff:.2f}": _box_agreement(reference_rows, rows, cutoff)
            for cutoff in (0.0, 0.25)
        },
    }


def _box_agreement(
    reference: Sequence[Predictions], candidate: Sequence[Predictions], min_score: float
) -> dict[str, int | float | None]:
    reference_count = candidate_count = 0
    matched_ious = []
    for first, second in zip(reference, candidate, strict=True):
        left = first.boxes_xyxy[first.scores >= min_score]
        right = second.boxes_xyxy[second.scores >= min_score]
        ious = pairwise_iou(left, right)
        row, col = match_iou(ious, 0.50)
        reference_count += len(left)
        candidate_count += len(right)
        matched_ious.append(ious[row, col])
    matched = np.concatenate(matched_ious)
    return {
        "reference_boxes": reference_count,
        "candidate_boxes": candidate_count,
        "matched_boxes": len(matched),
        "matched_iou_p05": float(np.percentile(matched, 5)) if len(matched) else None,
    }


def _read_pt16(
    archive: Path, record: dict[str, Any], prepared: torch.Tensor
) -> dict[int, dict[str, np.ndarray]]:
    for field in ("pixels", "raw_outputs"):
        asset = record[field]
        if dual_sha256(archive / Path(asset["path"]).name) != asset["sha256"]:
            raise ValueError(f"Archived {field} SHA-256 mismatch: {archive}")
    archived = np.load(archive / "pixels-fp32.npy", allow_pickle=False)
    if archived.dtype != np.float32 or not np.array_equal(archived, prepared.numpy()):
        raise ValueError("Prepared speed images differ from the saved PT16 inputs")
    with np.load(archive / "raw-outputs.npz", allow_pickle=False) as arrays:
        result = {
            batch: {name: arrays[f"pt16_b{batch}_{name}"] for name in OUTPUT_NAMES}
            for batch in (1, 2)
        }
    for heads in result.values():
        if heads["pred_logits"].shape != (
            len(prepared),
            NUM_QUERIES,
            NUM_CLASSES,
        ) or heads["pred_boxes"].shape != (len(prepared), NUM_QUERIES, 4):
            raise ValueError("Unexpected archived PT16 head shape")
    return result


def _diagnose(args: argparse.Namespace, receipt: dict[str, Any]) -> None:
    assets = Assets(args.model_id, args.upstream, args.weights)
    assets.verify()
    evidence = json.loads(RECEIPTS.read_text())
    if args.manifest_sha256 != evidence["inputs"]["accuracy"]["manifest_sha256"]:
        raise ValueError("Pseudo-GT manifest differs from the archived diagnostic")
    prior = evidence["trt16_diagnosis"]
    archive = prior["runs"][args.model_id]
    base = detector_engine_bundle(assets, args.engine_dir, _export_settings())
    if (
        base.engine.plan_sha256 != archive["engine_sha256"]
        or base.export.onnx_files[ONNX_FILE] != archive["onnx_sha256"]
    ):
        raise ValueError("Original plan differs from the archived diagnostic")
    fixed = import_module("speed_inputs").InputSet(args.speed_inputs)
    if fixed.sha256 != args.speed_sha256:
        raise ValueError(f"Speed input SHA-256 mismatch: {args.speed_inputs}")
    selected = [fixed.rows("frames")[index] for index in FRAME_INDICES]
    ids = [row["id"] for row in selected]
    if ids != prior["inputs"]["ids"]:
        raise ValueError("Fixed speed frame IDs differ from the archived diagnostic")
    keys = tuple(
        FrameKey(row["clip_id"], row["camera"], row["frame"]) for row in selected
    )
    all_references = load_reference_frames(
        args.benchmark, expected_sha256=args.manifest_sha256
    )
    references = {key: all_references[key] for key in keys}
    prepared, native_sizes = preprocess([fixed.frame(id_) for id_ in ids])
    pt16_heads = _read_pt16(args.archive, archive, prepared)

    target = args.output / "engine"
    device = torch.device("cuda:0")
    export = measure_conversion(
        Conversion.ONNX_EXPORT, lambda: export_attention_variant(base, assets, target)
    )
    write_json_atomic(args.output / "onnx-export.json", asdict(export))
    if export.failure:
        raise ValueError(export.failure.message)
    build = measure_conversion(
        Conversion.ENGINE_BUILD,
        lambda: build_bundle(
            target, base.engine.profiles, device=device, workspace_bytes=GIB
        ),
        device=device,
    )
    write_json_atomic(args.output / "engine-build.json", asdict(build))
    if build.failure:
        raise RuntimeError(build.failure.message)
    candidate = detector_engine_bundle(assets, target, _attention_settings(base))
    receipt["engines"] = {
        name: {
            "plan_sha256": bundle.engine.plan_sha256,
            "onnx_files": bundle.export.onnx_files,
            "profiles": {
                key: profile.model_dump(mode="json")
                for key, profile in bundle.engine.profiles.items()
            },
            "environment": bundle.engine.environment.model_dump(mode="json"),
        }
        for name, bundle in (("trt16", base), (CONDITION, candidate))
    }

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    _, postprocessor = assets.config_and_postprocessor()
    raw: dict[str, np.ndarray] = {}
    receipt["cases"] = {}
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream(device=device)):
        pixels, sizes = prepared.to(device), native_sizes.to(device)
        postprocessor = postprocessor.to(device)
        for name, bundle in (("trt16", base), (CONDITION, candidate)):
            runner = TensorRTRunner(bundle, device)

            def forward(
                batch: torch.Tensor, engine: TensorRTRunner = runner
            ) -> ModelOutputs:
                outputs = engine({INPUT_NAME: batch})
                return outputs["pred_logits"], outputs["pred_boxes"]

            decoder = Detector(Backend.TRT16, device, forward, postprocessor)
            for batch_size in (1, 2):
                saved = pt16_heads[batch_size]
                pt16_rows = decode_rows(
                    decoder,
                    (
                        torch.from_numpy(saved["pred_logits"]),
                        torch.from_numpy(saved["pred_boxes"]),
                    ),
                    sizes,
                )
                for order_name, order in FRAME_ORDERS.items():
                    heads, rows = capture_in_frame_order(
                        forward, decoder, pixels, sizes, batch_size, order
                    )
                    case = f"{name}_b{batch_size}_{order_name}"
                    raw.update({f"{case}_{key}": value for key, value in heads.items()})
                    for index, row in enumerate(rows):
                        raw[f"{case}_xyxy_{index}"] = row.boxes_xyxy
                        raw[f"{case}_scores_{index}"] = row.scores
                    score = score_players(
                        references, dict(zip(keys, rows, strict=True))
                    )
                    detection = score["overall"]["detection"]
                    receipt["cases"][case] = {
                        "batch_size": batch_size,
                        "execution_order": [ids[index] for index in order],
                        "vs_saved_pt16": _compare(saved, pt16_rows, heads, rows),
                        "pseudo_gt_references": detection["references"],
                        "pseudo_gt_recall": detection["recall_by_iou"],
                        "pseudo_gt_recall_by_min_score": detection[
                            "recall_by_min_score"
                        ],
                    }
                    if order_name == "original":
                        original_heads, original_rows = heads, rows
                    else:
                        receipt["cases"][case]["vs_original_order"] = _compare(
                            original_heads, original_rows, heads, rows
                        )

    raw_path = args.output / "raw-outputs.npz"
    np.savez_compressed(raw_path, allow_pickle=False, **raw)
    receipt.update(
        status="ok",
        weight_sha256=assets.weight_sha256,
        source_diagnostic_job=archive["job_id"],
        source_pixels_sha256=archive["pixels"]["sha256"],
        source_pt16_archive_sha256=archive["raw_outputs"]["sha256"],
        input_ids=ids,
        raw_outputs_sha256=dual_sha256(raw_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=WEIGHT_SHA256, required=True)
    for name in (
        "upstream",
        "weights",
        "engine-dir",
        "archive",
        "speed-inputs",
        "benchmark",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--speed-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_REPRO_DIR"):
        raise ValueError("Run the FP32 attention probe in the shared training queue")
    args.output.mkdir(parents=True, exist_ok=False)
    (Path(os.environ["TENNIS_REPRO_DIR"]) / "output_dir.txt").write_text(
        f"{args.output.resolve()}\n"
    )
    receipt = {
        "condition": CONDITION,
        "model_id": args.model_id,
        "job_id": os.environ["TENNIS_RUN_ID"],
        "speed_manifest_sha256": args.speed_sha256,
        "accuracy_manifest_sha256": args.manifest_sha256,
        "parity_status": ParityStatus.NOT_EVALUATED,
        "parity_not_evaluated_reason": PT32_NOT_MEASURED,
    }
    path = args.output / "diagnosis.json"
    try:
        _diagnose(args, receipt)
    except (OSError, ValueError, RuntimeError) as error:
        receipt.update(status="infeasible", reason=f"{type(error).__name__}: {error}")
        write_json_atomic(path, receipt)
        raise
    write_json_atomic(path, receipt)
    print(
        f"{args.model_id}/{CONDITION}: {len(receipt['cases'])} cases at {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
