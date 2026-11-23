"""Numerical and layer checks for GVHMR's typed TensorRT export."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_FUSED_ATTENTION = ("_gemm_mha_v2_", "_gemv_mha_v1_")
_SMALL_PROJECTION = "/network/learned_pos_linear/MatMul"
_ONNX_OPERATION = re.compile(r"\[ONNX Layer: ([^\]]+)\]")
_DTYPE_CODES = "f16|f32|tf32|bf16|i8|e4m3"
_TACTIC_OPERANDS = re.compile(
    rf"_(?:gemm|gemvx)_(?:interleaved_|indexed_|wo_smem_)*"
    rf"({_DTYPE_CODES})(?:{_DTYPE_CODES}|i32)?_({_DTYPE_CODES})(?:{_DTYPE_CODES}|i32)?_",
)
_CUBLAS_OPERANDS = re.compile(rf"_cublas_gemvx_({_DTYPE_CODES})({_DTYPE_CODES})(?:{_DTYPE_CODES})_")
_HALF = re.compile(r"\b(?:FP16|Float16|Half)\b", re.IGNORECASE)
_MATRIX_LAYERS = {"gemm", "matmul", "matrixmultiply", "fullyconnected", "conv", "convolution"}


def _agree(*dimensions: int) -> bool:
    return len({dim for dim in dimensions if dim != -1}) <= 1


def _additive_attention(inputs: Sequence[Mapping[str, Any]]) -> bool:
    if len(inputs) != 5 or any(t["Format/Datatype"] != "Half" for t in inputs):
        return False
    query, key, local, padding, value = (tuple(t["Dimensions"]) for t in inputs)
    if tuple(map(len, (query, key, local, padding, value))) != (3, 3, 4, 4, 3):
        return False
    heads_batch, tokens, width = query
    batch = padding[0]
    return (
        value == query and local[:2] == (1, 1) and padding[1:3] == (1, 1)
        and _agree(heads_batch, key[0]) and _agree(width, key[1])
        and _agree(tokens, key[2], local[2], local[3], padding[3])
        and (-1 in (batch, heads_batch) or batch > 0 and heads_batch > 0 and heads_batch % batch == 0)
    )


def verify_fp16_layers(
    details: Mapping[str, Any], expected_ops: set[str], *, attention_blocks: int,
) -> dict[str, Any]:
    """Check matrix operands, including fused additive MHA and the scalar K=2 projection."""
    mapped: dict[str, str] = {}
    mha_layers: list[str] = []
    small_k: list[str] = []
    for layer in details["Layers"]:
        kind = layer["LayerType"].lower()
        tactic = layer.get("TacticName") or ""
        operations = expected_ops.intersection(_ONNX_OPERATION.findall(layer.get("Metadata") or ""))
        inputs, outputs = layer.get("Inputs", []), layer.get("Outputs", [])
        name = layer["Name"]
        pattern = _CUBLAS_OPERANDS if "_cublas_gemvx_" in tactic else _TACTIC_OPERANDS
        operands = pattern.search(tactic)
        if kind == "kgen" and tactic.startswith(_FUSED_ATTENTION):
            valid = _additive_attention(inputs) and len(operations) == 2
            mha_layers.append(name)
        elif kind == "kgen" and tactic.startswith("__myl_SlicSlicMulMulAdd_"):
            valid = (
                operations == {_SMALL_PROJECTION} and len(inputs) == 2 and len(outputs) == 1
                and all(t["Format/Datatype"] == "Half" for t in (*inputs, *outputs))
                and all(t["Dimensions"] and t["Dimensions"][-1] == 2 for t in inputs)
                and outputs[0]["Dimensions"] and outputs[0]["Dimensions"][-1] == 32
            )
            small_k.append(name)
        elif kind in _MATRIX_LAYERS or operands is not None:
            valid = (
                operands is not None and operands.groups() == ("f16", "f16")
                and bool(inputs) and _HALF.search(inputs[0]["Format/Datatype"]) is not None
            )
        else:
            continue
        if not valid or not operations:
            raise ValueError(f"GVHMR layer {name} lacks FP16 operand evidence ({tactic})")
        for op in operations:
            mapped[op] = name
    if missing := expected_ops - mapped.keys():
        raise ValueError(f"Missing FP16 compute evidence for GVHMR: {sorted(missing)}")
    if len(mha_layers) != attention_blocks:
        raise ValueError(f"Expected {attention_blocks} fused additive attention layers, found {len(mha_layers)}")
    return {"fp16_operations": dict(sorted(mapped.items())), "fused_additive_attention": mha_layers,
            "elementwise_small_k": small_k}


def probe_inputs(batch: int, frames: int) -> dict[str, np.ndarray]:
    """Deterministic, distinct full-length sequences with visible and missing keypoints."""
    if batch < 1 or frames < 1:
        raise ValueError("GVHMR probe requires positive batch and length")
    rng = np.random.default_rng(2481)
    values = {}
    for name, tail in (("obs", (17, 3)), ("f_cliffcam", (3,)), ("f_cam_angvel", (6,)), ("f_imgseq", (1024,))):
        values[name] = rng.normal(0., .3, (batch, frames, *tail)).astype(np.float16)
    values["obs"][..., 2] = rng.integers(0, 2, (batch, frames, 17))
    values["length"] = np.full(batch, frames, dtype=np.int64)
    return values


def verify_outputs(
    outputs: Mapping[str, np.ndarray], *, batch: int, frames: int,
) -> None:
    shapes = {"pred_x": (batch, frames, 151), "pred_cam": (batch, frames, 3),
              "static_conf_logits": (batch, frames, 6), "mask_out": (batch, frames), "length_out": (batch,)}
    if outputs.keys() != shapes.keys():
        raise ValueError(f"GVHMR emitted unexpected output names: {sorted(outputs)}")
    for name, shape in shapes.items():
        array = outputs[name]
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"GVHMR {name} at B={batch}, L={frames} has shape {array.shape} or non-finite values")
    if outputs["mask_out"].dtype != np.bool_ or not outputs["mask_out"].all():
        raise ValueError(f"GVHMR B={batch}, L={frames} changed the complete sequence mask")
    if outputs["length_out"].dtype != np.int64 or not (outputs["length_out"] == frames).all():
        raise ValueError(f"GVHMR B={batch}, L={frames} changed the complete sequence length")


def verify_bit_equal(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray]) -> None:
    for name, reference in before.items():
        result = after[name]
        if result.dtype != reference.dtype or result.shape != reference.shape or result.tobytes() != reference.tobytes():
            raise ValueError(f"GVHMR ONNX mask rewrite changed {name} for output shape {reference.shape}")
