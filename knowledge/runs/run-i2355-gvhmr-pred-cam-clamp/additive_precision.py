"""Verify FP16 compute precision for GVHMR engines with additive attention masks.

``convert.verify_compute_precision`` recognises fused masked attention whose two
masks enter the kernel as Bool tensors.  With the additive-mask rewrite they
enter as FP16 bias tensors instead.  This wrapper removes exactly those two
bias inputs from each fused attention layer after checking their shapes, so
the shared verifier still requires FP16 Q, K and V operands.
"""

from __future__ import annotations

import copy

from convert import ONNX_LAYER, verify_compute_precision

FUSED_ATTENTION = ("_gemm_mha_v2_", "_gemv_mha_v1_")
DYNAMIC = -1
SMALL_K_TACTIC = "__myl_SlicSlicMulMulAdd_"
SMALL_K_OPERATION = "/network/learned_pos_linear/MatMul"


def _agree(*dims: int) -> bool:
    """Known dimensions agree; -1 is a dynamic-profile dimension."""
    return len({dim for dim in dims if dim != DYNAMIC}) <= 1


def _mask_biases(query, key, local, padding, value) -> bool:
    """Q (BH,T,D), K (BH,D,T), local (1,1,T,T), padding (B,1,1,T), V = Q."""
    if not all(len(shape) == 3 for shape in (query, key, value)) or not all(
        len(shape) == 4 for shape in (local, padding)
    ):
        return False
    heads_batch, tokens, width = query
    batch = padding[0]
    known_split = DYNAMIC in (heads_batch, batch) or (
        batch > 0 and heads_batch % batch == 0
    )
    return (
        value == query
        and local[:2] == (1, 1)
        and padding[1:3] == (1, 1)
        and _agree(heads_batch, key[0])
        and _agree(width, key[1])
        and _agree(tokens, key[2], local[2], local[3], padding[3])
        and known_split
    )


def strip_additive_masks(details: dict) -> tuple[dict, int]:
    stripped = copy.deepcopy(details)
    count = 0
    for layer in stripped["Layers"]:
        tactic = layer.get("TacticName") or ""
        if layer["LayerType"].lower() != "kgen" or not tactic.startswith(
            FUSED_ATTENTION
        ):
            continue
        inputs = layer.get("Inputs", [])
        if len(inputs) != 5:
            raise RuntimeError(
                f"Fused attention {layer['Name']} has {len(inputs)} inputs"
            )
        query, key, local, padding, value = (
            tuple(tensor["Dimensions"]) for tensor in inputs
        )
        if not _mask_biases(query, key, local, padding, value) or any(
            inputs[i]["Format/Datatype"] != "Half" for i in (2, 3)
        ):
            raise RuntimeError(
                f"Fused attention {layer['Name']} does not take two FP16 mask biases"
            )
        layer["Inputs"] = [inputs[0], inputs[1], inputs[4]]
        count += 1
    if not count:
        raise RuntimeError("No fused attention layer with additive masks was found")
    return stripped, count


def elementwise_small_k(details: dict) -> dict[str, str]:
    """Find K=2 MatMuls that TensorRT lowered to FP16 scalar multiply-adds.

    GVHMR's ``learned_pos_linear`` maps 2 keypoint coordinates to 32 channels.
    The dynamic-batch engine computes it as ``__myl_SlicSlicMulMulAdd_`` with
    FP16 inputs and output instead of a GEMM.  Only that exact layer shape is
    accepted, and the evidence names it.
    """
    found: dict[str, str] = {}
    for layer in details["Layers"]:
        tactic = layer.get("TacticName") or ""
        if not tactic.startswith(SMALL_K_TACTIC):
            continue
        operations = set(ONNX_LAYER.findall(layer.get("Metadata") or ""))
        inputs = layer.get("Inputs", [])
        outputs = layer.get("Outputs", [])
        if (
            layer["LayerType"].lower() != "kgen"
            or operations != {SMALL_K_OPERATION}
            or len(inputs) != 2
            or len(outputs) != 1
            or any(t["Format/Datatype"] != "Half" for t in (*inputs, *outputs))
            or any(t["Dimensions"][-1] != 2 for t in inputs)
            or outputs[0]["Dimensions"][-1] != 32
        ):
            raise RuntimeError(
                f"Layer {layer['Name']} is not the FP16 K=2 learned_pos_linear"
            )
        found[SMALL_K_OPERATION] = layer["Name"]
    return found


def verify_additive_attention(details, required_precision, *, expected_operations):
    stripped, count = strip_additive_masks(details)
    small_k = elementwise_small_k(details)
    if required_precision != "fp16" or any(
        expected_operations.get(op) != "fp16" for op in small_k
    ):
        raise RuntimeError("Elementwise K=2 evidence only covers FP16 operations")
    remaining = {
        op: precision
        for op, precision in expected_operations.items()
        if op not in small_k
    }
    evidence = verify_compute_precision(
        stripped, required_precision, expected_operations=remaining
    )
    evidence["additive_mask_attention_layers"] = count
    if small_k:
        evidence["elementwise_small_k_operations"] = small_k
    return evidence
